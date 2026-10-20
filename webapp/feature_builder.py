"""Strict, in-memory conversion of one patient's 0–24 h event CSV to model inputs."""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


# Canonical names and units mirror the existing trained MIMIC-IV feature pipeline.
UNITS = {
    "heart_rate": "beats/min", "map": "mmHg", "respiratory_rate": "breaths/min",
    "temperature": "°C", "spo2": "%", "gcs_total": "points",
    "wbc": "10³/µL", "platelet": "10³/µL", "hemoglobin": "g/dL",
    "hematocrit": "%", "rbc": "10⁶/µL", "rdw": "%",
    "creatinine": "mg/dL", "bun": "mg/dL", "sodium": "mmol/L",
    "potassium": "mmol/L", "bicarbonate": "mmol/L", "aniongap": "mmol/L",
    "glucose": "mg/dL", "lactate": "mmol/L", "ph": "unitless",
    "pao2fio2ratio": "mmHg", "pao2": "mmHg", "pco2": "mmHg",
    "urine_output": "mL",  # Individual incremental urine volumes, summed over 0–24 h.
}

# Wide plausibility bounds catch unit-entry errors; they are not clinical cutoffs.
BOUNDS = {
    "heart_rate": (20, 300), "map": (20, 250), "respiratory_rate": (4, 80),
    "temperature": (25, 45), "spo2": (40, 100), "gcs_total": (3, 15),
    "wbc": (0.1, 500), "platelet": (1, 2000), "hemoglobin": (1, 25),
    "hematocrit": (5, 80), "rbc": (0.5, 10), "rdw": (5, 35),
    "creatinine": (0, 30), "bun": (0, 250), "sodium": (90, 200),
    "potassium": (1, 10), "bicarbonate": (0, 60), "aniongap": (0, 60),
    "glucose": (20, 1500), "lactate": (0, 30), "ph": (6.5, 8),
    "pao2fio2ratio": (20, 1000), "pao2": (20, 800), "pco2": (10, 150),
    "urine_output": (0, 3000),
}

REQUIRED_COLUMNS = ("variable", "hour", "value", "unit")
STATS = ("min", "max", "mean", "last")
CATEGORICAL = ("first_careunit", "gender", "admission_type", "race")


@dataclass(frozen=True)
class FeatureAudit:
    measured_variables: int
    available_variables: tuple[str, ...]
    missing_variables: tuple[str, ...]
    observation_count: int


def template_csv() -> str:
    return "variable,hour,value,unit\nheart_rate,0,86,beats/min\nheart_rate,12,94,beats/min\nmap,1,72,mmHg\nplatelet,8,180,10³/µL\nurine_output,4,250,mL\nurine_output,12,300,mL\n"


def validate_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.columns.duplicated().any():
        raise ValueError("CSV中存在重复列名。")
    missing = set(REQUIRED_COLUMNS) - set(events.columns)
    extra = set(events.columns) - set(REQUIRED_COLUMNS)
    if missing or extra:
        raise ValueError(f"CSV列必须恰好为 {', '.join(REQUIRED_COLUMNS)}；缺少 {sorted(missing)}，多出 {sorted(extra)}。")
    if events.empty or len(events) > 100_000:
        raise ValueError("记录数必须为1–100,000行，且仅包含一名患者。")
    data = events.copy().reset_index(drop=True)
    data["variable"] = data["variable"].astype(str).str.strip()
    data["unit"] = data["unit"].astype(str).str.strip()
    unknown = sorted(set(data.variable) - set(UNITS))
    if unknown:
        raise ValueError(f"无法识别的变量：{unknown}。请使用模板中的标准英文名称。")
    bad_units = data.loc[data.apply(lambda row: row.unit != UNITS[row.variable], axis=1)]
    if not bad_units.empty:
        examples = bad_units[["variable", "unit"]].drop_duplicates().head(5).to_dict("records")
        raise ValueError(f"单位与模型定义不一致：{examples}。不自动换算单位。")
    for column in ("hour", "value"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
        if not np.isfinite(data[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{column}列含空值或非数值。")
    if data.hour.lt(0).any() or data.hour.gt(24).any():
        raise ValueError("hour必须是入ICU后0–24小时；不能使用预测时点之后的数据。")
    outside = data.apply(lambda row: not BOUNDS[row.variable][0] <= row.value <= BOUNDS[row.variable][1], axis=1)
    if outside.any():
        examples = data.loc[outside, ["variable", "hour", "value", "unit"]].head(5).to_dict("records")
        raise ValueError(f"数值超出宽松的录入核查范围：{examples}。请核对单位和记录。")
    # Stable sort preserves uploaded order for tied observation times.
    return data.sort_values("hour", kind="stable").reset_index(drop=True)


def build_features(events: pd.DataFrame, metadata: dict, model_features: list[str]) -> tuple[pd.DataFrame, FeatureAudit]:
    data = validate_events(events)
    age = float(metadata.get("age", np.nan))
    if not np.isfinite(age) or not 18 <= age <= 120:
        raise ValueError("年龄须为18–120岁。")
    for field in CATEGORICAL:
        if not isinstance(metadata.get(field), str) or not metadata[field].strip():
            raise ValueError(f"缺少分类变量：{field}。")
    result: dict[str, object] = {feature: np.nan for feature in model_features}
    result["age"] = age
    for field in CATEGORICAL:
        result[field] = metadata[field].strip()

    measured = set(data.variable)
    for variable, group in data.groupby("variable", sort=False):
        values = group.value.to_numpy(dtype=float)
        if variable == "urine_output":
            result["urine_output_24h"] = float(values.sum())
        else:
            stats = {"min": float(values.min()), "max": float(values.max()),
                     "mean": float(values.mean()), "last": float(values[-1])}
            for stat, value in stats.items():
                column = f"{variable}_{stat}"
                if column in result:
                    result[column] = value
    missing = tuple(sorted(set(UNITS) - measured))
    audit = FeatureAudit(len(measured), tuple(sorted(measured)), missing, len(data))
    return pd.DataFrame([result], columns=model_features), audit


def grouped_shap(model, features: pd.DataFrame) -> pd.DataFrame:
    """Return feature contributions on the raw model log-odds scale."""
    from catboost import Pool

    transformed = model.named_steps["preprocess"].transform(features)
    catboost = model.named_steps["model"]
    values = catboost.get_feature_importance(Pool(transformed), type="ShapValues")[0]
    names = model.named_steps["preprocess"].get_feature_names_out()
    if len(values) != len(names) + 1:
        raise ValueError("SHAP贡献维度与模型预处理列不一致。")
    grouped: dict[str, float] = {}
    for name, contribution in zip(names, values[:-1]):
        name = str(name)
        if name.startswith("numeric__"):
            base = name.removeprefix("numeric__").removeprefix("missingindicator_")
            group = re.sub(r"_(min|max|mean|last)$", "", base)
        else:
            base = name.removeprefix("categorical__")
            group = next((field for field in CATEGORICAL if base.startswith(field + "_")), base)
        grouped[group] = grouped.get(group, 0.0) + float(contribution)
    frame = pd.DataFrame({"variable": list(grouped), "shap_log_odds": list(grouped.values())})
    frame["absolute_contribution"] = frame.shap_log_odds.abs()
    return frame.sort_values("absolute_contribution", ascending=False).reset_index(drop=True)
