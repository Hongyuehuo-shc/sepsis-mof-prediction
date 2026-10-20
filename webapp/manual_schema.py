"""User-facing definitions for the independently trained manual-entry model."""
from __future__ import annotations

from feature_builder import BOUNDS

FIELDS = [
    ("heart_rate_last", "心率", "beats/min", "生命体征"),
    ("map_last", "平均动脉压", "mmHg", "生命体征"),
    ("respiratory_rate_last", "呼吸频率", "breaths/min", "生命体征"),
    ("temperature_last", "体温", "°C", "生命体征"),
    ("spo2_last", "指脉氧饱和度", "%", "生命体征"),
    ("gcs_total_last", "GCS总分", "points", "生命体征"),
    ("wbc_last", "白细胞计数", "10³/µL", "血常规"),
    ("platelet_last", "血小板计数", "10³/µL", "血常规"),
    ("hemoglobin_last", "血红蛋白", "g/dL", "血常规"),
    ("rdw_last", "红细胞分布宽度", "%", "血常规"),
    ("creatinine_last", "肌酐", "mg/dL", "生化"),
    ("bun_last", "尿素氮", "mg/dL", "生化"),
    ("sodium_last", "钠", "mmol/L", "生化"),
    ("potassium_last", "钾", "mmol/L", "生化"),
    ("bicarbonate_last", "碳酸氢根", "mmol/L", "生化"),
    ("glucose_last", "血糖", "mg/dL", "生化"),
    ("lactate_last", "乳酸", "mmol/L", "血气与氧合"),
    ("ph_last", "动脉血pH", "unitless", "血气与氧合"),
    ("pao2fio2ratio_last", "PaO₂/FiO₂比值", "mmHg", "血气与氧合"),
    ("pco2_last", "PaCO₂", "mmHg", "血气与氧合"),
    ("urine_output_24h", "前24小时总尿量", "mL", "尿量"),
]

FEATURES = ["age", "gender", *[field[0] for field in FIELDS]]


def bounds_for(feature: str) -> tuple[float, float]:
    if feature == "urine_output_24h":
        return 0.0, 20_000.0
    base = feature.removesuffix("_last")
    return tuple(float(x) for x in BOUNDS[base])


def validate_manual_values(age: int | None, gender: str, values: dict[str, float | None]) -> dict[str, object]:
    if age is None or not 18 <= age <= 120:
        raise ValueError("请填写18–120岁的年龄。")
    if gender not in {"F", "M"}:
        raise ValueError("请选择患者性别。")
    if set(values) != {field[0] for field in FIELDS}:
        raise ValueError("输入字段与已验证的手工录入模型不一致。")
    row: dict[str, object] = {"age": int(age), "gender": gender}
    supplied = 0
    for field, value in values.items():
        if value is None:
            row[field] = float("nan")
            continue
        low, high = bounds_for(field)
        if not low <= float(value) <= high:
            raise ValueError(f"{field}超出宽松的录入核查范围，请核对单位。")
        row[field] = float(value)
        supplied += 1
    if supplied < 10:
        raise ValueError("至少填写10项生命体征或检验结果后才能演示预测；请勿用大量空白字段推断个体风险。")
    return row
