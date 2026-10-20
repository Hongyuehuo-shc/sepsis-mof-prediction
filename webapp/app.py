"""Local research-only Streamlit interface for the locked full CatBoost model."""
from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_builder import CATEGORICAL, UNITS, build_features, grouped_shap, template_csv  # noqa: E402

MODEL_PATH = PROJECT / "results/full_sepsis3/final_catboost/final_catboost_model.joblib"
MODEL_SHA256 = "1f55cdb89256f7ad4f7c0c74b191e29a0d24951f3faab4f248cea32280423321"


@st.cache_resource
def load_locked_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"找不到锁定模型：{MODEL_PATH}")
    digest = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    if digest != MODEL_SHA256:
        raise ValueError("模型文件哈希值与已验证版本不符，已停止预测。")
    model = joblib.load(MODEL_PATH)  # Only load this verified, trusted local artifact.
    if len(model.feature_names_in_) != 102:
        raise ValueError("模型输入列数不符，已停止预测。")
    return model


st.set_page_config(page_title="Sepsis-associated MOF | 研究演示", page_icon="🩺", layout="wide")
st.title("Sepsis-associated MOF：早期风险预测")
st.caption("锁定CatBoost完整模型 · MIMIC-IV开发 · eICU外部验证 · 仅供本机科研演示")
st.error("非医疗器械，未完成本地前瞻性验证；不得用于诊断、治疗决定或自动提醒。请勿上传含姓名、病历号等身份信息的文件。")

with st.expander("研究适用范围与结果含义", expanded=True):
    st.write("适用于研究方案中符合Sepsis-3、入ICU满24小时、且在预测时点尚未发生研究定义MOF的成人ICU住院。网页**不能独立验证这些入组条件**。模型输出的是此后48小时内发生研究定义MOF的估计概率，而非临床诊断。")
    st.write("内部锁定测试集AUROC 0.808；eICU外部验证AUROC 0.705。外部表现下降，固定阈值的敏感度亦下降，因此本网页不显示临床风险等级或治疗建议。")

try:
    model = load_locked_model()
except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

st.subheader("1. 准备观测记录")
st.write("上传**单名患者**入ICU后0–24小时的标准化CSV。每行是一条观测：`variable,hour,value,unit`。同一指标可有多次记录；尿量应为每次记录的**增量**，程序将求和。时间为入ICU后的小时数。")
st.download_button("下载不含患者信息的示例模板", template_csv().encode("utf-8-sig"),
                   file_name="single_patient_events_template.csv", mime="text/csv")
with st.expander("可用变量与规定单位"):
    st.dataframe(pd.DataFrame([{"variable": name, "unit": unit} for name, unit in UNITS.items()]),
                 hide_index=True, width="stretch")

encoder = model.named_steps["preprocess"].named_transformers_["categorical"].named_steps["onehot"]
categories = {name: list(values) for name, values in zip(CATEGORICAL, encoder.categories_)}

with st.form("single_patient_prediction"):
    uploaded = st.file_uploader("单名患者观测CSV", type="csv", help="仅在本机内存中处理，不保存上传文件。")
    st.subheader("2. 基本信息与入组确认")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("年龄（岁）", min_value=18, max_value=120, value=65, step=1)
        gender = st.selectbox("生理性别（原始数据库编码）", categories["gender"])
    with c2:
        first_careunit = st.selectbox("首次ICU类型", categories["first_careunit"])
        admission_type = st.selectbox("入院类型", categories["admission_type"])
    with c3:
        race = st.selectbox("种族/族裔（原始数据库编码）", categories["race"])
    eligible = st.checkbox("我已按研究方案人工核对：成人Sepsis-3、入ICU满24小时、预测时点尚无MOF，且CSV不含24小时之后数据。")
    submitted = st.form_submit_button("计算研究预测概率", type="primary")

if submitted:
    if uploaded is None:
        st.warning("请先上传CSV。")
        st.stop()
    if not eligible:
        st.warning("必须先人工确认研究入组条件；本网页不能自动判定Sepsis-3或已有MOF。")
        st.stop()
    try:
        if len(uploaded.getvalue()) > 5_000_000:
            raise ValueError("文件超过5 MB；单名患者的观测CSV不应如此大。")
        events = pd.read_csv(io.BytesIO(uploaded.getvalue()), encoding="utf-8-sig")
        metadata = {"age": age, "gender": gender, "first_careunit": first_careunit,
                    "admission_type": admission_type, "race": race}
        features, audit = build_features(events, metadata, list(model.feature_names_in_))
        probability = float(model.predict_proba(features)[0, 1])
        if not 0 <= probability <= 1:
            raise ValueError("预测概率超出有效范围。")
        explanation = grouped_shap(model, features)
    except (ValueError, KeyError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error(f"未进行预测：{exc}")
        st.stop()

    st.subheader("3. 研究预测结果")
    st.metric("未来48小时研究定义MOF的估计概率", f"{probability:.1%}")
    st.caption("此为原始锁定模型概率。不能据此单独判断个体患者是否会发生MOF。")
    st.write(f"输入记录：{audit.observation_count}条；测得变量：{audit.measured_variables}/{len(UNITS)}类。")
    if audit.missing_variables:
        st.warning("未测得的变量将由模型训练期确定的中位数及缺失指示机制处理：" + ", ".join(audit.missing_variables))
    if audit.measured_variables < 12:
        st.error("可用变量少于一半，预测可能不可靠；请核查数据提取与单位。")
    st.caption("网页未连接医院电子病历；CSV规范化和入组条件仍需人工核对。")

    st.subheader("4. 个体模型解释")
    top = explanation.head(12).sort_values("shap_log_odds")
    top = top.assign(direction=top.shap_log_odds.map(lambda x: "推高模型风险" if x > 0 else "降低模型风险"))
    chart = px.bar(top, x="shap_log_odds", y="variable", color="direction", orientation="h",
                   color_discrete_map={"推高模型风险": "#b53a45", "降低模型风险": "#2d6fa3"},
                   labels={"shap_log_odds": "SHAP贡献（原始模型log-odds）", "variable": "临床变量"})
    chart.update_layout(font_family="Arial", legend_title_text="")
    st.plotly_chart(chart, width="stretch")
    st.caption("SHAP说明模型如何使用这些数据，不代表病因、治疗效应或可干预阈值。")
    with st.expander("查看汇总后的102项模型输入（仅本机显示）"):
        display = features.T.reset_index()
        display.columns = ["feature", "value"]
        display["value"] = display["value"].map(lambda x: "缺失" if pd.isna(x) else str(x))
        st.dataframe(display, hide_index=True, width="stretch")

st.divider()
st.caption("版本：2026-09研究原型；仅加载本地锁定模型。上传数据只在当前Streamlit会话内处理，不写入磁盘。请勿将此程序部署到公网或接入真实临床工作流。")
