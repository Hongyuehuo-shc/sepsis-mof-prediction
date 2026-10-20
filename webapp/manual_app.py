"""Direct hand-entry risk calculator for the exploratory single-value model."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_builder import grouped_shap  # noqa: E402
from manual_schema import FEATURES, FIELDS, bounds_for, validate_manual_values  # noqa: E402

MODEL_PATH = PROJECT / "results/manual_entry_model/manual_entry_catboost.joblib"
MODEL_SHA256 = "72a3bf41665d2ddccb03ee3923d4f292fbc191a1e1eca56e7d7c7ed6a90509ef"


@st.cache_resource
def load_manual_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("未找到手工录入模型，请先运行研究脚本训练模型。")
    if hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() != MODEL_SHA256:
        raise ValueError("模型文件与已验证版本不符，已停止预测。")
    model = joblib.load(MODEL_PATH)  # Trusted, hash-checked local artifact only.
    if list(model.feature_names_in_) != FEATURES:
        raise ValueError("模型输入顺序或字段不一致，已停止预测。")
    return model


st.set_page_config(page_title="MOF风险概率｜手工输入研究原型", page_icon="🩺", layout="wide")
st.title("脓毒症相关多器官衰竭风险预测")
st.caption("直接填写检查结果 · 23项输入 · 研究演示版")
st.error("仅供本机科研演示。不是临床诊断工具，不得据此作出治疗决定或自动提醒。请勿输入姓名、病历号等身份信息。")
st.write("请填写**入ICU满24小时之前，各指标最近一次检查值**；尿量填写前24小时累计总量。未检测的指标可以留空。页面估计的是此后48小时内发生本研究定义MOF的概率。")

with st.expander("研究证据和适用范围", expanded=True):
    st.write("该23项手工录入模型是在原研究的MIMIC-IV开发数据上重新训练的探索性模型，**不是论文主模型**。MIMIC-IV既往测试集AUROC为0.783；eICU既往外部数据AUROC为0.700。两套数据此前已用于主模型分析，不能把本次比较称为全新的独立确认性验证。")
    st.write("仅适用于已按研究方案确认Sepsis-3、入ICU满24小时且预测时点未发生MOF的成人住院。网页无法自动确认入组条件，且未经本院前瞻性和临床流程验证。")

try:
    model = load_manual_model()
except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

with st.form("manual_risk_form"):
    st.subheader("患者基本信息")
    left, right = st.columns(2)
    with left:
        age = st.number_input("年龄（岁）", min_value=18, max_value=120, value=None,
                              step=1, placeholder="必填", key="age")
    with right:
        gender = st.selectbox("生理性别", ["请选择", "F", "M"], key="gender")

    entered: dict[str, float | None] = {}
    for section in ("生命体征", "血常规", "生化", "血气与氧合", "尿量"):
        st.subheader(section)
        fields = [item for item in FIELDS if item[3] == section]
        columns = st.columns(3)
        for index, (feature, label, unit, _) in enumerate(fields):
            low, high = bounds_for(feature)
            with columns[index % 3]:
                entered[feature] = st.number_input(
                    f"{label}（{unit}）", min_value=low, max_value=high,
                    value=None, step=1.0 if feature == "urine_output_24h" else 0.1,
                    format="%.0f" if feature == "urine_output_24h" else "%.2f",
                    placeholder="未检测可留空", key=feature,
                    help="除尿量外，请输入入ICU后24小时内最近一次测量值；单位必须与标签一致。",
                )
    eligible = st.checkbox("我已人工核对：患者符合研究中的Sepsis-3与24小时预测时点条件，且此前尚未发生MOF。", key="eligible")
    submitted = st.form_submit_button("计算MOF风险概率", type="primary")

if submitted:
    if not eligible:
        st.warning("请先人工确认研究适用条件。")
        st.stop()
    try:
        row = validate_manual_values(age, gender, entered)
        frame = pd.DataFrame([row], columns=FEATURES)
        probability = float(model.predict_proba(frame)[0, 1])
        if not 0 <= probability <= 1:
            raise ValueError("模型未返回有效概率。")
        explanation = grouped_shap(model, frame)
    except (ValueError, KeyError) as exc:
        st.error(f"暂不能计算：{exc}")
        st.stop()

    st.subheader("研究预测结果")
    st.metric("未来48小时发生研究定义MOF的估计概率", f"{probability:.1%}")
    filled = sum(value is not None for value in entered.values())
    st.caption(f"已输入{filled}/{len(FIELDS)}项检查结果；其余字段按训练期缺失值处理。此为原始模型概率，不是确诊或临床风险等级。")
    if filled < 15:
        st.warning("部分检查项目未提供。虽然模型允许缺失，但当前缺失模式的个体预测可靠性未经单独验证。")
    st.subheader("哪些输入影响了这次预测？")
    top = explanation.head(10).sort_values("shap_log_odds")
    top = top.assign(direction=top.shap_log_odds.map(lambda x: "推高模型风险" if x > 0 else "降低模型风险"))
    chart = px.bar(top, x="shap_log_odds", y="variable", color="direction", orientation="h",
                   color_discrete_map={"推高模型风险": "#b53a45", "降低模型风险": "#2d6fa3"},
                   labels={"shap_log_odds": "SHAP贡献（模型log-odds）", "variable": "输入变量"})
    chart.update_layout(font_family="Arial", legend_title_text="")
    st.plotly_chart(chart, width="stretch")
    st.caption("SHAP表示模型计算中的贡献，不等于病因、治疗效果或指标之间的比值。")

st.divider()
st.caption("本页面只在当前电脑运行，输入值不写入磁盘，也不上传公网。任何实际临床应用均需额外的本地/前瞻性验证、伦理治理与安全评估。")
