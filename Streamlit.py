import streamlit as st
import catboost as cb
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# 加载模型
CAT_model = joblib.load('Catboost.pkl')

# 创建SHAP解释器
explainer = shap.TreeExplainer(CAT_model)

# Streamlit 用户界面
# 添加团队logo
st.image("jsszyylogo.png", width=50)  # 更改url_to_your_logo.png为你的logo图片链接，调整width为适当的大小

# 使用Markdown来定制标题的字体大小
st.markdown('<h1 style="font-size:36px;">“通督调神”针法治疗失眠症疗效预测</h1>', unsafe_allow_html=True)
# 添加团队logo和标题
best_threshold = 0.72  # 这是你确定的最佳阈值

# 创建列布局
col1, col2 = st.columns(2)
with col1:
    DUR = st.number_input("病程（月）:", min_value=0.0, max_value=500.0, value=1.0)
    LPRDR = st.number_input("记录期间最低脉率（次/分钟）:", min_value=0.0, max_value=200.0, value=1.0)
    DOM = st.number_input("微觉醒持续时间（分钟）:", min_value=0.0, max_value=100.0, value=1.0)
    MAPTS = st.number_input("微觉醒占总睡眠时长比例（%）:", min_value=0.0, max_value=100.0, value=1.0)
    SS20 = st.selectbox("是否口臭:", options=["否", "是"], index=0)
    
with col2:
    N1LOL = st.number_input("自关灯起的N1期潜伏期（分钟）:", min_value=0.0, max_value=500.0, value=1.0)
    N1P = st.number_input("N1期占总睡眠时长比例（%）:", min_value=0.0, max_value=100.0, value=1.0)
    N3P = st.number_input("N3期占总睡眠时长比例（%）:", min_value=0.0, max_value=100.0, value=1.0)
    RP = st.number_input("REM期占总睡眠时长比例（%）:", min_value=0.0, max_value=100.0, value=1.0)
    SS31 = st.selectbox("是否乳房胀痛:", options=["否", "是"], index=0)
    
DI17 = st.selectbox("是否患有胃炎:", options=["否", "是"], index=0)

# 进行预测
if st.button("预测"):
    feature_values = [DUR, LPRDR, N1P, 1 if SS20 == "是" else 0, N1LOL, N3P, DOM, 1 if SS31 == "是" else 0, MAPTS, RP, 1 if DI17 == "是" else 0]
    feature_names = ["DUR", "LPRDR", "N1P", "SS20", "N1LOL", "N3P", "DOM", "SS31", "MAPTS", "RP", "DI17"]
    prediction_proba = CAT_model.predict_proba([feature_values])[0, 1]
    st.write(f"该患者经“通督调神”针法治疗后PSQI减分率≥50%的概率: {prediction_proba:.2%}")
    
    if prediction_proba >= best_threshold:
        st.write("该患者可能经“通督调神”针法治疗后显效")
    else:
        st.write("该患者可能经“通督调神”针法治疗后不显效")

    # 计算 SHAP 值并生成力图
    shap_values = explainer.shap_values([feature_values])
    plt.figure(figsize=(12, 4))  # 调整图的尺寸
    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True, show=False)
    plt.savefig("shap_force_plot.png", bbox_inches='tight')
    st.image("shap_force_plot.png")
