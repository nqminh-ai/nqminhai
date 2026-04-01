import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ─── Config ───────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide"
)

# ─── Load model ───────────────────────────────────
@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model('models/xgb_tuned.json')
    with open('models/best_threshold.json') as f:
        threshold = json.load(f)['threshold']
    return model, threshold

@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)

model, threshold = load_model()
explainer = load_explainer(model)

# ─── Load sample data ─────────────────────────────
@st.cache_data
def load_sample():
    df = pd.read_csv('data/test_final.csv')
    return df

test_df = load_sample()
feature_cols = [c for c in test_df.columns if c != 'isFraud']

# ─── Header ───────────────────────────────────────
st.title("🔍 Real-Time Fraud Detection System")
st.markdown("**AI Graduation Thesis** — XGBoost + SHAP Explainability")
st.divider()

# ─── Sidebar ──────────────────────────────────────
st.sidebar.header("⚙️ Cài đặt")
st.sidebar.metric("Model", "XGBoost Tuned")
st.sidebar.metric("Threshold", f"{threshold:.2f}")
st.sidebar.metric("F1 Score", "0.50")
st.sidebar.metric("AUPRC", "0.5813")
st.sidebar.metric("ROC-AUC", "0.8918")

# ─── Main tabs ────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔴 Predict Single Transaction",
    "📊 Batch Prediction",
    "📈 Model Performance"
])

# ══════════════════════════════════════════════════
# TAB 1 — Single Transaction
# ══════════════════════════════════════════════════
with tab1:
    st.subheader("Nhập thông tin giao dịch")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        trans_amt = st.number_input("💰 Số tiền giao dịch (USD)", 
                                     min_value=0.0, max_value=10000.0, 
                                     value=390.0, step=10.0)
        card1 = st.number_input("💳 Card1 ID", 
                                 min_value=0, max_value=20000, value=11862)
        card6 = st.selectbox("💳 Card6 Type", [0, 1, 2, 3, 4], index=1)
    
    with col2:
        c1 = st.number_input("C1", min_value=0, max_value=100, value=1)
        c2 = st.number_input("C2", min_value=0, max_value=100, value=1)
        c4 = st.number_input("C4", min_value=0, max_value=100, value=0)
    
    with col3:
        card1_freq = st.number_input("🔄 Tần suất giao dịch card", 
                                      min_value=1, max_value=500, value=6)
        hour = st.slider("🕐 Giờ giao dịch", 0, 23, 10)
        dayofweek = st.slider("📅 Thứ trong tuần", 1, 7, 2)

    predict_btn = st.button("🚀 Dự đoán", type="primary", use_container_width=True)
    
    if predict_btn:
        # Tạo input từ sample giao dịch thật
        sample = test_df[feature_cols].iloc[0].copy()
        
        # Override bằng giá trị user nhập
        sample['TransactionAmt'] = trans_amt
        sample['card1'] = card1
        sample['card6'] = card6
        sample['C1'] = c1
        sample['C2'] = c2
        sample['C4'] = c4
        sample['card1_freq'] = card1_freq
        sample['hour'] = hour
        sample['dayofweek'] = dayofweek
        sample['card1_avg_amt'] = trans_amt * 0.8
        sample['amt_diff_from_avg'] = trans_amt - trans_amt * 0.8
        
        X_input = pd.DataFrame([sample])
        prob = model.predict_proba(X_input)[0][1]
        pred = int(prob >= threshold)
        
        st.divider()
        
        # Kết quả
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            if pred == 1:
                st.error("🚨 FRAUD DETECTED!")
            else:
                st.success("✅ LEGITIMATE")
        with col_r2:
            st.metric("Xác suất Fraud", f"{prob*100:.1f}%")
        with col_r3:
            risk = "🔴 Cao" if prob > 0.7 else "🟡 Trung bình" if prob > threshold else "🟢 Thấp"
            st.metric("Mức độ rủi ro", risk)
        
        # SHAP explanation
        st.subheader("🔍 SHAP Explanation — Tại sao model đưa ra kết quả này?")
        
        shap_vals = explainer.shap_values(X_input)
        
        explanation = shap.Explanation(
            values=shap_vals[0],
            base_values=explainer.expected_value,
            data=X_input.iloc[0],
            feature_names=feature_cols
        )
        
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.waterfall_plot(explanation, max_display=12, show=False)
        plt.title(f"SHAP Waterfall — {'FRAUD' if pred==1 else 'LEGIT'} (prob={prob:.3f})")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ══════════════════════════════════════════════════
# TAB 2 — Batch Prediction
# ══════════════════════════════════════════════════
with tab2:
    st.subheader("Dự đoán hàng loạt từ file CSV")
    
    uploaded = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded:
        df_upload = pd.read_csv(uploaded)
        st.write(f"File có {len(df_upload)} giao dịch")
        st.dataframe(df_upload.head())
        
        if st.button("🚀 Predict tất cả", type="primary"):
            # Align columns
            missing_cols = set(feature_cols) - set(df_upload.columns)
            for col in missing_cols:
                df_upload[col] = 0
            
            X_batch = df_upload[feature_cols]
            probs = model.predict_proba(X_batch)[:, 1]
            preds = (probs >= threshold).astype(int)
            
            df_upload['fraud_probability'] = probs.round(4)
            df_upload['prediction'] = preds
            df_upload['result'] = df_upload['prediction'].map({0: '✅ Legit', 1: '🚨 Fraud'})
            
            n_fraud = preds.sum()
            st.error(f"🚨 Phát hiện {n_fraud} giao dịch gian lận / {len(preds)} tổng!")
            st.dataframe(df_upload[['fraud_probability', 'prediction', 'result']].head(20))
            
            csv = df_upload.to_csv(index=False)
            st.download_button("📥 Download kết quả", csv, 
                             "fraud_predictions.csv", "text/csv")
    else:
        # Demo với test data
        if st.button("🎯 Demo với 20 giao dịch mẫu"):
            sample_20 = test_df[feature_cols].head(20)
            probs = model.predict_proba(sample_20)[:, 1]
            preds = (probs >= threshold).astype(int)
            
            result_df = pd.DataFrame({
                'TransactionAmt': sample_20['TransactionAmt'].values,
                'card1': sample_20['card1'].values,
                'Fraud Probability': probs.round(4),
                'Prediction': ['🚨 Fraud' if p==1 else '✅ Legit' for p in preds]
            })
            
            st.dataframe(result_df, use_container_width=True)
            st.metric("Fraud phát hiện", f"{preds.sum()}/20 giao dịch")

# ══════════════════════════════════════════════════
# TAB 3 — Model Performance
# ══════════════════════════════════════════════════
with tab3:
    st.subheader("📊 So sánh hiệu suất các model")
    
    results = pd.DataFrame([
        {'Model': 'Logistic Regression', 'F1': 0.1074, 'AUPRC': 0.0830, 'AUC': 0.6414},
        {'Model': 'LSTM',               'F1': 0.2388, 'AUPRC': 0.2229, 'AUC': 0.6694},
        {'Model': 'Decision Tree',      'F1': 0.4421, 'AUPRC': 0.2236, 'AUC': 0.7415},
        {'Model': 'LightGBM',           'F1': 0.3958, 'AUPRC': 0.5574, 'AUC': 0.8543},
        {'Model': 'XGBoost Default',    'F1': 0.3579, 'AUPRC': 0.5649, 'AUC': 0.8812},
        {'Model': 'XGBoost Tuned',      'F1': 0.4554, 'AUPRC': 0.5813, 'AUC': 0.8918},
        {'Model': '🏆 XGBoost+Threshold','F1': 0.5000, 'AUPRC': 0.5813, 'AUC': 0.8918},
    ])
    
    st.dataframe(results, use_container_width=True, hide_index=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image('reports/shap_summary.png', caption='SHAP Summary Plot')
    with col2:
        st.image('reports/confusion_matrix.png', caption='Confusion Matrix')
    
    st.image('reports/drift_simulation.png', caption='Concept Drift Simulation')