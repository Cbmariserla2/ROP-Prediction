"""
streamlit_app.py
-----------------
Interactive dashboard for the ROP Prediction project.

Run with:
    streamlit run app/streamlit_app.py

Features:
- Model performance comparison (R2, RMSE, MAE)
- Predicted vs Actual ROP scatter (interactive, per model)
- Feature importance chart
- "What-if" ROP predictor: adjust drilling parameters with sliders and
  get a live predicted ROP from the trained XGBoost model.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

BASE = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE / "models"
REPORTS_DIR = BASE / "reports"
DATA_PATH = BASE / "data" / "processed" / "drilling_data_clean.csv"

st.set_page_config(page_title="ROP Prediction Dashboard", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    metrics = pd.read_csv(REPORTS_DIR / "model_metrics.csv")
    preds = pd.read_csv(REPORTS_DIR / "predictions_vs_actual.csv")
    importance = pd.read_csv(REPORTS_DIR / "feature_importance.csv", index_col=0)
    return df, metrics, preds, importance


@st.cache_resource
def load_models():
    xgb_model = joblib.load(MODELS_DIR / "xgboost.joblib")
    rf_model = joblib.load(MODELS_DIR / "random_forest.joblib")
    lr_model = joblib.load(MODELS_DIR / "linear_regression.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    with open(MODELS_DIR / "feature_names.json") as f:
        feature_names = json.load(f)
    return xgb_model, rf_model, lr_model, scaler, feature_names


df, metrics, preds, importance = load_data()
xgb_model, rf_model, lr_model, scaler, feature_names = load_models()

st.title("🛢️ Rate of Penetration (ROP) Prediction Dashboard")
st.caption(
    "Drilling optimization dashboard built on drilling parameters "
    "(WOB, RPM, flow rate, torque, mud weight, depth, formation)."
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Model Performance", "🎯 Predicted vs Actual", "🔑 Feature Importance", "🧮 What-If Predictor"]
)

# ---------------- Tab 1: Model performance ----------------
with tab1:
    st.subheader("Model Comparison")
    st.dataframe(metrics.set_index("model"), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(metrics, x="model", y="R2", color="model", title="R² by Model", range_y=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(metrics, x="model", y="RMSE", color="model", title="RMSE by Model")
        st.plotly_chart(fig, use_container_width=True)

# ---------------- Tab 2: Predicted vs actual ----------------
with tab2:
    st.subheader("Predicted vs Actual ROP")
    model_choice = st.selectbox("Choose a model", ["XGBoost", "Random Forest", "Linear Regression"])
    fig = px.scatter(
        preds, x="y_test", y=model_choice, opacity=0.3,
        labels={"y_test": "Actual ROP (m/hr)", model_choice: "Predicted ROP (m/hr)"},
        title=f"{model_choice}: Predicted vs Actual ROP",
    )
    min_v, max_v = preds["y_test"].min(), preds["y_test"].max()
    fig.add_shape(type="line", x0=min_v, y0=min_v, x1=max_v, y1=max_v,
                  line=dict(color="red", dash="dash"))
    st.plotly_chart(fig, use_container_width=True)

    residuals = preds["y_test"] - preds[model_choice]
    fig2 = px.histogram(residuals, nbins=50, title=f"{model_choice} Residual Distribution")
    st.plotly_chart(fig2, use_container_width=True)

# ---------------- Tab 3: Feature importance ----------------
with tab3:
    st.subheader("Which parameters influence ROP the most?")
    imp_long = importance.reset_index().melt(id_vars="index", var_name="Model", value_name="Importance")
    imp_long = imp_long.rename(columns={"index": "Feature"})
    fig = px.bar(
        imp_long, x="Importance", y="Feature", color="Model", orientation="h",
        barmode="group", title="Feature Importance: Random Forest vs XGBoost",
        height=600,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        "Formation type and Mechanical Specific Energy (MSE) dominate importance "
        "in this dataset — consistent with drilling-engineering intuition that "
        "rock drillability and drilling efficiency drive ROP more than any single "
        "surface parameter alone."
    )

# ---------------- Tab 4: What-if predictor ----------------
with tab4:
    st.subheader("Interactive ROP Predictor (XGBoost)")
    st.write("Adjust drilling parameters to see the predicted ROP update live.")

    formation_cols = [c for c in feature_names if c.startswith("Formation_")]
    formation_options = [c.replace("Formation_", "") for c in formation_cols]

    c1, c2, c3 = st.columns(3)
    with c1:
        wob = st.slider("Weight on Bit (klbs)", 4.0, 45.0, 20.0)
        rpm = st.slider("RPM", 40.0, 220.0, 110.0)
        flow_rate = st.slider("Flow Rate (gpm)", 250.0, 900.0, 500.0)
    with c2:
        torque = st.slider("Torque (kft-lbs)", 0.5, 28.0, 8.0)
        mud_weight = st.slider("Mud Weight (ppg)", 8.5, 16.5, 10.5)
        depth = st.slider("Depth (m)", 1500.0, 5200.0, 2800.0)
    with c3:
        bit_hours = st.slider("Bit Hours (wear)", 0.0, 40.0, 10.0)
        formation = st.selectbox("Formation", formation_options)

    # Approximate MSE the same way as in preprocessing
    bit_diameter_in = 8.5
    area_in2 = np.pi * (bit_diameter_in / 2) ** 2
    assumed_rop_for_mse = 15.0  # seed estimate; MSE depends circularly on ROP in reality
    mse = (wob * 1000 / area_in2) + (120 * np.pi * rpm * torque * 1000) / (area_in2 * assumed_rop_for_mse)

    input_row = {col: 0 for col in feature_names}
    input_row.update({
        "WOB_klbs": wob, "RPM": rpm, "Flow_Rate_gpm": flow_rate,
        "Torque_kftlbs": torque, "Mud_Weight_ppg": mud_weight,
        "Depth_m": depth, "Bit_Hours": bit_hours, "MSE_psi": mse,
        f"Formation_{formation}": 1,
    })
    input_df = pd.DataFrame([input_row])[feature_names]

    predicted_rop = xgb_model.predict(input_df)[0]
    st.metric("Predicted ROP (XGBoost)", f"{predicted_rop:.2f} m/hr")

    st.caption(
        "Note: MSE is approximated using an assumed baseline ROP since it is "
        "normally computed *from* ROP; treat this predictor as illustrative "
        "of parameter sensitivity rather than a precision drilling-advisory tool."
    )

st.divider()
st.caption(
    "Built on a synthetic, physics-informed drilling dataset styled after the Volve field "
    "(see README for how to plug in the real Equinor Volve dataset)."
)
