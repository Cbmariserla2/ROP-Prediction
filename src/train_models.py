import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

BASE = Path(__file__).resolve().parents[1]
PROCESSED_PATH = BASE / "data" / "processed" / "drilling_data_clean.csv"
MODELS_DIR = BASE / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = BASE / "reports"

FEATURE_COLS = [
    "WOB_klbs", "RPM", "Flow_Rate_gpm", "Torque_kftlbs",
    "Mud_Weight_ppg", "Depth_m", "Bit_Hours", "MSE_psi",
]
TARGET_COL = "ROP_m_hr"
RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(PROCESSED_PATH)
    formation_cols = [c for c in df.columns if c.startswith("Formation_") and c != "Formation_Label"]
    features = FEATURE_COLS + formation_cols
    X = df[features]
    y = df[TARGET_COL]
    return X, y, features


def evaluate(model, X_test, y_test, name):
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    print(f"[{name}] R2={r2:.4f}  RMSE={rmse:.3f}  MAE={mae:.3f}")
    return {"model": name, "R2": round(r2, 4), "RMSE": round(rmse, 3), "MAE": round(mae, 3)}, preds


def main():
    X, y, feature_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Scale features for Linear Regression (tree models don't need it,
    # but we keep one shared scaler for consistency / the dashboard)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []
    predictions = {"y_test": y_test.reset_index(drop=True)}

    # --- 1. Linear Regression ---
    
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    res, preds = evaluate(lr, X_test_scaled, y_test, "Linear Regression")
    results.append(res)
    predictions["Linear Regression"] = preds
    joblib.dump(lr, MODELS_DIR / "linear_regression.joblib")

    # --- 2. Random Forest (small grid search) ---

    # min_samples_leaf is capped to keep tree sizes (and the saved model
    # file) reasonable while still getting strong accuracy.
    rf_param_grid = {
        "n_estimators": [150, 300],
        "max_depth": [10, 15],
    }
    rf_base = RandomForestRegressor(
        random_state=RANDOM_STATE, n_jobs=-1, min_samples_leaf=5
    )
    rf_search = GridSearchCV(rf_base, rf_param_grid, cv=2, scoring="r2", n_jobs=-1, verbose=1)
    rf_search.fit(X_train, y_train)
    rf = rf_search.best_estimator_
    print("Best RF params:", rf_search.best_params_)
    res, preds = evaluate(rf, X_test, y_test, "Random Forest")
    results.append(res)
    predictions["Random Forest"] = preds
    joblib.dump(rf, MODELS_DIR / "random_forest.joblib")

    # --- 3. XGBoost (small grid search) ---
    xgb_param_grid = {
        "n_estimators": [150, 300],
        "max_depth": [4, 6],
        "learning_rate": [0.1],
    }
    xgb_base = XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, objective="reg:squarederror")
    xgb_search = GridSearchCV(xgb_base, xgb_param_grid, cv=2, scoring="r2", n_jobs=-1, verbose=1)
    xgb_search.fit(X_train, y_train)
    xgb_model = xgb_search.best_estimator_
    print("Best XGB params:", xgb_search.best_params_)
    res, preds = evaluate(xgb_model, X_test, y_test, "XGBoost")
    results.append(res)
    predictions["XGBoost"] = preds
    joblib.dump(xgb_model, MODELS_DIR / "xgboost.joblib")

    # Save scaler + feature list for downstream use (dashboard)
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    with open(MODELS_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    # Save metrics
    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(REPORTS_DIR / "model_metrics.csv", index=False)
    print("\n=== Final Metrics ===")
    print(metrics_df.to_string(index=False))

    # Save predictions vs actual for the dashboard / plotting
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(REPORTS_DIR / "predictions_vs_actual.csv", index=False)

    # Feature importance (Random Forest & XGBoost)
    rf_importance = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
    xgb_importance = pd.Series(xgb_model.feature_importances_, index=feature_names).sort_values(ascending=False)
    importance_df = pd.DataFrame({
        "Random_Forest_Importance": rf_importance,
        "XGBoost_Importance": xgb_importance,
    }).fillna(0)
    importance_df.to_csv(REPORTS_DIR / "feature_importance.csv")
    print("\nSaved feature_importance.csv, model_metrics.csv, predictions_vs_actual.csv")


if __name__ == "__main__":
    main()
