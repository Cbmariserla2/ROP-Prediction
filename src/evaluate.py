import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
REPORTS_DIR = BASE / "reports"
FIG_DIR = REPORTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def main():
    metrics = pd.read_csv(REPORTS_DIR / "model_metrics.csv")
    preds = pd.read_csv(REPORTS_DIR / "predictions_vs_actual.csv")
    importance = pd.read_csv(REPORTS_DIR / "feature_importance.csv", index_col=0)

    model_names = ["Linear Regression", "Random Forest", "XGBoost"]

    # 1. Metrics bar chart (R2 and RMSE side by side)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(data=metrics, x="model", y="R2", ax=axes[0], palette="crest")
    axes[0].set_title("Model Comparison: R² (higher is better)")
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("")
    sns.barplot(data=metrics, x="model", y="RMSE", ax=axes[1], palette="flare")
    axes[1].set_title("Model Comparison: RMSE (lower is better)")
    axes[1].set_xlabel("")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "07_model_metrics_comparison.png", dpi=150)
    plt.close()

    # 2. Predicted vs Actual scatter, one panel per model
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    for ax, name in zip(axes, model_names):
        ax.scatter(preds["y_test"], preds[name], s=4, alpha=0.25, color="teal")
        lims = [preds["y_test"].min(), preds["y_test"].max()]
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_title(name)
        ax.set_xlabel("Actual ROP (m/hr)")
    axes[0].set_ylabel("Predicted ROP (m/hr)")
    plt.suptitle("Predicted vs Actual ROP")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "08_predicted_vs_actual.png", dpi=150)
    plt.close()

    # 3. Residuals for best model (XGBoost)
    residuals = preds["y_test"] - preds["XGBoost"]
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, bins=50, kde=True, color="purple")
    plt.title("XGBoost Residual Distribution (Actual - Predicted)")
    plt.xlabel("Residual (m/hr)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "09_xgboost_residuals.png", dpi=150)
    plt.close()

    # 4. Feature importance (XGBoost, sorted)
    imp_sorted = importance["XGBoost_Importance"].sort_values(ascending=True)
    plt.figure(figsize=(8, 6))
    plt.barh(imp_sorted.index, imp_sorted.values, color="darkcyan")
    plt.title("Feature Importance (XGBoost) for ROP Prediction")
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "10_feature_importance.png", dpi=150)
    plt.close()

    print("Saved evaluation figures: 07-10 in reports/figures/")
    print("\nTop 5 most influential parameters (XGBoost):")
    print(importance["XGBoost_Importance"].sort_values(ascending=False).head(5))


if __name__ == "__main__":
    main()
