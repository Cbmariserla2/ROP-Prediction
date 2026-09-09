import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
PROCESSED_PATH = BASE / "data" / "processed" / "drilling_data_clean.csv"
FIG_DIR = BASE / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

FEATURES = ["WOB_klbs", "RPM", "Flow_Rate_gpm", "Torque_kftlbs",
            "Mud_Weight_ppg", "Depth_m", "Bit_Hours", "MSE_psi"]
TARGET = "ROP_m_hr"


def main():
    df = pd.read_csv(PROCESSED_PATH)

    # 1. Summary statistics
    summary = df[FEATURES + [TARGET]].describe().T
    summary.to_csv(BASE / "reports" / "summary_statistics.csv")
    print("Saved summary_statistics.csv")

    # 2. Target distribution
    plt.figure(figsize=(7, 5))
    sns.histplot(df[TARGET], bins=50, kde=True, color="steelblue")
    plt.title("Distribution of Rate of Penetration (ROP)")
    plt.xlabel("ROP (m/hr)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_rop_distribution.png", dpi=150)
    plt.close()

    # 3. Correlation heatmap
    plt.figure(figsize=(9, 7))
    corr = df[FEATURES + [TARGET]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap: Drilling Parameters vs ROP")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_correlation_heatmap.png", dpi=150)
    plt.close()

    # 4. ROP vs Depth (colored by well)
    plt.figure(figsize=(10, 6))
    for well, grp in df.groupby("Well_ID"):
        plt.scatter(grp["Depth_m"], grp[TARGET], s=4, alpha=0.4, label=well)
    plt.xlabel("Depth (m)")
    plt.ylabel("ROP (m/hr)")
    plt.title("ROP vs Depth by Well")
    plt.legend(markerscale=4, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_rop_vs_depth.png", dpi=150)
    plt.close()

    # 5. ROP by formation (boxplot)
    plt.figure(figsize=(9, 6))
    order = df.groupby("Formation_Label")[TARGET].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="Formation_Label", y=TARGET, order=order, palette="viridis")
    plt.title("ROP Distribution by Formation Type")
    plt.xlabel("Formation")
    plt.ylabel("ROP (m/hr)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_rop_by_formation.png", dpi=150)
    plt.close()

    # 6. Pairwise scatter of key params vs ROP
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    pairs = ["WOB_klbs", "RPM", "Flow_Rate_gpm", "Torque_kftlbs"]
    for ax, col in zip(axes.flat, pairs):
        ax.scatter(df[col], df[TARGET], s=3, alpha=0.15, color="darkorange")
        ax.set_xlabel(col)
        ax.set_ylabel("ROP (m/hr)")
        ax.set_title(f"ROP vs {col}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_rop_vs_key_params.png", dpi=150)
    plt.close()

    # 7. Bit wear effect
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df.sample(min(3000, len(df)), random_state=1),
                     x="Bit_Hours", y=TARGET, hue="Formation_Label", s=10, alpha=0.5)
    plt.title("Bit Wear (Hours on Bit) vs ROP")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_bit_wear_vs_rop.png", dpi=150)
    plt.close()

    print(f"Saved 6 figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
