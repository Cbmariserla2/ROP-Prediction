"""
data_preprocessing.py
----------------------
Cleans and merges raw drilling parameters into a modeling-ready dataset.

Steps:
1. Load raw CSV.
2. Drop exact duplicate rows.
3. Handle missing values (median imputation per well, since values are
   depth-continuous and should be locally interpolated rather than
   globally imputed).
4. Remove physically impossible outliers (domain-based clipping).
5. Encode categorical Formation column.
6. Feature engineering: Mechanical Specific Energy (MSE) — a standard
   drilling-efficiency metric — and normalized bit wear.
7. Save cleaned dataset to data/processed/.
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
RAW_PATH = BASE / "data" / "raw" / "drilling_data_raw.csv"
PROCESSED_PATH = BASE / "data" / "processed" / "drilling_data_clean.csv"

NUMERIC_COLS = [
    "Depth_m", "WOB_klbs", "RPM", "Flow_Rate_gpm",
    "Torque_kftlbs", "Mud_Weight_ppg", "Bit_Hours", "ROP_m_hr"
]

# Domain-plausible physical bounds used to clip/remove sensor outliers
PHYSICAL_BOUNDS = {
    "WOB_klbs": (0, 60),
    "RPM": (0, 300),
    "Flow_Rate_gpm": (0, 1200),
    "Torque_kftlbs": (0, 35),
    "Mud_Weight_ppg": (7.5, 19),
    "ROP_m_hr": (0, 120),
}


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df)} duplicate rows.")

    # Sort by well & depth so interpolation is depth-continuous
    df = df.sort_values(["Well_ID", "Depth_m"]).reset_index(drop=True)

    # Median-impute missing numeric values PER WELL (local behavior),
    # falling back to global median if a well has no valid values at all.
    for col in NUMERIC_COLS:
        if df[col].isna().any():
            df[col] = df.groupby("Well_ID")[col].transform(lambda s: s.fillna(s.median()))
            df[col] = df[col].fillna(df[col].median())

    # Clip physically impossible values
    for col, (low, high) in PHYSICAL_BOUNDS.items():
        n_out = ((df[col] < low) | (df[col] > high)).sum()
        if n_out:
            print(f"Clipping {n_out} out-of-range values in {col} to [{low}, {high}]")
        df[col] = df[col].clip(low, high)

    # Drop rows still missing the target
    df = df.dropna(subset=["ROP_m_hr"])

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Mechanical Specific Energy (simplified rotary drilling MSE, standard in the industry)
    # MSE = (WOB / Area) + (120 * pi * RPM * Torque) / (Area * ROP)   [psi]
    bit_diameter_in = 8.5  # typical bit size assumption (in), constant across dataset here
    area_in2 = np.pi * (bit_diameter_in / 2) ** 2

    wob_lbs = df["WOB_klbs"] * 1000
    rop_safe = df["ROP_m_hr"].replace(0, np.nan)  # avoid div by zero
    df["MSE_psi"] = (
        (wob_lbs / area_in2)
        + (120 * np.pi * df["RPM"] * df["Torque_kftlbs"] * 1000) / (area_in2 * rop_safe)
    )
    df["MSE_psi"] = df["MSE_psi"].fillna(df["MSE_psi"].median())

    # Normalized bit wear (0-1) as a proxy dull-bit indicator
    df["Bit_Wear_Norm"] = df["Bit_Hours"] / df["Bit_Hours"].max()

    # One-hot encode Formation (kept alongside a copy of the raw label for EDA)
    df["Formation_Label"] = df["Formation"]
    formation_dummies = pd.get_dummies(df["Formation"], prefix="Formation")
    df = pd.concat([df, formation_dummies], axis=1)

    return df


def main():
    df = load_raw()
    df = clean(df)
    df = engineer_features(df)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Saved cleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns -> {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
