import numpy as np
import pandas as pd
from pathlib import Path

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "drilling_data_raw.csv"

COLUMNS = [
    "Well_ID", "Depth_m", "WOB_klbs", "RPM", "Flow_Rate_gpm", "Torque_kftlbs",
    "Mud_Weight_ppg", "Formation", "Bit_Hours", "ROP_m_hr"
]

FORMATIONS = {
    # formation: (base drillability factor, hardness noise std)
    "Shale":      (1.15, 0.08),
    "Sandstone":  (1.35, 0.10),
    "Limestone":  (0.85, 0.07),
    "Dolomite":   (0.65, 0.06),
    "Claystone":  (1.45, 0.09),
    "Chalk":      (1.60, 0.11),
}

def generate_well(well_id: str, n_rows: int, start_depth: float, rng: np.random.Generator) -> pd.DataFrame:
    depth = start_depth + np.cumsum(rng.uniform(0.2, 1.0, size=n_rows))  # depth increases downhole

    # Formation changes in contiguous blocks (like real geology), not row-by-row randomness
    formation_names = list(FORMATIONS.keys())
    n_blocks = max(1, n_rows // rng.integers(150, 400))
    block_bounds = np.sort(rng.choice(np.arange(1, n_rows), size=max(0, n_blocks - 1), replace=False))
    block_bounds = np.concatenate(([0], block_bounds, [n_rows]))
    formation = np.empty(n_rows, dtype=object)
    for i in range(len(block_bounds) - 1):
        f = rng.choice(formation_names)
        formation[block_bounds[i]:block_bounds[i + 1]] = f

    # Drilling parameters — correlated with depth & driller behavior, not iid noise
    wob = np.clip(10 + depth / 250 + rng.normal(0, 2.5, n_rows), 4, 45)                # klbs
    rpm = np.clip(90 + rng.normal(0, 15, n_rows) - depth / 400, 40, 220)               # rev/min
    flow_rate = np.clip(450 + rng.normal(0, 40, n_rows) + depth / 500, 250, 900)       # gpm
    mud_weight = np.clip(9.5 + depth / 3000 + rng.normal(0, 0.15, n_rows), 8.5, 16.5)  # ppg
    bit_hours = np.mod(np.cumsum(rng.uniform(0.01, 0.05, n_rows)), 40)                 # bit wear resets on trips
    torque = np.clip(
        2 + 0.09 * wob + 0.015 * rpm + rng.normal(0, 1.2, n_rows), 0.5, 28
    )  # kft-lbs, correlated with WOB/RPM as in real rigs

    # --- Bourgoyne & Young inspired ROP model ---
    drill_factor = np.array([FORMATIONS[f][0] for f in formation])
    hardness_noise = np.array([rng.normal(0, FORMATIONS[f][1]) for f in formation])

    bit_wear_penalty = 1 - 0.010 * bit_hours          # ROP drops as bit dulls
    mud_weight_penalty = 1 - 0.04 * (mud_weight - 9.5)  # overbalance suppresses ROP
    depth_penalty = np.exp(-depth / 6000)              # compaction effect with depth

    rop = (
        drill_factor
        * depth_penalty
        * bit_wear_penalty
        * mud_weight_penalty
        * (0.6 * np.log1p(wob))
        * (0.5 * np.log1p(rpm))
        * (1 + 0.0006 * flow_rate)
        * (1 + 0.01 * torque)
    )
    rop = rop * 6.0 + hardness_noise + rng.normal(0, 1.0, n_rows)  # scale + noise
    rop = np.clip(rop, 0.5, 90)  # realistic ROP bounds (m/hr)

    df = pd.DataFrame({
        "Well_ID": well_id,
        "Depth_m": np.round(depth, 2),
        "WOB_klbs": np.round(wob, 2),
        "RPM": np.round(rpm, 1),
        "Flow_Rate_gpm": np.round(flow_rate, 1),
        "Torque_kftlbs": np.round(torque, 2),
        "Mud_Weight_ppg": np.round(mud_weight, 2),
        "Formation": formation,
        "Bit_Hours": np.round(bit_hours, 2),
        "ROP_m_hr": np.round(rop, 2),
    })
    return df


def main():
    rng = np.random.default_rng(42)
    wells = [
        ("VOLVE_15_9_F1", 2600, 2200),
        ("VOLVE_15_9_F5", 2400, 1800),
        ("VOLVE_15_9_F11", 2800, 2500),
        ("VOLVE_15_9_F14", 2300, 2000),
        ("VOLVE_15_9_F15", 2500, 2100),
    ]
    dfs = [generate_well(wid, n, sd, rng) for wid, n, sd in wells]
    data = pd.concat(dfs, ignore_index=True)

    # Inject a small amount of realistic messiness (missing values, duplicates)
    # so the preprocessing step has real work to do, like actual field data.
    n_missing = int(0.015 * len(data))
    missing_idx = rng.choice(data.index, size=n_missing, replace=False)
    missing_cols = rng.choice(
        ["WOB_klbs", "RPM", "Flow_Rate_gpm", "Torque_kftlbs", "Mud_Weight_ppg"],
        size=n_missing
    )
    for idx, col in zip(missing_idx, missing_cols):
        data.loc[idx, col] = np.nan

    dup_rows = data.sample(n=int(0.005 * len(data)), random_state=1)
    data = pd.concat([data, dup_rows], ignore_index=True)

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(RAW_PATH, index=False)
    print(f"Generated {len(data)} rows across {len(wells)} wells -> {RAW_PATH}")


if __name__ == "__main__":
    main()
