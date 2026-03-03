from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SEXES = ["F", "M"]
PAYER_TYPES = ["Commercial", "Medicare", "Medicaid", "SelfPay"]
DISCHARGE = ["Home", "HomeHealth", "SNF", "Rehab"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_dataset(rows: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    patient_ids = [f"P{idx:06d}" for idx in rng.integers(1, rows // 2 + 5, size=rows)]
    encounter_ids = [f"E{idx:07d}" for idx in range(1, rows + 1)]

    age = rng.integers(18, 91, size=rows)
    sex = rng.choice(SEXES, p=[0.52, 0.48], size=rows)
    payer_type = rng.choice(PAYER_TYPES, p=[0.45, 0.30, 0.20, 0.05], size=rows)

    comorbidity_score = np.clip(rng.gamma(shape=2.2, scale=1.1, size=rows), 0, 10)
    prior_admissions_180d = rng.poisson(lam=np.clip((age - 35) / 42, 0.1, 2.5), size=rows)
    ed_visits_90d = rng.poisson(lam=np.clip((age - 30) / 65, 0.05, 1.8), size=rows)

    avg_length_of_stay = np.clip(rng.normal(loc=3.8, scale=1.9, size=rows), 0.5, 18)
    med_count = np.clip(rng.poisson(lam=np.clip(3 + comorbidity_score * 1.5, 1, 18), size=rows), 1, 40)
    discharge_disposition = rng.choice(DISCHARGE, p=[0.62, 0.17, 0.14, 0.07], size=rows)
    zip_svi = np.clip(rng.beta(a=2.0, b=2.8, size=rows), 0, 1)

    ts = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 730, size=rows), unit="D")

    logits = (
        -4.4
        + 0.028 * (age - 50)
        + 0.42 * prior_admissions_180d
        + 0.31 * ed_visits_90d
        + 0.24 * comorbidity_score
        + 0.15 * (avg_length_of_stay > 5).astype(float)
        + 0.36 * (discharge_disposition == "SNF").astype(float)
        + 0.18 * (payer_type == "Medicaid").astype(float)
        + 0.95 * zip_svi
    )

    probs = _sigmoid(logits)
    readmit_30d = rng.binomial(1, probs)

    split_rand = rng.random(rows)
    split = np.where(split_rand < 0.70, "train", np.where(split_rand < 0.85, "val", "test"))

    df = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "encounter_id": encounter_ids,
            "event_timestamp": ts,
            "age": age,
            "sex": sex,
            "payer_type": payer_type,
            "comorbidity_score": np.round(comorbidity_score, 3),
            "prior_admissions_180d": prior_admissions_180d,
            "ed_visits_90d": ed_visits_90d,
            "avg_length_of_stay": np.round(avg_length_of_stay, 3),
            "med_count": med_count,
            "discharge_disposition": discharge_disposition,
            "zip_svi": np.round(zip_svi, 4),
            "readmit_30d": readmit_30d,
            "split": split,
        }
    )

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic healthcare readmission dataset.")
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/raw/claims_events.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = generate_dataset(rows=args.rows, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    prevalence = df["readmit_30d"].mean()
    print(f"Wrote {len(df)} rows to {args.output}")
    print(f"Readmission prevalence: {prevalence:.3f}")


if __name__ == "__main__":
    main()
