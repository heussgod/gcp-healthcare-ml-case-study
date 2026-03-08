from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SEXES = ["F", "M"]
PAYER_TYPES = ["Commercial", "Medicare", "Medicaid", "SelfPay"]
DISCHARGE = ["Home", "HomeHealth", "SNF", "Rehab"]
INTERACTION_CHANNELS = ["discharge_note", "nurse_followup_call", "care_manager_message"]

LOW_RISK_UPDATES = [
    "reports feeling better today",
    "follow up visit already scheduled",
    "taking medication as directed",
    "denies chest pain or shortness of breath",
    "understands discharge plan",
]

MEDICATION_BARRIER_UPDATES = [
    "cannot afford rx copay",
    "confused about medication list",
    "has not picked up rx from pharmacy",
    "missed evening dose after discharge",
]

FOLLOW_UP_BARRIER_UPDATES = [
    "missed fu appointment",
    "needs help rescheduling cardiology follow up",
    "no ride to primary care visit",
    "transportation barrier for clinic visit",
]

URGENT_SYMPTOM_UPDATES = [
    "reports sob walking across room",
    "worsening lower extremity swelling",
    "fever overnight with chills",
    "returned to ed for dizziness",
]

SOCIAL_RISK_UPDATES = [
    "caregiver unavailable this week",
    "housing is unstable after discharge",
    "food insecurity affecting recovery",
]


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


def _sample_statements(
    rng: np.random.Generator,
    base_choices: list[str],
    minimum: int = 2,
    maximum: int = 3,
) -> list[str]:
    count = int(rng.integers(minimum, maximum + 1))
    count = min(count, len(base_choices))
    return list(rng.choice(base_choices, size=count, replace=False))


def generate_interaction_dataset(claims_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 97)
    interaction_rows: list[dict[str, object]] = []

    for claim in claims_df.itertuples(index=False):
        event_ts = pd.Timestamp(claim.event_timestamp)
        interaction_count = int(rng.integers(1, 4))

        high_risk = (
            int(claim.readmit_30d) == 1
            or float(claim.prior_admissions_180d) >= 2
            or float(claim.ed_visits_90d) >= 2
            or str(claim.discharge_disposition) == "SNF"
        )

        for interaction_idx in range(interaction_count):
            timestamp = event_ts + pd.to_timedelta(int(rng.integers(0, 9)), unit="D")
            channel = str(rng.choice(INTERACTION_CHANNELS))

            statements = _sample_statements(rng, LOW_RISK_UPDATES)
            if float(claim.med_count) >= 10 or float(claim.comorbidity_score) >= 4.5:
                statements.extend(_sample_statements(rng, MEDICATION_BARRIER_UPDATES, minimum=1, maximum=1))
            if float(claim.prior_admissions_180d) >= 2 or float(claim.ed_visits_90d) >= 1:
                statements.extend(_sample_statements(rng, FOLLOW_UP_BARRIER_UPDATES, minimum=1, maximum=1))
            if high_risk:
                statements.extend(_sample_statements(rng, URGENT_SYMPTOM_UPDATES, minimum=1, maximum=2))
            if float(claim.zip_svi) >= 0.7:
                statements.extend(_sample_statements(rng, SOCIAL_RISK_UPDATES, minimum=1, maximum=1))

            if not high_risk and int(claim.readmit_30d) == 0:
                statements = [s for s in statements if s not in URGENT_SYMPTOM_UPDATES]
                if rng.random() < 0.45:
                    statements.append("no new concerns reported")

            header = f"Pt {claim.patient_id} follow-up after encounter {claim.encounter_id}."
            if channel == "discharge_note":
                header = f"Discharge note for patient {claim.patient_id}."
            elif channel == "care_manager_message":
                header = f"Care manager outreach to patient {claim.patient_id}."

            raw_text = "  ".join([header, *statements])
            if rng.random() < 0.35:
                raw_text = raw_text.replace("follow up", "fu")
            if rng.random() < 0.40:
                raw_text = raw_text.replace("medication", "rx")
            if rng.random() < 0.30:
                raw_text = raw_text.replace("shortness of breath", "sob")

            interaction_rows.append(
                {
                    "interaction_id": f"{claim.encounter_id}_I{interaction_idx + 1:02d}",
                    "patient_id": claim.patient_id,
                    "encounter_id": claim.encounter_id,
                    "interaction_timestamp": timestamp,
                    "channel": channel,
                    "language_code": "en",
                    "raw_text": raw_text,
                }
            )

    return pd.DataFrame(interaction_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic healthcare readmission dataset.")
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/raw/claims_events.csv"))
    parser.add_argument(
        "--interactions-output",
        type=Path,
        default=Path("data/raw/care_interactions.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = generate_dataset(rows=args.rows, seed=args.seed)
    interactions_df = generate_interaction_dataset(df, seed=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.interactions_output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    interactions_df.to_csv(args.interactions_output, index=False)

    prevalence = df["readmit_30d"].mean()
    print(f"Wrote {len(df)} rows to {args.output}")
    print(f"Wrote {len(interactions_df)} rows to {args.interactions_output}")
    print(f"Readmission prevalence: {prevalence:.3f}")


if __name__ == "__main__":
    main()
