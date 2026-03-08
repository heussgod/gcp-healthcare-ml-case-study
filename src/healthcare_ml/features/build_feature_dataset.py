from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


TEXT_FEATURE_COLUMNS = [
    "interaction_count",
    "avg_sentiment_score",
    "urgent_symptom_mentions",
    "medication_barrier_flag",
    "followup_barrier_flag",
    "social_barrier_flag",
    "positive_recovery_flag",
]


def load_jsonl(input_path: Path) -> list[dict[str, object]]:
    with input_path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_interaction_feature_table(enriched_df: pd.DataFrame) -> pd.DataFrame:
    if enriched_df.empty:
        return pd.DataFrame(columns=["encounter_id", *TEXT_FEATURE_COLUMNS])

    grouped = (
        enriched_df.groupby("encounter_id", as_index=False)
        .agg(
            interaction_count=("interaction_id", "count"),
            avg_sentiment_score=("sentiment_score", "mean"),
            urgent_symptom_mentions=("urgent_symptom_mentions", "sum"),
            medication_barrier_flag=("medication_barrier_flag", "max"),
            followup_barrier_flag=("followup_barrier_flag", "max"),
            social_barrier_flag=("social_barrier_flag", "max"),
            positive_recovery_flag=("positive_recovery_flag", "max"),
        )
    )
    return grouped


def merge_claims_with_text_features(claims_df: pd.DataFrame, interaction_features_df: pd.DataFrame) -> pd.DataFrame:
    merged = claims_df.merge(interaction_features_df, on="encounter_id", how="left")

    for column in TEXT_FEATURE_COLUMNS:
        if column not in merged.columns:
            merged[column] = 0.0

    fill_defaults = {
        "interaction_count": 0,
        "avg_sentiment_score": 0.0,
        "urgent_symptom_mentions": 0,
        "medication_barrier_flag": 0,
        "followup_barrier_flag": 0,
        "social_barrier_flag": 0,
        "positive_recovery_flag": 0,
    }
    merged = merged.fillna(fill_defaults)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge claims events with interaction enrichment into a training-ready feature dataset."
    )
    parser.add_argument("--claims-csv", type=Path, required=True)
    parser.add_argument("--enrichment-jsonl", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--interaction-features-output",
        type=Path,
        default=Path("data/processed/interaction_features.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    claims_df = pd.read_csv(args.claims_csv)
    enriched_records = load_jsonl(args.enrichment_jsonl)
    enriched_df = pd.DataFrame(enriched_records)

    interaction_features_df = build_interaction_feature_table(enriched_df)
    merged_df = merge_claims_with_text_features(claims_df, interaction_features_df)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.interaction_features_output.parent.mkdir(parents=True, exist_ok=True)

    merged_df.to_csv(args.output_csv, index=False)
    interaction_features_df.to_csv(args.interaction_features_output, index=False)

    print(f"Wrote merged training dataset to {args.output_csv}")
    print(f"Wrote interaction feature table to {args.interaction_features_output}")


if __name__ == "__main__":
    main()
