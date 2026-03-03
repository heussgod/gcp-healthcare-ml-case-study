from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from healthcare_ml.training.train import FEATURE_COLUMNS, build_model_pipeline, evaluate_by_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train model locally from generated CSV.")
    parser.add_argument("--input-csv", type=Path, default=Path("data/raw/claims_events.csv"))
    parser.add_argument("--model-output", type=Path, default=Path("model/model_local.joblib"))
    parser.add_argument("--metrics-output", type=Path, default=Path("model/metrics_local.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input_csv)
    df = df.rename(columns={"readmit_30d": "label"})

    train_df = df[df["split"] == "train"].copy()
    if train_df.empty:
        raise ValueError("No train rows in input CSV")

    model = build_model_pipeline()
    model.fit(train_df[FEATURE_COLUMNS], train_df["label"].astype(int))

    metrics = evaluate_by_split(model, df)

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, args.model_output)
    args.metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved local model: {args.model_output}")
    print(f"Saved local metrics: {args.metrics_output}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
