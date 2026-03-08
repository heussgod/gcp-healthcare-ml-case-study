from __future__ import annotations

from healthcare_ml.data.generate_synthetic import generate_dataset
from healthcare_ml.training.train import (
    FEATURE_COLUMNS,
    build_model_pipeline,
    ensure_feature_columns,
    evaluate_by_split,
)


def test_model_pipeline_trains_and_scores() -> None:
    df = generate_dataset(rows=4000, seed=5).rename(columns={"readmit_30d": "label"})
    df = ensure_feature_columns(df)

    model = build_model_pipeline()
    train_df = df[df["split"] == "train"]

    model.fit(train_df[FEATURE_COLUMNS], train_df["label"].astype(int))

    metrics = evaluate_by_split(model, df)

    assert "test" in metrics
    assert metrics["test"]["roc_auc"] > 0.60
    assert metrics["test"]["pr_auc"] > 0.15
