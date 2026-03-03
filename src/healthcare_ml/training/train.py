from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FEATURE_COLUMNS = [
    "age",
    "sex",
    "payer_type",
    "comorbidity_score",
    "prior_admissions_180d",
    "ed_visits_90d",
    "avg_length_of_stay",
    "med_count",
    "discharge_disposition",
    "zip_svi",
]

NUMERIC_FEATURES = [
    "age",
    "comorbidity_score",
    "prior_admissions_180d",
    "ed_visits_90d",
    "avg_length_of_stay",
    "med_count",
    "zip_svi",
]

CATEGORICAL_FEATURES = [
    "sex",
    "payer_type",
    "discharge_disposition",
]


def build_model_pipeline() -> Pipeline:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )

    model = LogisticRegression(max_iter=1500, class_weight="balanced")

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def load_training_data(project_id: str, dataset_id: str, table_id: str, location: str) -> pd.DataFrame:
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id, location=location)
    query = f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}`"
    return client.query(query).to_dataframe(create_bqstorage_client=False)


def _compute_metrics(y_true: pd.Series, y_prob: pd.Series, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "prevalence": float(y_true.mean()),
    }


def evaluate_by_split(model: Pipeline, df: pd.DataFrame) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}

    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split]
        if split_df.empty:
            continue
        y_true = split_df["label"].astype(int)
        y_prob = model.predict_proba(split_df[FEATURE_COLUMNS])[:, 1]
        metrics[split] = _compute_metrics(y_true, y_prob)

    return metrics


def _upload_to_gcs(local_path: Path, gcs_uri: str) -> str:
    from google.cloud import storage

    if not gcs_uri.startswith("gs://"):
        raise ValueError("gcs_uri must start with gs://")

    bucket_name, _, blob_name = gcs_uri[5:].partition("/")
    if not bucket_name or not blob_name:
        raise ValueError("gcs_uri must include bucket and object path")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(local_path))
    return gcs_uri


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train readmission model from BigQuery feature table.")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--table-id", default="training_features")
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--model-output", type=Path, default=Path("model/model.joblib"))
    parser.add_argument("--metrics-output", type=Path, default=Path("model/metrics.json"))
    parser.add_argument("--model-output-gcs", default="")
    parser.add_argument("--metrics-output-gcs", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_training_data(
        project_id=args.project_id,
        dataset_id=args.dataset_id,
        table_id=args.table_id,
        location=args.location,
    )

    train_df = df[df["split"] == "train"].copy()
    if train_df.empty:
        raise ValueError("No train rows found in split column.")

    model = build_model_pipeline()
    model.fit(train_df[FEATURE_COLUMNS], train_df["label"].astype(int))

    metrics = evaluate_by_split(model, df)

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, args.model_output)
    args.metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model to {args.model_output}")
    print(f"Saved metrics to {args.metrics_output}")
    print(json.dumps(metrics, indent=2))

    if args.model_output_gcs:
        _upload_to_gcs(args.model_output, args.model_output_gcs)
        print(f"Uploaded model to {args.model_output_gcs}")

    if args.metrics_output_gcs:
        _upload_to_gcs(args.metrics_output, args.metrics_output_gcs)
        print(f"Uploaded metrics to {args.metrics_output_gcs}")

    # Vertex custom training jobs can persist artifacts from these env paths.
    aip_model_dir_env = os.getenv("AIP_MODEL_DIR")
    if aip_model_dir_env:
        aip_model_dir = Path(aip_model_dir_env)
        aip_model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, aip_model_dir / "model.joblib")

    aip_tb_dir_env = os.getenv("AIP_TENSORBOARD_LOG_DIR")
    if aip_tb_dir_env:
        metrics_path = Path(aip_tb_dir_env) / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
