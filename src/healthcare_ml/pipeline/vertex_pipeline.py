import argparse
from pathlib import Path

from kfp import compiler, dsl
from kfp.dsl import Output, Metrics, Model, component


@component(
    base_image="python:3.11",
    packages_to_install=["google-cloud-bigquery>=3.25,<4"],
)
def build_features_component(
    project_id: str,
    dataset_id: str,
    location: str,
    sql_template: str,
) -> str:
    from google.cloud import bigquery

    sql = sql_template.format(project_id=project_id, dataset_id=dataset_id)
    client = bigquery.Client(project=project_id, location=location)
    client.query(sql).result()
    return f"{project_id}.{dataset_id}.training_features"


@component(
    base_image="python:3.11",
    packages_to_install=[
        "google-cloud-bigquery>=3.25,<4",
        "pandas>=2.2,<3",
        "scikit-learn>=1.5,<2",
        "joblib>=1.4,<2",
    ],
)
def train_component(
    project_id: str,
    dataset_id: str,
    location: str,
    table_id: str,
    model_artifact: Output[Model],
    metrics_artifact: Output[Metrics],
) -> str:
    import json
    import os

    import joblib
    from google.cloud import bigquery
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    feature_columns = [
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
    numeric_features = [
        "age",
        "comorbidity_score",
        "prior_admissions_180d",
        "ed_visits_90d",
        "avg_length_of_stay",
        "med_count",
        "zip_svi",
    ]
    categorical_features = ["sex", "payer_type", "discharge_disposition"]

    client = bigquery.Client(project=project_id, location=location)
    query = f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}`"
    df = client.query(query).to_dataframe(create_bqstorage_client=False)

    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    num_transform = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    cat_transform = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transform, numeric_features),
            ("cat", cat_transform, categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", LogisticRegression(max_iter=1500, class_weight="balanced")),
        ]
    )

    model.fit(train_df[feature_columns], train_df["label"].astype(int))

    test_probs = model.predict_proba(test_df[feature_columns])[:, 1]
    test_auc = float(roc_auc_score(test_df["label"].astype(int), test_probs))
    test_pr_auc = float(average_precision_score(test_df["label"].astype(int), test_probs))

    metrics_artifact.log_metric("test_roc_auc", test_auc)
    metrics_artifact.log_metric("test_pr_auc", test_pr_auc)

    os.makedirs(model_artifact.path, exist_ok=True)
    model_file = os.path.join(model_artifact.path, "model.joblib")
    joblib.dump(model, model_file)

    summary_file = os.path.join(model_artifact.path, "metrics.json")
    with open(summary_file, "w", encoding="utf-8") as fp:
        json.dump({"test_roc_auc": test_auc, "test_pr_auc": test_pr_auc}, fp, indent=2)

    return model_file


@dsl.pipeline(name="healthcare-readmission-vertex-pipeline")
def healthcare_pipeline(
    project_id: str,
    dataset_id: str,
    location: str,
    sql_template: str,
) -> None:
    features = build_features_component(
        project_id=project_id,
        dataset_id=dataset_id,
        location=location,
        sql_template=sql_template,
    )

    train_task = train_component(
        project_id=project_id,
        dataset_id=dataset_id,
        location=location,
        table_id="training_features",
    )

    train_task.after(features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile and optionally submit Vertex pipeline.")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--dataset-id", default="healthcare_ml")
    parser.add_argument("--pipeline-root", required=True)
    parser.add_argument("--sql-path", type=Path, default=Path("sql/build_training_features.sql"))
    parser.add_argument("--template-output", type=Path, default=Path("pipeline_spec.json"))
    parser.add_argument("--job-id", default="healthcare-readmission-pipeline")
    parser.add_argument("--submit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sql_template = args.sql_path.read_text(encoding="utf-8")

    compiler.Compiler().compile(
        pipeline_func=healthcare_pipeline,
        package_path=str(args.template_output),
    )
    print(f"Compiled pipeline template: {args.template_output}")

    if not args.submit:
        return

    from google.cloud import aiplatform

    aiplatform.init(project=args.project_id, location=args.region)

    job = aiplatform.PipelineJob(
        display_name=args.job_id,
        template_path=str(args.template_output),
        pipeline_root=args.pipeline_root,
        parameter_values={
            "project_id": args.project_id,
            "dataset_id": args.dataset_id,
            "location": args.region,
            "sql_template": sql_template,
        },
    )

    job.submit()
    print(f"Submitted Vertex Pipeline: {job.resource_name}")


if __name__ == "__main__":
    main()
