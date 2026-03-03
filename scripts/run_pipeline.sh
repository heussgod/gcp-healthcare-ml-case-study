#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ID:?PROJECT_ID is required}"
BQ_DATASET="${BQ_DATASET:-healthcare_ml}"
BQ_LOCATION="${BQ_LOCATION:-us-central1}"

python -m healthcare_ml.data.generate_synthetic --rows 12000 --output data/raw/claims_events.csv
python -m healthcare_ml.data.load_to_bigquery --project-id "${PROJECT_ID}" --dataset-id "${BQ_DATASET}" --table-id claims_events --csv-path data/raw/claims_events.csv --location "${BQ_LOCATION}" --replace
python -m healthcare_ml.features.build_features --project-id "${PROJECT_ID}" --dataset-id "${BQ_DATASET}" --sql-path sql/build_training_features.sql --location "${BQ_LOCATION}"
python -m healthcare_ml.training.train --project-id "${PROJECT_ID}" --dataset-id "${BQ_DATASET}" --table-id training_features --location "${BQ_LOCATION}" --model-output model/model.joblib --metrics-output model/metrics.json
