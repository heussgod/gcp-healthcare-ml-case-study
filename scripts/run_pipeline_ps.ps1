$ErrorActionPreference = "Stop"

if (-not $env:PROJECT_ID) { throw "PROJECT_ID env var is required." }
if (-not $env:BQ_DATASET) { $env:BQ_DATASET = "healthcare_ml" }
if (-not $env:BQ_LOCATION) { $env:BQ_LOCATION = "us-central1" }

python -m healthcare_ml.data.generate_synthetic --rows 12000 --output data/raw/claims_events.csv
python -m healthcare_ml.data.load_to_bigquery --project-id $env:PROJECT_ID --dataset-id $env:BQ_DATASET --table-id claims_events --csv-path data/raw/claims_events.csv --location $env:BQ_LOCATION --replace
python -m healthcare_ml.features.build_features --project-id $env:PROJECT_ID --dataset-id $env:BQ_DATASET --sql-path sql/build_training_features.sql --location $env:BQ_LOCATION
python -m healthcare_ml.training.train --project-id $env:PROJECT_ID --dataset-id $env:BQ_DATASET --table-id training_features --location $env:BQ_LOCATION --model-output model/model.joblib --metrics-output model/metrics.json
