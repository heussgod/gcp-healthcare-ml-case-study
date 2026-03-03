# Healthcare Readmission Risk on Google Cloud (End-to-End)

This project is a portfolio-grade, end-to-end healthcare ML system built on Google Cloud.
It predicts 30-day readmission risk and demonstrates Professional ML Engineer skills across data prep, feature engineering, training, evaluation, deployment, serving, and monitoring.

## What This Shows

- BigQuery data warehouse ingestion and feature tables
- Vertex AI compatible training pipeline
- Model artifact management and endpoint deployment
- Cloud Run API for online inference
- Drift/performance monitoring workflow
- Infrastructure as code with Terraform
- CI/CD style automation with Cloud Build configs

## Architecture

1. Synthetic encounter data generated locally (`src/healthcare_ml/data/generate_synthetic.py`)
2. Raw data loaded to BigQuery (`src/healthcare_ml/data/load_to_bigquery.py`)
3. SQL feature build in BigQuery (`sql/build_training_features.sql`)
4. Model training and metrics (`src/healthcare_ml/training/train.py`)
5. Vertex pipeline compile/submit (`src/healthcare_ml/pipeline/vertex_pipeline.py`)
6. Model upload + endpoint deployment (`src/healthcare_ml/deploy/deploy_endpoint.py`)
7. Cloud Run inference API (`src/healthcare_ml/serving/app.py`)
8. Drift report job (`src/healthcare_ml/monitoring/drift_report.py`)

## Repository Layout

```text
gcp_healthcare_ml_case_study/
  config/
  sql/
  infra/terraform/
  cloudbuild/
  scripts/
  src/healthcare_ml/
  tests/
```

## Setup

### 1) Local Python

```powershell
cd gcp_healthcare_ml_case_study
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

### 2) Google Cloud Setup

```powershell
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable bigquery.googleapis.com aiplatform.googleapis.com run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

### 3) Configure Env Vars

```powershell
$env:PROJECT_ID="YOUR_PROJECT_ID"
$env:BQ_DATASET="healthcare_ml"
$env:BQ_LOCATION="us-central1"
$env:GCS_BUCKET="YOUR_ARTIFACT_BUCKET"
```

## Quick End-to-End Run

### A) Generate synthetic data

```powershell
python -m healthcare_ml.data.generate_synthetic --rows 12000 --output data/raw/claims_events.csv
```

### B) Load to BigQuery

```powershell
python -m healthcare_ml.data.load_to_bigquery --project-id $env:PROJECT_ID --dataset-id $env:BQ_DATASET --table-id claims_events --csv-path data/raw/claims_events.csv --location $env:BQ_LOCATION
```

### C) Build feature table

```powershell
python -m healthcare_ml.features.build_features --project-id $env:PROJECT_ID --dataset-id $env:BQ_DATASET --sql-path sql/build_training_features.sql --location $env:BQ_LOCATION
```

### D) Train locally against BigQuery

```powershell
python -m healthcare_ml.training.train --project-id $env:PROJECT_ID --dataset-id $env:BQ_DATASET --table-id training_features --model-output model/model.joblib --metrics-output model/metrics.json --location $env:BQ_LOCATION
```

### E) Compile and submit Vertex pipeline

```powershell
python -m healthcare_ml.pipeline.vertex_pipeline --project-id $env:PROJECT_ID --region us-central1 --dataset-id $env:BQ_DATASET --pipeline-root gs://$env:GCS_BUCKET/pipeline-root --sql-path sql/build_training_features.sql --submit
```

### F) Deploy model endpoint

```powershell
python -m healthcare_ml.deploy.deploy_endpoint --project-id $env:PROJECT_ID --region us-central1 --model-display-name readmission-risk-model --artifact-uri gs://$env:GCS_BUCKET/models/latest --create-endpoint
```

## Cloud Run Serving API

Build and deploy a FastAPI service that proxies to Vertex Endpoint.

```powershell
gcloud builds submit --config cloudbuild/cloudbuild-deploy.yaml --substitutions _REGION=us-central1,_REPO=healthcare-ml,_IMAGE=readmission-api
```

Set service env vars:
- `PROJECT_ID`
- `REGION`
- `VERTEX_ENDPOINT_ID`

POST request body:

```json
{
  "age": 68,
  "sex": "F",
  "payer_type": "Medicare",
  "comorbidity_score": 4.1,
  "prior_admissions_180d": 2,
  "ed_visits_90d": 1,
  "avg_length_of_stay": 5.2,
  "med_count": 11,
  "discharge_disposition": "SNF",
  "zip_svi": 0.64
}
```

## Monitoring

Run drift report comparing training features vs production prediction logs:

```powershell
python -m healthcare_ml.monitoring.drift_report --project-id $env:PROJECT_ID --location $env:BQ_LOCATION --baseline-table "$env:PROJECT_ID.$env:BQ_DATASET.training_features" --production-table "$env:PROJECT_ID.$env:BQ_DATASET.prediction_log" --output monitoring/drift_report.md
```

## Terraform Bootstrap

```powershell
cd infra/terraform
terraform init
terraform apply -var="project_id=YOUR_PROJECT_ID" -var="region=us-central1"
```

## Mapping to Professional ML Engineer Domains

- Framing and architecture: clinical readmission risk workflow design
- Data prep: synthetic cohort generation, BQ ingestion, SQL feature engineering
- ML modeling: train/validate/test split, metrics, thresholding, reproducibility
- MLOps: pipeline orchestration, model artifact lifecycle, endpoint deployment
- Responsible operations: drift monitoring and explicit logging surface

## Notes

- Data in this repo is synthetic and not PHI.
- Replace synthetic source with de-identified healthcare data in production.
- Add IAM least-privilege and VPC-SC policies for hardened environments.
