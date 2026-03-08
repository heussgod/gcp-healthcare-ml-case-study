# Healthcare Outreach Triage on Google Cloud

This project is a healthcare ML system I built on Google Cloud. It is structured around the workflow taught in **Prepare Data for ML APIs on Google Cloud**: messy synthetic discharge follow-up text is prepared for analysis, enriched with ML API style signals, joined to structured encounter data, and used to train a readmission-risk model that can be deployed on Vertex AI and served through Cloud Run.

The healthcare use case is post-discharge outreach triage:

- synthetic discharge notes, nurse follow-up calls, and care-manager messages
- text preparation and light de-identification
- Natural Language API style enrichment for sentiment, symptom urgency, medication barriers, and follow-up barriers
- merged structured + text-derived features for 30-day readmission risk
- Vertex AI / Cloud Run deployment path for scoring

## Certification Skills Mapped

- `Dataprep / ETL thinking`: normalize messy clinical text, standardize abbreviations, redact identifiers
- `ML APIs`: enrich prepared text with sentiment and entity-like signals
- `BigQuery`: store claims and aggregated interaction features for downstream training
- `Vertex AI`: train and deploy the downstream risk model
- `Cloud Run`: expose an inference API for online scoring
- `Monitoring`: compare baseline and production feature distributions

## Repository Layout

```text
gcp_healthcare_ml_case_study/
  cloudbuild/
  config/
  data/
  infra/terraform/
  model/
  scripts/
  sql/
  src/healthcare_ml/
  tests/
```

## Local Setup

```powershell
cd gcp_healthcare_ml_case_study
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## Local End-to-End Run

This is the fastest way to run the entire project locally.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_local_demo.ps1
```

That script does all of the following:

1. Generates synthetic structured encounter data and synthetic care-interaction text
2. Prepares messy text into normalized JSONL records
3. Enriches text with ML API style signals
4. Builds merged model features
5. Trains the local readmission model
6. Writes metrics to `model/metrics_local.json`

## Manual Pipeline

### 1) Generate structured and text data

```powershell
python -m healthcare_ml.data.generate_synthetic `
  --rows 12000 `
  --output data/raw/claims_events.csv `
  --interactions-output data/raw/care_interactions.csv
```

### 2) Prepare text for ML APIs

```powershell
python -m healthcare_ml.prep.prepare_interactions `
  --input-csv data/raw/care_interactions.csv `
  --output-jsonl data/prepared/care_interactions.jsonl
```

What happens here:

- patient and encounter IDs are redacted from free text
- abbreviations like `sob`, `fu`, and `rx` are expanded
- interaction channels are standardized
- JSONL output is created for downstream enrichment

### 3) Enrich prepared text

Local fallback:

```powershell
python -m healthcare_ml.apis.text_enrichment `
  --input-jsonl data/prepared/care_interactions.jsonl `
  --output-jsonl data/enriched/care_interactions.jsonl `
  --provider heuristic
```

Google Cloud Natural Language API path:

```powershell
gcloud auth application-default login
python -m healthcare_ml.apis.text_enrichment `
  --input-jsonl data/prepared/care_interactions.jsonl `
  --output-jsonl data/enriched/care_interactions.jsonl `
  --provider google
```

The enrichment output captures:

- `sentiment_score`
- `urgent_symptom_mentions`
- `medication_barrier_flag`
- `followup_barrier_flag`
- `social_barrier_flag`
- `positive_recovery_flag`

### 4) Build training-ready features

```powershell
python -m healthcare_ml.features.build_feature_dataset `
  --claims-csv data/raw/claims_events.csv `
  --enrichment-jsonl data/enriched/care_interactions.jsonl `
  --output-csv data/processed/training_dataset.csv `
  --interaction-features-output data/processed/interaction_features.csv
```

### 5) Train locally

```powershell
python -m healthcare_ml.training.train_local `
  --input-csv data/processed/training_dataset.csv `
  --model-output model/model_local.joblib `
  --metrics-output model/metrics_local.json
```

## Google Cloud Runbook

### 1) Authenticate and set project config

```powershell
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable bigquery.googleapis.com aiplatform.googleapis.com run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com language.googleapis.com

$env:PROJECT_ID="YOUR_PROJECT_ID"
$env:BQ_DATASET="healthcare_ml"
$env:BQ_LOCATION="us-central1"
$env:GCS_BUCKET="YOUR_ARTIFACT_BUCKET"
```

### 2) Load structured claims data to BigQuery

```powershell
python -m healthcare_ml.data.load_to_bigquery `
  --project-id $env:PROJECT_ID `
  --dataset-id $env:BQ_DATASET `
  --table-id claims_events `
  --csv-path data/raw/claims_events.csv `
  --location $env:BQ_LOCATION `
  --replace
```

### 3) Load aggregated text features to BigQuery

```powershell
python -m healthcare_ml.data.load_interaction_features_to_bigquery `
  --project-id $env:PROJECT_ID `
  --dataset-id $env:BQ_DATASET `
  --table-id interaction_features `
  --csv-path data/processed/interaction_features.csv `
  --location $env:BQ_LOCATION `
  --replace
```

### 4) Build the joined BigQuery training table

```powershell
python -m healthcare_ml.features.build_features `
  --project-id $env:PROJECT_ID `
  --dataset-id $env:BQ_DATASET `
  --sql-path sql/build_training_features.sql `
  --location $env:BQ_LOCATION
```

### 5) Train on BigQuery data

```powershell
python -m healthcare_ml.training.train `
  --project-id $env:PROJECT_ID `
  --dataset-id $env:BQ_DATASET `
  --table-id training_features `
  --model-output model/model.joblib `
  --metrics-output model/metrics.json `
  --location $env:BQ_LOCATION
```

### 6) Compile and submit Vertex pipeline

```powershell
python -m healthcare_ml.pipeline.vertex_pipeline `
  --project-id $env:PROJECT_ID `
  --region us-central1 `
  --dataset-id $env:BQ_DATASET `
  --pipeline-root gs://$env:GCS_BUCKET/pipeline-root `
  --sql-path sql/build_training_features.sql `
  --submit
```

### 7) Deploy the endpoint

```powershell
python -m healthcare_ml.deploy.deploy_endpoint `
  --project-id $env:PROJECT_ID `
  --region us-central1 `
  --model-display-name outreach-triage-model `
  --artifact-uri gs://$env:GCS_BUCKET/models/latest `
  --create-endpoint
```

## Cloud Run API

Build and deploy:

```powershell
gcloud builds submit --config cloudbuild/cloudbuild-deploy.yaml --substitutions _REGION=us-central1,_REPO=healthcare-ml,_IMAGE=readmission-api
```

Set service environment variables:

- `PROJECT_ID`
- `REGION`
- `VERTEX_ENDPOINT_ID`

Example request:

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
  "zip_svi": 0.64,
  "interaction_count": 3,
  "avg_sentiment_score": -0.55,
  "urgent_symptom_mentions": 2,
  "medication_barrier_flag": 1,
  "followup_barrier_flag": 1,
  "social_barrier_flag": 0,
  "positive_recovery_flag": 0
}
```

## Monitoring

Generate a drift report from BigQuery:

```powershell
python -m healthcare_ml.monitoring.drift_report `
  --project-id $env:PROJECT_ID `
  --location $env:BQ_LOCATION `
  --baseline-table "$env:PROJECT_ID.$env:BQ_DATASET.training_features" `
  --production-table "$env:PROJECT_ID.$env:BQ_DATASET.prediction_log" `
  --output monitoring/drift_report.md
```

## Tests

```powershell
python -m pytest
```

## Notes

- All data in this repo is synthetic and contains no PHI.
- The `heuristic` provider is for local development and testing.
- The `google` provider is the certification-aligned ML API path.
- For production use, replace synthetic text generation with de-identified clinical communication data and harden IAM, networking, and audit logging.
