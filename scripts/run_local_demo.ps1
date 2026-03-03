python -m healthcare_ml.data.generate_synthetic --rows 8000 --output data/raw/claims_events.csv
python -m healthcare_ml.training.train_local --input-csv data/raw/claims_events.csv --model-output model/model_local.joblib --metrics-output model/metrics_local.json
Get-Content model/metrics_local.json
