python -m healthcare_ml.data.generate_synthetic --rows 8000 --output data/raw/claims_events.csv --interactions-output data/raw/care_interactions.csv
python -m healthcare_ml.prep.prepare_interactions --input-csv data/raw/care_interactions.csv --output-jsonl data/prepared/care_interactions.jsonl
python -m healthcare_ml.apis.text_enrichment --input-jsonl data/prepared/care_interactions.jsonl --output-jsonl data/enriched/care_interactions.jsonl --provider heuristic
python -m healthcare_ml.features.build_feature_dataset --claims-csv data/raw/claims_events.csv --enrichment-jsonl data/enriched/care_interactions.jsonl --output-csv data/processed/training_dataset.csv --interaction-features-output data/processed/interaction_features.csv
python -m healthcare_ml.training.train_local --input-csv data/processed/training_dataset.csv --model-output model/model_local.joblib --metrics-output model/metrics_local.json
Get-Content model/metrics_local.json
