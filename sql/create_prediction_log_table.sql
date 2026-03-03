CREATE TABLE IF NOT EXISTS `{project_id}.{dataset_id}.prediction_log` (
  inference_timestamp TIMESTAMP,
  model_version STRING,
  age FLOAT64,
  sex STRING,
  payer_type STRING,
  comorbidity_score FLOAT64,
  prior_admissions_180d FLOAT64,
  ed_visits_90d FLOAT64,
  avg_length_of_stay FLOAT64,
  med_count FLOAT64,
  discharge_disposition STRING,
  zip_svi FLOAT64,
  predicted_risk FLOAT64,
  predicted_label INT64,
  observed_label INT64
);
