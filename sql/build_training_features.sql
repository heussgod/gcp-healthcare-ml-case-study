CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.training_features` AS
SELECT
  patient_id,
  encounter_id,
  event_timestamp,
  split,
  CAST(age AS FLOAT64) AS age,
  sex,
  payer_type,
  CAST(comorbidity_score AS FLOAT64) AS comorbidity_score,
  CAST(prior_admissions_180d AS FLOAT64) AS prior_admissions_180d,
  CAST(ed_visits_90d AS FLOAT64) AS ed_visits_90d,
  CAST(avg_length_of_stay AS FLOAT64) AS avg_length_of_stay,
  CAST(med_count AS FLOAT64) AS med_count,
  discharge_disposition,
  CAST(zip_svi AS FLOAT64) AS zip_svi,
  CAST(readmit_30d AS INT64) AS label
FROM `{project_id}.{dataset_id}.claims_events`
WHERE age BETWEEN 18 AND 95;
