CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.training_features` AS
SELECT
  claims.patient_id,
  claims.encounter_id,
  claims.event_timestamp,
  claims.split,
  CAST(claims.age AS FLOAT64) AS age,
  claims.sex,
  claims.payer_type,
  CAST(claims.comorbidity_score AS FLOAT64) AS comorbidity_score,
  CAST(claims.prior_admissions_180d AS FLOAT64) AS prior_admissions_180d,
  CAST(claims.ed_visits_90d AS FLOAT64) AS ed_visits_90d,
  CAST(claims.avg_length_of_stay AS FLOAT64) AS avg_length_of_stay,
  CAST(claims.med_count AS FLOAT64) AS med_count,
  claims.discharge_disposition,
  CAST(claims.zip_svi AS FLOAT64) AS zip_svi,
  COALESCE(CAST(text.interaction_count AS FLOAT64), 0.0) AS interaction_count,
  COALESCE(CAST(text.avg_sentiment_score AS FLOAT64), 0.0) AS avg_sentiment_score,
  COALESCE(CAST(text.urgent_symptom_mentions AS FLOAT64), 0.0) AS urgent_symptom_mentions,
  COALESCE(CAST(text.medication_barrier_flag AS FLOAT64), 0.0) AS medication_barrier_flag,
  COALESCE(CAST(text.followup_barrier_flag AS FLOAT64), 0.0) AS followup_barrier_flag,
  COALESCE(CAST(text.social_barrier_flag AS FLOAT64), 0.0) AS social_barrier_flag,
  COALESCE(CAST(text.positive_recovery_flag AS FLOAT64), 0.0) AS positive_recovery_flag,
  CAST(claims.readmit_30d AS INT64) AS label
FROM `{project_id}.{dataset_id}.claims_events` AS claims
LEFT JOIN `{project_id}.{dataset_id}.interaction_features` AS text
  ON claims.encounter_id = text.encounter_id
WHERE claims.age BETWEEN 18 AND 95;
