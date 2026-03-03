output "dataset" {
  value = google_bigquery_dataset.healthcare_ml.dataset_id
}

output "artifact_bucket" {
  value = google_storage_bucket.artifacts.name
}

output "artifact_repo" {
  value = google_artifact_registry_repository.containers.repository_id
}

output "service_account_email" {
  value = google_service_account.ml_runner.email
}
