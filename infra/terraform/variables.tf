variable "project_id" {
  type        = string
  description = "GCP project ID"
}

variable "region" {
  type        = string
  default     = "us-central1"
  description = "Primary region"
}

variable "bq_location" {
  type        = string
  default     = "us-central1"
  description = "BigQuery dataset location"
}

variable "bucket_location" {
  type        = string
  default     = "US"
  description = "GCS bucket location"
}

variable "dataset_id" {
  type        = string
  default     = "healthcare_ml"
  description = "BigQuery dataset ID"
}

variable "artifact_bucket" {
  type        = string
  description = "Unique bucket name for model and pipeline artifacts"
}

variable "artifact_repo" {
  type        = string
  default     = "healthcare-ml"
  description = "Artifact Registry repo name"
}

variable "service_account_id" {
  type        = string
  default     = "healthcare-ml-runner"
  description = "Service account id for ML jobs"
}
