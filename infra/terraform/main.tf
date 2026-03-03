terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.30.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_project_service" "services" {
  for_each = toset([
    "aiplatform.googleapis.com",
    "bigquery.googleapis.com",
    "run.googleapis.com",
    "cloudbuild.googleapis.com",
    "artifactregistry.googleapis.com",
    "storage.googleapis.com"
  ])

  project            = var.project_id
  service            = each.key
  disable_on_destroy = false
}

resource "google_bigquery_dataset" "healthcare_ml" {
  dataset_id                 = var.dataset_id
  location                   = var.bq_location
  delete_contents_on_destroy = true

  depends_on = [google_project_service.services]
}

resource "google_storage_bucket" "artifacts" {
  name                        = var.artifact_bucket
  location                    = var.bucket_location
  force_destroy               = true
  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  depends_on = [google_project_service.services]
}

resource "google_artifact_registry_repository" "containers" {
  location      = var.region
  repository_id = var.artifact_repo
  description   = "Container images for healthcare ml case study"
  format        = "DOCKER"

  depends_on = [google_project_service.services]
}

resource "google_service_account" "ml_runner" {
  account_id   = var.service_account_id
  display_name = "Healthcare ML Runner"
}

resource "google_project_iam_member" "ml_runner_roles" {
  for_each = toset([
    "roles/aiplatform.user",
    "roles/bigquery.dataEditor",
    "roles/bigquery.jobUser",
    "roles/storage.objectAdmin",
    "roles/run.developer",
    "roles/artifactregistry.writer"
  ])

  project = var.project_id
  role    = each.key
  member  = "serviceAccount:${google_service_account.ml_runner.email}"
}
