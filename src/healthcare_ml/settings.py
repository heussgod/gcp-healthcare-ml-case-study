from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectSettings:
    project_id: str
    region: str = "us-central1"
    bq_dataset: str = "healthcare_ml"
    bq_location: str = "us-central1"
    gcs_bucket: str = ""


    @property
    def pipeline_root(self) -> str:
        if not self.gcs_bucket:
            return ""
        return f"gs://{self.gcs_bucket}/pipeline-root"


def from_env() -> ProjectSettings:
    project_id = os.getenv("PROJECT_ID", "")
    if not project_id:
        raise ValueError("PROJECT_ID is required.")

    return ProjectSettings(
        project_id=project_id,
        region=os.getenv("REGION", "us-central1"),
        bq_dataset=os.getenv("BQ_DATASET", "healthcare_ml"),
        bq_location=os.getenv("BQ_LOCATION", "us-central1"),
        gcs_bucket=os.getenv("GCS_BUCKET", ""),
    )
