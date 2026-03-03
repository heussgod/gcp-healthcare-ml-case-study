from __future__ import annotations

import os
from typing import Any

from google.cloud.aiplatform.gapic import PredictionServiceClient


class VertexEndpointPredictor:
    def __init__(self, project_id: str, region: str, endpoint_id: str) -> None:
        self.project_id = project_id
        self.region = region
        self.endpoint_id = endpoint_id
        self.client = PredictionServiceClient(
            client_options={"api_endpoint": f"{region}-aiplatform.googleapis.com"}
        )

    @property
    def endpoint_path(self) -> str:
        return self.client.endpoint_path(
            project=self.project_id,
            location=self.region,
            endpoint=self.endpoint_id,
        )

    def predict_probability(self, instance: dict[str, Any]) -> float:
        response = self.client.predict(endpoint=self.endpoint_path, instances=[instance])
        pred = response.predictions[0]

        if isinstance(pred, (int, float)):
            return float(pred)

        if isinstance(pred, dict):
            for key in ["probability", "score", "scores"]:
                if key in pred:
                    value = pred[key]
                    if isinstance(value, list):
                        return float(value[-1])
                    return float(value)

        if isinstance(pred, list):
            return float(pred[-1])

        raise ValueError(f"Unsupported prediction payload: {pred}")


def predictor_from_env() -> VertexEndpointPredictor:
    project_id = os.getenv("PROJECT_ID", "")
    region = os.getenv("REGION", "us-central1")
    endpoint_id = os.getenv("VERTEX_ENDPOINT_ID", "")

    if not project_id or not endpoint_id:
        raise ValueError("PROJECT_ID and VERTEX_ENDPOINT_ID env vars are required.")

    return VertexEndpointPredictor(project_id=project_id, region=region, endpoint_id=endpoint_id)
