from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload sklearn model and deploy to Vertex Endpoint.")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--model-display-name", required=True)
    parser.add_argument("--artifact-uri", required=True)
    parser.add_argument(
        "--serving-container-image",
        default="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-5:latest",
    )
    parser.add_argument("--endpoint-id", default="")
    parser.add_argument("--create-endpoint", action="store_true")
    parser.add_argument("--endpoint-display-name", default="readmission-risk-endpoint")
    parser.add_argument("--machine-type", default="n1-standard-2")
    parser.add_argument("--traffic-percentage", type=int, default=100)
    return parser.parse_args()


def _endpoint_resource_name(project_id: str, region: str, endpoint_id: str) -> str:
    if endpoint_id.startswith("projects/"):
        return endpoint_id
    return f"projects/{project_id}/locations/{region}/endpoints/{endpoint_id}"


def main() -> None:
    args = parse_args()

    from google.cloud import aiplatform

    aiplatform.init(project=args.project_id, location=args.region)

    model = aiplatform.Model.upload(
        display_name=args.model_display_name,
        artifact_uri=args.artifact_uri,
        serving_container_image_uri=args.serving_container_image,
        sync=True,
    )
    print(f"Uploaded model: {model.resource_name}")

    endpoint = None
    if args.create_endpoint:
        endpoint = aiplatform.Endpoint.create(
            display_name=args.endpoint_display_name,
            sync=True,
        )
        print(f"Created endpoint: {endpoint.resource_name}")
    elif args.endpoint_id:
        endpoint = aiplatform.Endpoint(_endpoint_resource_name(args.project_id, args.region, args.endpoint_id))
    else:
        raise ValueError("Provide --endpoint-id or set --create-endpoint.")

    model.deploy(
        endpoint=endpoint,
        machine_type=args.machine_type,
        traffic_percentage=args.traffic_percentage,
        sync=True,
    )

    print(f"Deployed model to endpoint: {endpoint.resource_name}")


if __name__ == "__main__":
    main()
