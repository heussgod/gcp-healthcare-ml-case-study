from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BigQuery feature table using SQL template.")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--sql-path", type=Path, default=Path("sql/build_training_features.sql"))
    parser.add_argument("--location", default="us-central1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from google.cloud import bigquery

    sql_template = args.sql_path.read_text(encoding="utf-8")
    sql = sql_template.format(project_id=args.project_id, dataset_id=args.dataset_id)

    client = bigquery.Client(project=args.project_id, location=args.location)
    job = client.query(sql)
    job.result()

    target = f"{args.project_id}.{args.dataset_id}.training_features"
    print(f"Built feature table: {target}")


if __name__ == "__main__":
    main()
