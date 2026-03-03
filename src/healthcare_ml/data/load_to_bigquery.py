from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load local CSV to BigQuery table.")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--table-id", required=True)
    parser.add_argument("--csv-path", type=Path, required=True)
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from google.cloud import bigquery

    client = bigquery.Client(project=args.project_id, location=args.location)
    dataset_ref = bigquery.DatasetReference(args.project_id, args.dataset_id)

    try:
        client.get_dataset(dataset_ref)
        print(f"Dataset exists: {args.project_id}.{args.dataset_id}")
    except Exception:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = args.location
        client.create_dataset(dataset, exists_ok=True)
        print(f"Created dataset: {args.project_id}.{args.dataset_id}")

    schema = [
        bigquery.SchemaField("patient_id", "STRING"),
        bigquery.SchemaField("encounter_id", "STRING"),
        bigquery.SchemaField("event_timestamp", "TIMESTAMP"),
        bigquery.SchemaField("age", "INT64"),
        bigquery.SchemaField("sex", "STRING"),
        bigquery.SchemaField("payer_type", "STRING"),
        bigquery.SchemaField("comorbidity_score", "FLOAT64"),
        bigquery.SchemaField("prior_admissions_180d", "INT64"),
        bigquery.SchemaField("ed_visits_90d", "INT64"),
        bigquery.SchemaField("avg_length_of_stay", "FLOAT64"),
        bigquery.SchemaField("med_count", "INT64"),
        bigquery.SchemaField("discharge_disposition", "STRING"),
        bigquery.SchemaField("zip_svi", "FLOAT64"),
        bigquery.SchemaField("readmit_30d", "INT64"),
        bigquery.SchemaField("split", "STRING"),
    ]

    table_ref = dataset_ref.table(args.table_id)
    write_disposition = (
        bigquery.WriteDisposition.WRITE_TRUNCATE
        if args.replace
        else bigquery.WriteDisposition.WRITE_APPEND
    )

    job_config = bigquery.LoadJobConfig(
        schema=schema,
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        write_disposition=write_disposition,
    )

    with args.csv_path.open("rb") as fp:
        load_job = client.load_table_from_file(fp, table_ref, job_config=job_config)
    load_job.result()

    table = client.get_table(table_ref)
    print(f"Loaded {table.num_rows} rows into {table.full_table_id}")


if __name__ == "__main__":
    main()
