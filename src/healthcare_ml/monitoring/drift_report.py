from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


NUMERIC_COLUMNS = [
    "age",
    "comorbidity_score",
    "prior_admissions_180d",
    "ed_visits_90d",
    "avg_length_of_stay",
    "med_count",
    "zip_svi",
    "interaction_count",
    "avg_sentiment_score",
    "urgent_symptom_mentions",
    "medication_barrier_flag",
    "followup_barrier_flag",
    "social_barrier_flag",
    "positive_recovery_flag",
]


def population_stability_index(baseline: pd.Series, production: pd.Series, bins: int = 10) -> float:
    baseline_values = baseline.astype(float).dropna().to_numpy()
    production_values = production.astype(float).dropna().to_numpy()

    if len(baseline_values) == 0 or len(production_values) == 0:
        return 0.0

    edges = np.quantile(baseline_values, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0

    baseline_hist, _ = np.histogram(baseline_values, bins=edges)
    production_hist, _ = np.histogram(production_values, bins=edges)

    baseline_pct = baseline_hist / max(baseline_hist.sum(), 1)
    production_pct = production_hist / max(production_hist.sum(), 1)

    eps = 1e-6
    psi = np.sum((production_pct - baseline_pct) * np.log((production_pct + eps) / (baseline_pct + eps)))
    return float(psi)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate drift report from BigQuery tables.")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--baseline-table", required=True)
    parser.add_argument("--production-table", required=True)
    parser.add_argument("--output", type=Path, default=Path("monitoring/drift_report.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from google.cloud import bigquery

    client = bigquery.Client(project=args.project_id, location=args.location)

    baseline_df = client.query(f"SELECT * FROM `{args.baseline_table}`").to_dataframe(create_bqstorage_client=False)
    production_df = client.query(f"SELECT * FROM `{args.production_table}`").to_dataframe(create_bqstorage_client=False)

    rows: list[str] = []
    rows.append("# Drift Report")
    rows.append("")
    rows.append(f"Baseline table: `{args.baseline_table}`")
    rows.append(f"Production table: `{args.production_table}`")
    rows.append("")
    rows.append("| feature | baseline_mean | production_mean | psi |")
    rows.append("|---|---:|---:|---:|")

    for col in NUMERIC_COLUMNS:
        if col not in baseline_df.columns or col not in production_df.columns:
            continue

        b_mean = float(pd.to_numeric(baseline_df[col], errors="coerce").mean())
        p_mean = float(pd.to_numeric(production_df[col], errors="coerce").mean())
        psi = population_stability_index(baseline_df[col], production_df[col])

        rows.append(f"| {col} | {b_mean:.4f} | {p_mean:.4f} | {psi:.4f} |")

    report = "\n".join(rows) + "\n"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote drift report: {args.output}")


if __name__ == "__main__":
    main()
