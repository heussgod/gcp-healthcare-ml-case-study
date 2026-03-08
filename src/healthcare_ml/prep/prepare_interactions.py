from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


PATIENT_ID_PATTERN = re.compile(r"\b(?:patient|pt)\s+P\d+\b", flags=re.IGNORECASE)
ENCOUNTER_ID_PATTERN = re.compile(r"\bencounter\s+E\d+\b", flags=re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")

ABBREVIATIONS = {
    r"\bsob\b": "shortness of breath",
    r"\bfu\b": "follow up",
    r"\brx\b": "medication",
    r"\bpt\b": "patient",
    r"\bed\b": "emergency department",
}

CHANNEL_MAP = {
    "discharge_note": "discharge_summary",
    "nurse_followup_call": "nurse_call",
    "care_manager_message": "care_management",
}


def normalize_text(text: str) -> str:
    normalized = PATIENT_ID_PATTERN.sub("patient [redacted]", text)
    normalized = ENCOUNTER_ID_PATTERN.sub("encounter [redacted]", normalized)

    for pattern, replacement in ABBREVIATIONS.items():
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

    normalized = normalized.lower()
    normalized = WHITESPACE_PATTERN.sub(" ", normalized).strip()
    return normalized


def prepare_record(record: dict[str, Any]) -> dict[str, Any]:
    prepared_text = normalize_text(str(record["raw_text"]))
    standardized_channel = CHANNEL_MAP.get(str(record["channel"]), str(record["channel"]))

    return {
        "interaction_id": record["interaction_id"],
        "patient_id": record["patient_id"],
        "encounter_id": record["encounter_id"],
        "interaction_timestamp": record["interaction_timestamp"],
        "channel": standardized_channel,
        "language_code": record.get("language_code", "en"),
        "prepared_text": prepared_text,
        "token_count": len(prepared_text.split()),
    }


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    prepared_rows = [prepare_record(record) for record in df.to_dict(orient="records")]
    return pd.DataFrame(prepared_rows)


def _write_jsonl(df: pd.DataFrame, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for row in df.to_dict(orient="records"):
            handle.write(json.dumps(row, default=str) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare clinical interaction text for downstream ML API enrichment."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_df = pd.read_csv(args.input_csv)
    prepared_df = prepare_dataframe(input_df)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(prepared_df, args.output_jsonl)
    print(f"Wrote {len(prepared_df)} prepared interaction records to {args.output_jsonl}")


if __name__ == "__main__":
    main()
