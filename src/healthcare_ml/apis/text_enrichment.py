from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


NEGATIVE_TERMS = [
    "cannot",
    "worried",
    "worsening",
    "missed",
    "barrier",
    "shortness of breath",
    "fever",
    "dizziness",
    "emergency department",
    "unstable",
]

POSITIVE_TERMS = [
    "feeling better",
    "no new concerns",
    "follow up visit already scheduled",
    "taking medication as directed",
    "understands discharge plan",
    "denies chest pain",
]

URGENT_SYMPTOM_TERMS = [
    "shortness of breath",
    "chest pain",
    "worsening",
    "fever",
    "dizziness",
    "swelling",
]

MEDICATION_BARRIER_TERMS = [
    "cannot afford medication",
    "confused about medication list",
    "has not picked up medication",
    "missed evening dose",
]

FOLLOWUP_BARRIER_TERMS = [
    "missed follow up appointment",
    "rescheduling cardiology follow up",
    "no ride to primary care visit",
    "transportation barrier",
]

SOCIAL_BARRIER_TERMS = [
    "housing is unstable",
    "food insecurity",
    "caregiver unavailable",
]


def _term_hits(text: str, terms: list[str]) -> int:
    return sum(1 for term in terms if term in text)


def heuristic_enrich_record(record: dict[str, Any]) -> dict[str, Any]:
    prepared_text = str(record["prepared_text"])

    negative_hits = _term_hits(prepared_text, NEGATIVE_TERMS)
    positive_hits = _term_hits(prepared_text, POSITIVE_TERMS)
    sentiment_score = (positive_hits - negative_hits) / max(positive_hits + negative_hits, 1)

    urgent_symptom_mentions = _term_hits(prepared_text, URGENT_SYMPTOM_TERMS)
    medication_barrier_flag = int(_term_hits(prepared_text, MEDICATION_BARRIER_TERMS) > 0)
    followup_barrier_flag = int(_term_hits(prepared_text, FOLLOWUP_BARRIER_TERMS) > 0)
    social_barrier_flag = int(_term_hits(prepared_text, SOCIAL_BARRIER_TERMS) > 0)
    positive_recovery_flag = int(positive_hits > negative_hits and positive_hits > 0)

    matched_entities: list[str] = []
    if urgent_symptom_mentions:
        matched_entities.append("urgent_symptom")
    if medication_barrier_flag:
        matched_entities.append("medication_barrier")
    if followup_barrier_flag:
        matched_entities.append("follow_up_barrier")
    if social_barrier_flag:
        matched_entities.append("social_barrier")
    if positive_recovery_flag:
        matched_entities.append("recovery_signal")

    return {
        **record,
        "sentiment_score": round(sentiment_score, 4),
        "urgent_symptom_mentions": urgent_symptom_mentions,
        "medication_barrier_flag": medication_barrier_flag,
        "followup_barrier_flag": followup_barrier_flag,
        "social_barrier_flag": social_barrier_flag,
        "positive_recovery_flag": positive_recovery_flag,
        "matched_entities": matched_entities,
        "provider": "heuristic",
    }


def _load_google_language_module() -> Any:
    try:
        from google.cloud import language_v2
    except ImportError as exc:
        raise ImportError(
            "google-cloud-language is required for --provider google. Install it first."
        ) from exc

    return language_v2


def google_enrich_record(record: dict[str, Any], client: Any | None = None) -> dict[str, Any]:
    language_v2 = _load_google_language_module()
    prepared_text = str(record["prepared_text"])
    active_client = client or language_v2.LanguageServiceClient()
    document = language_v2.Document(
        content=prepared_text,
        type_=language_v2.Document.Type.PLAIN_TEXT,
        language_code=str(record.get("language_code", "en")),
    )

    sentiment_response = active_client.analyze_sentiment(
        request={"document": document, "encoding_type": language_v2.EncodingType.UTF8}
    )
    entity_response = active_client.analyze_entities(
        request={"document": document, "encoding_type": language_v2.EncodingType.UTF8}
    )

    entity_names = [entity.name.lower() for entity in entity_response.entities]
    medication_barrier_flag = int(any("medication" in name or "pharmacy" in name for name in entity_names))
    followup_barrier_flag = int(any("appointment" in name or "follow up" in name for name in entity_names))
    social_barrier_flag = int(
        any("housing" in name or "food" in name or "caregiver" in name for name in entity_names)
    )
    urgent_symptom_mentions = sum(
        1
        for name in entity_names
        if any(term in name for term in ["shortness of breath", "chest pain", "fever", "dizziness", "swelling"])
    )

    return {
        **record,
        "sentiment_score": round(float(sentiment_response.document_sentiment.score), 4),
        "urgent_symptom_mentions": urgent_symptom_mentions,
        "medication_barrier_flag": medication_barrier_flag,
        "followup_barrier_flag": followup_barrier_flag,
        "social_barrier_flag": social_barrier_flag,
        "positive_recovery_flag": int(float(sentiment_response.document_sentiment.score) > 0.25),
        "matched_entities": entity_names,
        "provider": "google",
    }


def google_enrich_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    language_v2 = _load_google_language_module()
    client = language_v2.LanguageServiceClient()
    return [google_enrich_record(record, client=client) for record in records]


def _write_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")


def _load_jsonl(input_path: Path) -> list[dict[str, Any]]:
    with input_path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich prepared healthcare text with ML API style features."
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--provider", choices=["heuristic", "google"], default="heuristic")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = _load_jsonl(args.input_jsonl)
    if args.provider == "google":
        enriched_rows = google_enrich_records(records)
    else:
        enriched_rows = [heuristic_enrich_record(record) for record in records]

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(enriched_rows, args.output_jsonl)
    print(f"Wrote {len(enriched_rows)} enriched interaction records to {args.output_jsonl}")


if __name__ == "__main__":
    main()
