from __future__ import annotations

import pandas as pd

from healthcare_ml.apis.text_enrichment import heuristic_enrich_record
from healthcare_ml.features.build_feature_dataset import (
    TEXT_FEATURE_COLUMNS,
    build_interaction_feature_table,
    merge_claims_with_text_features,
)
from healthcare_ml.prep.prepare_interactions import normalize_text, prepare_dataframe


def test_normalize_text_redacts_ids_and_expands_abbreviations() -> None:
    normalized = normalize_text(
        "Pt P000321 follow-up after encounter E0001456. Reports sob and cannot afford rx."
    )

    assert "p000321" not in normalized
    assert "e0001456" not in normalized
    assert "shortness of breath" in normalized
    assert "medication" in normalized


def test_heuristic_enrichment_flags_high_risk_signals() -> None:
    record = {
        "interaction_id": "E0000001_I01",
        "encounter_id": "E0000001",
        "prepared_text": (
            "patient [redacted] missed follow up appointment and reports shortness of breath. "
            "cannot afford medication."
        ),
    }

    enriched = heuristic_enrich_record(record)

    assert enriched["urgent_symptom_mentions"] >= 1
    assert enriched["medication_barrier_flag"] == 1
    assert enriched["followup_barrier_flag"] == 1
    assert enriched["sentiment_score"] < 0


def test_merge_claims_with_interaction_features_preserves_rows() -> None:
    claims_df = pd.DataFrame(
        [
            {"encounter_id": "E1", "split": "train", "readmit_30d": 0},
            {"encounter_id": "E2", "split": "test", "readmit_30d": 1},
        ]
    )
    prepared_df = prepare_dataframe(
        pd.DataFrame(
            [
                {
                    "interaction_id": "E1_I01",
                    "patient_id": "P1",
                    "encounter_id": "E1",
                    "interaction_timestamp": "2024-01-01T00:00:00Z",
                    "channel": "nurse_followup_call",
                    "language_code": "en",
                    "raw_text": "Pt P1 reports feeling better and taking medication as directed.",
                },
                {
                    "interaction_id": "E2_I01",
                    "patient_id": "P2",
                    "encounter_id": "E2",
                    "interaction_timestamp": "2024-01-02T00:00:00Z",
                    "channel": "nurse_followup_call",
                    "language_code": "en",
                    "raw_text": "Pt P2 missed fu appointment and reports sob.",
                },
            ]
        )
    )
    enriched_df = pd.DataFrame(
        [heuristic_enrich_record(record) for record in prepared_df.to_dict(orient="records")]
    )

    interaction_features_df = build_interaction_feature_table(enriched_df)
    merged_df = merge_claims_with_text_features(claims_df, interaction_features_df)

    assert len(merged_df) == len(claims_df)
    assert set(TEXT_FEATURE_COLUMNS).issubset(merged_df.columns)
    assert merged_df.loc[merged_df["encounter_id"] == "E2", "urgent_symptom_mentions"].item() >= 1
