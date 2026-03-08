from __future__ import annotations

from healthcare_ml.data.generate_synthetic import generate_dataset, generate_interaction_dataset


def test_generate_dataset_shape_and_columns() -> None:
    df = generate_dataset(rows=1000, seed=7)

    expected_columns = {
        "patient_id",
        "encounter_id",
        "event_timestamp",
        "age",
        "sex",
        "payer_type",
        "comorbidity_score",
        "prior_admissions_180d",
        "ed_visits_90d",
        "avg_length_of_stay",
        "med_count",
        "discharge_disposition",
        "zip_svi",
        "readmit_30d",
        "split",
    }

    assert len(df) == 1000
    assert expected_columns.issubset(df.columns)


def test_readmission_prevalence_reasonable() -> None:
    df = generate_dataset(rows=5000, seed=13)
    prevalence = float(df["readmit_30d"].mean())

    assert 0.05 <= prevalence <= 0.45


def test_split_distribution_present() -> None:
    df = generate_dataset(rows=2000, seed=11)
    counts = df["split"].value_counts(normalize=True)

    assert set(counts.index) == {"train", "val", "test"}
    assert counts["train"] > counts["val"]
    assert counts["train"] > counts["test"]


def test_interaction_dataset_links_back_to_encounters() -> None:
    claims_df = generate_dataset(rows=250, seed=19)
    interactions_df = generate_interaction_dataset(claims_df, seed=19)

    assert len(interactions_df) >= len(claims_df)
    assert set(interactions_df["encounter_id"]).issubset(set(claims_df["encounter_id"]))
    assert {"interaction_id", "channel", "raw_text"}.issubset(interactions_df.columns)
