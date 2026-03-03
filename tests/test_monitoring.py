from __future__ import annotations

import pandas as pd

from healthcare_ml.monitoring.drift_report import population_stability_index


def test_psi_zero_for_identical_series() -> None:
    s = pd.Series([1, 2, 3, 4, 5, 6])
    psi = population_stability_index(s, s)

    assert psi < 1e-6


def test_psi_increases_when_distributions_shift() -> None:
    baseline = pd.Series([1, 1, 2, 2, 3, 3, 4, 4])
    shifted = pd.Series([4, 4, 5, 5, 6, 6, 7, 7])

    psi = population_stability_index(baseline, shifted)

    assert psi > 0.1
