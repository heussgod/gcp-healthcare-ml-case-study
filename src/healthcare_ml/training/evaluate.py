from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from healthcare_ml.training.train import FEATURE_COLUMNS


@dataclass
class BatchPredictionResult:
    probability_mean: float
    positive_rate: float
    count: int


def batch_score(model_path: Path, df: pd.DataFrame, threshold: float = 0.5) -> BatchPredictionResult:
    model = joblib.load(model_path)
    probs = model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    preds = (probs >= threshold).astype(int)

    return BatchPredictionResult(
        probability_mean=float(probs.mean()),
        positive_rate=float(preds.mean()),
        count=int(len(df)),
    )
