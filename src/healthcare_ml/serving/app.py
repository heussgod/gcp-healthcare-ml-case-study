from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from healthcare_ml.serving.predictor import predictor_from_env


class PatientFeatures(BaseModel):
    age: float = Field(ge=18, le=100)
    sex: str
    payer_type: str
    comorbidity_score: float = Field(ge=0)
    prior_admissions_180d: float = Field(ge=0)
    ed_visits_90d: float = Field(ge=0)
    avg_length_of_stay: float = Field(ge=0)
    med_count: float = Field(ge=0)
    discharge_disposition: str
    zip_svi: float = Field(ge=0, le=1)


app = FastAPI(title="Healthcare Readmission Risk API")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PatientFeatures) -> dict[str, float | str]:
    try:
        predictor = predictor_from_env()
        probability = predictor.predict_probability(payload.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if probability >= 0.70:
        band = "high"
    elif probability >= 0.40:
        band = "medium"
    else:
        band = "low"

    return {
        "readmission_probability": round(probability, 4),
        "risk_band": band,
    }
