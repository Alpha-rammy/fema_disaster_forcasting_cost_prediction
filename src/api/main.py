import os
import joblib
import uvicorn
import pandas as pd
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel


# PATHS

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

MODEL_DIR = os.path.join(BASE_DIR, "models")


# LOAD MODEL

model = joblib.load(
    os.path.join(MODEL_DIR, "terranova_disaster_cost_model.pkl")
)

print("TerraNova cost model loaded successfully!")


# FASTAPI

app = FastAPI(
    title="TerraNova Cost Recovery Prediction API",
    description="Predict disaster recovery cost using trained XGBoost model",
    version="0.1"
)


# INPUT SCHEMA

class TerraNovaFeatures(BaseModel):

    # Numeric features
    designated_area_count: int
    declaration_delay_days: float
    ongoing_at_declaration: int
    declaration_year: int
    declaration_month: int
    declaration_weekday: int
    previous_state_disasters: int
    previous_incident_disasters: int
    previous_state_incident_disasters: int
    days_since_previous_state_disaster: float
    days_since_previous_incident: float

    # Categorical features
    state: str
    declarationtype: str
    incidenttype: str
    declaration_season: str


# ROOT ENDPOINT

@app.get("/")
def welcome_root():

    return {
        "message": "Welcome to TerraNova Disaster Cost Recovery Prediction API"
    }


# PREDICTION ENDPOINT

@app.post("/predict")
def predict_cost(disaster: TerraNovaFeatures):

    # Convert API input to dataframe
    data = pd.DataFrame([disaster.model_dump()])

    # Predict log recovery cost
    predicted_log_cost = model.predict(data)[0]

    # Convert log prediction back to original cost scale
    predicted_cost = np.expm1(predicted_log_cost)

    return {
        "predicted_log_cost": round(float(predicted_log_cost), 4),
        "predicted_recovery_cost": round(float(predicted_cost), 2)
    }


# RUN SERVER

if __name__ == "__main__":

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )

    print("TerraNova cost model loaded successfully!")