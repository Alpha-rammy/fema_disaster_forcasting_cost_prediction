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
    os.path.join(MODEL_DIR, "fema_cost_model.pkl")
)

print("FEMA cost model loaded successfully!")


# FAST API

app = FastAPI(
    title="TerraNova FEMA Cost Prediction API",
    description="Predict FEMA disaster recovery cost using trained XGBoost model",
    version="0.1"
)



# INPUT SCHEMA


class FEMAFeatures(BaseModel):

    # Numeric features
    fydeclared: int
    avg_delay_days: float
    declaration_delay_days: float
    declaration_year: int
    declaration_month: int
    declaration_quarter: int

    # Categorical features
    state: str
    declarationtype: str
    incidenttype: str
    designatedarea: str
    declaration_season: str

    

# ROOT ENDPOINT


@app.get("/")
def welcome_root():

    return {
        "message": "Welcome to TerraNova FEMA Cost Prediction API"
    }



# PREDICTION ENDPOINT

@app.post("/predict")
def predict_cost(disaster: FEMAFeatures):

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