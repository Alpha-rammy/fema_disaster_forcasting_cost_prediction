"""
TerraNova Disaster Cost Recovery Forecasting
Model Training Script
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor



# Paths


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "features_terranova.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)



# Load Data


df = pd.read_csv(
    DATA_PATH,
    parse_dates=["declarationdate"]
)

df = df[df["totalobligated"] > 0].copy()

df = df.sort_values("declarationdate")



# Features

FEATURE_COLS = [
    "state",
    "declarationtype",
    "incidenttype",
    "designated_area_count",
    "declaration_delay_days",
    "ongoing_at_declaration",
    "declaration_year",
    "declaration_month",
    "declaration_weekday",
    "declaration_season",
    "previous_state_disasters",
    "previous_incident_disasters",
    "previous_state_incident_disasters",
    "days_since_previous_state_disaster",
    "days_since_previous_incident"
]

TARGET = "log_totalobligated"

X = df[FEATURE_COLS]

y = df[TARGET]



# Chronological Train/Test Split


split = int(len(df) * 0.80)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]



# Preprocessing


numeric_features = X_train.select_dtypes(
    include=np.number
).columns

categorical_features = X_train.select_dtypes(
    exclude=np.number
).columns

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])



# Models


models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        random_state=42
    ),

    "XGBoost": XGBRegressor(
        random_state=42
    )
}


results = []

best_model = None
best_r2 = -999



# Training Loop


for name, estimator in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2
    })

    if r2 > best_r2:
        best_r2 = r2
        best_model = pipeline



# Results


results_df = pd.DataFrame(results)

print(results_df.sort_values(
    "R²",
    ascending=False
))



# Save Model


joblib.dump(
    best_model,
    os.path.join(
        MODEL_DIR,
        "terranova_disaster_cost_model.pkl"
    )
)

joblib.dump(
    FEATURE_COLS,
    os.path.join(
        MODEL_DIR,
        "feature_columns.pkl"
    )
)

print("\nModel saved successfully.")