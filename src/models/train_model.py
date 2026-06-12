import os
import warnings
import joblib
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor


DATA_PROCESSED = os.path.join("data", "processed")
MODEL_DIR = os.path.join("models")

os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(os.path.join(DATA_PROCESSED, "features_fema.csv"))

    print("Feature data loaded")
    print(df.shape)
    print("Categorical columns:")
    print(df.select_dtypes(include=["object"]).columns.tolist())

    return df


def prepare_target(df):
    df = df.dropna(subset=["totalobligated"]).copy()
    df["log_totalobligated"] = np.log1p(df["totalobligated"])

    print(f"Disasters: {len(df):,}")
    print(df["log_totalobligated"].describe())

    return df


def select_features(df):
    exclude = [
        # Target
        "totalobligated",
        "log_totalobligated",

        # Identifier
        "disasternumber",

        # Direct obligation leakage
        "pa_obligated_total",
        "pa_obligated_mean",
        "pa_obligated_max",

        # Federal share leakage
        "federalShareObligated",
        "federalShareObligated_total",
        "avg_federalShareObligated",

        # Derived from obligations
        "avg_obligation_per_project",
        "funding_intensity",

        # Project amount leakage
        "pa_project_amount_total",
        "pa_project_amount_mean",
        "pa_project_amount_median",
        "pa_project_amount_max",
        "pa_project_amount_std",

        # Disaster summary dollar amounts
        "totalAmountIhpApproved",
        "totalAmountHaApproved",
        "totalAmountOnaApproved",
        "log_totalAmountIhpApproved",
        "log_totalAmountHaApproved",
        "log_totalAmountOnaApproved",
        "totalObligatedAmountPa",
        "totalObligatedAmountCatAb",
        "totalObligatedAmountCatC2g",
        "totalObligatedAmountHmg",
        "log_totalObligatedAmountPa",
        "log_totalObligatedAmountCatAb",
        "log_totalObligatedAmountCatC2g",
        "log_totalObligatedAmountHmg",

        # DSF leakage
        "dsf_combined_score",
        "dsf_funding_score",
        "dsf_scale_score",
        "dsf_project_amount_score",

        # Strong financial proxy features
        "large_project_ratio",
        "small_project_ratio",
        "pa_project_count",
        "large_project_count",
        "small_project_count",
    ]

    feature_cols = [c for c in df.columns if c not in exclude]

    print("Features selected:", len(feature_cols))
    print("Remaining obligated columns:")
    print([c for c in feature_cols if "oblig" in c.lower()])
    print("Remaining amount columns:")
    print([c for c in feature_cols if "amount" in c.lower()])

    X = df[feature_cols]
    y = df["log_totalobligated"]

    return X, y, feature_cols


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print(f"Train: {len(X_train):,}")
    print(f"Test : {len(X_test):,}")

    return X_train, X_test, y_train, y_test


def build_preprocessor(X_train):
    numeric_features = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object"]
    ).columns.tolist()

    print("Numeric features:", len(numeric_features))
    print("Categorical features:", len(categorical_features))
    print(categorical_features)

    numeric_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features)
        ]
    )

    return preprocessor


def define_models():
    models = {
        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),

        "XGBoost": XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        )
    }

    return models


def train_and_evaluate(models, preprocessor, X_train, X_test, y_train, y_test):
    results = {}
    trained_models = {}

    for name, model in models.items():

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )

        pipeline.fit(X_train, y_train)

        preds_log = pipeline.predict(X_test)

        mae_log = mean_absolute_error(y_test, preds_log)
        rmse_log = np.sqrt(mean_squared_error(y_test, preds_log))
        r2 = r2_score(y_test, preds_log)

        y_test_original = np.expm1(y_test)
        preds_original = np.expm1(preds_log)

        mae_original = mean_absolute_error(y_test_original, preds_original)
        rmse_original = np.sqrt(
            mean_squared_error(y_test_original, preds_original)
        )

        results[name] = {
            "MAE_log": mae_log,
            "RMSE_log": rmse_log,
            "R2": r2,
            "MAE_original": mae_original,
            "RMSE_original": rmse_original
        }

        trained_models[name] = pipeline

        print(
            f"{name:25s} | "
            f"RMSE log: {rmse_log:.4f} | "
            f"R2: {r2:.4f} | "
            f"MAE original: {mae_original:,.2f}"
        )

    results_df = (
        pd.DataFrame(results)
        .T
        .sort_values("RMSE_log")
    )

    return results_df, trained_models


def save_best_model(results_df, trained_models):
    best_name = results_df.index[0]
    best_model = trained_models[best_name]

    model_path = os.path.join(MODEL_DIR, "fema_cost_model.pkl")

    joblib.dump(best_model, model_path)

    print("\nModel Comparison:")
    print(results_df)

    print(f"\nBest model saved: {best_name}")
    print("Saved to:", model_path)

    return best_model, best_name


def main():
    df = load_data()
    df = prepare_target(df)

    X, y, feature_cols = select_features(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    preprocessor = build_preprocessor(X_train)

    models = define_models()

    results_df, trained_models = train_and_evaluate(
        models,
        preprocessor,
        X_train,
        X_test,
        y_train,
        y_test
    )

    save_best_model(results_df, trained_models)


if __name__ == "__main__":
    main()