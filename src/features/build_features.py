import os
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")



# PATHS


DATA_PROCESSED = os.path.join("data", "processed")

FEATURE_FILE = os.path.join(DATA_PROCESSED, "features_fema.csv")



# LOAD CLEAN DATA


def load_clean_data():
    declarations = pd.read_csv(
        os.path.join(DATA_PROCESSED, "declarations_clean.csv"),
        parse_dates=[
            "declarationdate",
            "incidentbegindate",
            "incidentenddate"
        ]
    )

    public_assistance = pd.read_csv(
        os.path.join(DATA_PROCESSED, "public_assistance_clean.csv")
    )

    print("Clean datasets loaded successfully.")
    print("Declarations:", declarations.shape)
    print("Public Assistance:", public_assistance.shape)

    return declarations, public_assistance



# DECLARATION FEATURE ENGINEERING


def engineer_declaration_features(declarations):
    declarations = declarations.copy()

    # Duration of disaster
    declarations["disaster_duration_days"] = (
        declarations["incidentenddate"] -
        declarations["incidentbegindate"]
    ).dt.days

    # Delay between incident start and declaration
    declarations["declaration_delay_days"] = (
        declarations["declarationdate"] -
        declarations["incidentbegindate"]
    ).dt.days

    # Clean negative and missing values
    declarations["disaster_duration_days"] = (
        declarations["disaster_duration_days"]
        .clip(lower=0)
        .fillna(0)
    )

    declarations["declaration_delay_days"] = (
        declarations["declaration_delay_days"]
        .clip(lower=0)
        .fillna(0)
    )

    # Date-derived features
    declarations["declaration_year"] = declarations["declarationdate"].dt.year
    declarations["declaration_month"] = declarations["declarationdate"].dt.month
    declarations["declaration_quarter"] = declarations["declarationdate"].dt.quarter

    declarations["declaration_season"] = (
        declarations["declaration_month"]
        .apply(get_season)
    )

    # Count-based engineered features
    declarations["fy_declaration_count"] = (
        declarations
        .groupby("fydeclared")["disasternumber"]
        .transform("count")
    )

    declarations["state_declaration_count"] = (
        declarations
        .groupby("state")["disasternumber"]
        .transform("count")
    )

    declarations["incident_frequency"] = (
        declarations
        .groupby("incidenttype")["disasternumber"]
        .transform("count")
    )

    declarations["state_incident_count"] = (
        declarations
        .groupby(["state", "incidenttype"])["disasternumber"]
        .transform("count")
    )

    declarations["state_year_count"] = (
        declarations
        .groupby(["state", "fydeclared"])["disasternumber"]
        .transform("count")
    )

    # Disaster-level aggregate features
    agg_features = (
        declarations
        .groupby("disasternumber")
        .agg(
            avg_duration_days=("disaster_duration_days", "mean"),
            avg_delay_days=("declaration_delay_days", "mean")
        )
        .round(2)
        .reset_index()
    )

    # Drop raw date columns
    declarations = declarations.drop(
        columns=[
            "declarationdate",
            "incidentbegindate",
            "incidentenddate"
        ]
    )

    declaration_features = declarations.merge(
        agg_features,
        on="disasternumber",
        how="left"
    )

    declaration_features = declaration_features.drop_duplicates(
        subset=["disasternumber"]
    )

    print("Declaration features created.")
    print(declaration_features.shape)

    return declaration_features


def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"



# TARGET CREATION FROM PUBLIC ASSISTANCE


def create_target(public_assistance):
    pa_target = (
        public_assistance
        .groupby("disasternumber", as_index=False)
        .agg(
            totalobligated=("totalobligated", "sum")
        )
    )

    pa_target["log_totalobligated"] = np.log1p(
        pa_target["totalobligated"]
    )

    pa_target = pa_target.drop_duplicates(
        subset=["disasternumber"]
    )

    print("Target created from Public Assistance.")
    print(pa_target.shape)

    return pa_target


# =====================================================
# MERGE FINAL FEATURE MATRIX
# =====================================================

def build_feature_matrix(declaration_features, pa_target):
    features_fema = (
        declaration_features
        .merge(
            pa_target,
            on="disasternumber",
            how="left"
        )
    )

    features_fema = features_fema.fillna(0)

    print("Final feature matrix created.")
    print(features_fema.shape)

    return features_fema



# SAVE FEATURES


def save_features(features_fema):
    os.makedirs(DATA_PROCESSED, exist_ok=True)

    features_fema.to_csv(
        FEATURE_FILE,
        index=False
    )

    print("Feature matrix saved successfully.")
    print("Saved to:", FEATURE_FILE)



# MAIN


def main():
    declarations, public_assistance = load_clean_data()

    declaration_features = engineer_declaration_features(
        declarations
    )

    pa_target = create_target(
        public_assistance
    )

    features_fema = build_feature_matrix(
        declaration_features,
        pa_target
    )

    save_features(features_fema)

    print("Feature engineering completed successfully.")


if __name__ == "__main__":
    main()