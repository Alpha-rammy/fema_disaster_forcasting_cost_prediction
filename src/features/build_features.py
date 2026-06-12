import os
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")


DATA_PROCESSED = os.path.join("data", "processed")


def load_clean_data():
    declarations = pd.read_csv(
        os.path.join(DATA_PROCESSED, "declarations_clean.csv")
    )

    pa = pd.read_csv(
        os.path.join(DATA_PROCESSED, "public_assistance_clean.csv")
    )

    summaries = pd.read_csv(
        os.path.join(DATA_PROCESSED, "disaster_summaries_clean.csv")
    )

    print("Clean data loaded successfully")
    print("Declarations:", declarations.shape)
    print("Public Assistance:", pa.shape)
    print("Disaster Summaries:", summaries.shape)

    return declarations, pa, summaries


def convert_declaration_dates(declarations):
    declarations = declarations.copy()

    date_cols = [
        "declarationdate",
        "incidentbegindate",
        "incidentenddate"
    ]

    for col in date_cols:
        if col in declarations.columns:
            declarations[col] = pd.to_datetime(
                declarations[col],
                errors="coerce"
            )

    declarations["fydeclared"] = declarations["fydeclared"].fillna(
        declarations["declarationdate"].dt.year
    )

    return declarations


def create_base_features(declarations):
    base = declarations[
        [
            "disasternumber",
            "state",
            "declarationtype",
            "incidenttype",
            "fydeclared",
            "designatedarea",
            "declarationdate",
            "incidentbegindate",
            "incidentenddate"
        ]
    ].copy()

    base = base.drop_duplicates(subset=["disasternumber"])

    return base


def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"


def create_temporal_features(base):
    base = base.copy()

    base["disaster_duration_days"] = (
        base["incidentenddate"] - base["incidentbegindate"]
    ).dt.days

    base["declaration_delay_days"] = (
        base["declarationdate"] - base["incidentbegindate"]
    ).dt.days

    base["disaster_duration_days"] = base["disaster_duration_days"].clip(lower=0)
    base["declaration_delay_days"] = base["declaration_delay_days"].clip(lower=0)

    base["declaration_year"] = base["declarationdate"].dt.year
    base["declaration_month"] = base["declarationdate"].dt.month
    base["declaration_quarter"] = base["declarationdate"].dt.quarter

    base["declaration_season"] = base["declaration_month"].apply(get_season)

    return base


def create_pa_aggregation_features(pa):
    pa_features = (
        pa.groupby("disasternumber")
        .agg(
            pa_project_count=("projectamount", "count"),
            pa_project_amount_total=("projectamount", "sum"),
            pa_project_amount_mean=("projectamount", "mean"),
            pa_project_amount_median=("projectamount", "median"),
            pa_project_amount_max=("projectamount", "max"),
            pa_project_amount_std=("projectamount", "std"),
            pa_obligated_total=("federalshareobligated", "sum"),
            pa_obligated_mean=("federalshareobligated", "mean"),
            pa_obligated_max=("federalshareobligated", "max"),
        )
        .reset_index()
    )

    pa_features = pa_features.fillna(0)

    return pa_features


def create_project_size_features(pa, pa_features):
    pa_features = pa_features.copy()

    large_threshold = pa["projectamount"].quantile(0.75)
    small_threshold = pa["projectamount"].quantile(0.25)

    large_counts = (
        pa[pa["projectamount"] >= large_threshold]
        .groupby("disasternumber")
        .size()
    )

    small_counts = (
        pa[pa["projectamount"] <= small_threshold]
        .groupby("disasternumber")
        .size()
    )

    pa_features["large_project_count"] = (
        pa_features["disasternumber"]
        .map(large_counts)
        .fillna(0)
    )

    pa_features["small_project_count"] = (
        pa_features["disasternumber"]
        .map(small_counts)
        .fillna(0)
    )

    return pa_features


def create_ratio_features(pa_features):
    pa_features = pa_features.copy()

    pa_features["avg_obligation_per_project"] = (
        pa_features["pa_obligated_total"] /
        (pa_features["pa_project_count"] + 1)
    )

    pa_features["funding_intensity"] = (
        pa_features["pa_obligated_total"] /
        (pa_features["pa_project_amount_total"] + 1)
    )

    pa_features["large_project_ratio"] = (
        pa_features["large_project_count"] /
        (pa_features["pa_project_count"] + 1)
    )

    pa_features["small_project_ratio"] = (
        pa_features["small_project_count"] /
        (pa_features["pa_project_count"] + 1)
    )

    return pa_features


def create_target_features(pa):
    summary_features = (
        pa.groupby("disasternumber")
        .agg(
            totalobligated=("totalobligated", "sum")
        )
        .reset_index()
    )

    summary_features["log_totalobligated"] = np.log1p(
        summary_features["totalobligated"]
    )

    return summary_features


def create_dsf_features(pa_features):
    dsf = pa_features[
        [
            "disasternumber",
            "pa_project_count",
            "pa_project_amount_total",
            "pa_obligated_total"
        ]
    ].copy()

    dsf["dsf_scale_score"] = pd.qcut(
        dsf["pa_project_count"].rank(method="first"),
        q=5,
        labels=[1, 2, 3, 4, 5]
    ).astype(int)

    dsf["dsf_project_amount_score"] = pd.qcut(
        dsf["pa_project_amount_total"].rank(method="first"),
        q=5,
        labels=[1, 2, 3, 4, 5]
    ).astype(int)

    dsf["dsf_funding_score"] = pd.qcut(
        dsf["pa_obligated_total"].rank(method="first"),
        q=5,
        labels=[1, 2, 3, 4, 5]
    ).astype(int)

    dsf["dsf_combined_score"] = (
        dsf["dsf_scale_score"] +
        dsf["dsf_project_amount_score"] +
        dsf["dsf_funding_score"]
    ) / 3

    dsf["dsf_combined_score"] = dsf["dsf_combined_score"].round()

    return dsf[
        [
            "disasternumber",
            "dsf_scale_score",
            "dsf_project_amount_score",
            "dsf_funding_score",
            "dsf_combined_score"
        ]
    ]


def merge_features(base, pa_features, dsf_features, target_features):
    features_fema = (
        base
        .merge(pa_features, on="disasternumber", how="left")
        .merge(dsf_features, on="disasternumber", how="left")
        .merge(target_features, on="disasternumber", how="left")
    )

    features_fema = features_fema.fillna(0)

    print(
        f"Feature matrix: {features_fema.shape[0]:,} disasters x "
        f"{features_fema.shape[1]} columns"
    )

    print(f"Null count: {features_fema.isnull().sum().sum()}")

    return features_fema


def drop_unused_columns(features_fema):
    features_fema = features_fema.drop(
        columns=[
            "declarationdate",
            "incidentbegindate",
            "incidentenddate",
            "paloaddate",
            "ialoaddate"
        ],
        errors="ignore"
    )

    return features_fema


def save_features(features_fema):
    output_path = os.path.join(DATA_PROCESSED, "features_fema.csv")

    features_fema.to_csv(output_path, index=False)

    print("Saved unencoded features to:", output_path)
    print("Shape:", features_fema.shape)
    print("Categorical columns:")
    print(features_fema.select_dtypes(include=["object"]).columns.tolist())


def main():
    declarations, pa, summaries = load_clean_data()

    declarations = convert_declaration_dates(declarations)

    base = create_base_features(declarations)
    base = create_temporal_features(base)

    pa_features = create_pa_aggregation_features(pa)
    pa_features = create_project_size_features(pa, pa_features)
    pa_features = create_ratio_features(pa_features)

    target_features = create_target_features(pa)
    dsf_features = create_dsf_features(pa_features)

    features_fema = merge_features(
        base=base,
        pa_features=pa_features,
        dsf_features=dsf_features,
        target_features=target_features
    )

    features_fema = drop_unused_columns(features_fema)

    save_features(features_fema)


if __name__ == "__main__":
    main()