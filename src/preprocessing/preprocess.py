import os
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")


DATA_RAW = os.path.join("data", "raw")
DATA_PROCESSED = os.path.join("data", "processed")

os.makedirs(DATA_PROCESSED, exist_ok=True)


def standardize_columns(df):
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.lower()
    )
    return df


def load_datasets():
    declarations = pd.read_csv(os.path.join(DATA_RAW, "declarations.csv"))
    public_assistance = pd.read_csv(os.path.join(DATA_RAW, "public_assistance.csv"))
    disaster_summaries = pd.read_csv(os.path.join(DATA_RAW, "disaster_summaries.csv"))

    print("Raw data loaded")
    print("Declarations:", declarations.shape)
    print("Public Assistance:", public_assistance.shape)
    print("Disaster Summaries:", disaster_summaries.shape)

    return declarations, public_assistance, disaster_summaries


def clean_declarations(declarations):
    declarations_clean = standardize_columns(declarations)

    date_cols = [
        "declarationdate",
        "incidentbegindate",
        "incidentenddate"
    ]

    for col in date_cols:
        if col in declarations_clean.columns:
            declarations_clean[col] = pd.to_datetime(
                declarations_clean[col],
                errors="coerce"
            )

    cat_cols = [
        "state",
        "declarationtype",
        "incidenttype",
        "designatedarea",
        "region"
    ]

    for col in cat_cols:
        if col in declarations_clean.columns:
            declarations_clean[col] = (
                declarations_clean[col]
                .astype(str)
                .str.strip()
                .str.title()
            )

    if "disasternumber" in declarations_clean.columns:
        declarations_clean["disasternumber"] = declarations_clean["disasternumber"].astype(int)

    declarations_clean = declarations_clean.drop_duplicates()

    return declarations_clean


def clean_public_assistance(public_assistance):
    pa_clean = standardize_columns(public_assistance)

    if "disasternumber" in pa_clean.columns:
        pa_clean["disasternumber"] = pa_clean["disasternumber"].astype(int)

    funding_cols = [
        "projectamount",
        "federalshareobligated",
        "totalobligated"
    ]

    for col in funding_cols:
        if col in pa_clean.columns:
            pa_clean[col] = pd.to_numeric(pa_clean[col], errors="coerce")
            pa_clean[col] = pa_clean[col].fillna(0)
            pa_clean[col] = pa_clean[col].clip(lower=0)

    pa_cat_cols = [
        "projectsize",
        "damagecategorycode",
        "applicationtitle",
        "stateabbreviation"
    ]

    for col in pa_cat_cols:
        if col in pa_clean.columns:
            pa_clean[col] = (
                pa_clean[col]
                .astype(str)
                .str.strip()
                .str.title()
            )

    pa_clean = pa_clean.drop_duplicates()

    return pa_clean


def clean_disaster_summaries(disaster_summaries):
    summaries_clean = standardize_columns(disaster_summaries)

    summary_date_cols = [
        "paloaddate",
        "ialoaddate"
    ]

    for col in summary_date_cols:
        if col in summaries_clean.columns:
            summaries_clean[col] = pd.to_datetime(
                summaries_clean[col],
                errors="coerce"
            )

    if "disasternumber" in summaries_clean.columns:
        summaries_clean["disasternumber"] = summaries_clean["disasternumber"].astype(int)

    money_cols = [
        "totalnumberiaapproved",
        "totalamountihpapproved",
        "totalamounthaapproved",
        "totalamountonaapproved",
        "totalobligatedamountpa",
        "totalobligatedamountcatab",
        "totalobligatedamountcatc2g",
        "totalobligatedamounthmgp"
    ]

    for col in money_cols:
        if col in summaries_clean.columns:
            summaries_clean[col] = pd.to_numeric(
                summaries_clean[col],
                errors="coerce"
            )
            summaries_clean[col] = summaries_clean[col].fillna(0)
            summaries_clean[col] = summaries_clean[col].clip(lower=0)

    summaries_clean = summaries_clean.drop_duplicates()

    return summaries_clean


def check_referential_integrity(declarations_clean, pa_clean, summaries_clean):
    valid_disasters = set(declarations_clean["disasternumber"])

    before_pa = len(pa_clean)
    before_summaries = len(summaries_clean)

    pa_clean = pa_clean[pa_clean["disasternumber"].isin(valid_disasters)]
    summaries_clean = summaries_clean[summaries_clean["disasternumber"].isin(valid_disasters)]

    print("Referential integrity check completed")
    print(f"Public Assistance rows dropped: {before_pa - len(pa_clean)}")
    print(f"Disaster Summary rows dropped: {before_summaries - len(summaries_clean)}")

    return pa_clean, summaries_clean


def save_clean_data(declarations_clean, pa_clean, summaries_clean):
    files = {
        "declarations_clean.csv": declarations_clean,
        "public_assistance_clean.csv": pa_clean,
        "disaster_summaries_clean.csv": summaries_clean
    }

    for filename, df in files.items():
        output_path = os.path.join(DATA_PROCESSED, filename)
        df.to_csv(output_path, index=False)
        print(f"{filename} saved successfully")


def main():
    declarations, public_assistance, disaster_summaries = load_datasets()

    declarations_clean = clean_declarations(declarations)
    pa_clean = clean_public_assistance(public_assistance)
    summaries_clean = clean_disaster_summaries(disaster_summaries)

    pa_clean, summaries_clean = check_referential_integrity(
        declarations_clean,
        pa_clean,
        summaries_clean
    )

    print("\nRow count - raw vs clean")
    print(f"declarations         : {len(declarations):>7,} -> {len(declarations_clean):>7,}")
    print(f"public_assistance    : {len(public_assistance):>7,} -> {len(pa_clean):>7,}")
    print(f"disaster_summaries   : {len(disaster_summaries):>7,} -> {len(summaries_clean):>7,}")

    save_clean_data(declarations_clean, pa_clean, summaries_clean)


if __name__ == "__main__":
    main()