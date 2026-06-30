import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")


# PATHS


DATA_RAW = os.path.join("data", "raw")
DATA_PROCESSED = os.path.join("data", "processed")

os.makedirs(DATA_PROCESSED, exist_ok=True)



# LOAD DATA


def load_data():
    declarations = pd.read_csv(
        os.path.join(DATA_RAW, "declarations.csv")
    )

    public_assistance = pd.read_csv(
        os.path.join(DATA_RAW, "public_assistance.csv")
    )

    disaster_summaries = pd.read_csv(
        os.path.join(DATA_RAW, "disaster_summaries.csv")
    )

    print("Raw datasets loaded successfully.")
    print("Declarations:", declarations.shape)
    print("Public Assistance:", public_assistance.shape)
    print("Disaster Summaries:", disaster_summaries.shape)

    return declarations, public_assistance, disaster_summaries



# STANDARDIZE COLUMN NAMES


def standardize_columns(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.lower()
    )

    return df



# CLEAN DECLARATIONS

def clean_declarations(declarations):
    declarations_clean = declarations.copy()

    date_cols = [
        "declarationdate",
        "incidentbegindate",
        "incidentenddate"
    ]

    for col in date_cols:
        if col in declarations_clean.columns:
            declarations_clean[col] = (
                pd.to_datetime(
                    declarations_clean[col],
                    utc=True,
                    errors="coerce"
                )
                .dt.tz_localize(None)
            )

    declarations_clean = declarations_clean.drop_duplicates()

    print("\nDeclarations cleaned.")
    print(declarations_clean.shape)
    print(declarations_clean.isnull().sum())

    return declarations_clean



# CLEAN PUBLIC ASSISTANCE


def clean_public_assistance(public_assistance):
    pa_clean = public_assistance.copy()

    funding_cols = [
        "projectamount",
        "federalshareobligated",
        "totalobligated"
    ]

    for col in funding_cols:
        if col in pa_clean.columns:
            pa_clean[col] = pd.to_numeric(
                pa_clean[col],
                errors="coerce"
            )

            pa_clean[col] = pa_clean[col].fillna(0)
            pa_clean[col] = pa_clean[col].clip(lower=0)

    pa_clean = pa_clean.drop_duplicates()

    print("\nPublic Assistance cleaned.")
    print(pa_clean.shape)
    print(pa_clean.isnull().sum())

    return pa_clean



# CLEAN DISASTER SUMMARIES


def clean_disaster_summaries(disaster_summaries):
    summaries_clean = disaster_summaries.copy()

    date_cols = [
        "paloaddate",
        "ialoaddate"
    ]

    for col in date_cols:
        if col in summaries_clean.columns:
            summaries_clean[col] = (
                pd.to_datetime(
                    summaries_clean[col],
                    utc=True,
                    errors="coerce"
                )
                .dt.tz_localize(None)
            )

    numeric_cols = summaries_clean.select_dtypes(
        include=["int64", "float64"]
    ).columns

    for col in numeric_cols:
        summaries_clean[col] = pd.to_numeric(
            summaries_clean[col],
            errors="coerce"
        )

        summaries_clean[col] = summaries_clean[col].fillna(0)
        summaries_clean[col] = summaries_clean[col].clip(lower=0)

    summaries_clean = summaries_clean.drop_duplicates()

    print("\nDisaster Summaries cleaned.")
    print(summaries_clean.shape)
    print(summaries_clean.isnull().sum())

    return summaries_clean



# REFERENTIAL INTEGRITY CHECK


def check_referential_integrity(
    declarations_clean,
    pa_clean,
    summaries_clean
):
    valid_disasters = set(
        declarations_clean["disasternumber"]
    )

    before_pa = len(pa_clean)
    before_summaries = len(summaries_clean)

    pa_clean = pa_clean[
        pa_clean["disasternumber"].isin(valid_disasters)
    ]

    summaries_clean = summaries_clean[
        summaries_clean["disasternumber"].isin(valid_disasters)
    ]

    print("\nReferential integrity check completed.")
    print(f"Public Assistance rows dropped: {before_pa - len(pa_clean)}")
    print(f"Disaster Summary rows dropped: {before_summaries - len(summaries_clean)}")

    return pa_clean, summaries_clean



# SAVE CLEAN DATA


def save_clean_data(
    declarations_clean,
    pa_clean,
    summaries_clean
):
    files = {
        "declarations_clean.csv": declarations_clean,
        "public_assistance_clean.csv": pa_clean,
        "disaster_summaries_clean.csv": summaries_clean
    }

    for filename, df in files.items():
        path = os.path.join(DATA_PROCESSED, filename)
        df.to_csv(path, index=False)
        print(f"{filename} saved successfully.")



# MAIN FUNCTION


def main():
    declarations, public_assistance, disaster_summaries = load_data()

    declarations = standardize_columns(declarations)
    public_assistance = standardize_columns(public_assistance)
    disaster_summaries = standardize_columns(disaster_summaries)

    declarations_clean = clean_declarations(declarations)

    pa_clean = clean_public_assistance(public_assistance)

    summaries_clean = clean_disaster_summaries(disaster_summaries)

    pa_clean, summaries_clean = check_referential_integrity(
        declarations_clean,
        pa_clean,
        summaries_clean
    )

    print("\nRow count - raw vs clean")
    print(f"Declarations        : {len(declarations):,} -> {len(declarations_clean):,}")
    print(f"Public Assistance   : {len(public_assistance):,} -> {len(pa_clean):,}")
    print(f"Disaster Summaries  : {len(disaster_summaries):,} -> {len(summaries_clean):,}")

    save_clean_data(
        declarations_clean,
        pa_clean,
        summaries_clean
    )

    print("\nPreprocessing completed successfully.")


if __name__ == "__main__":
    main()