import requests
import pandas as pd
import time
from pathlib import Path

BASE_URL = "https://www.fema.gov/api/open"

ENDPOINTS = {
    "declarations": f"{BASE_URL}/v2/DisasterDeclarationsSummaries",
    "public_assistance": f"{BASE_URL}/v2/PublicAssistanceFundedProjectsDetails",
    "disaster_summaries": f"{BASE_URL}/v1/FemaWebDisasterSummaries",
}

FIELDS = {
    "declarations": [
        "disasterNumber",
        "state",
        "declarationType",
        "incidentType",
        "declarationDate",
        "incidentBeginDate",
        "incidentEndDate",
        "fyDeclared",
        "designatedArea",
        "declarationRequestNumber",
    ],

    "public_assistance": [
        "disasterNumber",
        "stateAbbreviation",
        "projectAmount",
        "federalShareObligated",
        "totalObligated",
        "projectSize",
        "damageCategoryCode",
        "applicationTitle",
    ],

    "disaster_summaries": [
    "disasterNumber",
    "totalNumberIaApproved",
    "totalAmountIhpApproved",
    "totalAmountHaApproved",
    "totalAmountOnaApproved",
    "totalObligatedAmountPa",
    "totalObligatedAmountCatAb",
    "totalObligatedAmountCatC2g",
    "totalObligatedAmountHmgp",
    "paLoadDate",
    "iaLoadDate",
],

}

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def get_existing_rows(file_path):
    if file_path.exists():
        return len(pd.read_csv(file_path))
    return 0


def append_batch_to_csv(df, file_path):
    file_exists = file_path.exists()

    df.to_csv(
        file_path,
        mode="a",
        header=not file_exists,
        index=False
    )


def fetch_api(endpoint: str, fields: list[str], name: str) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    save_path = DATA_DIR / f"{name}.csv"

    page_size = 1000
    existing_rows = get_existing_rows(save_path)

    params = {
        "$format": "json",
        "$select": ",".join(fields),
        "$top": page_size,
        "$skip": existing_rows,
    }

    print(f"Starting {name} from skip={existing_rows}")

    while True:
        try:
            resp = requests.get(endpoint, params=params, timeout=60)
            print(f"Status: {resp.status_code} | skip={params['$skip']}")
            resp.raise_for_status()

        except requests.exceptions.SSLError:
            print(f"SSL error at skip={params['$skip']}. Waiting and retrying...")
            time.sleep(20)
            continue

        except requests.exceptions.HTTPError:
            print(f"HTTP error at skip={params['$skip']}. Waiting and retrying...")
            time.sleep(20)
            continue

        except requests.exceptions.RequestException:
            print(f"Request error at skip={params['$skip']}. Waiting and retrying...")
            time.sleep(20)
            continue

        payload = resp.json()

        data_key = next(
            k for k in payload.keys()
            if k not in ("metadata", "count")
        )

        data = payload[data_key]

        if len(data) == 0:
            print(f"No more records for {name}.")
            break

        batch_df = pd.DataFrame(data)

        append_batch_to_csv(batch_df, save_path)

        print(f"Saved batch. Total fetched up to skip={params['$skip'] + len(data)}")

        if len(data) < page_size:
            break

        params["$skip"] += page_size

        time.sleep(5)

    return pd.read_csv(save_path)


def fetch_dataset(name: str):
    endpoint = ENDPOINTS[name]
    fields = FIELDS[name]

    return fetch_api(endpoint, fields, name)


def run_ingestion():
    for name in ENDPOINTS:
        print(f"\nFetching {name} dataset...")
        fetch_dataset(name)


if __name__ == "__main__":
    run_ingestion()