"""Load and clean the IBM Telco Customer Churn dataset."""
from pathlib import Path

import pandas as pd


DEFAULT_DATA_PATH = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
TARGET = "Churn"


def load_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and apply basic, reproducible cleaning."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download the IBM Telco Customer Churn CSV "
            "and place it at data/WA_Fn-UseC_-Telco-Customer-Churn.csv."
        )
    data = pd.read_csv(path)

    required = {TARGET, "customerID"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
    

    data = data.drop_duplicates().copy()
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"].replace(" ", pd.NA), errors="coerce")  #["100", "200", "abc"] -> [100, 200, NaN]
    data = data.dropna(subset=[TARGET]).copy()
    return data
