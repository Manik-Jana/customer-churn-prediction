"""Create a small set of exploratory charts from the cleaned dataset."""
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from src.data import DEFAULT_DATA_PATH, TARGET, load_data


def main():
    data = load_data(DEFAULT_DATA_PATH)
    out = Path("reports/figures")
    out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=data, x=TARGET, hue=TARGET, palette="Set2", legend=False, ax=ax)
    ax.set_title("Customer churn distribution")
    fig.tight_layout()
    fig.savefig(out / "churn_distribution.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=data, x="tenure", hue=TARGET, bins=30, multiple="stack", ax=ax)
    ax.set_title("Tenure by churn status")
    fig.tight_layout()
    fig.savefig(out / "tenure_by_churn.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=data, x=TARGET, y="MonthlyCharges", hue=TARGET, palette="Set2", legend=False, ax=ax)
    ax.set_title("Monthly charges by churn status")
    fig.tight_layout()
    fig.savefig(out / "monthly_charges_by_churn.png", dpi=160)
    plt.close(fig)
    print(f"Saved EDA charts to {out}")


if __name__ == "__main__":
    main()
