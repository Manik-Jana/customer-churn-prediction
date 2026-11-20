"""Train and compare churn classifiers, then save the best pipeline."""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.data import DEFAULT_DATA_PATH, TARGET, load_data


def build_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                  class_weight="balanced", random_state=42,
                                                  n_jobs=-1),
    }


def main(data_path=DEFAULT_DATA_PATH):
    data = load_data(data_path)
    X = data.drop(columns=[TARGET, "customerID"])
    y = data[TARGET].map({"No": 0, "Yes": 1})
    if y.isna().any():
        raise ValueError("Churn must contain only 'Yes' and 'No'.")

    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()


    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                 ("encode", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("numeric", numeric_pipe, numeric),
                              ("categorical", categorical_pipe, categorical)])

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = []
    candidates = build_models()

    try:
        from xgboost import XGBClassifier
        candidates["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=42, n_jobs=-1,
        )
    except ImportError:
        print("XGBoost is not installed; skipping it.")


    fitted = {}
    for name, estimator in candidates.items():
        model = Pipeline([("preprocess", prep), ("model", estimator)])
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        results.append({
            "model": name,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probabilities),
        })
        fitted[name] = model

    report = pd.DataFrame(results).sort_values("roc_auc", ascending=False)

    print(report.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    best_name = report.iloc[0]["model"]
    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    
    joblib.dump(fitted[best_name], "models/churn_pipeline.joblib")
    report.to_csv("reports/model_metrics.csv", index=False)
    Path("reports/best_model.json").write_text(json.dumps({"model": best_name}, indent=2))
    print(f"Saved {best_name} pipeline to models/churn_pipeline.joblib")


if __name__ == "__main__":
    main()
