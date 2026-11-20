# Customer Churn Prediction System

Predict whether a customer is likely to leave using the IBM Telco Customer Churn dataset. The target is `Churn` (`0` = stay, `1` = churn).

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. For model retraining or EDA, put `WA_Fn-UseC_-Telco-Customer-Churn.csv` in `data/`.

The raw dataset is intentionally excluded from version control. The trained inference pipeline is included at `models/churn_pipeline.joblib`, so the prediction app does not need the CSV at runtime. `customerID` is excluded from model features. Blank `TotalCharges` values are converted to missing and imputed inside the model pipeline.

## Run the project

```bash
python -m src.eda
python -m src.train
streamlit run app.py
```

Training compares Logistic Regression, Decision Tree, Random Forest, and XGBoost (when installed), using a stratified 80/20 holdout. It reports accuracy, precision, recall, F1, and ROC-AUC; the highest ROC-AUC model is saved as `models/churn_pipeline.joblib` and metrics as `reports/model_metrics.csv`.

## Project structure

```text
app.py                  Streamlit prediction interface
data/                   Place the downloaded CSV here
models/                 Trained pipeline output
reports/                Metrics and generated charts
src/data.py             Loading and cleaning
src/eda.py              Exploratory charts
src/train.py            Preprocessing, training, evaluation
```

The probability is an estimate from the trained model, not a guarantee of customer behavior. Choose an outreach threshold based on business costs and validation results rather than treating 50% as a universal decision boundary.

## Deploy to Streamlit Community Cloud

The app entrypoint is `app.py`; the model file and `requirements.txt` must be present in the GitHub repository. The model was serialized with scikit-learn 1.9.0, which is pinned in the requirements file for compatibility. The raw customer CSV is not required for deployment and remains excluded.

1. Push this project to a GitHub repository. Include `models/churn_pipeline.joblib`; do not add the raw CSV from `data/`.
2. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub and choose **Create app**.
3. Select the repository, branch, and `app.py` as the entrypoint. Choose Python 3.12 in the advanced settings.
4. Deploy. Check the app logs if dependency installation or model loading fails.

Community Cloud deploys from GitHub and requires admin access to the selected repository. See the [official deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).
