import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Dataset not found. Download the Telco Customer Churn CSV and "
        "place it in the project root as "
        "'WA_Fn-UseC_-Telco-Customer-Churn.csv'."
    )

df = pd.read_csv(DATA_PATH)

# Convert TotalCharges to numeric; blank values become NaN.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Keep a compact set of useful, understandable features.
features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "InternetService",
    "OnlineSecurity",
    "TechSupport",
    "PaperlessBilling",
    "PaymentMethod"
]

X = df[features].copy()
y = df["Churn"].astype(str).str.strip()

numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = [
    "Contract",
    "InternetService",
    "OnlineSecurity",
    "TechSupport",
    "PaperlessBilling",
    "PaymentMethod"
]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, predictions)
report = classification_report(
    y_test, predictions, output_dict=True, zero_division=0
)
matrix = confusion_matrix(y_test, predictions).tolist()

joblib.dump(model, MODEL_DIR / "churn_model.joblib")

metrics = {
    "accuracy": accuracy,
    "classification_report": report,
    "confusion_matrix": matrix,
    "training_samples": len(X_train),
    "test_samples": len(X_test),
    "features": features
}

(REPORT_DIR / "metrics.json").write_text(
    json.dumps(metrics, indent=4),
    encoding="utf-8"
)

print("=" * 60)
print("CUSTOMER CHURN MODEL TRAINING COMPLETE")
print("=" * 60)
print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")
print(f"Test accuracy    : {accuracy:.2%}\n")
print(classification_report(y_test, predictions, zero_division=0))
print(f"Model saved to: {MODEL_DIR / 'churn_model.joblib'}")
print(f"Metrics saved to: {REPORT_DIR / 'metrics.json'}")
