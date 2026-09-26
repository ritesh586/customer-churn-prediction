import sys
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path("models/churn_model.joblib")

if not MODEL_PATH.exists():
    print("Model not found. Run: python3 train_model.py")
    raise SystemExit(1)

if len(sys.argv) != 10:
    print("Usage:")
    print(
        'python3 predict.py TENURE MONTHLY TOTAL CONTRACT INTERNET '
        'SECURITY SUPPORT PAPERLESS PAYMENT'
    )
    raise SystemExit(1)

_, tenure, monthly, total, contract, internet, security, support, paperless, payment = sys.argv

customer = pd.DataFrame([{
    "tenure": float(tenure),
    "MonthlyCharges": float(monthly),
    "TotalCharges": float(total),
    "Contract": contract,
    "InternetService": internet,
    "OnlineSecurity": security,
    "TechSupport": support,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment
}])

model = joblib.load(MODEL_PATH)
prediction = model.predict(customer)[0]
probability = model.predict_proba(customer)[0][1]

print(f"Prediction: {'LIKELY TO CHURN' if prediction == 'Yes' else 'LIKELY TO STAY'}")
print(f"Churn probability: {probability:.2%}")
