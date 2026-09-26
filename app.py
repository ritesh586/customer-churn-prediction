import os
import pandas as pd
import joblib

MODEL_PATH = "models/churn_model.joblib"

def main():
    print("=" * 60)
    print("             CUSTOMER CHURN PREDICTION")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print("\nModel not found.")
        print("First run: python3 train_model.py")
        return

    model = joblib.load(MODEL_PATH)

    print("\nEnter customer information.")
    print("Type 'exit' at any prompt to close.\n")

    try:
        tenure = input("Tenure in months: ")
        if tenure.lower() == "exit":
            return

        monthly_charges = input("Monthly charges: ")
        if monthly_charges.lower() == "exit":
            return

        total_charges = input("Total charges: ")
        if total_charges.lower() == "exit":
            return

        contract = input("Contract (Month-to-month / One year / Two year): ")
        if contract.lower() == "exit":
            return

        internet_service = input("Internet service (DSL / Fiber optic / No): ")
        if internet_service.lower() == "exit":
            return

        online_security = input("Online security (Yes / No): ")
        if online_security.lower() == "exit":
            return

        tech_support = input("Tech support (Yes / No): ")
        if tech_support.lower() == "exit":
            return

        paperless_billing = input("Paperless billing (Yes / No): ")
        if paperless_billing.lower() == "exit":
            return

        payment_method = input(
            "Payment method (Electronic check / Mailed check / "
            "Bank transfer (automatic) / Credit card (automatic)): "
        )
        if payment_method.lower() == "exit":
            return

        customer = pd.DataFrame([{
            "tenure": float(tenure),
            "MonthlyCharges": float(monthly_charges),
            "TotalCharges": float(total_charges),
            "Contract": contract,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "TechSupport": tech_support,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method
        }])

        prediction = model.predict(customer)[0]
        probability = model.predict_proba(customer)[0][1]

        print("\n" + "-" * 60)
        if prediction == "Yes":
            print("Prediction: CUSTOMER IS LIKELY TO CHURN")
        else:
            print("Prediction: CUSTOMER IS LIKELY TO STAY")

        print(f"Churn probability: {probability:.2%}")
        print("-" * 60)

    except ValueError:
        print("\nInvalid numeric input. Please enter numbers for tenure and charges.")

if __name__ == "__main__":
    main()
