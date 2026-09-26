# 📊 Customer Churn Prediction

A machine learning project that predicts whether a telecommunications customer is likely to **churn (leave the service)**.

## Project Overview

This project demonstrates a complete supervised machine learning workflow:

- Data loading and cleaning
- Handling missing values
- Numerical feature scaling
- Categorical feature encoding
- Train/test splitting
- Logistic Regression
- Model evaluation
- Churn probability prediction
- Command-line application

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- joblib

## Machine Learning Workflow

```text
Customer Data
     ↓
Data Cleaning
     ↓
Missing Value Handling
     ↓
Train / Test Split
     ↓
Numerical Scaling + Categorical Encoding
     ↓
Logistic Regression
     ↓
Evaluation
     ↓
Churn Prediction
```

## Project Structure

```text
customer-churn-prediction/
├── app.py
├── train_model.py
├── predict.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── README.md
├── models/
└── reports/
    └── Project_Report.docx
```

## Dataset

Use the **Telco Customer Churn** dataset.

Place the downloaded CSV in the project root with this exact name:

```text
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The dataset contains customer information such as tenure, contract type, monthly charges, internet service and churn status.

## Installation

Clone your repository:

```bash
git clone https://github.com/ritesh586/Customer_Churn_Prediction_Project.git
cd Customer_Churn_Prediction_Project
```

Create a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## Train the Model

After placing the dataset in the project root:

```bash
python3 train_model.py
```

The script creates:

```text
models/churn_model.joblib
reports/metrics.json
```

It also prints:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## Run the Application

```bash
python3 app.py
```

The program will ask for customer information and return:

```text
Prediction: CUSTOMER IS LIKELY TO CHURN
Churn probability: 72.34%
```

or:

```text
Prediction: CUSTOMER IS LIKELY TO STAY
Churn probability: 18.21%
```

## Example

Example customer:

```text
Tenure: 5
Monthly charges: 85
Total charges: 425
Contract: Month-to-month
Internet service: Fiber optic
Online security: No
Tech support: No
Paperless billing: Yes
Payment method: Electronic check
```

The model then estimates the customer's probability of churn.

## Why Logistic Regression?

Logistic Regression is a suitable baseline for binary classification because:

- The target has two outcomes: Churn / No Churn.
- It is fast to train.
- It is easy to interpret.
- It provides probability estimates.
- It works well with encoded tabular data.

## Evaluation

The project evaluates the classifier using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Because customer churn datasets can be imbalanced, the project uses `class_weight="balanced"` so that the minority churn class receives appropriate importance.

## Important Note

This is an educational machine learning project. Predictions should not be treated as guaranteed outcomes. Real customer-retention systems would require more data, validation, monitoring and business-specific features.

## Author

**Artificial Intelligence / Machine Learning Internship Project**
