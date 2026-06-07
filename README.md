# Telco Customer Churn Prediction

A supervised binary classification project that predicts whether a telecom customer will churn, using the IBM Telco Customer Churn dataset.

---

## Problem Statement

Customer churn is one of the most costly problems in the telecom industry — acquiring a new customer costs **5–7× more** than retaining an existing one. This project builds a machine learning pipeline that flags at-risk customers early, enabling proactive retention campaigns.

---

## Project Structure

```
.
├── TelcoChurnPrediction_Fixed.ipynb   # Main notebook (full pipeline)
├── TelcoCustomerChurn.csv             # Raw dataset (place here before running)
├── churn_pipeline.pkl                 # Saved best model (generated on run)
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

---

## Dataset

**IBM Telco Customer Churn**  
- **Source:** [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
- **Rows:** 7,043 customers  
- **Features:** 21 (demographics, services, contract details, charges)  
- **Target:** `Churn` — `Yes` (1) or `No` (0)  
- **Class balance:** ~73% No-Churn, ~27% Churn (imbalanced)

> Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from Kaggle and rename it to `TelcoCustomerChurn.csv` in the project root.

---

## Notebook Outline

| Section | Description |
|---|---|
| 1. Imports | All libraries loaded upfront |
| 2. Dataset Loading | Load CSV, inspect shape, types, statistics |
| 3. EDA | Churn distribution, feature distributions, churn rates by category, correlation heatmap |
| 4. Preprocessing | Fix `TotalCharges`, drop nulls, encode target, StandardScaler + OneHotEncoder pipeline |
| 5. Model Training | Logistic Regression and Random Forest inside sklearn Pipelines |
| 6. Evaluation | Accuracy, ROC-AUC, Precision/Recall/F1, confusion matrices, ROC curves, PR curves, cross-validation |
| 7. Hyperparameter Tuning | GridSearchCV on Random Forest optimising for Recall |
| 8. Feature Importance | Top 15 features by Gini importance |
| 9. Model Comparison | Side-by-side metric table and bar chart |
| 10. Save Model | Best pipeline saved with `joblib` |
| 11. Summary & Insights | Business findings and recommended next steps |

---

## Setup & Usage

### 1. Clone / download the project

```bash
git clone <your-repo-url>
cd telco-churn-prediction
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Place `TelcoCustomerChurn.csv` in the project root directory.

### 5. Run the notebook

```bash
jupyter notebook TelcoChurnPrediction_Fixed.ipynb
```

Run all cells in order (`Kernel → Restart & Run All`).

---

## Results

| Model | Accuracy | ROC-AUC | Recall (Churn) |
|---|---|---|---|
| Logistic Regression | ~80% | ~0.85 | ~56% |
| Random Forest (default) | ~79% | ~0.83 | ~49% |
| **Tuned Random Forest** | **~80%** | **~0.84** | **~55%** |

> Recall on the Churn class is the primary metric — missing a churner is more costly than a false alarm.

---

## Key Findings

1. **Contract type** is the strongest churn predictor — month-to-month customers churn at ~43% vs ~3% for two-year contracts.
2. **Fibre optic customers** churn more than DSL customers, likely due to higher charges or competition.
3. **Short-tenure customers** are at the highest risk — early loyalty rewards can reduce first-year churn.
4. **Lack of TechSupport** is a significant churn signal — bundling support may improve retention.
5. **Class imbalance** means accuracy alone is misleading; Precision-Recall and ROC-AUC are more informative.

---

## Recommended Next Steps

- Try **XGBoost / LightGBM** — typically outperform Random Forest on tabular churn data
- Apply **SMOTE** or `class_weight='balanced'` to improve recall on the minority class
- **Optimise the decision threshold** to trade precision for higher recall
- **Deploy** `churn_pipeline.pkl` as a REST API (e.g. FastAPI + Docker) for real-time scoring

---

## Dependencies

| Package | Version |
|---|---|
| Python | ≥ 3.9 |
| numpy | ≥ 1.24 |
| pandas | ≥ 2.0 |
| scikit-learn | ≥ 1.3 |
| matplotlib | ≥ 3.7 |
| seaborn | ≥ 0.12 |
| joblib | ≥ 1.3 |
