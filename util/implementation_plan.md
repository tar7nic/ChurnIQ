# Customer Churn Prediction System — Implementation Plan

## Overview

Build a production-quality, end-to-end ML project for telecom customer churn prediction. The system includes a full data science pipeline (EDA → feature engineering → modeling → evaluation → deployment) plus a Streamlit dashboard for real-time scoring.

---

## Proposed Folder Structure

```
customer-churn-project/
├── data/
│   ├── raw/                  # synthetic dataset (churn_raw.csv)
│   └── processed/            # cleaned + engineered features
├── notebooks/
│   └── eda_analysis.ipynb    # (optional standalone notebook)
├── src/
│   ├── __init__.py
│   ├── data_generator.py     # synthetic dataset generation
│   ├── preprocessing.py      # cleaning + feature engineering
│   ├── train.py              # model training + HPT
│   ├── evaluate.py           # metrics, confusion matrix, cross-val
│   └── sql_queries.py        # churn cohort SQL queries
├── outputs/
│   ├── figures/              # EDA + evaluation plots (PNG)
│   ├── model.pkl             # best trained model
│   ├── scaler.pkl
│   ├── encoder.pkl
│   └── metrics_report.json
├── app.py                    # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Proposed Changes

### Component 1: Project Scaffolding

#### [NEW] requirements.txt
All dependencies: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, streamlit, joblib, shap, imbalanced-learn

---

### Component 2: Data Generation

#### [NEW] src/data_generator.py
- Generate 5,000+ row realistic telecom dataset
- All 24 columns as specified
- Controlled correlations (e.g., month-to-month → higher churn, low satisfaction → higher churn)
- Reproducible with `random_seed=42`

---

### Component 3: Preprocessing Pipeline

#### [NEW] src/preprocessing.py
- Load raw CSV
- Handle missing values (median/mode imputation)
- Fix dtypes (TotalCharges → float)
- Remove duplicates
- Outlier treatment (IQR capping on MonthlyCharges, TotalCharges)
- Feature engineering:
  - `tenure_group` (buckets: 0-12, 13-24, 25-48, 49-72)
  - `avg_charge_per_tenure`
  - `late_payment_risk_score`
  - `engagement_score`
- Encode categoricals (OrdinalEncoder + OneHotEncoder via ColumnTransformer)
- Scale numerics (StandardScaler)
- Stratified train/test split (80/20)

---

### Component 4: EDA & Visualizations

#### [NEW] src/eda.py
Generate and save 8+ publication-quality plots:
1. Churn distribution (pie + bar)
2. Churn by contract type
3. Churn by tenure buckets
4. Churn vs monthly charges (box/violin)
5. Churn vs support tickets
6. Correlation heatmap
7. Churn by payment method
8. Churn by satisfaction score

Each plot saved to `outputs/figures/`.

---

### Component 5: Model Training

#### [NEW] src/train.py
Train 5 models:
- Logistic Regression
- Random Forest
- XGBoost (or GradientBoosting fallback)
- Support Vector Machine
- Decision Tree

Handle class imbalance with `class_weight='balanced'` and/or SMOTE.
Save all trained models + best model artifacts.

---

### Component 6: Evaluation

#### [NEW] src/evaluate.py
For each model compute:
- Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
- Confusion Matrix (plotted)
- 5-fold cross-validation score
- ROC curve overlay plot
- PR curve overlay plot
- Summary comparison table

---

### Component 7: Hyperparameter Tuning

Included in `train.py`:
- RandomizedSearchCV on best model (XGBoost/RF)
- Optimize for ROC-AUC scorer
- Save tuned model as `model.pkl`

---

### Component 8: Feature Importance & SHAP

Included in `evaluate.py`:
- Feature importance bar chart (tree-based models)
- SHAP summary plot + waterfall for a sample prediction
- Business-labeled top-10 churn drivers

---

### Component 9: SQL Queries

#### [NEW] src/sql_queries.py
- Churn cohort analysis queries (SQLite in-memory)
- A/B test framework skeleton
- Model drift monitoring concept

---

### Component 10: Streamlit App

#### [NEW] app.py
Multi-page Streamlit dashboard:
- **Page 1 - Dashboard**: KPIs, churn rate, key metrics
- **Page 2 - Predict**: Form inputs → churn probability, risk category, top reasons, retention action
- **Page 3 - EDA**: Interactive charts
- **Page 4 - Model Performance**: Metrics comparison table + ROC curves
- **Page 5 - Business Insights**: Recommendations + resume bullets

---

### Component 11: README

#### [NEW] README.md
Full GitHub-ready README with badges, workflow diagram, metrics table, business impact, run instructions, and resume bullets.

---

## Verification Plan

### Automated
- Run `python src/data_generator.py` → verify CSV created with 5000+ rows
- Run `python src/preprocessing.py` → verify processed data + artifacts saved
- Run `python src/train.py` → verify model.pkl saved, metrics printed
- Run `python src/evaluate.py` → verify all plots saved
- Run `streamlit run app.py` → verify app loads with browser subagent

### Manual
- Review generated plots for quality
- Test Streamlit prediction form with edge-case inputs
