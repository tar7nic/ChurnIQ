# 📡 Customer Churn Prediction System

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-189A35?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=flat&logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)

An end-to-end machine learning system for predicting telecom customer churn — from synthetic data generation and feature engineering through model training, evaluation, SQL cohort analysis, and a live 5-page Streamlit dashboard for real-time scoring.

> **Best Model:** Logistic Regression — ROC-AUC **0.7842** | Recall **0.7449** | F1 **0.5201**

---

## 🖥 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://churniq-tn-019.streamlit.app/)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Pipeline Workflow](#pipeline-workflow)
- [Features](#features)
- [Model Results](#model-results)
- [Business Impact](#business-impact)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Quick Start](#quick-start)
- [Run Order](#run-order)
- [Tech Stack](#tech-stack)

---

## Overview

This project builds a production-quality churn prediction system for a telecom company using a fully synthetic dataset of **6,000 customers** with **24 features**. The system identifies customers likely to churn, quantifies revenue at risk, and recommends personalized retention actions — all accessible through an interactive dashboard.

**Key highlights:**
- Logistic Regression outperforms XGBoost on ROC-AUC (0.7842 vs 0.7745), demonstrating that rigorous feature engineering can make simpler models competitive
- 5 business-driven engineered features improve model recall by ~8% over baseline
- Full class-imbalance handling via `class_weight='balanced'` and `scale_pos_weight`
- Real-time churn scoring with risk tier classification and retention action recommendations
- SQL-based cohort analysis identifying Month-to-month contracts as the top revenue risk segment

---

## Project Structure

```
Customer-Churn-Prediction/
├── app.py                        # Streamlit dashboard (5 pages)
├── requirements.txt              # Python dependencies
├── data/
│   ├── raw/
│   │   └── churn_raw.csv         # Synthetic dataset (6,000 rows x 24 cols)
│   └── processed/
│       ├── X_train.csv           # Encoded + scaled training features
│       ├── X_test.csv            # Encoded + scaled test features
│       ├── y_train.csv           # Training labels
│       └── y_test.csv            # Test labels
├── src/
│   ├── __init__.py
│   ├── data_generator.py         # Synthetic dataset generation
│   ├── preprocessing.py          # Cleaning + feature engineering + encoding
│   ├── eda.py                    # 8+ publication-quality EDA plots
│   ├── train.py                  # Model training + hyperparameter tuning
│   ├── evaluate.py               # Metrics, confusion matrix, SHAP, cross-val
│   └── sql_queries.py            # Churn cohort SQL queries (SQLite)
└── outputs/
    ├── model.pkl                 # Best tuned model (Logistic Regression)
    ├── scaler.pkl                # Fitted StandardScaler
    ├── encoder.pkl               # Fitted LabelEncoders + OHE column map
    ├── feature_names.pkl         # Feature column names list
    ├── metrics_summary.json      # All model metrics + best model metadata
    ├── data_quality_report.json  # Preprocessing quality report
    └── figures/                  # All generated plots (PNG)
```

---

## Pipeline Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                     PIPELINE SEQUENCE                           │
├──────────────┬──────────────┬─────────────┬────────────────────┤
│   GENERATE   │  PREPROCESS  │    TRAIN    │     EVALUATE       │
│              │              │             │                    │
│ data_        │ preprocessing│  train.py   │  evaluate.py       │
│ generator.py │ .py          │             │                    │
│              │              │  5 models   │  Metrics           │
│ 6,000 rows   │  Clean       │  + HPT      │  Confusion Matrix  │
│ 24 features  │  Engineer    │             │  ROC / PR Curves   │
│ seed=42      │  Encode      │  model.pkl  │  SHAP Values       │
│              │  Scale       │             │                    │
├──────────────┴──────────────┴─────────────┴────────────────────┤
│                         DEPLOY                                  │
│                                                                 │
│   sql_queries.py  →  Cohort Analysis                           │
│   app.py          →  Streamlit Dashboard (5 pages)             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Features

### Data & Feature Engineering

| Feature | Description | Business Rationale |
|---|---|---|
| `tenure_group` | Ordinal bins: 0–12, 13–24, 25–48, 49–72 months | Early customers churn most |
| `avg_charge_per_tenure` | Total charges / tenure months | Price-to-value perception proxy |
| `late_payment_risk_score` | Weighted composite of late payments + support tickets | Payment friction indicator |
| `engagement_score` | Count of active add-on services (0–6) | Low engagement → high churn |
| `high_value_customer` | Binary flag: monthly charges ≥ 75th percentile | Retention priority segmentation |

### Preprocessing Steps
- Duplicate removal + `customer_id` drop
- `total_charges` dtype fix (string → float)
- Median/mode imputation for missing values
- 5th/95th percentile outlier capping
- Binary yes/no encoding via `LabelEncoder`
- Multi-class encoding via `pd.get_dummies` (one-hot)
- `StandardScaler` normalization
- Stratified 80/20 train/test split

---

## Model Results

All models trained with class-imbalance handling (`class_weight='balanced'`, `scale_pos_weight=4` for XGBoost) and evaluated on a held-out stratified test set (1,200 rows, ~20% churn rate).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | CV AUC (5-fold) |
|---|---|---|---|---|---|---|---|
| **Logistic Regression ⭐** | 0.7217 | 0.3996 | **0.7449** | **0.5201** | **0.7842** | **0.5337** | 0.7936 ± 0.0109 |
| Random Forest | 0.7967 | 0.4980 | 0.5185 | 0.5081 | 0.7818 | 0.5191 | 0.7845 ± 0.0053 |
| XGBoost | 0.8167 | **0.5920** | 0.6008 | 0.5961 | 0.7745 | 0.5021 | 0.7674 ± 0.0086 |
| SVM | 0.7200 | 0.3880 | 0.6626 | 0.4894 | 0.7641 | 0.4496 | 0.7718 ± 0.0141 |
| Decision Tree | 0.6975 | 0.3673 | 0.6831 | 0.4777 | 0.7245 | 0.4325 | 0.6955 ± 0.0183 |

> **Why Logistic Regression wins:** Selected on ROC-AUC (industry standard for churn). Despite XGBoost having higher accuracy, LR achieves the best discrimination ability (0.7842 AUC) and highest recall (0.7449) — catching 74% of actual churners, which directly maximises retention ROI.

> **Hyperparameter tuning:** `RandomizedSearchCV` (30 iterations, 5-fold stratified CV, ROC-AUC scorer) applied to the best model.

---

## Business Impact

Based on the synthetic dataset of 6,000 customers:

| Metric | Value |
|---|---|
| Overall Churn Rate | ~20% |
| Monthly Revenue at Risk | ~$18,400 |
| Estimated Annual Revenue at Risk | ~$220,800 |
| Avg CLV — Churned Customers | lower by ~40% vs retained |
| Customer Acquisition Cost (assumed) | $300/customer |
| Total Replacement Cost | ~$360,000 |

### Top Churn Drivers
1. **Month-to-month contract** — 3–4x higher churn than annual contracts
2. **Fiber optic internet** — highest charges + highest churn (price-value gap)
3. **Low tenure (< 12 months)** — ~45% of all churns occur in year one
4. **No online security / tech support** — low switching cost, low engagement
5. **Electronic check payment** — highest churn of all payment methods
6. **Low engagement score (0–1 add-ons)** — 2.3x more likely to churn

### Recommended Retention Strategies
- Convert M2M customers to annual contracts with 10% loyalty discount
- Proactive onboarding program for customers in months 1–6
- Free 3-month trial of Security + Tech Support for low-engagement users
- Auto-pay incentive ($5/month discount) for electronic check payers
- Monthly high-risk scoring using the deployed model (threshold: >60% probability)

---

## Streamlit Dashboard

A 5-page interactive dashboard built with Streamlit:

| Page | Description |
|---|---|
| 📊 **Dashboard** | KPI cards, churn distribution, contract breakdown, tenure analysis, pipeline figures gallery |
| 🔮 **Predict** | Customer input form → churn probability, risk tier (High/Medium/Low), top 5 churn drivers, retention actions |
| 🔍 **EDA** | 4 tabs: Distribution, Charges & Tenure, Correlations, Pipeline Figures |
| 📈 **Model Performance** | Metrics table, ROC/PR curves, confusion matrix, sensitivity/specificity breakdown |
| 💡 **Business Insights** | Revenue impact, key findings, retention strategies, copy-paste resume bullets |

---

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2. Create and activate virtual environment
```bash
python -m venv churnenv
# Windows
churnenv\Scripts\activate
# Mac/Linux
source churnenv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the pipeline (in order)
```bash
python src/data_generator.py
python src/eda.py
python src/preprocessing.py
python src/train.py
python src/evaluate.py
python src/sql_queries.py
```

### 5. Launch the dashboard
```bash
streamlit run app.py
```

---

## Run Order

```
1. python src/data_generator.py   → generates data/raw/churn_raw.csv
2. python src/eda.py              → saves 8+ plots to outputs/figures/
3. python src/preprocessing.py   → saves processed splits + artifacts
4. python src/train.py           → trains 5 models, saves model.pkl
5. python src/evaluate.py        → confusion matrix, ROC/PR, SHAP plots
6. python src/sql_queries.py     → cohort analysis (standalone)
7. streamlit run app.py          → launches dashboard
```

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+, R 4.2+ |
| Data | Pandas, NumPy |
| ML | Scikit-Learn, XGBoost, imbalanced-learn |
| Visualization | Matplotlib, Seaborn, ggplot2, survminer |
| Statistical Analysis | survival, coin, rstatix, ggpubr |
| Explainability | SHAP |
| Dashboard | Streamlit |
| Database | SQLite (via Python `sqlite3`) |
| Serialization | Joblib |
| Environment | Anaconda / virtualenv |

---

## 📊 R Statistical Analysis

Beyond the Python ML pipeline, two R scripts provide rigorous statistical validation of the churn drivers identified by the model.

### Survival Analysis (`r/survival_analysis.R`)

Treats churn as a time-to-event problem using classical survival analysis methods.

| Method | Purpose |
|---|---|
| Kaplan-Meier Curves | Visualize retention probability over tenure by contract type & tenure group |
| Log-Rank Test | Statistically compare survival distributions across contract groups |
| Cox PH Model | Quantify hazard ratios for monthly charges, support tickets, satisfaction score, contract type, payment method |
| Schoenfeld Residuals | Validate proportional hazards assumption |

**Outputs:**

| Plot | Insight |
|---|---|
| ![KM by Contract](outputs/figures/km_survival_contract.png) | Month-to-month customers drop off sharply — survival curve confirms Python model findings |
| ![KM by Tenure Group](outputs/figures/km_survival_tenure_group.png) | 0–12 month cohort has lowest retention — year-one is the critical risk window |
| ![Hazard Ratios](outputs/figures/cox_hazard_ratios.png) | Forest plot of Cox PH hazard ratios — features with HR > 1 directly increase churn risk |
| ![Schoenfeld Residuals](outputs/figures/cox_ph_schoenfeld.png) | Residual plot validating Cox model assumptions |

---

### Hypothesis Testing (`r/hypothesis_testing.R`)

Formally tests whether feature distributions differ significantly between churned and retained customers.

| Test | Applied To | Purpose |
|---|---|---|
| Wilcoxon Rank-Sum | monthly_charges, total_charges, tenure, num_support_tickets, satisfaction_score | Non-parametric test — no normality assumption |
| Chi-Squared + Cramér's V | contract, payment_method, internet_service, tenure_group | Tests independence between category and churn |
| One-Way ANOVA + Tukey HSD | satisfaction_score across contract types | Pairwise mean comparison across 3 contract groups |

**Outputs:**

| Plot | Insight |
|---|---|
| ![Violin Plots](outputs/figures/hypothesis_violin_plots.png) | Distribution shape differences between churned vs retained across all numeric features |
| ![Churn Rate by Category](outputs/figures/hypothesis_churn_rate_categorical.png) | Churn rate % per category — confirms contract type and payment method as top drivers |
| ![ANOVA Satisfaction](outputs/figures/hypothesis_anova_satisfaction.png) | Satisfaction score varies significantly across contract types (p < 0.05) |

> All test results with p-values, effect sizes, and significance flags saved to `outputs/hypothesis_test_results.csv`.

---

## License

MIT License — free to use for portfolio, learning, and personal projects.

---

<div align="center">
Built with 🤖 ML + 📊 Data Science + 💡 Business Intelligence
</div>
