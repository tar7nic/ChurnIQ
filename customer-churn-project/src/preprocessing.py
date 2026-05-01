"""
preprocessing.py
================
End-to-end data cleaning, feature engineering, encoding, and scaling pipeline
for the Customer Churn Prediction System.

Steps:
    1. Load raw data
    2. Data quality report (shape, dtypes, nulls, duplicates)
    3. Clean data (impute, fix dtypes, cap outliers)
    4. Feature engineering (tenure groups, risk scores, engagement score)
    5. Encode categoricals + scale numerics
    6. Stratified train/test split
    7. Save processed artifacts

Usage:
    python src/preprocessing.py

Outputs:
    data/processed/X_train.csv, X_test.csv, y_train.csv, y_test.csv
    outputs/scaler.pkl, encoder.pkl, feature_names.pkl
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "churn_raw.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RANDOM_SEED = 42

# ─────────────────────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────────────────────
def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw CSV and return as DataFrame."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. "
            "Run 'python src/data_generator.py' first."
        )
    df = pd.read_csv(path)
    print(f"[✓] Loaded raw data: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────────────────────
# 2. Data Quality Report
# ─────────────────────────────────────────────────────────────
def data_quality_report(df: pd.DataFrame) -> dict:
    """
    Generate a comprehensive data quality report.

    Returns
    -------
    dict with keys: shape, dtypes, null_counts, null_pct, duplicates,
                    class_distribution, summary_stats
    """
    report = {}

    report["shape"] = {"rows": df.shape[0], "columns": df.shape[1]}
    report["dtypes"] = df.dtypes.astype(str).to_dict()

    null_counts = df.isnull().sum()
    report["null_counts"] = null_counts[null_counts > 0].to_dict()
    report["null_pct"] = (
        (null_counts[null_counts > 0] / len(df) * 100).round(2).to_dict()
    )

    report["duplicates"] = int(df.duplicated().sum())

    churn_dist = df["churn"].value_counts().to_dict()
    report["class_distribution"] = churn_dist
    total = sum(churn_dist.values())
    report["class_imbalance_ratio"] = {
        k: f"{v / total * 100:.2f}%" for k, v in churn_dist.items()
    }

    report["summary_stats"] = df.describe().round(2).to_dict()

    print("\n" + "=" * 60)
    print("  DATA QUALITY REPORT")
    print("=" * 60)
    print(f"  Shape          : {report['shape']['rows']} rows × {report['shape']['columns']} cols")
    print(f"  Duplicates     : {report['duplicates']}")
    print(f"  Null columns   : {list(report['null_counts'].keys())}")
    print(f"  Class balance  : {report['class_imbalance_ratio']}")
    print("=" * 60)

    return report


# ─────────────────────────────────────────────────────────────
# 3. Data Cleaning
# ─────────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform data cleaning:
      - Remove duplicates
      - Fix dtypes (total_charges → float)
      - Impute missing values (median for numeric, mode for categorical)
      - Cap outliers via IQR method on numeric columns
    """
    df = df.copy()

    # 3a. Drop exact duplicates (keep first)
    n_dups = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    if n_dups:
        print(f"[✓] Dropped {n_dups} duplicate rows.")

    # 3b. Drop customer_id (identifier, not a feature)
    if "customer_id" in df.columns:
        df.drop(columns=["customer_id"], inplace=True)

    # 3c. Fix total_charges dtype
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")

    # 3d. Impute missing numeric values with median
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "senior_citizen"]  # binary flag

    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"[✓] Imputed '{col}' missing values with median ({median_val:.2f}).")

    # 3e. Impute missing categorical values with mode
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    cat_cols = [c for c in cat_cols if c != "churn"]

    for col in cat_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"[✓] Imputed '{col}' missing values with mode ('{mode_val}').")

    # 3f. Outlier capping via IQR (Winsorization) on key numeric cols
    outlier_cols = ["monthly_charges", "total_charges", "num_support_tickets", "late_payments"]
    for col in outlier_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.01)
            Q3 = df[col].quantile(0.99)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            n_capped = ((df[col] < lower) | (df[col] > upper)).sum()
            df[col] = df[col].clip(lower=lower, upper=upper)
            if n_capped:
                print(f"[✓] Capped {n_capped} outliers in '{col}' [{lower:.2f}, {upper:.2f}].")

    print(f"[✓] Cleaning complete. Shape: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────
# 4. Feature Engineering
# ─────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create business-meaningful derived features:
      - tenure_group            : binned tenure for segmentation
      - avg_charge_per_tenure   : cost efficiency proxy
      - late_payment_risk_score : normalized composite risk
      - engagement_score        : count of active add-on services
      - high_value_customer     : flag for top monthly charge quartile
    """
    df = df.copy()

    # 4a. Tenure group (months → category)
    bins = [0, 12, 24, 48, 72]
    labels = ["0-12 months", "13-24 months", "25-48 months", "49-72 months"]
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=bins, labels=labels, include_lowest=True
    )

    # 4b. Average charge per tenure month (avoid divide-by-zero)
    df["avg_charge_per_tenure"] = np.where(
        df["tenure"] > 0,
        (df["total_charges"] / df["tenure"]).round(2),
        df["monthly_charges"],
    )

    # 4c. Late payment risk score (0–1 normalized composite)
    max_late = max(df["late_payments"].max(), 1)
    max_tickets = max(df["num_support_tickets"].max(), 1)
    df["late_payment_risk_score"] = (
        0.6 * (df["late_payments"] / max_late)
        + 0.4 * (df["num_support_tickets"] / max_tickets)
    ).round(3)

    # 4d. Engagement score: count of Yes across add-on services
    addon_cols = [
        "online_security", "online_backup", "device_protection",
        "tech_support", "streaming_tv", "streaming_movies",
    ]
    engagement = pd.DataFrame()
    for col in addon_cols:
        engagement[col] = (df[col] == "Yes").astype(int)
    df["engagement_score"] = engagement.sum(axis=1)

    # 4e. High-value customer flag
    q75 = df["monthly_charges"].quantile(0.75)
    df["high_value_customer"] = (df["monthly_charges"] >= q75).astype(int)

    print(f"[✓] Feature engineering complete. New columns: tenure_group, avg_charge_per_tenure,")
    print(f"     late_payment_risk_score, engagement_score, high_value_customer")
    return df


# ─────────────────────────────────────────────────────────────
# 5. Encoding + Scaling
# ─────────────────────────────────────────────────────────────
def encode_and_scale(
    df: pd.DataFrame,
    fit: bool = True,
    scaler: StandardScaler = None,
    cat_encoder: dict = None,
) -> tuple:
    """
    Encode categorical variables and scale numeric features.

    Strategy:
      - Binary yes/no columns → LabelEncoder (0/1)
      - Multi-class categoricals → pd.get_dummies (one-hot)
      - Numeric features → StandardScaler

    Parameters
    ----------
    df        : Cleaned + engineered DataFrame (without 'churn')
    fit       : If True, fit new scaler/encoder. If False, use provided ones.
    scaler    : Pre-fitted StandardScaler (used when fit=False)
    cat_encoder : Dict of pre-fitted LabelEncoders (used when fit=False)

    Returns
    -------
    X_encoded : np.ndarray of encoded + scaled features
    feature_names : list of feature column names
    scaler    : fitted StandardScaler
    cat_encoder : dict of fitted LabelEncoders
    """
    df = df.copy()

    # Encode target separately
    target = None
    if "churn" in df.columns:
        target = (df["churn"] == "Yes").astype(int).values
        df.drop(columns=["churn"], inplace=True)

    # Drop tenure_group (ordinal — we use the numeric tenure directly)
    if "tenure_group" in df.columns:
        df.drop(columns=["tenure_group"], inplace=True)

    # ── Binary yes/no columns ─────────────────────────────────
    binary_yes_no = [
        "partner", "dependents", "phone_service", "paperless_billing",
    ]
    if fit:
        cat_encoder = {}
    for col in binary_yes_no:
        if col in df.columns:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                cat_encoder[col] = le
            else:
                le = cat_encoder.get(col)
                if le:
                    df[col] = le.transform(df[col].astype(str))

    # ── Binary senior_citizen — already 0/1 ───────────────────

    # ── Multi-value categoricals → one-hot ────────────────────
    ohe_cols = [
        "gender", "multiple_lines", "internet_service",
        "online_security", "online_backup", "device_protection",
        "tech_support", "streaming_tv", "streaming_movies",
        "contract", "payment_method",
    ]
    ohe_cols = [c for c in ohe_cols if c in df.columns]

    if fit:
        df_encoded = pd.get_dummies(df, columns=ohe_cols, drop_first=False)
        cat_encoder["ohe_columns"] = [
            c for c in df_encoded.columns if c not in df.columns or c in ohe_cols
        ]
        cat_encoder["final_columns"] = df_encoded.columns.tolist()
    else:
        df_encoded = pd.get_dummies(df, columns=ohe_cols, drop_first=False)
        # Align columns to training set
        final_columns = cat_encoder.get("final_columns", df_encoded.columns.tolist())
        for col in final_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[final_columns]

    feature_names = df_encoded.columns.tolist()

    # ── Scale numeric columns ─────────────────────────────────
    num_cols = df_encoded.select_dtypes(include=np.number).columns.tolist()
    if fit:
        scaler = StandardScaler()
        df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols].astype(float))
    else:
        df_encoded[num_cols] = scaler.transform(df_encoded[num_cols].astype(float))

    X = df_encoded.values
    return X, feature_names, target, scaler, cat_encoder


# ─────────────────────────────────────────────────────────────
# 6. Train/Test Split
# ─────────────────────────────────────────────────────────────
def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.20,
    seed: int = RANDOM_SEED,
) -> tuple:
    """Stratified split preserving class balance in both splits."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    print(
        f"[✓] Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows"
    )
    print(
        f"    Train churn rate: {y_train.mean():.2%} | Test churn rate: {y_test.mean():.2%}"
    )
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────
# 7. Save Artifacts
# ─────────────────────────────────────────────────────────────
def save_processed_data(
    X_train, X_test, y_train, y_test,
    feature_names, scaler, cat_encoder, report
) -> None:
    """Persist all processed data and pipeline artifacts."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "figures").mkdir(parents=True, exist_ok=True)

    pd.DataFrame(X_train, columns=feature_names).to_csv(
        PROCESSED_DIR / "X_train.csv", index=False
    )
    pd.DataFrame(X_test, columns=feature_names).to_csv(
        PROCESSED_DIR / "X_test.csv", index=False
    )
    pd.DataFrame(y_train, columns=["churn"]).to_csv(
        PROCESSED_DIR / "y_train.csv", index=False
    )
    pd.DataFrame(y_test, columns=["churn"]).to_csv(
        PROCESSED_DIR / "y_test.csv", index=False
    )

    joblib.dump(scaler, OUTPUTS_DIR / "scaler.pkl")
    joblib.dump(cat_encoder, OUTPUTS_DIR / "encoder.pkl")
    joblib.dump(feature_names, OUTPUTS_DIR / "feature_names.pkl")

    with open(OUTPUTS_DIR / "data_quality_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n[✓] Saved processed splits to data/processed/")
    print("[✓] Saved scaler.pkl, encoder.pkl, feature_names.pkl to outputs/")


# ─────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────
def run_preprocessing_pipeline() -> tuple:
    """
    Execute the full preprocessing pipeline end-to-end.

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names
    """
    print("\n" + "=" * 60)
    print("  PREPROCESSING PIPELINE")
    print("=" * 60)

    # Step 1: Load
    df_raw = load_raw_data()

    # Step 2: Quality report
    report = data_quality_report(df_raw)

    # Step 3: Clean
    df_clean = clean_data(df_raw)

    # Step 4: Feature engineering
    df_feat = engineer_features(df_clean)

    # Step 5: Encode + scale
    X, feature_names, y, scaler, cat_encoder = encode_and_scale(df_feat, fit=True)

    # Step 6: Split
    X_train, X_test, y_train, y_test = stratified_split(X, y)

    # Step 7: Save
    save_processed_data(X_train, X_test, y_train, y_test, feature_names, scaler, cat_encoder, report)

    print("\n[✓] Preprocessing pipeline complete.\n")
    return X_train, X_test, y_train, y_test, feature_names


if __name__ == "__main__":
    run_preprocessing_pipeline()
