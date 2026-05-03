"""
train.py
========
Model training pipeline for Customer Churn Prediction.

Trains 5 classifiers, applies class-imbalance handling, performs
hyperparameter tuning on the best model, and saves all artifacts.

Models:
    1. Logistic Regression
    2. Random Forest
    3. XGBoost (fallback: Gradient Boosting)
    4. Support Vector Machine
    5. Decision Tree

Usage:
    python src/train.py

Outputs:
    outputs/model.pkl          (best tuned model)
    outputs/all_models.pkl     (dict of all trained models)
    outputs/metrics_summary.json
"""

import sys
import warnings
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (
    RandomizedSearchCV, cross_val_score, StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[!] XGBoost not available — using GradientBoostingClassifier instead.")

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Paths & Constants
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
RANDOM_SEED = 42

PALETTE = {
    "bg": "#0f1117",
    "surface": "#1a1d2e",
    "text": "#e8e8f0",
    "grid": "#2a2d3e",
    "colors": ["#7c6af7", "#f7706a", "#4ecdc4", "#ffd166", "#a8dadc"],
}

plt.rcParams.update({
    "figure.facecolor": PALETTE["bg"],
    "axes.facecolor": PALETTE["surface"],
    "axes.edgecolor": PALETTE["grid"],
    "axes.labelcolor": PALETTE["text"],
    "axes.titlecolor": PALETTE["text"],
    "xtick.color": PALETTE["text"],
    "ytick.color": PALETTE["text"],
    "text.color": PALETTE["text"],
    "grid.color": PALETTE["grid"],
    "grid.linewidth": 0.5,
    "font.family": "DejaVu Sans",
})


# ─────────────────────────────────────────────────────────────
# Load Processed Data
# ─────────────────────────────────────────────────────────────
def load_processed_data() -> tuple:
    """Load train/test splits from processed CSVs."""
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv").values
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv").values
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").values.ravel()
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").values.ravel()
    feature_names = joblib.load(OUTPUTS_DIR / "feature_names.pkl")

    print(f"[OK] Loaded processed data:")
    print(f"    X_train: {X_train.shape}  |  y_train churn rate: {y_train.mean():.2%}")
    print(f"    X_test:  {X_test.shape}  |  y_test  churn rate: {y_test.mean():.2%}")
    return X_train, X_test, y_train, y_test, feature_names


# ─────────────────────────────────────────────────────────────
# Define Models
# ─────────────────────────────────────────────────────────────
def get_model_definitions() -> dict:
    """Return a dict of model_name -> model_instance."""
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_SEED,
            C=0.5,
            solver="lbfgs",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
            max_depth=12,
            min_samples_leaf=4,
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_SEED,
            max_depth=8,
            min_samples_leaf=10,
        ),
        "SVM": SVC(
            kernel="rbf",
            class_weight="balanced",
            random_state=RANDOM_SEED,
            probability=True,
            C=1.0,
            gamma="scale",
        ),
    }

    if XGBOOST_AVAILABLE:
        # Calculate scale_pos_weight for imbalance handling
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            random_state=RANDOM_SEED,
            eval_metric="logloss",
            use_label_encoder=False,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            verbosity=0,
        )
    else:
        models["Gradient Boosting"] = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=RANDOM_SEED,
            subsample=0.8,
        )

    return models


# ─────────────────────────────────────────────────────────────
# Evaluate a Single Model
# ─────────────────────────────────────────────────────────────
def evaluate_model(
    model, X_train: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray, model_name: str
) -> dict:
    """Train model and compute all evaluation metrics."""
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                scoring="roc_auc", n_jobs=-1)

    metrics = {
        "model": model_name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "pr_auc": round(average_precision_score(y_test, y_prob), 4),
        "cv_roc_auc_mean": round(cv_scores.mean(), 4),
        "cv_roc_auc_std": round(cv_scores.std(), 4),
        "train_time_sec": round(train_time, 2),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "y_prob": y_prob.tolist(),
        "y_pred": y_pred.tolist(),
    }

    return metrics


# ─────────────────────────────────────────────────────────────
# Train All Models
# ─────────────────────────────────────────────────────────────
def train_all_models(
    X_train, X_test, y_train, y_test
) -> tuple:
    """Train all models and return results."""
    models = get_model_definitions()
    results = {}
    trained_models = {}
    metrics_list = []

    print("\n" + "=" * 60)
    print("  MODEL TRAINING")
    print("=" * 60)

    for name, model in models.items():
        print(f"\n[->] Training: {name} ...")
        m = evaluate_model(model, X_train, X_test, y_train, y_test, name)
        results[name] = m
        trained_models[name] = model
        metrics_list.append({
            k: v for k, v in m.items()
            if k not in ["confusion_matrix", "y_prob", "y_pred"]
        })
        print(
            f"    Accuracy: {m['accuracy']:.4f} | Recall: {m['recall']:.4f} | "
            f"ROC-AUC: {m['roc_auc']:.4f} | PR-AUC: {m['pr_auc']:.4f} | "
            f"CV-AUC: {m['cv_roc_auc_mean']:.4f}±{m['cv_roc_auc_std']:.4f} | "
            f"Time: {m['train_time_sec']}s"
        )

    return results, trained_models, metrics_list


# ─────────────────────────────────────────────────────────────
# Select Best Model
# ─────────────────────────────────────────────────────────────
def select_best_model(results: dict, trained_models: dict) -> tuple:
    """Select the best model by ROC-AUC score."""
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = trained_models[best_name]
    best_auc = results[best_name]["roc_auc"]
    print(f"\n[*] Best model: {best_name} (ROC-AUC = {best_auc:.4f})")
    return best_name, best_model


# ─────────────────────────────────────────────────────────────
# Hyperparameter Tuning
# ─────────────────────────────────────────────────────────────
def tune_best_model(
    best_name: str, best_model, X_train: np.ndarray, y_train: np.ndarray
):
    """
    Run RandomizedSearchCV on the best model.
    Optimizes for ROC-AUC with 5-fold stratified CV.
    """
    print(f"\n[->] Hyperparameter tuning: {best_name} ...")

    param_grids = {
        "XGBoost": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 4, 5, 6, 7],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "min_child_weight": [1, 3, 5, 7],
            "gamma": [0, 0.1, 0.3, 0.5],
        },
        "Random Forest": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [6, 8, 10, 12, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", "log2", 0.5],
        },
        "Gradient Boosting": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 4, 5, 6],
            "subsample": [0.6, 0.8, 1.0],
            "min_samples_leaf": [1, 5, 10],
        },
        "Logistic Regression": {
            "C": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear", "saga"],
        },
        "Decision Tree": {
            "max_depth": [4, 6, 8, 10, 12, None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 5, 10],
            "criterion": ["gini", "entropy"],
        },
        "SVM": {
            "C": [0.1, 0.5, 1.0, 5.0, 10.0],
            "gamma": ["scale", "auto", 0.01, 0.001],
            "kernel": ["rbf", "poly"],
        },
    }

    param_grid = param_grids.get(best_name, {})
    if not param_grid:
        print(f"  [!] No param grid for {best_name}. Skipping tuning.")
        return best_model

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    search = RandomizedSearchCV(
        estimator=best_model.__class__(**{
            k: v for k, v in best_model.get_params().items()
            if k not in param_grid
        }) if hasattr(best_model, "get_params") else best_model,
        param_distributions=param_grid,
        n_iter=30,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=0,
        refit=True,
    )
    search.fit(X_train, y_train)

    print(f"  [OK] Best CV ROC-AUC: {search.best_score_:.4f}")
    print(f"  [OK] Best params: {search.best_params_}")
    return search.best_estimator_


# ─────────────────────────────────────────────────────────────
# Plot: ROC Curves (all models)
# ─────────────────────────────────────────────────────────────
def plot_roc_curves(results: dict, y_test: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))

    for (name, m), color in zip(results.items(), PALETTE["colors"]):
        y_prob = np.array(m["y_prob"])
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = m["roc_auc"]
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})",
                color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], "w--", linewidth=1, alpha=0.4, label="Random Classifier")
    ax.fill_between([0, 1], [0, 1], alpha=0.03, color="white")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(facecolor=PALETTE["surface"], fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = FIGURES_DIR / "11_roc_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  [OK] Saved: {path.name}")


# ─────────────────────────────────────────────────────────────
# Plot: PR Curves (all models)
# ─────────────────────────────────────────────────────────────
def plot_pr_curves(results: dict, y_test: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    baseline = y_test.mean()

    for (name, m), color in zip(results.items(), PALETTE["colors"]):
        y_prob = np.array(m["y_prob"])
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = m["pr_auc"]
        ax.plot(recall, precision, label=f"{name} (PR-AUC={pr_auc:.3f})",
                color=color, linewidth=2)

    ax.axhline(baseline, color="white", linestyle="--", alpha=0.4,
               label=f"Baseline ({baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(facecolor=PALETTE["surface"], fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = FIGURES_DIR / "12_pr_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  [OK] Saved: {path.name}")


# ─────────────────────────────────────────────────────────────
# Plot: Metrics Comparison Bar Chart
# ─────────────────────────────────────────────────────────────
def plot_metrics_comparison(metrics_list: list) -> None:
    df_m = pd.DataFrame(metrics_list).set_index("model")
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    df_m = df_m[metric_cols]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(df_m))
    width = 0.13
    offsets = np.linspace(-2.5 * width, 2.5 * width, len(metric_cols))

    for i, (col, color) in enumerate(zip(metric_cols, PALETTE["colors"] + ["#ff9a3c"])):
        bars = ax.bar(x + offsets[i], df_m[col], width,
                      label=col.replace("_", " ").title(), color=color, alpha=0.85,
                      edgecolor=PALETTE["bg"])

    ax.set_xticks(x)
    ax.set_xticklabels(df_m.index, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.legend(facecolor=PALETTE["surface"], fontsize=9, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = FIGURES_DIR / "13_model_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  [OK] Saved: {path.name}")


# ─────────────────────────────────────────────────────────────
# Save Artifacts
# ─────────────────────────────────────────────────────────────
def save_artifacts(
    best_model, tuned_model, trained_models: dict,
    results: dict, metrics_list: list, best_name: str
) -> None:
    """Persist all model artifacts and metrics."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save best (tuned) model
    joblib.dump(tuned_model, OUTPUTS_DIR / "model.pkl")

    # Save all trained models
    joblib.dump(trained_models, OUTPUTS_DIR / "all_models.pkl")

    # Save metrics summary (JSON-serializable)
    serializable_results = {}
    for name, m in results.items():
        serializable_results[name] = {
            k: v for k, v in m.items() if k not in ["y_prob", "y_pred"]
        }

    with open(OUTPUTS_DIR / "metrics_summary.json", "w") as f:
        json.dump(
            {
                "best_model": best_name,
                "results": serializable_results,
                "metrics_list": metrics_list,
            },
            f, indent=2,
        )

    print(f"\n[OK] Saved model.pkl, all_models.pkl, metrics_summary.json to outputs/")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  CUSTOMER CHURN — MODEL TRAINING PIPELINE")
    print("=" * 60)

    # Load data
    X_train, X_test, y_train, y_test, feature_names = load_processed_data()

    # Train all models
    results, trained_models, metrics_list = train_all_models(X_train, X_test, y_train, y_test)

    # Select best model
    best_name, best_model = select_best_model(results, trained_models)

    # Hyperparameter tuning
    tuned_model = tune_best_model(best_name, best_model, X_train, y_train)

    # Re-evaluate tuned model
    print(f"\n[->] Evaluating tuned {best_name} ...")
    tuned_metrics = evaluate_model(tuned_model, X_train, X_test, y_train, y_test, f"{best_name} (Tuned)")
    print(
        f"    ROC-AUC: {tuned_metrics['roc_auc']:.4f} | "
        f"Recall: {tuned_metrics['recall']:.4f} | "
        f"F1: {tuned_metrics['f1']:.4f}"
    )

    # Generate plots
    print("\n[->] Generating evaluation plots ...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_roc_curves(results, y_test)
    plot_pr_curves(results, y_test)
    plot_metrics_comparison(metrics_list)

    # Save artifacts
    save_artifacts(best_model, tuned_model, trained_models, results, metrics_list, best_name)

    print(f"\n{'=' * 60}")
    print(f"  TRAINING COMPLETE — Best Model: {best_name}")
    print(f"  ROC-AUC: {results[best_name]['roc_auc']:.4f}  |  Recall: {results[best_name]['recall']:.4f}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
