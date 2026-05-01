"""
data_generator.py
=================
Generates a realistic synthetic telecom customer churn dataset with 6,000+ rows.
Controlled correlations ensure the dataset is suitable for ML modeling:
  - Month-to-month contracts → higher churn probability
  - Low satisfaction scores → higher churn
  - High support tickets → higher churn
  - Long tenure → lower churn
  - Electronic check payment → higher churn

Usage:
    python src/data_generator.py

Output:
    data/raw/churn_raw.csv
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
# Ensure UTF-8 output on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RANDOM_SEED = 42
N_CUSTOMERS = 6000
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
OUTPUT_FILE = OUTPUT_DIR / "churn_raw.csv"


def generate_churn_dataset(n: int = N_CUSTOMERS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate a synthetic telecom churn dataset with realistic distributions
    and controlled feature-target correlations.

    Parameters
    ----------
    n : int
        Number of customer records to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with 24 columns matching the project specification.
    """
    rng = np.random.default_rng(seed)

    # ── Demographics ──────────────────────────────────────────
    customer_id = [f"CUST-{str(i).zfill(5)}" for i in range(1, n + 1)]
    gender = rng.choice(["Male", "Female"], size=n)
    senior_citizen = rng.choice([0, 1], size=n, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], size=n, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], size=n, p=[0.30, 0.70])

    # ── Service tenure (months) ───────────────────────────────
    # Heavy left-skew — many newer customers
    tenure = rng.integers(1, 73, size=n)

    # ── Phone & Internet Services ─────────────────────────────
    phone_service = rng.choice(["Yes", "No"], size=n, p=[0.90, 0.10])
    multiple_lines = np.where(
        phone_service == "No",
        "No phone service",
        rng.choice(["Yes", "No"], size=n, p=[0.42, 0.58]),
    )
    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"], size=n, p=[0.34, 0.44, 0.22]
    )

    def internet_addon(prob_yes=0.45, prob_no=0.55):
        """Helper for internet-dependent features."""
        choices = []
        for svc in internet_service:
            if svc == "No":
                choices.append("No internet service")
            else:
                choices.append(rng.choice(["Yes", "No"], p=[prob_yes, prob_no]))
        return np.array(choices)

    online_security = internet_addon(0.30, 0.70)
    online_backup = internet_addon(0.35, 0.65)
    device_protection = internet_addon(0.34, 0.66)
    tech_support = internet_addon(0.29, 0.71)
    streaming_tv = internet_addon(0.38, 0.62)
    streaming_movies = internet_addon(0.39, 0.61)

    # ── Contract & Billing ────────────────────────────────────
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n,
        p=[0.55, 0.24, 0.21],
    )
    paperless_billing = rng.choice(["Yes", "No"], size=n, p=[0.59, 0.41])
    payment_method = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        size=n,
        p=[0.34, 0.22, 0.22, 0.22],
    )

    # ── Charges ───────────────────────────────────────────────
    # Monthly charges depend on internet service tier
    monthly_base = np.where(
        internet_service == "Fiber optic",
        rng.uniform(70, 120, size=n),
        np.where(
            internet_service == "DSL",
            rng.uniform(40, 80, size=n),
            rng.uniform(18, 45, size=n),
        ),
    )
    monthly_charges = np.round(monthly_base + rng.normal(0, 5, size=n), 2)
    monthly_charges = np.clip(monthly_charges, 18, 120)

    # Total charges = tenure × monthly (with some noise for plan changes)
    total_charges = np.round(
        tenure * monthly_charges * rng.uniform(0.92, 1.08, size=n), 2
    )
    # Inject ~1% missing values in total_charges (mimics real Telco dataset)
    missing_mask = rng.choice([True, False], size=n, p=[0.01, 0.99])
    total_charges = total_charges.astype(float)
    total_charges[missing_mask] = np.nan

    # ── Behavioral Features ───────────────────────────────────
    # Support tickets: higher for fiber optic / low-tenure customers
    ticket_base = np.where(
        internet_service == "Fiber optic",
        rng.poisson(3, size=n),
        rng.poisson(1.5, size=n),
    )
    num_support_tickets = np.clip(ticket_base, 0, 12)

    # Late payments: month-to-month + electronic check → more late payments
    late_base = np.where(
        (contract == "Month-to-month") & (payment_method == "Electronic check"),
        rng.poisson(2.5, size=n),
        np.where(contract == "Month-to-month", rng.poisson(1.5, size=n), rng.poisson(0.5, size=n)),
    )
    late_payments = np.clip(late_base, 0, 10)

    # Satisfaction score (1-5): inversely correlated with tickets & late payments
    raw_satisfaction = 5.0 - (num_support_tickets * 0.15) - (late_payments * 0.20)
    satisfaction_noise = rng.normal(0, 0.5, size=n)
    satisfaction_score = np.clip(
        np.round(raw_satisfaction + satisfaction_noise, 1), 1.0, 5.0
    )

    # ── Churn Probability (controlled correlations) ────────────
    # Logistic-style probability with meaningful weights
    churn_logit = (
        -2.0                                                          # base intercept
        + 1.8 * (contract == "Month-to-month").astype(float)         # contract risk
        + 0.5 * (contract == "One year").astype(float)               # moderate risk
        - 0.04 * tenure                                               # longer tenure → lower churn
        + 0.015 * monthly_charges                                     # higher bills → more churn
        + 0.25 * num_support_tickets                                  # ticket friction
        + 0.30 * late_payments                                        # payment risk
        - 0.60 * satisfaction_score                                   # dissatisfaction
        + 0.80 * (payment_method == "Electronic check").astype(float) # payment method
        + 0.40 * (internet_service == "Fiber optic").astype(float)   # fiber churn risk
        + rng.logistic(0, 1, size=n)                                  # noise
    )
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    churn_binary = (churn_prob > rng.uniform(0, 1, size=n)).astype(int)
    churn = np.where(churn_binary == 1, "Yes", "No")

    # ── Assemble DataFrame ────────────────────────────────────
    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "gender": gender,
            "senior_citizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,
            "tenure": tenure,
            "phone_service": phone_service,
            "multiple_lines": multiple_lines,
            "internet_service": internet_service,
            "online_security": online_security,
            "online_backup": online_backup,
            "device_protection": device_protection,
            "tech_support": tech_support,
            "streaming_tv": streaming_tv,
            "streaming_movies": streaming_movies,
            "contract": contract,
            "paperless_billing": paperless_billing,
            "payment_method": payment_method,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "num_support_tickets": num_support_tickets,
            "late_payments": late_payments,
            "satisfaction_score": satisfaction_score,
            "churn": churn,
        }
    )

    return df


def save_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """Save the generated dataset to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[OK] Dataset saved to: {output_path}")
    print(f"    Shape: {df.shape}")
    print(f"    Churn rate: {(df['churn'] == 'Yes').mean():.2%}")
    print(f"    Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")


def main():
    print("=" * 60)
    print("  Customer Churn Dataset Generator")
    print("=" * 60)
    df = generate_churn_dataset(n=N_CUSTOMERS, seed=RANDOM_SEED)
    save_dataset(df, OUTPUT_FILE)
    print("\nSample records:")
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()
