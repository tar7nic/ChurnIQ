"""
sql_queries.py
==============
SQL-based churn cohort analysis using SQLite (in-memory).
Demonstrates production-level SQL analytics for churn insights.

Sections:
    1. Churn cohort analysis
    2. Revenue-at-risk calculations
    3. A/B test framework skeleton
    4. Model drift monitoring concept

Usage:
    python src/sql_queries.py
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "churn_raw.csv"


# ─────────────────────────────────────────────────────────────
# Setup SQLite connection with dataset
# ─────────────────────────────────────────────────────────────
def get_connection(df: pd.DataFrame) -> sqlite3.Connection:
    """Load DataFrame into an in-memory SQLite database."""
    conn = sqlite3.connect(":memory:")
    df.to_sql("customers", conn, index=False, if_exists="replace")
    return conn


def run_query(conn: sqlite3.Connection, query: str, title: str) -> pd.DataFrame:
    """Execute a query, print it, and display results."""
    print(f"\n{'─' * 60}")
    print(f"  📊 {title}")
    print(f"{'─' * 60}")
    print(f"  SQL:\n{query}")
    result = pd.read_sql_query(query, conn)
    print(f"\n  Results:")
    print(result.to_string(index=False))
    return result


# ─────────────────────────────────────────────────────────────
# 1. Churn Cohort Analysis Queries
# ─────────────────────────────────────────────────────────────
QUERY_1 = """
-- Overall churn rate
SELECT
    churn,
    COUNT(*) AS customer_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS percentage
FROM customers
GROUP BY churn
ORDER BY churn DESC;
"""

QUERY_2 = """
-- Churn rate by contract type
SELECT
    contract,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY contract
ORDER BY churn_rate_pct DESC;
"""

QUERY_3 = """
-- Churn rate by tenure bucket
SELECT
    CASE
        WHEN tenure BETWEEN 0  AND 12 THEN '0-12 months'
        WHEN tenure BETWEEN 13 AND 24 THEN '13-24 months'
        WHEN tenure BETWEEN 25 AND 48 THEN '25-48 months'
        ELSE '49-72 months'
    END AS tenure_group,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charge
FROM customers
GROUP BY tenure_group
ORDER BY churn_rate_pct DESC;
"""

QUERY_4 = """
-- Revenue at risk by segment
SELECT
    contract,
    COUNT(*) AS churned_customers,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charge,
    ROUND(SUM(monthly_charges), 2) AS monthly_revenue_at_risk,
    ROUND(SUM(monthly_charges) * 12, 2) AS annual_revenue_at_risk
FROM customers
WHERE churn = 'Yes'
GROUP BY contract
ORDER BY annual_revenue_at_risk DESC;
"""

QUERY_5 = """
-- High-risk customer segments (churn propensity > 60%)
SELECT
    payment_method,
    internet_service,
    COUNT(*) AS total,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS churn_pct,
    ROUND(AVG(satisfaction_score), 2) AS avg_satisfaction,
    ROUND(AVG(num_support_tickets), 2) AS avg_tickets
FROM customers
GROUP BY payment_method, internet_service
HAVING churn_pct >= 30
ORDER BY churn_pct DESC
LIMIT 10;
"""

QUERY_6 = """
-- Monthly cohort retention (simplified: churn by signup month proxy via tenure)
SELECT
    CASE
        WHEN tenure BETWEEN 0  AND 12 THEN 'Month 0-12'
        WHEN tenure BETWEEN 13 AND 24 THEN 'Month 13-24'
        WHEN tenure BETWEEN 25 AND 36 THEN 'Month 25-36'
        WHEN tenure BETWEEN 37 AND 48 THEN 'Month 37-48'
        ELSE 'Month 49+'
    END AS cohort,
    COUNT(*) AS cohort_size,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(SUM(CASE WHEN churn = 'No' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS retention_rate_pct
FROM customers
GROUP BY cohort
ORDER BY cohort;
"""

QUERY_7 = """
-- Top 10 highest-value churned customers
SELECT
    customer_id,
    tenure,
    monthly_charges,
    total_charges,
    contract,
    satisfaction_score,
    num_support_tickets
FROM customers
WHERE churn = 'Yes'
ORDER BY total_charges DESC
LIMIT 10;
"""

QUERY_8 = """
-- Average metrics: churned vs retained
SELECT
    churn,
    ROUND(AVG(tenure), 1)                AS avg_tenure_months,
    ROUND(AVG(monthly_charges), 2)        AS avg_monthly_charges,
    ROUND(AVG(total_charges), 2)          AS avg_total_charges,
    ROUND(AVG(satisfaction_score), 2)     AS avg_satisfaction,
    ROUND(AVG(num_support_tickets), 2)    AS avg_support_tickets,
    ROUND(AVG(late_payments), 2)          AS avg_late_payments
FROM customers
GROUP BY churn;
"""


# ─────────────────────────────────────────────────────────────
# 2. A/B Test Framework Skeleton
# ─────────────────────────────────────────────────────────────
def ab_test_framework(conn: sqlite3.Connection) -> None:
    """
    A/B Test Framework for Retention Offer Analysis.

    Concept:
        - Control group: no intervention
        - Treatment group A: 15% discount on annual plan
        - Treatment group B: Free month + price lock
    
    Metrics tracked:
        - Conversion rate (month-to-month → annual)
        - Churn rate post-intervention (30/60/90 day)
        - Revenue impact per customer
        - Statistical significance (chi-squared test)
    """
    print("\n" + "=" * 60)
    print("  🧪 A/B TEST FRAMEWORK — Retention Offer Analysis")
    print("=" * 60)

    # Simulate assigning at-risk customers to A/B groups
    query = """
    SELECT 
        customer_id,
        contract,
        monthly_charges,
        satisfaction_score,
        CASE 
            WHEN ROWID % 3 = 0 THEN 'Control'
            WHEN ROWID % 3 = 1 THEN 'Treatment_A (15% Discount)'
            ELSE 'Treatment_B (Free Month)'
        END AS ab_group
    FROM customers
    WHERE churn = 'Yes' AND contract = 'Month-to-month'
    LIMIT 300;
    """
    ab_df = pd.read_sql_query(query, conn)

    print(f"\n  At-risk customers in A/B test: {len(ab_df):,}")
    print(f"  Group distribution:\n{ab_df['ab_group'].value_counts().to_string()}")

    # Simulate outcomes (in real deployment, this would be tracked post-intervention)
    np.random.seed(42)
    # Conversion rates per group (simulated business estimates)
    conversion_rates = {
        "Control": 0.05,
        "Treatment_A (15% Discount)": 0.22,
        "Treatment_B (Free Month)": 0.18,
    }

    print("\n  📈 Simulated Conversion Results (would be measured post-rollout):")
    print(f"  {'Group':<35} {'Customers':>10} {'Conv. Rate':>12} {'Retained':>10}")
    print(f"  {'-'*70}")
    for group, rate in conversion_rates.items():
        n = ab_df[ab_df["ab_group"] == group].shape[0]
        retained = int(n * rate)
        print(f"  {group:<35} {n:>10} {rate * 100:>11.1f}% {retained:>10}")

    print("\n  Statistical Testing: Use chi-squared test on conversion counts")
    print("  Minimum detectable effect: 5% lift | Power: 80% | Alpha: 0.05")
    print("  Required sample size per group: ~385 customers (Cohen's h = 0.2)")

    print("\n  Implementation Notes:")
    print("    1. Assign customers randomly to groups using hash(customer_id) % 3")
    print("    2. Track churn status at 30, 60, 90 days post-intervention")
    print("    3. Run chi-squared test when n >= 385 per group")
    print("    4. Ship winning variant to all eligible customers")


# ─────────────────────────────────────────────────────────────
# 3. Model Drift Monitoring Concept
# ─────────────────────────────────────────────────────────────
def model_drift_monitoring(conn: sqlite3.Connection) -> None:
    """
    Model Drift Monitoring Framework.

    Concept:
        - Monitor feature distributions over time (data drift)
        - Monitor prediction score distributions (model drift)
        - Alert when PSI (Population Stability Index) > 0.2
    """
    print("\n" + "=" * 60)
    print("  📡 MODEL DRIFT MONITORING FRAMEWORK")
    print("=" * 60)

    # Simulate two time windows of data
    all_df = pd.read_sql_query("SELECT * FROM customers", conn)
    n = len(all_df)

    baseline = all_df.iloc[:n // 2].copy()   # Training-era data
    current = all_df.iloc[n // 2:].copy()     # Production-era data

    key_features = ["monthly_charges", "tenure", "satisfaction_score", "num_support_tickets"]

    print("\n  PSI (Population Stability Index) per Feature:")
    print(f"  {'Feature':<30} {'PSI':>8} {'Status':>10}")
    print(f"  {'-'*55}")

    for feature in key_features:
        # Compute PSI
        base = baseline[feature].dropna()
        curr = current[feature].dropna()

        bins = np.percentile(base, np.linspace(0, 100, 11))
        bins[0] -= 1e-6
        bins[-1] += 1e-6

        base_pct = np.histogram(base, bins=bins)[0] / len(base)
        curr_pct = np.histogram(curr, bins=bins)[0] / len(curr)

        # Avoid log(0)
        base_pct = np.where(base_pct == 0, 0.0001, base_pct)
        curr_pct = np.where(curr_pct == 0, 0.0001, curr_pct)

        psi = np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct))

        if psi < 0.1:
            status = "✅ Stable"
        elif psi < 0.2:
            status = "⚠️  Warning"
        else:
            status = "🚨 Alert"

        print(f"  {feature:<30} {psi:>8.4f} {status:>10}")

    print("\n  PSI Thresholds:")
    print("    < 0.10: No significant change — model is stable")
    print("    0.10-0.20: Slight shift — investigate and monitor")
    print("    > 0.20: Significant shift — RETRAIN MODEL IMMEDIATELY")

    print("\n  Monitoring Implementation:")
    print("    1. Run PSI check weekly on production feature distributions")
    print("    2. Alert data science team when any feature PSI > 0.1")
    print("    3. Track churn prediction score distribution monthly")
    print("    4. Automate retraining trigger when ROC-AUC drops > 3%")
    print("    5. Log all predictions with timestamps for drift analysis")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  SQL COHORT ANALYSIS & ADVANCED ANALYTICS")
    print("=" * 60)

    df = pd.read_csv(RAW_DATA_PATH)
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    df["total_charges"].fillna(df["total_charges"].median(), inplace=True)

    conn = get_connection(df)

    # Run cohort queries
    run_query(conn, QUERY_1, "Overall Churn Rate")
    run_query(conn, QUERY_2, "Churn Rate by Contract Type")
    run_query(conn, QUERY_3, "Churn Rate by Tenure Bucket")
    run_query(conn, QUERY_4, "Revenue at Risk by Segment")
    run_query(conn, QUERY_5, "High-Risk Customer Segments")
    run_query(conn, QUERY_6, "Cohort Retention Analysis")
    run_query(conn, QUERY_7, "Top 10 Highest-Value Churned Customers")
    run_query(conn, QUERY_8, "Average Metrics: Churned vs Retained")

    # A/B test framework
    ab_test_framework(conn)

    # Model drift monitoring
    model_drift_monitoring(conn)

    conn.close()
    print("\n[✓] SQL analysis complete.\n")


if __name__ == "__main__":
    main()
