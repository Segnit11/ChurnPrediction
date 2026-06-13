"""Portfolio-level analytics computed from the historical churn dataset.

These power the marketing/dashboard sections of the frontend (overall churn
rate, churn drivers by segment, etc.). Everything is computed once and cached
so the /analytics endpoint stays fast.
"""

import pandas as pd


def _rate(group):
    """Return churn rate (%) rounded to 1dp for a grouped frame."""
    return round(float(group["Exited"].mean()) * 100, 1)


def compute_portfolio_analytics(df: pd.DataFrame) -> dict:
    """Build an aggregate analytics summary from the labelled dataset."""
    df = df.copy()
    total = int(len(df))
    churned = int(df["Exited"].sum())
    retained = total - churned

    # Age buckets
    bins = [17, 30, 40, 50, 60, 200]
    labels = ["18-30", "31-40", "41-50", "51-60", "60+"]
    df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels)

    # Credit score bands
    cs_bins = [0, 580, 670, 740, 800, 1000]
    cs_labels = ["Poor", "Fair", "Good", "Very Good", "Excellent"]
    df["CreditBand"] = pd.cut(df["CreditScore"], bins=cs_bins, labels=cs_labels)

    def by(col):
        g = df.groupby(col, observed=True)
        return [
            {
                "label": str(k),
                "churnRate": _rate(sub),
                "customers": int(len(sub)),
            }
            for k, sub in g
        ]

    summary = {
        "totalCustomers": total,
        "churned": churned,
        "retained": retained,
        "overallChurnRate": round(churned / total * 100, 1) if total else 0.0,
        "retentionRate": round(retained / total * 100, 1) if total else 0.0,
        "avgBalanceChurned": round(float(df[df["Exited"] == 1]["Balance"].mean()), 2),
        "avgBalanceRetained": round(float(df[df["Exited"] == 0]["Balance"].mean()), 2),
        "avgAgeChurned": round(float(df[df["Exited"] == 1]["Age"].mean()), 1),
        "avgAgeRetained": round(float(df[df["Exited"] == 0]["Age"].mean()), 1),
        "byGeography": by("Geography"),
        "byGender": by("Gender"),
        "byAgeGroup": by("AgeGroup"),
        "byNumProducts": by("NumOfProducts"),
        "byCreditBand": by("CreditBand"),
        "byActivity": [
            {
                "label": "Active" if k == 1 else "Inactive",
                "churnRate": _rate(sub),
                "customers": int(len(sub)),
            }
            for k, sub in df.groupby("IsActiveMember", observed=True)
        ],
    }
    return summary


# Static, model-derived feature importances (from the trained stacking ensemble).
FEATURE_IMPORTANCE = [
    {"feature": "Number of Products", "importance": 0.3239},
    {"feature": "Active Membership", "importance": 0.1641},
    {"feature": "Age", "importance": 0.1096},
    {"feature": "Geography: Germany", "importance": 0.0914},
    {"feature": "Balance", "importance": 0.0528},
    {"feature": "Geography: France", "importance": 0.0465},
    {"feature": "Gender: Female", "importance": 0.0453},
    {"feature": "Geography: Spain", "importance": 0.0361},
    {"feature": "Credit Score", "importance": 0.0351},
    {"feature": "Estimated Salary", "importance": 0.0327},
]


def risk_tier(probability: float) -> dict:
    """Map a churn probability to a human-readable risk tier + action plan."""
    p = float(probability)
    if p >= 0.7:
        return {
            "tier": "Critical",
            "color": "danger",
            "headline": "Immediate intervention required",
            "actions": [
                "Trigger a personal outreach call from a relationship manager within 48 hours",
                "Offer a tailored retention package (fee waiver or rate boost)",
                "Flag the account for weekly monitoring",
            ],
        }
    if p >= 0.4:
        return {
            "tier": "Elevated",
            "color": "warn",
            "headline": "At risk — proactive engagement advised",
            "actions": [
                "Send the personalized retention email below",
                "Recommend a complementary product to deepen the relationship",
                "Invite to a financial review session",
            ],
        }
    if p >= 0.2:
        return {
            "tier": "Watch",
            "color": "brand",
            "headline": "Stable, with light monitoring",
            "actions": [
                "Include in the next loyalty rewards campaign",
                "Surface relevant cross-sell offers in-app",
            ],
        }
    return {
        "tier": "Loyal",
        "color": "safe",
        "headline": "Healthy & engaged",
        "actions": [
            "Nurture with appreciation perks",
            "Consider for referral / advocacy programs",
        ],
    }
