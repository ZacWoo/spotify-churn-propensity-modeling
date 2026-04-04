"""
preprocessing.py
----------------
Feature engineering and reusable sklearn preprocessing pipelines for the
three churn-propensity models.

Proxy-target definitions (no true churn labels exist in this dataset):

1. Inactivity risk  — target: `inactive_3_months_flag` (1 = user had >= 3
   consecutive months of inactivity).  This flag is provided directly in the
   dataset.

2. Free-to-subscribe likelihood — target: `ad_conversion_to_subscription`
   (1 = a Free-tier user converted after interacting with an ad).  This is a
   proxy because it captures demonstrated willingness to pay, not a future
   prediction ground truth.

3. Premium cancel / downgrade risk — target: `subscription_status == 'Inactive'`
   among users whose subscription_type is one of the paid tiers (Premium
   Individual, Premium Duo, Premium Family, Student).  "Inactive" status on a
   paid plan is used as a proxy for cancellation or downgrade intent.
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns useful across all three models."""
    df = df.copy()

    # Account tenure in days (from signup_date to the dataset's implicit snapshot)
    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
    snapshot_date = df["signup_date"].max() + pd.Timedelta(days=30)
    df["account_age_days"] = (snapshot_date - df["signup_date"]).dt.days

    # Binary flag: is the user on a paid plan?
    df["is_premium"] = (~df["subscription_type"].isin(["Free"])).astype(int)

    # Engagement ratio: listening hours relative to skips
    df["listen_to_skip_ratio"] = (
        df["avg_listening_hours_per_week"] / (df["avg_skips_per_day"] + 1)
    )

    return df


# ---------------------------------------------------------------------------
# Task-specific dataset builders
# ---------------------------------------------------------------------------

# Columns never used as features (identifiers, targets, leakage)
_DROP_ALWAYS = [
    "user_id",
    "signup_date",
    "subscription_status",
    "inactive_3_months_flag",
    "months_inactive",
    "ad_conversion_to_subscription",
]


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _DROP_ALWAYS]


def build_inactivity_dataset(df: pd.DataFrame):
    """Task 1 — predict which users become inactive (3+ months)."""
    features = df[_feature_cols(df)]
    target = df["inactive_3_months_flag"]
    return features, target


def build_free_conversion_dataset(df: pd.DataFrame):
    """Task 2 — among Free users, predict who is likely to subscribe."""
    mask = df["subscription_type"] == "Free"
    subset = df[mask]
    features = subset[_feature_cols(subset)]
    target = (subset["ad_conversion_to_subscription"] == "Yes").astype(int)
    return features, target


def build_premium_churn_dataset(df: pd.DataFrame):
    """Task 3 — among paid users, predict who becomes Inactive."""
    paid_tiers = ["Premium Individual", "Premium Duo", "Premium Family", "Student"]
    mask = df["subscription_type"].isin(paid_tiers)
    subset = df[mask]
    features = subset[_feature_cols(subset)]
    target = (subset["subscription_status"] == "Inactive").astype(int)
    return features, target


# ---------------------------------------------------------------------------
# Sklearn preprocessing pipeline (reusable across models)
# ---------------------------------------------------------------------------

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Return a ColumnTransformer that scales numerics and one-hot encodes
    categoricals based on the dtypes present in *X*."""
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )
