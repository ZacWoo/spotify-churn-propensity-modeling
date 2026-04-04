#!/usr/bin/env python3
"""
Spotify Churn Propensity Modeling
=================================
Entry point — downloads the dataset, engineers features, trains three
logistic-regression models, and saves metrics + plots.

Usage:
    python main.py
"""

from src.data_loader import load_dataset, print_schema
from src.preprocessing import (
    engineer_features,
    build_inactivity_dataset,
    build_free_conversion_dataset,
    build_premium_churn_dataset,
)
from src.models import train_and_evaluate
from src.visualization import save_all_plots


def main():
    # ------------------------------------------------------------------
    # 1. Load & inspect
    # ------------------------------------------------------------------
    df = load_dataset()
    print_schema(df)

    # ------------------------------------------------------------------
    # 2. Feature engineering
    # ------------------------------------------------------------------
    df = engineer_features(df)

    # ------------------------------------------------------------------
    # 3. Build task-specific datasets
    # ------------------------------------------------------------------
    tasks = {
        "Inactivity Risk": build_inactivity_dataset(df),
        "Free to Subscribe": build_free_conversion_dataset(df),
        "Premium Churn Risk": build_premium_churn_dataset(df),
    }

    # ------------------------------------------------------------------
    # 4. Train, evaluate, and visualize each model
    # ------------------------------------------------------------------
    all_results = {}
    for task_name, (X, y) in tasks.items():
        results = train_and_evaluate(X, y, task_name)
        save_all_plots(results)
        all_results[task_name] = results

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for name, r in all_results.items():
        print(f"  {name:25s}  AUC={r['roc_auc']:.4f}  F1={r['f1']:.4f}")
    print("\nAll metrics  -> outputs/metrics/")
    print("All plots    -> outputs/plots/")
    print("Done.")


if __name__ == "__main__":
    main()
