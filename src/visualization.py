"""
visualization.py
----------------
Generate and save evaluation plots for each model.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
from pathlib import Path


PLOT_DIR = Path("outputs/plots")


def save_all_plots(results: dict) -> None:
    """Generate ROC curve, confusion matrix, and coefficient bar chart."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    slug = results["task"].lower().replace(" ", "_")

    _plot_roc(results, slug)
    _plot_confusion(results, slug)
    _plot_coefficients(results, slug)


def _plot_roc(results: dict, slug: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(
        results["y_test"], results["y_prob"], ax=ax, name=results["task"]
    )
    ax.set_title(f"ROC Curve — {results['task']}\nAUC = {results['roc_auc']:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    fig.tight_layout()
    path = PLOT_DIR / f"{slug}_roc.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def _plot_confusion(results: dict, slug: str) -> None:
    cm = np.array(results["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"]).plot(
        ax=ax, cmap="Blues"
    )
    ax.set_title(f"Confusion Matrix — {results['task']}")
    fig.tight_layout()
    path = PLOT_DIR / f"{slug}_confusion.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def _plot_coefficients(results: dict, slug: str) -> None:
    coefs = results["top_coefficients"]
    df = pd.DataFrame(coefs).sort_values("coefficient")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in df["coefficient"]]
    ax.barh(df["feature"], df["coefficient"], color=colors)
    ax.set_xlabel("Logistic Regression Coefficient")
    ax.set_title(f"Top Feature Coefficients — {results['task']}")
    ax.axvline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    path = PLOT_DIR / f"{slug}_coefficients.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")
