"""
models.py
---------
Train and evaluate logistic-regression models for each churn-propensity task.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.preprocessing import build_preprocessor


OUTPUTS = Path("outputs/metrics")


def _get_feature_names(preprocessor) -> list[str]:
    """Extract feature names from a fitted ColumnTransformer."""
    names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(trans, "get_feature_names_out"):
            names.extend(trans.get_feature_names_out(cols))
        else:
            names.extend(cols)
    return list(names)


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    task_name: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Build a preprocessing + logistic-regression pipeline, train on 80 %,
    evaluate on 20 %, and return a results dict.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    preprocessor = build_preprocessor(X_train)

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=random_state,
        )),
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Top coefficients
    feature_names = _get_feature_names(pipe.named_steps["preprocessor"])
    coefs = pipe.named_steps["classifier"].coef_[0]
    coef_df = (
        pd.DataFrame({"feature": feature_names, "coefficient": coefs})
        .sort_values("coefficient", key=abs, ascending=False)
        .head(15)
    )

    results = {
        "task": task_name,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "pos_rate_train": float(y_train.mean()),
        "pos_rate_test": float(y_test.mean()),
        "roc_auc": round(auc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "confusion_matrix": cm.tolist(),
        "top_coefficients": coef_df.to_dict(orient="records"),
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"  {task_name}")
    print(f"{'='*60}")
    print(f"  Train size : {results['n_train']:,}  |  Test size: {results['n_test']:,}")
    print(f"  Positive rate (train): {results['pos_rate_train']:.2%}")
    print(f"  ROC-AUC    : {auc:.4f}")
    print(f"  Precision  : {prec:.4f}")
    print(f"  Recall     : {rec:.4f}")
    print(f"  F1         : {f1:.4f}")
    print(f"\n  Confusion Matrix:\n{cm}")
    print(f"\n  Classification Report:\n{classification_report(y_test, y_pred)}")
    print("  Top 10 coefficients (by magnitude):")
    for _, row in coef_df.head(10).iterrows():
        print(f"    {row['feature']:45s} {row['coefficient']:+.4f}")

    # Save to JSON
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS / f"{task_name.lower().replace(' ', '_')}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Metrics saved to {out_path}")

    return {**results, "pipeline": pipe, "X_test": X_test, "y_test": y_test, "y_prob": y_prob}
