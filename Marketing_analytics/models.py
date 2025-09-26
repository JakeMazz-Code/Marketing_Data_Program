"""Modeling helpers for marketing analytics (conversion propensity, uplift, etc.)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from Marketing_analytics.data_loader import DatasetBundle


@dataclass(slots=True)
class PropensityResults:
    model: Pipeline
    metrics: Dict[str, float]
    feature_importance: pd.DataFrame
    holdout_predictions: pd.DataFrame


def train_propensity_model(bundle: DatasetBundle, *, random_seed: int = 42) -> PropensityResults | None:
    mapping = bundle.mapping

    if not mapping.response or mapping.response not in bundle.frame.columns:
        return None

    df = bundle.frame.copy()
    df = df.dropna(subset=[mapping.response])
    if df.empty:
        return None

    numeric_features = [col for col in mapping.numeric_features() if col in df.columns]
    categorical_features = [col for col in mapping.categorical_features() if col in df.columns]
    candidate_features: List[str] = numeric_features + categorical_features

    if not candidate_features:
        return None

    X = df[candidate_features]
    y = df[mapping.response].astype(float).fillna(0)

    stratify = y if y.nunique() > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=random_seed,
        stratify=stratify,
    )

    transformers = []
    if numeric_features:
        transformers.append(("num", StandardScaler(), numeric_features))
    if categorical_features:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features))

    preprocess = ColumnTransformer(transformers=transformers, remainder="drop")

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    precision, recall, fbeta, _ = precision_recall_fscore_support(
        y_test, y_pred, zero_division=0, average=None
    )
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision[1]) if precision.size > 1 else float(precision[0]),
        "recall": float(recall[1]) if recall.size > 1 else float(recall[0]),
        "f1": float(fbeta[1]) if fbeta.size > 1 else float(fbeta[0]),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
    except ValueError:
        metrics["roc_auc"] = float("nan")

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    class_one = report.get("1", {})
    for key in ("precision", "recall", "f1-score", "support"):
        value = class_one.get(key)
        if isinstance(value, (int, float)):
            metrics[f"class_1_{key}"] = float(value)

    clf: LogisticRegression = pipeline.named_steps["clf"]
    pre: ColumnTransformer = pipeline.named_steps["preprocess"]
    feature_names = pre.get_feature_names_out() if hasattr(pre, "get_feature_names_out") else np.array(candidate_features)
    coefficients = clf.coef_.ravel()
    importance = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
        "odds_ratio": np.exp(coefficients),
    }).sort_values("odds_ratio", ascending=False)

    holdout = X_test.copy()
    holdout["actual"] = y_test.to_numpy()
    holdout["predicted_probability"] = y_prob
    holdout["predicted_label"] = y_pred

    return PropensityResults(
        model=pipeline,
        metrics=metrics,
        feature_importance=importance,
        holdout_predictions=holdout,
    )
