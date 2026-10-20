from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, brier_score_loss, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline


def select_predictors(frame: pd.DataFrame, excluded: Iterable[str]) -> list[str]:
    candidates = frame.select_dtypes(include="number").columns
    features = [column for column in candidates if column not in set(excluded)]
    if not features:
        raise ValueError("No numeric predictors remain after exclusions.")
    return features


def make_ensemble(seed: int, features: list[str]) -> VotingClassifier:
    preprocessor = ColumnTransformer(
        [("numeric", SimpleImputer(strategy="median", add_indicator=True), features)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    estimators = [
        ("rf", RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                        class_weight="balanced", n_jobs=-1, random_state=seed)),
        ("et", ExtraTreesClassifier(n_estimators=500, min_samples_leaf=4,
                                     class_weight="balanced", n_jobs=-1, random_state=seed + 1)),
        ("hgb", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                max_leaf_nodes=15, l2_regularization=1.0,
                                                random_state=seed + 2)),
    ]
    ensemble = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
    return Pipeline([("preprocessor", preprocessor), ("model", ensemble)])


def binary_metrics(y_true: pd.Series | np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true)
    predicted = (probabilities >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, probabilities),
        "average_precision": average_precision_score(y_true, probabilities),
        "brier_score": brier_score_loss(y_true, probabilities),
        "sensitivity": recall_score(y_true, predicted, zero_division=0),
        "precision": precision_score(y_true, predicted, zero_division=0),
    }


def bootstrap_auc(y_true: np.ndarray, probabilities: np.ndarray, seed: int, n_bootstrap: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y_true, probabilities = np.asarray(y_true), np.asarray(probabilities)
    scores: list[float] = []
    for _ in range(n_bootstrap):
        index = rng.integers(0, len(y_true), len(y_true))
        if np.unique(y_true[index]).size == 2:
            scores.append(roc_auc_score(y_true[index], probabilities[index]))
    return tuple(np.quantile(scores, [0.025, 0.975]).tolist())
