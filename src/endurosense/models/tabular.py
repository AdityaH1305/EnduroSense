"""Tabular Model A predictors: Linear Regression (ridge), Random Forest, XGBoost.

Each wraps a scikit-learn style estimator with median imputation (a few
features are undefined for some rows, e.g. internal resistance on the ground)
and, for the linear model, standardisation. Imputation and scaling statistics
come from the training fold only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from endurosense.features.model_a import FEATURES


class TabularModel:
    def __init__(self, name: str, estimator, features: list[str] = FEATURES, scale: bool = False):
        self.name, self.features = name, list(features)
        steps = [SimpleImputer(strategy="median")] + ([StandardScaler()] if scale else []) + [estimator]
        self.pipe = make_pipeline(*steps)

    def fit(self, train: pd.DataFrame, target: str) -> "TabularModel":
        self.pipe.fit(train[self.features], train[target])
        est = self.pipe.steps[-1][1]
        if isinstance(est, RandomForestRegressor):
            # Trees are fitted in parallel, but predictions add the trees up one by one. Summing them
            # across threads happens in a varying order, which changes results in the 15th digit.
            est.n_jobs = 1
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipe.predict(df[self.features])


class PhysicsInformedModel:
    """A tabular model that also receives a physics baseline's estimate as an input.

    The baseline (fitted on the training fold) supplies a good first guess from
    the discharge curve; the ML model learns where and by how much it is wrong,
    e.g. for weaker batteries. Plain idea: physics first, ML corrects it.
    """

    def __init__(self, baseline, model: TabularModel):
        self.baseline, self.model = baseline, model
        self.name = f"{model.name} + physics"
        self.model.features = self.model.features + ["baseline_wh"]

    def _with_baseline(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(baseline_wh=self.baseline.predict(df))

    def fit(self, train: pd.DataFrame, target: str) -> "PhysicsInformedModel":
        self.baseline.fit(train, target)
        self.model.fit(self._with_baseline(train), target)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._with_baseline(df))


def linear(alpha: float = 1.0, **_) -> TabularModel:
    return TabularModel("Linear Regression", Ridge(alpha=alpha), scale=True)


def random_forest(n_estimators=300, max_depth=None, min_samples_leaf=5, max_features=0.5,
                  max_samples=0.5, seed=42, **_) -> TabularModel:
    return TabularModel("Random Forest", RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
        max_features=max_features, max_samples=max_samples, n_jobs=-1, random_state=seed))


def xgboost(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, reg_lambda=1.0, seed=42, **_) -> TabularModel:
    return TabularModel("XGBoost", XGBRegressor(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, subsample=subsample,
        colsample_bytree=colsample_bytree, min_child_weight=min_child_weight, reg_lambda=reg_lambda,
        tree_method="hist", random_state=seed, n_jobs=-1))


SEARCH_SPACES = {
    "random_forest": {"max_depth": [None, 12, 20], "min_samples_leaf": [2, 5, 10, 20],
                      "max_features": [0.3, 0.5, 0.8], "max_samples": [0.3, 0.5]},
    "xgboost": {"n_estimators": [200, 400, 800], "max_depth": [3, 4, 6, 8], "learning_rate": [0.02, 0.05, 0.1],
                "subsample": [0.6, 0.8, 1.0], "colsample_bytree": [0.5, 0.8, 1.0],
                "min_child_weight": [1, 5, 20], "reg_lambda": [0.1, 1.0, 10.0]},
    "linear": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
}
FACTORIES = {"linear": linear, "random_forest": random_forest, "xgboost": xgboost}


def sample_configs(kind: str, n: int, seed: int) -> list[dict]:
    """``n`` distinct random configurations from ``SEARCH_SPACES[kind]`` (all of them for the linear model)."""
    space = SEARCH_SPACES[kind]
    if kind == "linear":
        return [{"alpha": a} for a in space["alpha"]]
    rng = np.random.default_rng(seed)
    seen, out = set(), []
    while len(out) < n:
        cfg = {k: v[rng.integers(len(v))] for k, v in space.items()}
        key = tuple(sorted(cfg.items(), key=lambda kv: kv[0]))
        if key not in seen:
            seen.add(key)
            out.append(cfg)
    return out
