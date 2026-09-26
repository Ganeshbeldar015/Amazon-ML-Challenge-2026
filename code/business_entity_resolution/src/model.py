import os
import joblib
import numpy as np
import lightgbm as lgb
from typing import Optional

class EntityMatchingModel:
    """
    High-Capacity Regularized LightGBM Classifier for Business Entity Resolution.
    Tuned for 45 non-linear dense features to achieve maximum discriminative precision.
    """
    def __init__(
        self,
        n_estimators: int = 350,
        learning_rate: float = 0.04,
        max_depth: int = 8,
        num_leaves: int = 63,
        min_child_samples: int = 40,
        reg_alpha: float = 0.05,
        reg_lambda: float = 0.5
    ):
        self.clf = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=42,
            n_jobs=-1
        )
        self.is_trained = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.clf.fit(X, y)
        self.is_trained = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model must be trained before predicting probabilities.")
        return self.clf.predict_proba(X)[:, 1]

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self.clf, filepath)

    @classmethod
    def load(cls, filepath: str) -> "EntityMatchingModel":
        model = cls()
        model.clf = joblib.load(filepath)
        model.is_trained = True
        return model
