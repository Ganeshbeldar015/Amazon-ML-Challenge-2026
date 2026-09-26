import os
import joblib
import numpy as np
import lightgbm as lgb
from typing import Optional

class EntityMatchingModel:
    """
    LightGBM Classifier wrapper for Business Entity Resolution pairwise classification.
    Optimized with L1/L2 regularization and minimum leaf sample constraints to prevent overfitting.
    """
    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_leaves: int = 31,
        min_child_samples: int = 50,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0
    ):
        self.clf = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            subsample=0.8,
            colsample_bytree=0.8,
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
