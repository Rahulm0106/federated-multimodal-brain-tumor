import numpy as np
from sklearn.ensemble import RandomForestClassifier
import torch


class RandomForestHead:
    def __init__(self, n_estimators=100, max_depth=None):
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=1
        )
        self.is_fitted = False

    def fit(self, features: torch.Tensor, labels: torch.Tensor):
        X = features.detach().cpu().numpy()
        y = labels.cpu().numpy()
        self.rf.fit(X, y)
        self.is_fitted = True

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted:
            raise RuntimeError("RF head not fitted.")
        X = features.detach().cpu().numpy()
        proba = self.rf.predict_proba(X)[:, 1]
        return torch.from_numpy(proba).to(features.device)

    def predict_class(self, features: torch.Tensor) -> torch.Tensor:
        X = features.detach().cpu().numpy()
        pred = self.rf.predict(X)
        return torch.from_numpy(pred).to(features.device)