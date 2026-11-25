import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score   # o la métrica que uses
from src.pbcre.tree.tree import PBCRETree
class PBCREHyperparameterSearch:
    def __init__(self, param_grid, metric, cv=5, random_state=42):
        self.param_grid = param_grid
        self.metric = metric
        self.cv = cv
        self.random_state = random_state

    def _iterate_params(self):
        from itertools import product
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())

        for combination in product(*values):
            yield dict(zip(keys, combination))

    def fit(self, X, y):
        best_score = -np.inf
        best_params = None

        skf = StratifiedKFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=self.random_state
        )

        for params in self._iterate_params():
            fold_scores = []
            print("params:      ",params)
            for train_idx, test_idx in skf.split(X, y):
                model = PBCRETree(**params)
                model.fit(X.iloc[train_idx], y.iloc[train_idx])

                y_pred = model.predict(X.iloc[test_idx])
                score = self.metric(y.iloc[test_idx], y_pred)
                fold_scores.append(score)

            avg_score = np.mean(fold_scores)
            print("avg_score:  ", avg_score)
            if avg_score > best_score:
                best_score = avg_score
                best_params = params

        self.best_params_ = best_params
        self.best_score_ = best_score
        return self
