import numpy as np
import pandas as pd
from itertools import product

from sklearn.model_selection import StratifiedKFold

from srcc.pbcre.tuning.cv_utils import evaluate_params


class PBCREHyperparameterSearch:
    """
    Grid search con CV estratificada.

    PbCRPE_v1: sampler dentro de cada fold, greater_is_better, estrategia de
    umbral y registro de FPR/FNR/BER por fold en history_.
    """

    def __init__(self, param_grid, metric, cv=5, random_state=42,
                 greater_is_better=True, sampler=None,
                 threshold_strategy=None, prune_metric=None, verbose=True):
        self.param_grid = param_grid
        self.metric = metric
        self.cv = cv
        self.random_state = random_state
        self.greater_is_better = greater_is_better
        self.sampler = sampler
        self.threshold_strategy = threshold_strategy
        self.prune_metric = prune_metric
        self.verbose = verbose
        self.history_ = []

    def _iterate_params(self):
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        for combination in product(*values):
            yield dict(zip(keys, combination))

    def fit(self, X, y):
        best_score = -np.inf
        best_params = None
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True,
                              random_state=self.random_state)

        for params in self._iterate_params():
            avg_score, reports = evaluate_params(
                params, X, y, skf, self.metric,
                greater_is_better=self.greater_is_better,
                sampler=self.sampler,
                threshold_strategy=self.threshold_strategy,
                prune_metric=self.prune_metric,
                random_state=self.random_state)

            df = pd.DataFrame(reports)
            self.history_.append({
                "params": params.copy(), "score": avg_score,
                "fpr_mean": df["fpr"].mean(), "fpr_std": df["fpr"].std(),
                "fnr_mean": df["fnr"].mean(), "fnr_std": df["fnr"].std(),
                "ber_mean": df["ber"].mean(), "folds": reports})

            if self.verbose:
                print(f"params: {params} | score: {avg_score:.4f} | "
                      f"FPR: {df['fpr'].mean():.4f} | FNR: {df['fnr'].mean():.4f}")

            if avg_score > best_score:
                best_score = avg_score
                best_params = params

        self.best_params_ = best_params
        self.best_score_ = best_score
        return self

    def history_frame(self):
        return pd.DataFrame([{k: v for k, v in h.items() if k != "folds"}
                             for h in self.history_])
