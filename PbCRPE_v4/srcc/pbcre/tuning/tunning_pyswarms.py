import numpy as np
import pandas as pd
import pyswarms as ps
from sklearn.model_selection import StratifiedKFold

from srcc.pbcre.tuning.cv_utils import evaluate_params


class PBCREHyperparameterSearchPSO:
    """
    Búsqueda PSO (pyswarms).

    param_bounds:
    {
        "max_depth": (2, 15),
        "min_samples_leaf": (5, 50),
        "alpha": (0.0, 1.0),
        "beta": (0.0, 1.0)
    }
    PbCRPE_v1: sampler dentro de cada fold, greater_is_better, estrategia de
    umbral, criterion configurable y registro de FPR/FNR.
    """

    INTEGER_PARAMS = {"max_depth", "min_samples_leaf", "max_features_sfs",
                      "n_quantiles"}

    def __init__(self, param_bounds, metric, criterion="hybrid", cv=5,
                 n_particles=25, iters=40, random_state=42,
                 greater_is_better=True, sampler=None,
                 threshold_strategy=None, prune_metric=None,
                 fixed_params=None):
        self.param_bounds = param_bounds
        self.metric = metric
        self.criterion = criterion
        self.cv = cv
        self.n_particles = n_particles
        self.iters = iters
        self.random_state = random_state
        self.greater_is_better = greater_is_better
        self.sampler = sampler
        self.threshold_strategy = threshold_strategy
        self.prune_metric = prune_metric
        self.fixed_params = fixed_params or {}
        self.history_ = []

        self.param_order_ = list(param_bounds.keys())
        self.dimensions = len(self.param_order_)
        self.lb = np.array([param_bounds[k][0] for k in self.param_order_], dtype=float)
        self.ub = np.array([param_bounds[k][1] for k in self.param_order_], dtype=float)

    def _vector_to_params(self, x):
        params = dict(self.fixed_params)
        for i, key in enumerate(self.param_order_):
            if key in self.INTEGER_PARAMS:
                params[key] = int(np.round(x[i]))
            else:
                params[key] = float(x[i])

        if "alpha" in params and "beta" in params:
            s = params["alpha"] + params["beta"]
            if s > 1:
                params["alpha"] /= s
                params["beta"] /= s

        params["criterion"] = self.criterion
        params.setdefault("random_state", self.random_state)
        return params

    def fit(self, X, y):
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True,
                              random_state=self.random_state)

        def objective_function(X_particles):
            costs = np.zeros(X_particles.shape[0])
            for i, particle in enumerate(X_particles):
                params = self._vector_to_params(particle)
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
                    "fpr_mean": df["fpr"].mean(), "fnr_mean": df["fnr"].mean(),
                    "ber_mean": df["ber"].mean()})
                costs[i] = -avg_score  # minimización
            return costs

        optimizer = ps.single.GlobalBestPSO(
            n_particles=self.n_particles,
            dimensions=self.dimensions,
            options={"c1": 2.05, "c2": 2.05, "w": 0.4},
            bounds=(self.lb, self.ub)
        )
        best_cost, best_pos = optimizer.optimize(objective_function,
                                                 iters=self.iters)

        self.best_params_ = self._vector_to_params(best_pos)
        self.best_score_ = -best_cost
        return self

    def history_frame(self):
        return pd.DataFrame(self.history_)
