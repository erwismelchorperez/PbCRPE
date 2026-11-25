import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

from mealpy.swarm_based.PSO import OriginalPSO
from mealpy.utils.space import IntegerVar, FloatVar

from srcc.pbcre.tuning.cv_utils import evaluate_params


class PBCREHyperparameterSearchPSO:
    """
    Búsqueda PSO (mealpy).

    PbCRPE_v1: sampler dentro de cada fold, greater_is_better,
    estrategia de umbral y registro de FPR/FNR por evaluación.
    Nota: en los criterios híbridos sólo importa la proporción
    beta : (1 - beta) * alpha, así que basta tunear uno de los dos.
    """

    INTEGER_PARAMS = {
        "max_depth",
        "min_samples_leaf",
        "max_features_sfs",
        "n_clusters",
        "n_patterns",
        "n_quantiles"
    }

    def __init__(self, param_bounds, metric, criterion="hybrid", cv=5,
                 pso_epochs=40, pso_pop_size=25, random_state=42,
                 greater_is_better=True, sampler=None,
                 threshold_strategy=None, prune_metric=None,
                 fixed_params=None):
        self.param_bounds = param_bounds
        self.metric = metric
        self.criterion = criterion
        self.cv = cv
        self.pso_epochs = pso_epochs
        self.pso_pop_size = pso_pop_size
        self.random_state = random_state
        self.greater_is_better = greater_is_better
        self.sampler = sampler
        self.threshold_strategy = threshold_strategy
        self.prune_metric = prune_metric
        self.fixed_params = fixed_params or {}
        self.history_ = []

    def _build_bounds(self):
        bounds = []
        self.param_order_ = []
        for key, (lb, ub) in self.param_bounds.items():
            self.param_order_.append(key)
            if key in self.INTEGER_PARAMS:
                bounds.append(IntegerVar(lb=lb, ub=ub))
            else:
                bounds.append(FloatVar(lb=lb, ub=ub))
        return bounds

    def _solution_to_params(self, solution):
        params = dict(self.fixed_params)
        for i, key in enumerate(self.param_order_):
            if key in self.INTEGER_PARAMS:
                params[key] = int(round(solution[i]))
            else:
                params[key] = float(solution[i])
        params["criterion"] = self.criterion
        params.setdefault("random_state", self.random_state)
        return params

    def fit(self, X, y):
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True,
                              random_state=self.random_state)

        def objective(solution):
            params = self._solution_to_params(solution)
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
            return -avg_score

        problem = {
            "bounds": self._build_bounds(),
            "minmax": "min",
            "obj_func": objective
        }

        optimizer = OriginalPSO(epoch=self.pso_epochs, pop_size=self.pso_pop_size)
        best_agent = optimizer.solve(problem)

        # mealpy >= 3 devuelve un Agent; versiones antiguas, una tupla
        if isinstance(best_agent, tuple):
            best_solution, best_fitness = best_agent
        else:
            best_solution = best_agent.solution
            best_fitness = best_agent.target.fitness

        self.best_params_ = self._solution_to_params(best_solution)
        self.best_score_ = -best_fitness
        return self

    def history_frame(self):
        return pd.DataFrame(self.history_)
