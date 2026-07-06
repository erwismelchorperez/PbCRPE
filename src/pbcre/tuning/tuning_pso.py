import numpy as np

from sklearn.model_selection import StratifiedKFold

from mealpy.swarm_based.PSO import OriginalPSO
from mealpy.utils.space import IntegerVar, FloatVar

from src.pbcre.tree.tree import PBCRETree


class PBCREHyperparameterSearchPSO:

    INTEGER_PARAMS = {
        "max_depth",
        "min_samples_leaf",
        "max_features_sfs",
        "n_clusters",
        "n_patterns"
    }

    def __init__(self,param_bounds,metric,criterion="hybrid",cv=5,pso_epochs=40,pso_pop_size=25,random_state=42):

        self.param_bounds = param_bounds
        self.metric = metric
        self.criterion = criterion

        self.cv = cv
        self.pso_epochs = pso_epochs
        self.pso_pop_size = pso_pop_size
        self.random_state = random_state

        self.history_ = []

    def _build_bounds(self):

        bounds = []
        self.param_order_ = []

        for key, (lb, ub) in self.param_bounds.items():
            self.param_order_.append(key)
            if key in self.INTEGER_PARAMS:
                bounds.append(IntegerVar(lb=lb,ub=ub))
            else:
                bounds.append(FloatVar(lb=lb,ub=ub))

        return bounds

    def _solution_to_params(self, solution):

        params = {}

        for i, key in enumerate(self.param_order_):
            if key in self.INTEGER_PARAMS:
                params[key] = int(round(solution[i]))
            else:
                params[key] = float(solution[i])

        return params

    def fit(self, X, y):

        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        def objective(solution):
            params = self._solution_to_params(solution)
            params["criterion"] = self.criterion
            fold_scores = []

            for train_idx, test_idx in skf.split(X, y):

                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]

                X_test = X.iloc[test_idx]
                y_test = y.iloc[test_idx]

                model = PBCRETree(**params)
                model.fit(X_train,y_train)
                y_pred = model.predict(X_test)

                score = self.metric(y_test,y_pred)
                fold_scores.append(score)

            avg_score = np.mean(fold_scores)

            self.history_.append({
                "params": params.copy(),
                "score": avg_score
            })

            return -avg_score

        problem = {
            "bounds": self._build_bounds(),
            "minmax": "min",
            "obj_func": objective
        }

        optimizer = OriginalPSO(epoch=self.pso_epochs,pop_size=self.pso_pop_size)

        best_solution, best_fitness = optimizer.solve(problem)

        self.best_params_ = self._solution_to_params(best_solution)
        self.best_params_["criterion"] = self.criterion
        self.best_score_ = -best_fitness
        return self