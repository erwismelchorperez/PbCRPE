import numpy as np
import pyswarms as ps
from sklearn.model_selection import StratifiedKFold
from src.pbcre.tree.tree import PBCRETree


class PBCREHyperparameterSearchPSO:
    def __init__(
        self,
        param_bounds,
        metric,
        cv=5,
        n_particles=25,
        iters=40,
        random_state=42
    ):
        """
        param_bounds:
        {
            "max_depth": (20, 150),
            "min_samples_leaf": (5, 50),
            "alpha": (0.0, 1.0),
            "beta": (0.0, 1.0)
        }
        """
        self.param_bounds = param_bounds
        self.metric = metric
        self.cv = cv
        self.n_particles = n_particles
        self.iters = iters
        self.random_state = random_state

        self.param_order_ = list(param_bounds.keys())
        self.dimensions = len(self.param_order_)

        self.lb = np.array([param_bounds[k][0] for k in self.param_order_])
        self.ub = np.array([param_bounds[k][1] for k in self.param_order_])

    def _vector_to_params(self, x):
        params = {}
        for i, key in enumerate(self.param_order_):
            if key in ["max_depth", "min_samples_leaf"]:
                params[key] = int(np.round(x[i]))
            else:
                params[key] = float(x[i])
        return params

    def fit(self, X, y):
        skf = StratifiedKFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=self.random_state
        )

        def objective_function(X_particles):
            scores = np.zeros(X_particles.shape[0])

            for i, particle in enumerate(X_particles):
                params = self._vector_to_params(particle)

                # 🔹 Normalizar α y β
                alpha = params["alpha"]
                beta = params["beta"]
                s = alpha + beta
                if s > 1:
                    alpha /= s
                    beta /= s

                params["alpha"] = alpha
                params["beta"] = beta
                params["criterion"] = "hybrid"
                #params["random_state"] = self.random_state

                fold_scores = []

                for train_idx, test_idx in skf.split(X, y):
                    model = PBCRETree(**params)
                    model.fit(X.iloc[train_idx], y.iloc[train_idx])
                    y_pred = model.predict(X.iloc[test_idx])
                    fold_scores.append(
                        self.metric(y.iloc[test_idx], y_pred)
                    )

                scores[i] = -np.mean(fold_scores)  # minimización

            return scores

        optimizer = ps.single.GlobalBestPSO(
            n_particles=self.n_particles,
            dimensions=self.dimensions,
            options={"c1": 2.05, "c2": 2.05, "w": 0.4},
            bounds=(self.lb, self.ub)
        )

        best_cost, best_pos = optimizer.optimize(
            objective_function,
            iters=self.iters
        )

        self.best_params_ = self._vector_to_params(best_pos)
        self.best_params_["criterion"] = "hybrid"

        # normalizar una vez más por seguridad
        a, b = self.best_params_["alpha"], self.best_params_["beta"]
        if a + b > 1:
            self.best_params_["alpha"] = a / (a + b)
            self.best_params_["beta"] = b / (a + b)

        self.best_score_ = -best_cost

        return self
