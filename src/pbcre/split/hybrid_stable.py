import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from collections import defaultdict

from .hybrid_twoing_functional import HybridTwoingFunctionalCriterion


class HybridTwoingFunctionalCriterionStable(HybridTwoingFunctionalCriterion):

    def __init__(
        self,
        alpha=0.5,
        beta=0.05,
        logistic_C=0.005,
        logistic_solver="liblinear",
        logistic_max_iter=200,
        n_quantiles=9,
        twoing_threshold=0.01,
        max_logistic_samples=500,
        cv_folds=5,
        random_state=42
    ):
        super().__init__(
            alpha=alpha,
            beta=beta,
            logistic_C=logistic_C,
            logistic_solver=logistic_solver,
            logistic_max_iter=logistic_max_iter
        )

        self.n_quantiles = n_quantiles
        self.twoing_threshold = twoing_threshold
        self.max_logistic_samples = max_logistic_samples
        self.cv_folds = cv_folds
        self.random_state = random_state

        # 🔥 CACHE GLOBAL POR NODO
        self._logistic_cache = {}

    # --------------------------------------------------
    # 🔥 LOGISTIC MÁS ESTABLE (CV EN LUGAR DE SPLIT)
    # --------------------------------------------------
    def _logistic_score(self, X, y, min_samples_leaf):

        y = np.asarray(y)

        if len(y) < 2 * min_samples_leaf:
            return 0.0, None

        if len(np.unique(y)) < 2:
            return 0.0, None

        X_enc = np.asarray(self._encode_X(X))

        # ---------------------------
        # 🔥 muestreo estable
        # ---------------------------
        if len(y) > self.max_logistic_samples:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(y), self.max_logistic_samples, replace=False)
            X_enc = X_enc[idx]
            y = y[idx]

        key = (X_enc.shape[0], tuple(np.unique(y)))

        # ---------------------------
        # 🔥 CACHE
        # ---------------------------
        if key in self._logistic_cache:
            return self._logistic_cache[key]

        model = LogisticRegression(
            solver=self.logistic_solver,
            C=self.logistic_C,
            max_iter=self.logistic_max_iter
        )

        # ---------------------------
        # 🔥 CV estable (en lugar de split)
        # ---------------------------
        if len(np.unique(y)) < self.cv_folds:
            model.fit(X_enc, y)
            pred = model.predict(X_enc)
            score = f1_score(y, pred, average="macro")
        else:
            skf = StratifiedKFold(
                n_splits=self.cv_folds,
                shuffle=True,
                random_state=self.random_state
            )

            scores = []

            for train_idx, val_idx in skf.split(X_enc, y):
                X_tr, X_val = X_enc[train_idx], X_enc[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]

                model.fit(X_tr, y_tr)
                pred = model.predict(X_val)
                scores.append(f1_score(y_val, pred, average="macro"))

            #score = float(np.mean(scores))
            cv_score = float(np.mean(scores))
            train_score = f1_score(y, model.fit(X_enc, y).predict(X_enc), average="macro")
            score = 0.75 * cv_score + 0.25 * train_score

        self._logistic_cache[key] = (score, model)
        return score, model

    # --------------------------------------------------
    # 🔥 SPLIT (misma lógica pero más estable)
    # --------------------------------------------------
    def best_split(self, X, y, min_samples_leaf):

        best_feat = None
        best_thr = None
        best_score = -np.inf
        best_models = None

        candidates = []
        scores_twoing = []
        scores_func = []

        for feat in range(X.shape[1]):

            col = X.iloc[:, feat].values

            if np.issubdtype(col.dtype, np.number):
                quantiles = np.linspace(0.1, 0.9, self.n_quantiles)
                thresholds = np.unique(np.quantile(col, quantiles))
            else:
                thresholds = np.unique(col)

            for thr in thresholds:

                mask = (
                    col <= thr
                    if np.issubdtype(col.dtype, np.number)
                    else col == thr
                )

                if mask.sum() < min_samples_leaf:
                    continue
                if (~mask).sum() < min_samples_leaf:
                    continue

                X_l, y_l = X[mask], y[mask]
                X_r, y_r = X[~mask], y[~mask]

                twoing = self._twoing(y_l, y_r)

                # 🔥 sin hard threshold (mejora importante)
                #if twoing <= 0:
                if twoing <= -0.0001:
                    continue

                score_l, model_l = self._logistic_score(X_l, y_l, min_samples_leaf)
                score_r, model_r = self._logistic_score(X_r, y_r, min_samples_leaf)

                func_score = (
                    len(y_l) * score_l + len(y_r) * score_r
                ) / (len(y_l) + len(y_r))

                scores_twoing.append(twoing)
                scores_func.append(func_score)

                candidates.append((feat, thr, twoing, func_score, model_l, model_r))

        if len(candidates) == 0:
            return None, None, -np.inf, None

        # ---------------------------
        # 🔥 normalización estable
        # ---------------------------
        twoing_arr = np.array(scores_twoing)
        func_arr = np.array(scores_func)
        
        """
        twoing_std = twoing_arr.std() if twoing_arr.std() > 0 else 1
        func_std = func_arr.std() if func_arr.std() > 0 else 1
        """
        twoing_std = np.std(twoing_arr) + 1e-8
        func_std = np.std(func_arr) + 1e-8

        for feat, thr, twoing, func, m_l, m_r in candidates:

            twoing_n = (twoing - twoing_arr.mean()) / twoing_std
            func_n = (func - func_arr.mean()) / func_std

            score = self.beta * twoing_n + (1 - self.beta) * (self.alpha * func_n)
            #score += np.random.normal(0, 0.003)

            if score > best_score:
                best_score = score
                best_feat = feat
                best_thr = thr
                best_models = (m_l, m_r)

        return best_feat, best_thr, best_score, best_models