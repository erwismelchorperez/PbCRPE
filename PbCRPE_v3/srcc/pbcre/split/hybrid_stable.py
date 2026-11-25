import hashlib

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .hybrid_twoing_functional import HybridTwoingFunctionalCriterion


class HybridTwoingFunctionalCriterionStable(HybridTwoingFunctionalCriterion):
    """
    Criterio híbrido con validación cruzada en el modelo local.

    Correcciones en PbCRPE_v1:
    - La caché usaba (n_muestras, clases) como clave: dos subconjuntos
      distintos del mismo tamaño recibían el mismo score. Ahora la clave es
      un hash del contenido exacto (X_enc, y).
    - La condición `len(np.unique(y)) < cv_folds` era siempre verdadera en
      binario (2 < 5), así que la CV nunca se ejecutaba. Ahora se usa el
      tamaño de la clase minoritaria.
    - Score con self._score (por defecto 1 - BER) y class_weight="balanced".
    - La caché se vacía en cada fit (reset_rng).
    """

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
        scoring="balanced",
        random_state=42,
        verbose=False
    ):
        self._logistic_cache = {}
        super().__init__(
            alpha=alpha,
            beta=beta,
            logistic_C=logistic_C,
            logistic_solver=logistic_solver,
            logistic_max_iter=logistic_max_iter,
            scoring=scoring,
            random_state=random_state,
            verbose=verbose
        )

        self.n_quantiles = n_quantiles
        self.twoing_threshold = twoing_threshold
        self.max_logistic_samples = max_logistic_samples
        self.cv_folds = cv_folds

    def reset_rng(self):
        super().reset_rng()
        self._logistic_cache = {}

    # --------------------------------------------------
    @staticmethod
    def _cache_key(X_enc, y):
        h = hashlib.sha1()
        h.update(np.ascontiguousarray(X_enc, dtype=float).tobytes())
        h.update(np.ascontiguousarray(y).tobytes())
        h.update(str(X_enc.shape).encode())
        return h.hexdigest()

    # --------------------------------------------------
    def _logistic_score(self, X, y, min_samples_leaf):

        y = np.asarray(y)

        if len(y) < 2 * min_samples_leaf:
            return 0.0, None
        if len(np.unique(y)) < 2:
            return 0.0, None

        X_enc = np.asarray(self._encode_X(X), dtype=float)

        if len(y) > self.max_logistic_samples:
            # semilla fija por llamada: el mismo nodo produce la misma muestra
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(y), self.max_logistic_samples, replace=False)
            X_enc = X_enc[idx]
            y = y[idx]
            if len(np.unique(y)) < 2:
                return 0.0, None

        key = self._cache_key(X_enc, y)
        if key in self._logistic_cache:
            return self._logistic_cache[key]

        def new_model():
            return LogisticRegression(
                solver=self.logistic_solver,
                C=self.logistic_C,
                max_iter=self.logistic_max_iter,
                class_weight="balanced"
            )

        _, counts = np.unique(y, return_counts=True)

        if counts.min() < self.cv_folds:
            model = new_model().fit(X_enc, y)
            score = self._score(y, model.predict(X_enc))
        else:
            skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                                  random_state=self.random_state)
            scores = []
            for train_idx, val_idx in skf.split(X_enc, y):
                m = new_model().fit(X_enc[train_idx], y[train_idx])
                scores.append(self._score(y[val_idx], m.predict(X_enc[val_idx])))

            cv_score = float(np.mean(scores))
            model = new_model().fit(X_enc, y)
            train_score = self._score(y, model.predict(X_enc))
            score = 0.75 * cv_score + 0.25 * train_score

        self._logistic_cache[key] = (score, model)
        return score, model

    # --------------------------------------------------
    def best_split(self, X, y, min_samples_leaf):

        best_feat = None
        best_thr = None
        best_score = -np.inf
        best_models = None

        candidates = []
        scores_twoing = []
        scores_func = []
        y = np.asarray(y)

        for feat in range(X.shape[1]):

            col = X.iloc[:, feat].values
            is_numeric = pd.api.types.is_numeric_dtype(X.iloc[:, feat])

            if is_numeric:
                quantiles = np.linspace(0.1, 0.9, self.n_quantiles)
                thresholds = np.unique(np.quantile(col.astype(float), quantiles))
            else:
                thresholds = np.unique(col)

            for thr in thresholds:

                mask = col <= thr if is_numeric else col == thr

                if mask.sum() < min_samples_leaf:
                    continue
                if (~mask).sum() < min_samples_leaf:
                    continue

                X_l, y_l = X[mask], y[mask]
                X_r, y_r = X[~mask], y[~mask]

                twoing = self._twoing(y_l, y_r)
                if twoing <= -0.0001:
                    continue

                score_l, model_l = self._logistic_score(X_l, y_l, min_samples_leaf)
                score_r, model_r = self._logistic_score(X_r, y_r, min_samples_leaf)

                func_score = (len(y_l) * score_l + len(y_r) * score_r) / (len(y_l) + len(y_r))

                scores_twoing.append(twoing)
                scores_func.append(func_score)
                candidates.append((feat, thr, twoing, func_score, model_l, model_r))

        if len(candidates) == 0:
            return None, None, -np.inf, None

        twoing_arr = np.array(scores_twoing)
        func_arr = np.array(scores_func)
        twoing_std = np.std(twoing_arr) + 1e-8
        func_std = np.std(func_arr) + 1e-8

        for feat, thr, twoing, func, m_l, m_r in candidates:
            twoing_n = (twoing - twoing_arr.mean()) / twoing_std
            func_n = (func - func_arr.mean()) / func_std

            score = self._combine(twoing_n, func_n)
            if score > best_score:
                best_score = score
                best_feat = feat
                best_thr = thr
                best_models = (m_l, m_r)

        return best_feat, best_thr, best_score, best_models
