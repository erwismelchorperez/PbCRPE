import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .hybrid_twoing_functional import HybridTwoingFunctionalCriterion


class HybridTwoingFunctionalCriterionFast(HybridTwoingFunctionalCriterion):
    """
    Versión rápida del criterio híbrido.

    Cambios en PbCRPE_v1:
    - Score funcional con self._score (por defecto 1 - BER).
    - purity_mode="balanced": pureza calculada sobre la distribución
      reponderada por el prior de la raíz (rango [0.5, 1]; vale 0.5 cuando
      el nodo reproduce el prior). Evita premiar splits que sólo aíslan
      la clase mayoritaria. purity_mode="majority" = comportamiento original.
    - class_quantiles=True: añade umbrales en cuantiles de la clase
      MINORITARIA (según el prior de la raíz), cuyas instancias suelen
      estar en las colas que los cuantiles 0.1-0.9 globales no exploran.
      En German (70 % positivos) la minoritaria es la clase 0.
    - Submuestreo con generador con semilla (antes np.random sin semilla).
    """

    def __init__(
        self,
        alpha=0.5,
        beta=0.05,
        logistic_C=0.005,
        logistic_solver="liblinear",
        logistic_max_iter=200,
        n_quantiles=7,
        twoing_threshold=0.01,
        max_logistic_samples=500,
        max_categories_grouping=8,
        scoring="balanced",
        purity_mode="balanced",
        class_quantiles=True,
        random_state=42,
        verbose=False
    ):
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
        if purity_mode not in ("balanced", "majority"):
            raise ValueError("purity_mode debe ser 'balanced' o 'majority'.")

        self.n_quantiles = n_quantiles
        self.twoing_threshold = twoing_threshold
        self.max_logistic_samples = max_logistic_samples
        self.max_categories_grouping = max_categories_grouping
        self.purity_mode = purity_mode
        self.class_quantiles = class_quantiles

    # -------------------------------------------------
    def _purity(self, y_node):
        y_node = np.asarray(y_node)
        p_pos = float(np.mean(y_node == 1))

        if self.purity_mode == "majority":
            return max(p_pos, 1.0 - p_pos)

        prior = self.class_prior_
        if prior is None or not 0.0 < prior < 1.0:
            prior = 0.5
        w_pos = p_pos / prior
        w_neg = (1.0 - p_pos) / (1.0 - prior)
        return max(w_pos, w_neg) / (w_pos + w_neg)

    # -------------------------------------------------
    def _logistic_score(self, X, y, min_samples_leaf):
        y = np.asarray(y)

        if len(y) < 2 * min_samples_leaf:
            return 0.0, None
        if len(np.unique(y)) < 2:
            return 0.0, None

        X_enc = np.asarray(self._encode_X(X))

        if len(y) > self.max_logistic_samples:
            idx = self._rng.choice(len(y), size=self.max_logistic_samples,
                                   replace=False)
            X_enc = X_enc[idx]
            y = y[idx]
            if len(np.unique(y)) < 2:
                return 0.0, None

        _, counts = np.unique(y, return_counts=True)

        model = LogisticRegression(
            solver=self.logistic_solver,
            C=self.logistic_C,
            max_iter=self.logistic_max_iter,
            class_weight="balanced"
        )

        if counts.min() >= 2:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_enc, y, test_size=0.3, stratify=y,
                random_state=self.random_state)
            model.fit(X_tr, y_tr)
            score = self._split_score(model, X_val, y_val)
        else:
            model.fit(X_enc, y)
            score = self._split_score(model, X_enc, y)

        return score, model

    def _split_score(self, model, X_eval, y_eval):
        """
        scoring="auc": AUC de las probabilidades (usa el orden completo de las
        probabilidades; mucho menos ruidoso que contar aciertos 0/1 en ~150
        instancias). Si la parte de evaluación tiene una sola clase, se usa
        balanced accuracy.
        """
        if self.scoring == "auc" and len(np.unique(y_eval)) == 2:
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(y_eval, model.predict_proba(X_eval)[:, 1])
        if self.scoring == "auc":
            from sklearn.metrics import balanced_accuracy_score
            return balanced_accuracy_score(y_eval, model.predict(X_eval))
        return self._score(y_eval, model.predict(X_eval))

    # -------------------------------------------------
    def _numeric_thresholds(self, series, y_array):
        values = series.dropna()
        quantiles = np.linspace(0.1, 0.9, self.n_quantiles)
        thr = [np.quantile(values, quantiles)]

        if self.class_quantiles:
            mask_valid = series.notna().to_numpy()
            prior = self.class_prior_ if self.class_prior_ is not None else 0.5
            minority = 1 if prior <= 0.5 else 0
            min_values = series.to_numpy()[mask_valid & (y_array == minority)]
            if len(min_values) >= 2:
                # cuantiles de la clase minoritaria, incluyendo las colas
                q_min = np.linspace(0.05, 0.95, self.n_quantiles)
                thr.append(np.quantile(min_values.astype(float), q_min))

        return np.unique(np.concatenate(thr))

    # -------------------------------------------------
    def _categorical_candidates(self, col, y=None):
        series = pd.Series(col)
        valid = series.notna()
        categories = list(pd.unique(series[valid]))
        n_categories = len(categories)

        if n_categories < 2:
            return categories

        candidates = list(categories)

        if y is None or n_categories > self.max_categories_grouping:
            return candidates

        y_array = np.asarray(y)
        if len(y_array) != len(series):
            raise ValueError(
                "Las columnas y las etiquetas deben tener la misma longitud."
            )

        category_rates = []
        for category in categories:
            mask = (series == category).to_numpy() & valid.to_numpy()
            y_category = y_array[mask]
            if len(y_category) == 0:
                continue
            category_rates.append((category, np.mean(y_category == 1)))

        category_rates.sort(key=lambda item: item[1])
        ordered_categories = [c for c, _ in category_rates]

        for split_point in range(1, len(ordered_categories)):
            left = tuple(ordered_categories[:split_point])
            if len(left) == 1:
                continue
            if len(ordered_categories) - len(left) < 2:
                continue
            candidates.append(left)

        return candidates

    # -------------------------------------------------
    def best_split(self, X, y, min_samples_leaf):

        best_feat = None
        best_thr = None
        best_score = -np.inf
        best_models = None

        candidates = []
        scores_twoing = []
        scores_func = []

        y_array = np.asarray(y)

        for feat in range(X.shape[1]):

            series = X.iloc[:, feat]
            col = series.to_numpy()
            is_numeric = pd.api.types.is_numeric_dtype(series)

            if is_numeric:
                thresholds = self._numeric_thresholds(series, y_array)
            else:
                thresholds = self._categorical_candidates(col, y_array)

            for thr in thresholds:

                if is_numeric:
                    mask = col <= thr
                elif isinstance(thr, tuple):
                    mask = pd.Series(col).isin(thr).to_numpy()
                else:
                    mask = col == thr

                left_size = int(mask.sum())
                right_size = int((~mask).sum())

                if left_size < min_samples_leaf or right_size < min_samples_leaf:
                    continue

                X_l = X.iloc[np.flatnonzero(mask)]
                y_l = y_array[mask]
                X_r = X.iloc[np.flatnonzero(~mask)]
                y_r = y_array[~mask]

                twoing = self._twoing(y_l, y_r)
                if twoing < self.twoing_threshold:
                    continue

                score_l, model_l = self._logistic_score(X_l, y_l, min_samples_leaf)
                score_r, model_r = self._logistic_score(X_r, y_r, min_samples_leaf)

                purity_l = self._purity(y_l)
                purity_r = self._purity(y_r)

                func_score = (
                    len(y_l) * score_l * purity_l
                    + len(y_r) * score_r * purity_r
                ) / (len(y_l) + len(y_r))

                scores_twoing.append(twoing)
                scores_func.append(func_score)
                candidates.append((feat, thr, twoing, func_score, model_l, model_r))

        if not candidates:
            return None, None, -np.inf, None

        twoing_min, twoing_max = min(scores_twoing), max(scores_twoing)
        func_min, func_max = min(scores_func), max(scores_func)

        for feat, thr, twoing, func, model_l, model_r in candidates:
            twoing_n = ((twoing - twoing_min) / (twoing_max - twoing_min)
                        if twoing_max > twoing_min else 0.0)
            func_n = ((func - func_min) / (func_max - func_min)
                      if func_max > func_min else 0.0)

            score = self._combine(twoing_n, func_n)
            if score > best_score:
                best_score = score
                best_feat = feat
                best_thr = thr
                best_models = (model_l, model_r)

        return best_feat, best_thr, best_score, best_models
