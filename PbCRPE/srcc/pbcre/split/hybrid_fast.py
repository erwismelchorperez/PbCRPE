import numpy as np
import pandas as pd

from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from .hybrid_twoing_functional import HybridTwoingFunctionalCriterion
from srcc.pbcre.evaluation.metrics import ClassificationMetrics
from sklearn.metrics import fbeta_score
from sklearn.metrics import recall_score



class HybridTwoingFunctionalCriterionFast(HybridTwoingFunctionalCriterion):

    def __init__(self,alpha=0.5,beta=0.05,logistic_C=0.005,logistic_solver="liblinear",logistic_max_iter=200,n_quantiles=7,twoing_threshold=0.01,max_logistic_samples=500,max_categories_grouping=8):
        super().__init__(alpha=alpha,beta=beta, logistic_C=logistic_C,logistic_solver=logistic_solver, logistic_max_iter=logistic_max_iter)

        self.n_quantiles = n_quantiles
        self.twoing_threshold = twoing_threshold
        self.max_logistic_samples = max_logistic_samples
        self.max_categories_grouping = max_categories_grouping
    
    def _logistic_score(self, X, y, min_samples_leaf):
        y = np.asarray(y)

        if len(y) < 2 * min_samples_leaf:
            return 0.0, None

        if len(np.unique(y)) < 2:
            return 0.0, None

        X_enc = np.asarray(self._encode_X(X))

        if len(y) > self.max_logistic_samples:
            idx = np.random.choice(
                len(y),
                size=self.max_logistic_samples,
                replace=False
            )
            X_enc = X_enc[idx]
            y = y[idx]

        _, counts = np.unique(y, return_counts=True)

        model = LogisticRegression(solver=self.logistic_solver,C=self.logistic_C,max_iter=self.logistic_max_iter,class_weight="balanced")

        if counts.min() >= 2:
            X_tr, X_val, y_tr, y_val = train_test_split(X_enc,y,test_size=0.2,stratify=y,random_state=42)

            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)

            score = f1_score(y_val,y_pred,average="weighted")
            #score = ClassificationMetrics.balanced_error_rate(y_val, y_pred)
            #score = fbeta_score(y_val, y_pred, beta=2, pos_label=1, zero_division=0)
            #score = ClassificationMetrics.auc(y_val, y_pred)
            recall_0 = recall_score(y_val, y_pred, pos_label=0, zero_division=0)
            recall_1 = recall_score(y_val, y_pred, pos_label=1, zero_division=0)

            #score = recall_1 if recall_0 >= 0.7 else 0.0
        else:
            model.fit(X_enc, y)

            y_pred = model.predict(X_enc)

            score = f1_score(y,y_pred,average="weighted")
            #score = ClassificationMetrics.balanced_error_rate(y, y_pred)
            #score = fbeta_score(y, y_pred, beta=2, pos_label=1, zero_division=0)
            #score = ClassificationMetrics.auc(y, y_pred)

        return score, model
    
  
    def _categorical_candidates(self, col, y=None):
        series = pd.Series(col)
        valid = series.notna()
        categories = list(pd.unique(series[valid]))

        n_categories = len(categories)

        if n_categories < 2:
            return categories

        # Conservar las divisiones individuales.
        candidates = list(categories)

        if y is None or n_categories > self.max_categories_grouping:
            return candidates

        y_array = np.asarray(y)

        if len(y_array) != len(series):
            raise ValueError(
                "Las columnas y las etiquetas deben tener la misma longitud."
            )

        # Calcular la proporción de la clase positiva para cada categoría.
        category_rates = []

        for category in categories:
            mask = (series == category).to_numpy() & valid.to_numpy()
            y_category = y_array[mask]

            if len(y_category) == 0:
                continue

            rate = np.mean(y_category == 1)

            category_rates.append((category, rate))

        # Ordenar categorías por proporción de la clase positiva.
        category_rates.sort(key=lambda item: item[1])

        ordered_categories = [
            category for category, _ in category_rates
        ]

        # Generar divisiones consecutivas entre categorías ordenadas.
        for split_point in range(1, len(ordered_categories)):
            left = tuple(ordered_categories[:split_point])

            # Las divisiones individuales ya están incluidas.
            if len(left) == 1:
                continue

            # Evitar duplicar divisiones representadas por su complemento.
            if len(ordered_categories) - len(left) < 2:
                continue

            candidates.append(left)

        return candidates
    
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

            col = X.iloc[:, feat].to_numpy()
            is_numeric = pd.api.types.is_numeric_dtype(X.iloc[:, feat])

            if is_numeric:
                # Mantener la generación actual de umbrales numéricos.
                quantiles = np.linspace(
                    0.1,
                    0.9,
                    self.n_quantiles
                )

                thresholds = np.unique(
                    np.quantile(
                        X.iloc[:, feat].dropna(),
                        quantiles
                    )
                )
            else:
                # Nueva generación de candidatos categóricos.
                #thresholds = self._categorical_candidates(col)
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

                if left_size < min_samples_leaf:
                    continue

                if right_size < min_samples_leaf:
                    continue

                X_l = X.iloc[np.flatnonzero(mask)]
                y_l = y_array[mask]

                X_r = X.iloc[np.flatnonzero(~mask)]
                y_r = y_array[~mask]

                twoing = self._twoing(y_l, y_r)

                if twoing < self.twoing_threshold:
                    continue

                score_l, model_l = self._logistic_score(
                    X_l,
                    y_l,
                    min_samples_leaf
                )

                score_r, model_r = self._logistic_score(
                    X_r,
                    y_r,
                    min_samples_leaf
                )

                # Conservar la puntuación funcional actual.
                purity_l = (
                    pd.Series(y_l).value_counts().max() / len(y_l)
                )

                purity_r = (
                    pd.Series(y_r).value_counts().max() / len(y_r)
                )

                func_score = (
                    len(y_l) * score_l * purity_l
                    + len(y_r) * score_r * purity_r
                ) / (len(y_l) + len(y_r))

                scores_twoing.append(twoing)
                scores_func.append(func_score)

                candidates.append(
                    (
                        feat,
                        thr,
                        twoing,
                        func_score,
                        model_l,
                        model_r
                    )
                )

        if not candidates:
            return None, None, -np.inf, None

        twoing_min = min(scores_twoing)
        twoing_max = max(scores_twoing)

        func_min = min(scores_func)
        func_max = max(scores_func)

        for feat, thr, twoing, func, model_l, model_r in candidates:

            twoing_n = (
                (twoing - twoing_min) / (twoing_max - twoing_min)
                if twoing_max > twoing_min
                else 0.0
            )

            func_n = (
                (func - func_min) / (func_max - func_min)
                if func_max > func_min
                else 0.0
            )

            # Mantener la fórmula actual, sin introducir otro cambio.
            score = (
                self.beta * twoing_n
                + (1 - self.beta) * (self.alpha * func_n)
            )

            if score > best_score:
                best_score = score
                best_feat = feat
                best_thr = thr
                best_models = (model_l, model_r)

        return best_feat, best_thr, best_score, best_models
