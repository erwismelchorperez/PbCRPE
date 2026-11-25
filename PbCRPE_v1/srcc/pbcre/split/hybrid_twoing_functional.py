import numpy as np
import pandas as pd
from .base import SplitCriterion
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, balanced_accuracy_score


class HybridTwoingFunctionalCriterion(SplitCriterion):
    """
    Criterio híbrido Twoing + funcional (regresión logística local).

    Cambios en PbCRPE_v1:
    - scoring: métrica del modelo funcional local.
        "balanced"    -> 1 - BER (balanced accuracy). Recomendado.
        "f1"          -> F1 de la clase positiva (comportamiento original).
        "f1_weighted" -> F1 ponderado por soporte (favorece la mayoritaria).
    - alpha se aplica UNA sola vez (antes se aplicaba dos veces: alpha^2).
    - Generador aleatorio con semilla (random_state) para reproducibilidad.
    - class_prior_: proporción de positivos en la raíz; la fija PBCRETree.
    """

    def __init__(self, alpha=0.5, beta=0.05, logistic_C=0.005,
                 logistic_solver="liblinear", logistic_max_iter=200,
                 scoring="balanced", random_state=42, verbose=False):
        """
        beta = 1.0  -> solo Twoing
        beta = 0.0  -> solo criterio funcional
        """
        if verbose:
            print("Split Hybrid")
        if scoring not in ("balanced", "f1", "f1_weighted", "auc"):
            raise ValueError("scoring debe ser 'balanced', 'f1', 'f1_weighted' o 'auc'.")
        # estandarizar las numéricas antes de la logística del split
        # (lo fija PBCRETree con split_standardize; por defecto no)
        self.standardize = False

        self.beta = beta
        self.alpha = alpha
        self.logistic_C = logistic_C
        self.logistic_solver = logistic_solver
        self.logistic_max_iter = logistic_max_iter
        self.scoring = scoring
        self.random_state = random_state
        self.class_prior_ = None
        self.reset_rng()

    # -------------------------------------------------
    def reset_rng(self):
        """Reinicia el generador; PBCRETree.fit lo llama al inicio."""
        self._rng = np.random.default_rng(self.random_state)

    def set_class_prior(self, prior_pos):
        self.class_prior_ = float(prior_pos)

    # -------------------------------------------------
    def _score(self, y_true, y_pred):
        """Métrica del modelo funcional local. Mayor es mejor."""
        if self.scoring == "balanced":
            # adjusted=False -> 1 - BER
            return balanced_accuracy_score(y_true, y_pred)
        if self.scoring == "f1":
            return f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        return f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # -------------------------------------------------
    def _encode_X(self, X):
        X = pd.DataFrame(X)

        num_cols = X.select_dtypes(include=[np.number]).columns
        cat_cols = X.select_dtypes(exclude=[np.number]).columns

        X_num = X[num_cols].values if len(num_cols) > 0 else None

        if len(cat_cols) > 0:
            enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            X_cat = enc.fit_transform(X[cat_cols])
        else:
            X_cat = None

        if X_num is not None and getattr(self, "standardize", False):
            X_num = np.asarray(X_num, dtype=float)
            mu = np.nanmean(X_num, axis=0)
            sd = np.nanstd(X_num, axis=0)
            sd[~np.isfinite(sd) | (sd == 0)] = 1.0
            X_num = (X_num - np.nan_to_num(mu)) / sd

        if X_num is not None and X_cat is not None:
            return np.hstack([X_num, X_cat])
        elif X_num is not None:
            return X_num
        else:
            return X_cat

    # -------------------------------------------------
    def _logistic_score(self, X, y, min_samples_leaf):
        y = np.asarray(y)
        if len(y) < 2 * min_samples_leaf or len(np.unique(y)) < 2:
            return 0.0, None

        X_enc = self._encode_X(X)
        _, counts = np.unique(y, return_counts=True)

        model = LogisticRegression(
            solver=self.logistic_solver,
            class_weight="balanced",
            C=self.logistic_C,
            max_iter=self.logistic_max_iter)

        if counts.min() >= 2:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_enc, y, test_size=0.3, stratify=y,
                random_state=self.random_state)
            model.fit(X_tr, y_tr)
            score = self._score(y_val, model.predict(X_val))
        else:
            model.fit(X_enc, y)
            score = self._score(y, model.predict(X_enc))

        return score, model

    # -------------------------------------------------
    def _twoing(self, y_l, y_r):
        """Twoing criterion (CART)."""
        n = len(y_l) + len(y_r)
        if n == 0 or len(y_l) == 0 or len(y_r) == 0:
            return 0.0

        classes = np.unique(np.concatenate([y_l, y_r]))
        p_l = np.array([np.mean(y_l == c) for c in classes])
        p_r = np.array([np.mean(y_r == c) for c in classes])

        return (len(y_l) / n) * (len(y_r) / n) * np.sum(np.abs(p_l - p_r)) ** 2

    # -------------------------------------------------
    def _combine(self, twoing_n, func_n):
        """Combinación única para todos los criterios híbridos."""
        return self.beta * twoing_n + (1 - self.beta) * self.alpha * func_n

    # -------------------------------------------------
    def best_split(self, X, y, min_samples_leaf):

        best_feat, best_thr = None, None
        best_score = -np.inf
        best_models = None

        scores_twoing = []
        scores_func = []
        candidates = []
        y = np.asarray(y)

        for feat in range(X.shape[1]):
            col = X.iloc[:, feat].values
            is_numeric = pd.api.types.is_numeric_dtype(X.iloc[:, feat])
            thresholds = np.unique(col)

            for thr in thresholds:
                mask = col <= thr if is_numeric else col == thr

                if mask.sum() < min_samples_leaf or (~mask).sum() < min_samples_leaf:
                    continue

                X_l, y_l = X[mask], y[mask]
                X_r, y_r = X[~mask], y[~mask]

                twoing = self._twoing(y_l, y_r)

                score_l, model_l = self._logistic_score(X_l, y_l, min_samples_leaf)
                score_r, model_r = self._logistic_score(X_r, y_r, min_samples_leaf)
                # promedio ponderado (antes era suma, que favorecía nodos
                # donde ambos lados entrenan modelo sin importar su tamaño)
                func_score = (len(y_l) * score_l + len(y_r) * score_r) / len(y)

                scores_twoing.append(twoing)
                scores_func.append(func_score)
                candidates.append((feat, thr, twoing, func_score, model_l, model_r))

        if len(candidates) == 0:
            return None, None, -np.inf, None

        twoing_min, twoing_max = min(scores_twoing), max(scores_twoing)
        func_min, func_max = min(scores_func), max(scores_func)

        for feat, thr, twoing, func, m_l, m_r in candidates:
            twoing_n = ((twoing - twoing_min) / (twoing_max - twoing_min)
                        if twoing_max > twoing_min else 0.0)
            func_n = ((func - func_min) / (func_max - func_min)
                      if func_max > func_min else 0.0)

            score = self._combine(twoing_n, func_n)
            if score > best_score:
                best_score = score
                best_feat = feat
                best_thr = thr
                best_models = (m_l, m_r)

        return best_feat, best_thr, best_score, best_models
