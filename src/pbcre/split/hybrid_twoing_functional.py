import numpy as np
import pandas as pd
from .base import SplitCriterion
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score


class HybridTwoingFunctionalCriterion(SplitCriterion):

    def __init__(self, alpha= 0.5, beta=0.05, logistic_C=0.005, logistic_solver="liblinear",logistic_max_iter=200):
        """
        beta = 1.0  -> solo Twoing
        beta = 0.0  -> solo criterio funcional
        """
        print("Split Hybrid")
        self.beta = beta
        self.alpha = alpha
        self.logistic_C = logistic_C
        self.logistic_solver = logistic_solver
        self.logistic_max_iter = logistic_max_iter


    # -------------------------------------------------
    def _encode_X(self, X):
        X = pd.DataFrame(X)

        num_cols = X.select_dtypes(include=[np.number]).columns
        cat_cols = X.select_dtypes(exclude=[np.number]).columns

        X_num = X[num_cols].values if len(num_cols) > 0 else None

        if len(cat_cols) > 0:
            enc = OneHotEncoder(sparse_output=False,handle_unknown="ignore")
            X_cat = enc.fit_transform(X[cat_cols])
        else:
            X_cat = None

        if X_num is not None and X_cat is not None:
            return np.hstack([X_num, X_cat])
        elif X_num is not None:
            return X_num
        else:
            return X_cat

    # -------------------------------------------------
    def _logistic_score(self, X, y, min_samples_leaf):
    
        if len(y) < 2 * min_samples_leaf or len(np.unique(y)) < 2:
            return 0.0, None

        X_enc = self._encode_X(X)
        _, counts = np.unique(y, return_counts=True)

        model = LogisticRegression(
            solver=self.logistic_solver,#liblinear,lbfgs
            #class_weight="balanced",
            C=self.logistic_C,#0.002, 0.005
            max_iter=self.logistic_max_iter)

        # Validación segura
        if counts.min() >= 2:
            X_tr, X_val, y_tr, y_val = train_test_split(X_enc,y,test_size=0.3,stratify=y,random_state=42)

            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            score = f1_score(y_val, y_pred, pos_label=1)
        else:
            model.fit(X_enc, y)
            y_pred = model.predict(X_enc)
            score = f1_score(y, y_pred, pos_label=1)

        return score, model

    # -------------------------------------------------
    def _twoing(self, y_l, y_r):
        """
        Twoing criterion (CART)
        """
        n = len(y_l) + len(y_r)
        if n == 0:
            return 0.0

        classes = np.unique(np.concatenate([y_l, y_r]))

        p_l = np.array([np.mean(y_l == c) for c in classes])
        p_r = np.array([np.mean(y_r == c) for c in classes])

        return (len(y_l) / n) * (len(y_r) / n) * np.sum(np.abs(p_l - p_r)) ** 2

    # -------------------------------------------------
    def best_split(self, X, y, min_samples_leaf):

        best_feat, best_thr = None, None
        best_score = -np.inf
        best_models = None

        scores_twoing = []
        scores_func = []
        candidates = []

        for feat in range(X.shape[1]):
            col = X.iloc[:, feat].values
            thresholds = np.unique(col)

            for thr in thresholds:

                mask = (
                    col <= thr
                    if np.issubdtype(col.dtype, np.number)
                    else col == thr
                )

                if mask.sum() < min_samples_leaf or (~mask).sum() < min_samples_leaf:
                    continue

                X_l, y_l = X[mask], y[mask]
                X_r, y_r = X[~mask], y[~mask]

                # --- Twoing
                twoing = self._twoing(y_l, y_r)

                # --- Funcional
                score_l, model_l = self._logistic_score(
                    X_l, y_l, min_samples_leaf
                )
                score_r, model_r = self._logistic_score(
                    X_r, y_r, min_samples_leaf
                )
                func_score = score_l + score_r

                scores_twoing.append(twoing)
                scores_func.append(func_score)

                candidates.append((feat, thr, twoing, func_score, model_l, model_r))

        # 🚫 No hay splits válidos
        if len(candidates) == 0:
            return None, None, -np.inf, None

        # -------------------------------------------------
        # Normalización local (por nodo)
        twoing_min, twoing_max = min(scores_twoing), max(scores_twoing)
        func_min, func_max = min(scores_func), max(scores_func)

        for feat, thr, twoing, func, m_l, m_r in candidates:

            twoing_n = (
                (twoing - twoing_min) / (twoing_max - twoing_min)
                if twoing_max > twoing_min else 0.0
            )
            func_n = (
                (func - func_min) / (func_max - func_min)
                if func_max > func_min else 0.0
            )

            func_score = self.alpha * func_n # aqui aggrego lo del alpha
            #score = self.beta * twoing_n + (1 - self.beta) * func_n # primera versión funcional 
            score = self.beta * twoing_n + (1 - self.beta)* self.alpha * func_score# segunda versión a probar
            #print(func_score, "             ", score, "         ", best_score)
            if score > best_score:
                best_score = score
                best_feat = feat
                best_thr = thr
                best_models = (m_l, m_r)

        return best_feat, best_thr, best_score, best_models
