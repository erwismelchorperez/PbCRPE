import numpy as np
import pandas as pd
from .base import SplitCriterion
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score


class FunctionalCriterion(SplitCriterion):

    def __init__(self, beta=0.5):
        self.beta = beta

    # -------------------------------------------------
    def _encode_X(self, X):
        """
        Codificación local por nodo
        """
        X = pd.DataFrame(X)

        num_cols = X.select_dtypes(include=[np.number]).columns
        cat_cols = X.select_dtypes(exclude=[np.number]).columns

        X_num = X[num_cols].values if len(num_cols) > 0 else None

        if len(cat_cols) > 0:
            enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
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
        """
            Entrena modelo funcional local si es posible.
            Devuelve (score, model) o (0.0, None)
        """
        # 🚫 nodo no separable
        if len(y) < 2 * min_samples_leaf or len(np.unique(y)) < 2:
            return 0.0, None

        X_enc = self._encode_X(X)

        # Conteo por clase (clave para árboles)
        _, counts = np.unique(y, return_counts=True)

        model = LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            C=0.002,              # 👈 más regularización: 0.002 funtiona mejor con f1_score
            max_iter=200
        )
        # -------------------------------------------------
        # Caso 1: validación segura
        if counts.min() >= 2:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_enc,
                y,
                test_size=0.3,
                stratify=y,
                random_state=42
            )

            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            score = f1_score(y_val, y_pred, pos_label=1)
            # 🔹 PROBABILIDADES (clave para AUC)
            #y_prob = model.predict_proba(X_val)[:, 1]
            #score = roc_auc_score(y_val, y_prob)

        # -------------------------------------------------
        # Caso 2: nodo pequeño → fallback (sin split)
        else:
            model.fit(X_enc, y)
            y_pred = model.predict(X_enc)
            score = f1_score(y, y_pred, pos_label=1)
            #y_prob = model.predict_proba(X_enc)[:, 1]
            #score = roc_auc_score(y, y_prob)

        return score, model
    # -------------------------------------------------
    def best_split(self, X, y, min_samples_leaf):
        best_feat, best_thr = None, None
        best_score = -np.inf
        best_models = None

        for feat in range(X.shape[1]):
            col = X.iloc[:, feat].values
            thresholds = np.unique(col)

            for thr in thresholds:
                mask = col <= thr if np.issubdtype(col.dtype, np.number) else col == thr

                if mask.sum() < min_samples_leaf or (~mask).sum() < min_samples_leaf:
                    continue

                X_l, y_l = X[mask], y[mask]
                X_r, y_r = X[~mask], y[~mask]

                score_l, model_l = self._logistic_score(
                    X_l, y_l, min_samples_leaf
                )
                score_r, model_r = self._logistic_score(
                    X_r, y_r, min_samples_leaf
                )

                score = score_l + score_r

                if score > best_score:
                    best_score = score
                    best_feat = feat
                    best_thr = thr
                    best_models = (model_l, model_r)

        return best_feat, best_thr, best_score, best_models
