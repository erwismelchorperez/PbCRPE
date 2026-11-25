import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .hybrid_twoing_functional import HybridTwoingFunctionalCriterion


class HybridTwoingFunctionalCriterionSFS(HybridTwoingFunctionalCriterion):
    """
    Criterio híbrido con selección de atributos por magnitud de coeficientes.
    PbCRPE_v1: score con self._score (por defecto 1 - BER).
    """

    def __init__(self, alpha=0.5, beta=0.05, max_features_sfs=3,
                 logistic_C=0.005, logistic_solver="liblinear",
                 logistic_max_iter=200, scoring="balanced",
                 random_state=42, verbose=False):
        super().__init__(alpha=alpha, beta=beta, logistic_C=logistic_C,
                         logistic_solver=logistic_solver,
                         logistic_max_iter=logistic_max_iter,
                         scoring=scoring, random_state=random_state,
                         verbose=verbose)
        self.max_features_sfs = max_features_sfs

    def _logistic_score(self, X, y, min_samples_leaf):
        y = np.asarray(y)
        if len(y) < 2 * min_samples_leaf:
            return 0.0, None
        if len(np.unique(y)) < 2:
            return 0.0, None

        X_enc = self._encode_X(X)
        _, counts = np.unique(y, return_counts=True)

        def new_model():
            return LogisticRegression(solver=self.logistic_solver,
                                      class_weight="balanced",
                                      C=self.logistic_C,
                                      max_iter=self.logistic_max_iter)

        # Selección de atributos
        selector_model = new_model().fit(X_enc, y)
        importance = np.abs(selector_model.coef_[0])
        n_features = min(self.max_features_sfs, X_enc.shape[1])
        selected_features = np.argsort(importance)[-n_features:]
        X_sel = X_enc[:, selected_features]

        model = new_model()
        if counts.min() >= 2:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_sel, y, test_size=0.30, stratify=y,
                random_state=self.random_state)
            model.fit(X_tr, y_tr)
            score = self._score(y_val, model.predict(X_val))
        else:
            model.fit(X_sel, y)
            score = self._score(y, model.predict(X_sel))

        return score, model
