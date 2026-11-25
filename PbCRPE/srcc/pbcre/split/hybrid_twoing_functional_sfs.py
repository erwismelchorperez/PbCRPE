import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from .hybrid_twoing_functional import HybridTwoingFunctionalCriterion


class HybridTwoingFunctionalCriterionSFS(HybridTwoingFunctionalCriterion):

    def __init__(self,alpha=0.5,beta=0.05,max_features_sfs=3, logistic_C=0.005, logistic_solver="liblinear", logistic_max_iter=200):
        super().__init__(alpha=alpha,beta=beta,logistic_C=logistic_C, logistic_solver=logistic_solver,logistic_max_iter=logistic_max_iter)

        self.max_features_sfs = max_features_sfs

    def _logistic_score(self,X,y,min_samples_leaf):
        if len(y) < 2 * min_samples_leaf:
            return 0.0, None
        if len(np.unique(y)) < 2:
            return 0.0, None
        X_enc = self._encode_X(X)
        _, counts = np.unique(y,return_counts=True)
        # -------------------------------------------------
        # Primera regresión para seleccionar atributos
        # -------------------------------------------------
        selector_model = LogisticRegression(solver=self.logistic_solver, class_weight="balanced",C=self.logistic_C,max_iter=self.logistic_max_iter)#0.005
        selector_model.fit(X_enc,y)
        importance = np.abs(selector_model.coef_[0])
        n_features = min(self.max_features_sfs,X_enc.shape[1])
        selected_features = np.argsort(importance)[-n_features:]

        X_sel = X_enc[:, selected_features]
        # -------------------------------------------------
        # Modelo final
        # -------------------------------------------------
        model = LogisticRegression(solver=self.logistic_solver,class_weight="balanced",C=self.logistic_C,max_iter=self.logistic_max_iter)

        if counts.min() >= 2:
            X_tr, X_val, y_tr, y_val = train_test_split(X_sel,y, test_size=0.30, stratify=y,random_state=42)
            model.fit(X_tr,y_tr)
            y_pred = model.predict(X_val)
            score = f1_score(y_val,y_pred,pos_label=1)

        else:
            model.fit(X_sel,y)
            y_pred = model.predict(X_sel)
            score = f1_score(y,y_pred,pos_label=1)

        return score, model