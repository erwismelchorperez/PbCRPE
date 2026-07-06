import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from .hybrid_twoing_functional import HybridTwoingFunctionalCriterion
class HybridTwoingFunctionalCriterionFast(HybridTwoingFunctionalCriterion):

    def __init__(self,alpha=0.5,beta=0.05,logistic_C=0.005,logistic_solver="liblinear",logistic_max_iter=200,n_quantiles=7,twoing_threshold=0.01,max_logistic_samples=500):
        super().__init__(alpha=alpha,beta=beta,logistic_C=logistic_C,logistic_solver=logistic_solver,logistic_max_iter=logistic_max_iter)

        self.n_quantiles = n_quantiles
        self.twoing_threshold = twoing_threshold
        self.max_logistic_samples = max_logistic_samples

    def _logistic_score(self,X,y,min_samples_leaf):
        y = np.asarray(y)

        if len(y) < 2 * min_samples_leaf:
            return 0.0,None

        if len(np.unique(y)) < 2:
            return 0.0,None

        X_enc = np.asarray(self._encode_X(X))

        if len(y) > self.max_logistic_samples:

            idx = np.random.choice(np.arange(len(y)),size=self.max_logistic_samples,replace=False)

            X_enc = X_enc[idx]
            y = y[idx]

        _,counts = np.unique(y,return_counts=True)

        model = LogisticRegression(solver=self.logistic_solver,C=self.logistic_C,max_iter=self.logistic_max_iter, class_weight="balanced")
        #model = LogisticRegression(solver=self.logistic_solver,C=self.logistic_C,max_iter=self.logistic_max_iter)

        if counts.min() >= 2:
            X_tr,X_val,y_tr,y_val = train_test_split(X_enc,y,test_size=0.3,stratify=y,random_state=42)
            model.fit(X_tr,y_tr)
            y_pred = model.predict(X_val)
            score = f1_score(y_val,y_pred,average="weighted")
            #score = f1_score(y_val,y_pred,average="macro")

        else:
            model.fit(X_enc,y)
            y_pred = model.predict(X_enc)
            score = f1_score(y,y_pred,average="weighted")
            #score = f1_score(y,y_pred,average="macro")

        return score,model
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
                quantiles = np.linspace(0.1,0.9,self.n_quantiles)
                thresholds = np.unique(np.quantile(col,quantiles))

            else:

                thresholds = np.unique(col)

            for thr in thresholds:

                mask = (
                    col <= thr
                    if np.issubdtype(col.dtype, np.number)
                    else col == thr
                )

                left_size = mask.sum()
                right_size = (~mask).sum()

                if left_size < min_samples_leaf:
                    continue

                if right_size < min_samples_leaf:
                    continue

                X_l = X[mask]
                y_l = y[mask]

                X_r = X[~mask]
                y_r = y[~mask]

                twoing = self._twoing(y_l,y_r)
                if twoing < self.twoing_threshold:
                    continue

                score_l, model_l = self._logistic_score(X_l,y_l,min_samples_leaf)
                score_r, model_r = self._logistic_score(X_r,y_r,min_samples_leaf)

                #func_score = (len(y_l) * score_l + len(y_r) * score_r) / (len(y_l) + len(y_r))
                
                purity_l = np.max(np.bincount(y_l))/len(y_l)
                purity_r = np.max(np.bincount(y_r))/len(y_r)
                func_score = (len(y_l)*score_l*purity_l + len(y_r)*score_r*purity_r) / (len(y_l)+len(y_r))
                

                scores_twoing.append(twoing)
                scores_func.append(func_score)

                candidates.append((feat,thr,twoing,func_score,model_l,model_r))

        if len(candidates) == 0:
            return None, None, -np.inf, None

        twoing_min = min(scores_twoing)
        twoing_max = max(scores_twoing)

        func_min = min(scores_func)
        func_max = max(scores_func)

        for feat, thr, twoing, func, m_l, m_r in candidates:
            twoing_n = (
                (twoing - twoing_min)
                / (twoing_max - twoing_min)
                if twoing_max > twoing_min
                else 0.0
            )

            func_n = (
                (func - func_min)
                / (func_max - func_min)
                if func_max > func_min
                else 0.0
            )

            score = (self.beta * twoing_n + (1 - self.beta) * ( self.alpha * func_n ))

            if score > best_score:

                best_score = score
                best_feat = feat
                best_thr = thr
                best_models = ( m_l, m_r)

        return ( best_feat, best_thr, best_score, best_models)