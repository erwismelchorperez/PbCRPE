import numpy as np
from .base import SplitCriterion

class TwoingCriterion(SplitCriterion):

    def _twoing_score(self, y_left, y_right):
        n_l, n_r = len(y_left), len(y_right)
        n = n_l + n_r

        if n_l == 0 or n_r == 0:
            return 0.0

        classes = np.unique(np.concatenate([y_left, y_right]))
        p_l = np.array([(y_left == c).mean() for c in classes])
        p_r = np.array([(y_right == c).mean() for c in classes])

        return (n_l / n) * (n_r / n) * (np.sum(np.abs(p_l - p_r)) ** 2)

    def best_split(self, X, y, min_samples_leaf):
        best_feat, best_thr = None, None
        best_score = -np.inf

        n_samples, n_features = X.shape

        for feat_idx in range(n_features):
            col = X.iloc[:, feat_idx].values
            thresholds = np.unique(col)

            for thr in thresholds:
                mask = col <= thr

                if mask.sum() < min_samples_leaf or (~mask).sum() < min_samples_leaf:
                    continue

                score = self._twoing_score(y[mask], y[~mask])

                if score > best_score:
                    best_score = score
                    best_feat = feat_idx
                    best_thr = thr

        return best_feat, best_thr, best_score, None
