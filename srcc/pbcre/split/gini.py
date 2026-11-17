import numpy as np
from .base import SplitCriterion

class GiniCriterion(SplitCriterion):

    def _gini_node(self, y):
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return 1.0 - np.sum(probs ** 2)

    def _gini_split(self, y_left, y_right):
        n = len(y_left) + len(y_right)
        return (
            len(y_left) / n * self._gini_node(y_left)
            + len(y_right) / n * self._gini_node(y_right)
        )

    def best_split(self, X, y, min_samples_leaf):
        best_feat, best_thr = None, None
        best_score = np.inf

        n_samples, n_features = X.shape

        for feat_idx in range(n_features):
            col = X.iloc[:, feat_idx].values
            thresholds = np.unique(col)

            for thr in thresholds:
                mask = col <= thr
                n_left = mask.sum()
                n_right = n_samples - n_left

                if n_left < min_samples_leaf or n_right < min_samples_leaf:
                    continue

                y_left = y[mask]
                y_right = y[~mask]

                score = self._gini_split(y_left, y_right)

                if score < best_score:
                    best_score = score
                    best_feat = feat_idx
                    best_thr = thr

        return best_feat, best_thr, best_score
