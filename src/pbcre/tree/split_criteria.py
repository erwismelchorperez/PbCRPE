# pbcre/tree/split_criteria.py
import numpy as np


class SplitCriterion:

    @staticmethod
    def find_best_split(X, y, criterion, min_samples_leaf):
        best_score = -np.inf
        best = None
        n = len(y)

        for feat in X.columns:
            values = np.unique(X[feat])

            # reducción brutal de thresholds
            if len(values) > 10:
                values = np.percentile(values, np.linspace(10, 90, 10))

            for thr in values:
                left = X[feat] <= thr
                right = ~left

                if left.sum() < min_samples_leaf or right.sum() < min_samples_leaf:
                    continue

                score = SplitCriterion._score(
                    y[left], y[right], criterion
                )

                if score > best_score:
                    best_score = score
                    best = (
                        feat,
                        thr,
                        np.where(left)[0],
                        np.where(right)[0]
                    )

        return best

    @staticmethod
    def _score(y_left, y_right, criterion):
        if criterion == "gini":
            return -SplitCriterion._gini(y_left, y_right)
        if criterion == "twoing":
            return SplitCriterion._twoing(y_left, y_right)
        raise ValueError("Criterion not supported")
