import numpy as np
from sklearn.linear_model import LogisticRegression

class LogisticNodeModel:

    def __init__(self, C=1.0, max_iter=200):
        self.model = LogisticRegression(C=C, max_iter=max_iter)

    def fit(self, X, y):
        self.model.fit(X, y)

    def score(self, X, y):
        """
        Usamos accuracy o log-loss (elige según paper)
        """
        return self.model.score(X, y)
