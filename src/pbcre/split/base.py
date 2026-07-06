from abc import ABC, abstractmethod

class SplitCriterion(ABC):
    """
    Contrato único para TODOS los criterios de split.
    """

    @abstractmethod
    def best_split(self, X, y, min_samples_leaf):
        """
        Returns
        -------
        best_feature : int
        best_threshold : float
        best_score : float
        extra_info : any (None or model)
        """
        pass
