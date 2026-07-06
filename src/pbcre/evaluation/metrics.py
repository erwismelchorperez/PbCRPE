# src/pbcre/metrics/metrics.py

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, roc_curve, auc

class ClassificationMetrics:
    """
    Métricas básicas de clasificación binaria/multiclase
    independientes del modelo.
    """

    @staticmethod
    def confusion_matrix(y_true, y_pred, labels=None):
        """
        Calcula la matriz de confusión.
        """
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        return cm

    @staticmethod
    def accuracy(y_true, y_pred):
        return round(accuracy_score(y_true, y_pred),4)

    @staticmethod
    def precision(y_true, y_pred):
        return round(precision_score(y_true, y_pred, average='weighted'),4)

    @staticmethod
    def recall(y_true, y_pred):
        return round(recall_score(y_true, y_pred, average='weighted'),4)

    @staticmethod
    def f1_score(y_true, y_pred):
        return round(f1_score(y_true, y_pred, average='weighted'),4)
    
    @staticmethod
    def matthews_corrcoef(y_true, y_pred):
        return round(matthews_corrcoef(y_true, y_pred),4)
    
    # -------------------------------------------------
    @staticmethod
    def auc(y_true, y_score, average="weighted"):
        """
        AUC ROC.
        y_score:
          - binario: probabilidad clase positiva
          - multiclase: matriz (n_samples, n_classes)
        """
        # Caso binario
        if y_score.ndim == 1:
            score = roc_auc_score(y_true, y_score)

        # Caso multiclase
        else:
            score = roc_auc_score(
                y_true,
                y_score,
                multi_class="ovr",
                average=average
            )

        return round(score, 4)

    @staticmethod
    def errorrateI(y_true, y_pred):
        # Calcular matriz de confusión
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        # Tasas de error
        total_negatives = tn + fp
        
        error_type_I_rate = fp / total_negatives if total_negatives > 0 else 0
        return round(error_type_I_rate,4)
    
    @staticmethod
    def errorrateII(y_true, y_pred):
        # Calcular matriz de confusión
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()        
        # Tasas de error
        total_positives = tp + fn
        
        error_type_II_rate = fn / total_positives if total_positives > 0 else 0
        return round(error_type_II_rate,4)