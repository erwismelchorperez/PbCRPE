# srcc/pbcre/evaluation/metrics.py
"""
Métricas de clasificación orientadas al control de errores tipo I y tipo II.

Cambios en PbCRPE_v1:
- Todas las matrices de confusión usan labels=[0, 1], de modo que no fallan
  cuando un nodo o un fold contiene una sola clase.
- Las funciones "raw" (fpr, fnr, ber, balanced_accuracy) no redondean, para
  poder usarse como criterio de optimización sin empates artificiales.
  Las funciones públicas originales conservan el redondeo para reportes.
- Se añade ConstrainedObjective: minimiza el BER penalizando que el error
  tipo I supere un valor de referencia (criterio tipo Neyman-Pearson).
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score
)


# =====================================================
# Funciones base (sin redondeo, para optimizar)
# =====================================================

def binary_confusion(y_true, y_pred):
    """Devuelve (tn, fp, fn, tp) aunque y_true/y_pred tengan una sola clase."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def fpr(y_true, y_pred):
    """Tasa de error tipo I: FP / (TN + FP)."""
    tn, fp, _, _ = binary_confusion(y_true, y_pred)
    return fp / (tn + fp) if (tn + fp) > 0 else 0.0


def fnr(y_true, y_pred):
    """Tasa de error tipo II: FN / (TP + FN)."""
    _, _, fn, tp = binary_confusion(y_true, y_pred)
    return fn / (tp + fn) if (tp + fn) > 0 else 0.0


def ber(y_true, y_pred):
    """Balanced Error Rate = (FPR + FNR) / 2. Menor es mejor."""
    return (fpr(y_true, y_pred) + fnr(y_true, y_pred)) / 2.0


def balanced_accuracy(y_true, y_pred):
    """1 - BER. Mayor es mejor (usar cuando el código MAXIMIZA)."""
    return 1.0 - ber(y_true, y_pred)


def error_report(y_true, y_pred):
    """Resumen de errores para registrar por fold."""
    tn, fp, fn, tp = binary_confusion(y_true, y_pred)
    _fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    _fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    return {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "fpr": _fpr,                 # error tipo I
        "fnr": _fnr,                 # error tipo II
        "ber": (_fpr + _fnr) / 2.0,
        "balanced_accuracy": 1.0 - (_fpr + _fnr) / 2.0,
    }


class ConstrainedObjective:
    """
    Objetivo para tuning (mayor es mejor):

        score = -( BER + lam * max(0, FPR - fpr_ref) )

    Con fpr_ref = error tipo I del modelo de referencia, el optimizador
    sólo gana si reduce el error tipo II sin pagar con error tipo I.
    fpr_ref=None equivale a minimizar el BER sin restricción.
    """

    def __init__(self, fpr_ref=None, lam=5.0):
        self.fpr_ref = fpr_ref
        self.lam = lam

    def __call__(self, y_true, y_pred):
        _fpr = fpr(y_true, y_pred)
        _fnr = fnr(y_true, y_pred)
        penalty = 0.0
        if self.fpr_ref is not None:
            penalty = self.lam * max(0.0, _fpr - self.fpr_ref)
        return -((_fpr + _fnr) / 2.0 + penalty)


# =====================================================
# API original (con redondeo, para reportes)
# =====================================================

class ClassificationMetrics:
    """
    Métricas básicas de clasificación binaria/multiclase
    independientes del modelo.
    """

    @staticmethod
    def confusion_matrix(y_true, y_pred, labels=None):
        if labels is None:
            labels = [0, 1]
        return confusion_matrix(y_true, y_pred, labels=labels)

    @staticmethod
    def accuracy(y_true, y_pred):
        return round(accuracy_score(y_true, y_pred), 4)

    @staticmethod
    def precision(y_true, y_pred):
        return round(precision_score(y_true, y_pred, average='weighted',
                                     zero_division=0), 4)

    @staticmethod
    def recall(y_true, y_pred):
        return round(recall_score(y_true, y_pred, average='weighted',
                                  zero_division=0), 4)

    @staticmethod
    def f1_score(y_true, y_pred):
        return round(f1_score(y_true, y_pred, average='weighted',
                              zero_division=0), 4)

    @staticmethod
    def matthews_corrcoef(y_true, y_pred):
        return round(matthews_corrcoef(y_true, y_pred), 4)

    @staticmethod
    def auc(y_true, y_score, average="weighted"):
        """
        AUC ROC.
        y_score:
          - binario: probabilidad clase positiva
          - multiclase: matriz (n_samples, n_classes)
        """
        y_score = np.asarray(y_score)
        if y_score.ndim == 1:
            score = roc_auc_score(y_true, y_score)
        else:
            score = roc_auc_score(y_true, y_score, multi_class="ovr",
                                  average=average)
        return round(score, 4)

    @staticmethod
    def errorrateI(y_true, y_pred):
        """Error tipo I (FPR). Menor es mejor."""
        return round(fpr(y_true, y_pred), 4)

    @staticmethod
    def errorrateII(y_true, y_pred):
        """Error tipo II (FNR). Menor es mejor."""
        return round(fnr(y_true, y_pred), 4)

    @staticmethod
    def balanced_error_rate(y_true, y_pred):
        """
        Balanced Error Rate (BER). MENOR es mejor.
        Para criterios que maximizan use balanced_accuracy (= 1 - BER).
        """
        return round(ber(y_true, y_pred), 4)

    @staticmethod
    def balanced_accuracy(y_true, y_pred):
        """1 - BER. MAYOR es mejor."""
        return round(balanced_accuracy(y_true, y_pred), 4)
