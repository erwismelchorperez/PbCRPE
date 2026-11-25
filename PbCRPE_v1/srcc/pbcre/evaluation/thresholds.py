# srcc/pbcre/evaluation/thresholds.py
"""
Selección del umbral de decisión global.

Relación clave:  Youden J = TPR - FPR = 1 - 2 * BER
=> maximizar J es exactamente minimizar el BER.

Restricciones (criterio tipo Neyman-Pearson):
    fpr_max -> minimizar error tipo II  sujeto a  error tipo I  <= fpr_max
    fnr_max -> minimizar error tipo I   sujeto a  error tipo II <= fnr_max
"""

import warnings

import numpy as np
from sklearn.metrics import roc_curve


def select_threshold(y_true, scores, fpr_max=None, fnr_max=None):
    """
    Parameters
    ----------
    y_true : etiquetas 0/1 de VALIDACIÓN (nunca del test final).
    scores : probabilidades/puntuaciones de la clase positiva.
    fpr_max : float -> minimiza FNR con FPR <= fpr_max.
    fnr_max : float -> minimiza FPR con FNR <= fnr_max.
    Ninguno -> minimiza BER (Youden).

    Returns
    -------
    threshold : float, se predice 1 si score >= threshold.
    info : dict con fpr/fnr/ber esperados en validación.
    """
    if fpr_max is not None and fnr_max is not None:
        raise ValueError("Use fpr_max o fnr_max, no ambos.")

    y_true = np.asarray(y_true).reshape(-1)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if len(np.unique(y_true)) < 2:
        raise ValueError("Se necesitan ambas clases para elegir el umbral.")

    fpr, tpr, thr = roc_curve(y_true, scores)

    # thr[0] = inf (nadie positivo): se descarta para no anular la clase 1
    finite = np.isfinite(thr)
    fpr, tpr, thr = fpr[finite], tpr[finite], thr[finite]
    fnr = 1.0 - tpr

    if fnr_max is not None:
        ok = np.flatnonzero(fnr <= fnr_max + 1e-12)
        if len(ok) == 0:
            warnings.warn(
                f"Ningún umbral cumple FNR <= {fnr_max:.4f} en validación; "
                f"se usa el de menor FNR ({fnr.min():.4f}).",
                RuntimeWarning, stacklevel=2)
            ok = np.flatnonzero(np.isclose(fnr, fnr.min()))
        best_val = fpr[ok].min()
        tied = ok[np.isclose(fpr[ok], best_val)]
        best = tied[np.argmin(fnr[tied])]          # empate: protege tipo II

    elif fpr_max is not None:
        ok = np.flatnonzero(fpr <= fpr_max + 1e-12)
        if len(ok) == 0:
            warnings.warn(
                f"Ningún umbral cumple FPR <= {fpr_max:.4f} en validación; "
                f"se usa el de menor FPR ({fpr.min():.4f}).",
                RuntimeWarning, stacklevel=2)
            ok = np.flatnonzero(np.isclose(fpr, fpr.min()))
        best_val = tpr[ok].max()
        tied = ok[np.isclose(tpr[ok], best_val)]
        best = tied[np.argmin(fpr[tied])]          # empate: protege tipo I

    else:
        j = tpr - fpr
        tied = np.flatnonzero(np.isclose(j, j.max()))
        best = tied[np.argmin(fpr[tied])]

    info = {
        "threshold": float(thr[best]),
        "fpr_val": float(fpr[best]),
        "fnr_val": float(fnr[best]),
        "ber_val": float((fpr[best] + fnr[best]) / 2.0),
        "fpr_max": fpr_max,
        "fnr_max": fnr_max,
    }
    return float(thr[best]), info


def resolve_current(y_true, scores, current_threshold, fpr_max, fnr_max):
    """Sustituye 'current' por el FPR/FNR del umbral actual en validación."""
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    pred = (np.asarray(scores) >= current_threshold).astype(int)
    for name in ("fpr_max", "fnr_max"):
        val = fpr_max if name == "fpr_max" else fnr_max
        if isinstance(val, str) and val != "current":
            raise ValueError(f"{name} debe ser None, un float o 'current'.")
    if fpr_max == "current":
        neg = y_true == 0
        fpr_max = float(pred[neg].mean()) if neg.any() else 0.0
    if fnr_max == "current":
        pos = y_true == 1
        fnr_max = float((pred[pos] == 0).mean()) if pos.any() else 0.0
    return fpr_max, fnr_max
