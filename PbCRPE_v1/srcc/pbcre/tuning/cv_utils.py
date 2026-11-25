# srcc/pbcre/tuning/cv_utils.py
"""
Evaluación por validación cruzada compartida por todos los buscadores.

- sampler: remuestreo (p. ej. SMOTE) aplicado SÓLO al train de cada fold.
- threshold_strategy:
    None          -> umbral fijo del árbol (0.5).
    "youden"      -> umbral que minimiza BER en una validación interna.
    "fpr_current" -> minimiza el error tipo II sin superar el error tipo I
                     del umbral 0.5 (validación interna).
    "fnr_current" -> minimiza el error tipo I sin superar el error tipo II
                     del umbral 0.5 (validación interna).
    float         -> minimiza el error tipo II con error tipo I <= float.
    ("fnr", x)    -> minimiza el error tipo I con error tipo II <= x.
  La validación interna se separa del train del fold (val_fraction), así el
  fold de test nunca participa en elegir el umbral.
- greater_is_better=False para métricas de error (BER, FNR...): se niegan
  internamente, evitando optimizar en la dirección equivocada.
"""

import numpy as np
from sklearn.model_selection import train_test_split

from srcc.pbcre.tree.tree import PBCRETree
from srcc.pbcre.evaluation.metrics import error_report


def _fit_one(params, X_tr, y_tr, sampler, threshold_strategy,
             val_fraction, prune_metric, random_state):
    X_fit, y_fit = X_tr, y_tr
    X_in, y_in = None, None

    needs_val = threshold_strategy is not None or prune_metric is not None
    if needs_val:
        X_fit, X_in, y_fit, y_in = train_test_split(
            X_tr, y_tr, test_size=val_fraction, stratify=y_tr,
            random_state=random_state)

    if sampler is not None:
        X_fit, y_fit = sampler(X_fit, y_fit)

    model = PBCRETree(**params)
    model.fit(X_fit, y_fit)

    if needs_val and len(np.unique(y_in)) == 2:
        if prune_metric is not None:
            model.cost_complexity_pruning(X_in, y_in, metric=prune_metric)
        if threshold_strategy == "youden":
            model.fit_threshold(X_in, y_in, fpr_max=None)
        elif threshold_strategy == "fpr_current":
            model.fit_threshold(X_in, y_in, fpr_max="current")
        elif threshold_strategy == "fnr_current":
            model.fit_threshold(X_in, y_in, fnr_max="current")
        elif (isinstance(threshold_strategy, tuple)
              and threshold_strategy[0] == "fnr"):
            model.fit_threshold(X_in, y_in, fnr_max=float(threshold_strategy[1]))
        elif isinstance(threshold_strategy, (int, float)):
            model.fit_threshold(X_in, y_in, fpr_max=float(threshold_strategy))
    return model


def evaluate_params(params, X, y, skf, metric, greater_is_better=True,
                    sampler=None, threshold_strategy=None, val_fraction=0.25,
                    prune_metric=None, random_state=42):
    """Devuelve (score_medio_a_maximizar, lista de reportes por fold)."""
    sign = 1.0 if greater_is_better else -1.0
    fold_scores, reports = [], []

    y = y.reset_index(drop=True) if hasattr(y, "reset_index") else y
    X = X.reset_index(drop=True)
    y_arr = np.asarray(y)

    for k, (train_idx, test_idx) in enumerate(skf.split(X, y_arr)):
        X_tr = X.iloc[train_idx].reset_index(drop=True)
        y_tr = y_arr[train_idx]
        X_te = X.iloc[test_idx].reset_index(drop=True)
        y_te = y_arr[test_idx]

        model = _fit_one(params, X_tr, y_tr, sampler, threshold_strategy,
                         val_fraction, prune_metric, random_state)
        y_pred = model.predict(X_te)

        fold_scores.append(sign * metric(y_te, y_pred))
        rep = error_report(y_te, y_pred)
        rep["fold"] = k
        rep["threshold"] = model.threshold_
        reports.append(rep)

    return float(np.mean(fold_scores)), reports
