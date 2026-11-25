"""
PbCRPE_v1 - Resultado FINAL (un solo modelo, una sola evaluación en test).

Objetivo: minimizar conjuntamente los errores tipo I y tipo II (BER).

Protocolo 80/20:
  1. Partición estratificada fija: ENTRENAMIENTO 80 % | TEST 20 %.
  2. Dentro del 80 % (el test no se toca):
       a) Umbral: validación cruzada interna de k folds (--inner_folds).
          Cada fold construye el modelo con el mismo procedimiento del paso b
          y predice su parte excluida. Con esas probabilidades fuera de fold
          (las 800 instancias de German, no 200) se elige el umbral.
       b) Modelo final: se crece con el 80 % del entrenamiento y se poda con
          el 20 % restante (--prune_frac); después las hojas se reestiman
          con TODO el entrenamiento (refit_leaves).
  3. TEST: una única evaluación con el umbral del paso a.

Remuestreo: sólo con --smote (SMOTENC si hay columnas categóricas, SMOTE si
no); se aplica únicamente a los datos con los que se entrena cada modelo.

Salidas en --out (por defecto results_final/):
  <dataset>_final_<fecha>.xlsx con las hojas
     Resultado        métricas en test + matriz de confusión
     Particion        tamaños y proporción de clases de cada conjunto
     Umbrales_OOF     cada estrategia evaluada con las predicciones fuera de fold
     Patrones         CIPs con soporte/confianza en train y su desempeño en test
     Hojas            camino raíz-hoja, FP/FN en test por hoja
     Modelos_hoja     coeficientes de la logística de cada hoja (si aplica)
     Instancias_test  predicción, probabilidad, tipo de acierto/error y hoja
     Configuracion    parámetros
  particion/train.csv, test.csv  (con el índice original de cada fila)

El umbral lo determina el algoritmo con validación cruzada interna:
  youden   mínimo BER (por defecto)
  closest  mínima distancia al clasificador perfecto
  minimax  minimiza el mayor de los dos errores
  np_fnr   mínimo error I con error II <= --fnr_max ('auto' o un número)
  np_fpr   mínimo error II con error I <= --fpr_max ('auto' o un número)
  constrained  ambas cotas a la vez; menor BER entre los umbrales que cumplen
  auto     elige entre --auto_candidates la que mejor GENERALIZA en el
           entrenamiento (CV 5x10 sobre las predicciones fuera de fold)

Además agrega/actualiza una fila en results_final/comparacion_trabajos.xlsx
(trabajo, dataset, Accuracy, Precision, Recall, F1-Score, AUC, MCC,
ErrorTipoI, ErrorTipoII, BER, TipoTrabajo).

Uso:
  python run_final.py --dataset german --strategy auto
  python run_final.py --dataset german
  python run_final.py --dataset german --strategy np_fnr --fnr_max auto
  python run_final.py --dataset german --strategy np_fnr --fnr_max 0.157 \
        --trabajo "Our Proposal V2 (FNR 0.157)"
  python run_final.py --dataset german --smote
  python run_final.py --dataset german --strategy np_fnr --fnr_max 0.25
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from srcc.pbcre.tree.tree import PBCRETree
from sklearn.model_selection import StratifiedKFold, train_test_split

from srcc.pbcre.evaluation.metrics import ClassificationMetrics as CM
from srcc.pbcre.evaluation.thresholds import select_threshold
from srcc.pbcre.evaluation.excel_report import (
    fold_metrics, _write_table, _autowidth, _style_header, _style_body, FONT,
    METRIC_NOTES)
from srcc.pbcre.utils.ProccessData import ProccessData

STRATEGIES = ("base", "prior", "youden", "closest", "minimax", "np_fnr", "np_fpr",
              "constrained", "auto")
LINE = "=" * 75


# -------------------------------------------------------------------------
def split_data(args):
    pdata = ProccessData(args.dataset, random_state=args.seed)
    path = os.path.join("dataset", args.dataset + ".csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró {os.path.abspath(path)}.")
    pdata.ReadDataset()

    # índice original para poder rastrear cada fila
    X_all = pdata.X.reset_index(drop=True)
    y_all = pdata.y.reset_index(drop=True)
    X_all.insert(0, "_id", np.arange(len(X_all)))
    pdata.X, pdata.y = X_all, y_all
    pdata.SplitDataset(test_size=args.test_size, val_size=None)

    parts = {}
    for name, X, y in (("train", pdata.X_train, pdata.y_train),
                       ("test", pdata.X_test, pdata.y_test)):
        ids = X["_id"].to_numpy()
        parts[name] = (X.drop(columns="_id").reset_index(drop=True),
                       np.asarray(y).astype(int), ids)
    cat_cols = ProccessData.categorical_columns(parts["train"][0])
    cat_idx = [parts["train"][0].columns.get_loc(c) for c in cat_cols]
    return parts, cat_idx


def build_model(X, y, args, sampler, seed):
    """
    Crece con (1 - prune_frac) del conjunto, poda con prune_frac y reestima
    las hojas con TODO el conjunto. Devuelve (modelo, hojas_antes_poda).
    """
    X = X.reset_index(drop=True)
    X_g, X_p, y_g, y_p = train_test_split(
        X, y, test_size=args.prune_frac, stratify=y, random_state=seed)
    X_g, X_p = X_g.reset_index(drop=True), X_p.reset_index(drop=True)
    if sampler is not None:
        X_g, y_g = sampler(X_g, y_g)

    model = PBCRETree(**args.tree_params).fit(X_g, y_g)
    leaves_before = len(model._get_leaves(model.root))
    if not args.no_prune:
        model.cost_complexity_pruning(X_p, y_p, metric=args.prune_metric,
                                      min_leaves=args.min_leaves)

    X_r, y_r = (sampler(X, y) if sampler is not None else (X, y))
    model.refit_leaves(X_r, y_r)
    return model, leaves_before


def oof_probabilities(X, y, args, sampler):
    """Probabilidades fuera de fold sobre todo el entrenamiento."""
    skf = StratifiedKFold(n_splits=args.inner_folds, shuffle=True,
                          random_state=args.seed)
    oof = np.zeros(len(y), dtype=float)
    for k, (tr, va) in enumerate(skf.split(X, y)):
        m, _ = build_model(X.iloc[tr], y[tr], args, sampler, args.seed + k)
        oof[va] = m.predict_proba(X.iloc[va])
        print(f"  fold interno {k + 1}/{args.inner_folds} listo")
    return oof


def tradeoff_table(y, probs, fnr_targets):
    """
    Compromiso entre errores con las probabilidades fuera de fold:
    para cada cota de error tipo II, el menor error tipo I alcanzable.
    """
    rows = []
    for t in fnr_targets:
        thr, info = select_threshold(y, probs, fnr_max=t)
        rows.append({"cota_error_tipo_II": t, "umbral": thr,
                     "error_tipo_I": info["fpr_val"],
                     "error_tipo_II": info["fnr_val"],
                     "ber": info["ber_val"]})
    return pd.DataFrame(rows)


def resolve_fnr_max(value, y, probs):
    """
    --fnr_max numérico se usa tal cual. "auto" lo determina el algoritmo:
    el error tipo II que tendría el clasificador con umbral 0.5 (fuera de
    fold), es decir, conservar la sensibilidad del clasificador estándar y
    reducir el error tipo I todo lo posible.
    """
    if isinstance(value, str) and value.lower() == "auto":
        pos = y == 1
        return float(np.mean(probs[pos] < 0.5)) if pos.any() else 0.0
    return float(value)


def resolve_fpr_max(value, y, probs):
    """
    --fpr_max numérico se usa tal cual. "auto": el error tipo I que tendría el
    clasificador con umbral 0.5 (fuera de fold). Si los negativos son la
    clase minoritaria (p. ej. German), este valor es alto y restringe poco.
    """
    if isinstance(value, str) and value.lower() == "auto":
        neg = y == 0
        return float(np.mean(probs[neg] >= 0.5)) if neg.any() else 0.0
    return float(value)


def _roc_finite(y, probs):
    from sklearn.metrics import roc_curve
    f, t, thr = roc_curve(y, probs)
    ok = np.isfinite(thr)
    return f[ok], 1.0 - t[ok], thr[ok]


def constrained_threshold(y, probs, fpr_cap, fnr_cap):
    """
    Umbral con ambas cotas: error I <= fpr_cap y error II <= fnr_cap.
    Si hay umbrales factibles, el de menor BER; si no, el que menos se pasa
    de las cotas (mínimo de la mayor violación). Devuelve (umbral, factible).
    """
    f, n, thr = _roc_finite(y, probs)
    ok = (f <= fpr_cap + 1e-12) & (n <= fnr_cap + 1e-12)
    if ok.any():
        idx = np.flatnonzero(ok)
        return float(thr[idx[np.argmin((f[idx] + n[idx]) / 2)]]), True
    viol = np.maximum(f - fpr_cap, n - fnr_cap)
    return float(thr[np.argmin(viol)]), False


def select_strategy_cv(y, probs, prior, candidates, fnr_cap, fpr_cap,
                       metric="ber", n_splits=5, n_repeats=10, seed=42):
    """
    Elige la estrategia de umbral SÓLO con el entrenamiento.

    Sobre las probabilidades fuera de fold se repite n_repeats veces una
    validación cruzada de n_splits: cada estrategia elige su umbral con
    k-1 partes y se evalúa en la parte excluida. Así se mide cuánto
    GENERALIZA cada regla (no cuánto se ajusta a los datos con que eligió).
    metric="ber" -> menor BER; "max_error" -> menor máximo(error I, error II).
    """
    from sklearn.model_selection import StratifiedKFold
    y = np.asarray(y).astype(int)
    scores = {c: [] for c in candidates}
    fprs = {c: [] for c in candidates}
    fnrs = {c: [] for c in candidates}
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=seed + r)
        preds = {c: np.zeros(len(y), dtype=int) for c in candidates}
        for tr, va in skf.split(probs.reshape(-1, 1), y):
            for c in candidates:
                t = threshold_for(c, y[tr], probs[tr], prior, fnr_cap, fpr_cap)
                preds[c][va] = (probs[va] >= t).astype(int)
        for c in candidates:
            fp_ = CM.errorrateI(y, preds[c])
            fn_ = CM.errorrateII(y, preds[c])
            fprs[c].append(fp_)
            fnrs[c].append(fn_)
            scores[c].append((fp_ + fn_) / 2 if metric == "ber" else max(fp_, fn_))
    rows = [{"estrategia": c,
             "error_tipo_I_cv": float(np.mean(fprs[c])),
             "error_tipo_II_cv": float(np.mean(fnrs[c])),
             f"{metric}_cv": float(np.mean(scores[c])),
             f"{metric}_cv_de": float(np.std(scores[c]))} for c in candidates]
    df = pd.DataFrame(rows).sort_values(f"{metric}_cv").reset_index(drop=True)
    return df.loc[0, "estrategia"], df


def threshold_for(strategy, y, probs, prior, fnr_max, fpr_max=None):
    """
    Umbral de cada estrategia a partir de probabilidades fuera de fold.
    Reglas automáticas (no requieren ningún valor del usuario):
      youden  -> mínimo BER = (error I + error II) / 2
      closest -> mínima distancia al clasificador perfecto,
                 sqrt(error_I^2 + error_II^2); penaliza más el error mayor
      minimax -> minimiza el MAYOR de los dos errores (deja ambos parejos)
    Con cotas:
      np_fnr      -> mínimo error I con error II <= fnr_max
      np_fpr      -> mínimo error II con error I <= fpr_max
      constrained -> ambas cotas; menor BER entre los umbrales que las cumplen
    """
    if strategy == "closest":
        f, n, thr = _roc_finite(y, probs)
        return float(thr[np.argmin(np.hypot(f, n))])
    if strategy == "minimax":
        f, n, thr = _roc_finite(y, probs)
        worst = np.maximum(f, n)
        tied = np.flatnonzero(np.isclose(worst, worst.min()))
        return float(thr[tied[np.argmin((f[tied] + n[tied]) / 2)]])
    if strategy == "constrained":
        return constrained_threshold(y, probs, fpr_max, fnr_max)[0]
    if strategy == "base":
        return 0.5
    if strategy == "prior":
        return float(prior)
    if strategy == "youden":
        return select_threshold(y, probs)[0]
    if strategy == "np_fnr":
        return select_threshold(y, probs, fnr_max=fnr_max)[0]
    if strategy == "np_fpr":
        return select_threshold(y, probs, fpr_max=fpr_max)[0]
    raise ValueError(strategy)


def evaluation_block(title, y_true, y_pred, scores):
    m = fold_metrics(y_true, y_pred, scores)
    cm = CM.confusion_matrix(y_true, y_pred)
    print(f"\n{LINE}\n{title}\n{LINE}\n")
    print("Matriz de confusión:")
    print(cm)
    print()
    labels = [("Accuracy", "accuracy"), ("Precision", "precision"),
              ("Recall", "recall"), ("F1-score", "f1_score"), ("MCC", "mcc"),
              ("Error tipo I", "error_tipo_I"), ("Error tipo II", "error_tipo_II"),
              ("Balanced Error Rate", "ber"), ("AUC", "auc")]
    for lab, key in labels:
        print(f"{lab + ':':22s}{m[key]:.4f}")
    return m, cm


# -------------------------------------------------------------------------
def pattern_table(model, cips, X_test, y_test, y_train):
    """CIPs con sus estadísticas de train y su comportamiento en test."""
    X_arr = model._to_array(X_test)
    n_neg = int(np.sum(y_train == 0))
    n_pos = int(np.sum(y_train == 1))
    rows = []
    for j, cip in enumerate(cips):
        covered = np.array([model._match_path(x, cip["path"]) for x in X_arr])
        yt = y_test[covered]
        cls = int(cip["class"])
        neg_c, pos_c = map(int, cip["class_counts"])
        rows.append({
            "id": j,
            "patron": model._path_to_human(cip["path"]),
            "clase": cls,
            "condiciones": cip["depth"],
            "train_neg": neg_c,
            "train_pos": pos_c,
            "soporte_neg": neg_c / n_neg if n_neg else 0.0,
            "soporte_pos": pos_c / n_pos if n_pos else 0.0,
            "confianza_train": cip["confidence"],
            "confianza_balanceada": cip["balanced_confidence"],
            "lift": cip["lift"],
            "test_cubiertas": int(covered.sum()),
            "test_neg": int(np.sum(yt == 0)),
            "test_pos": int(np.sum(yt == 1)),
            "confianza_test": float(np.mean(yt == cls)) if len(yt) else np.nan,
        })
    df = pd.DataFrame(rows)
    if len(df):
        df["caida_confianza"] = df["confianza_train"] - df["confianza_test"]
        df = df.sort_values(["clase", "lift"], ascending=[True, False])
    return df


def leaf_table(model, X_test, y_test, y_pred, p_test):
    paths = model.leaf_paths()
    leaf_ids = model.apply(X_test)
    rows = []
    for leaf in model._get_leaves(model.root):
        mask = leaf_ids == leaf.node_id
        yt, yp = y_test[mask], y_pred[mask]
        neg, pos = map(int, leaf.class_counts)
        rows.append({
            "hoja": leaf.node_id,
            "patron": model._path_to_human(paths[leaf.node_id]) or "(raíz)",
            "train_neg": neg,
            "train_pos": pos,
            "prop_pos_train": pos / max(neg + pos, 1),
            "modelo_logistico_de": getattr(leaf, "leaf_lr_source", None),
            "test_n": int(mask.sum()),
            "test_neg": int(np.sum(yt == 0)),
            "test_pos": int(np.sum(yt == 1)),
            "prob_media_test": float(p_test[mask].mean()) if mask.any() else np.nan,
            "FP": int(np.sum((yt == 0) & (yp == 1))),
            "FN": int(np.sum((yt == 1) & (yp == 0))),
        })
    return pd.DataFrame(rows).sort_values("FP", ascending=False)


def leaf_models_table(model, top):
    if model.leaf_model != "logistic":
        return pd.DataFrame()
    seen, rows = set(), []
    for leaf in model._get_leaves(model.root):
        src = getattr(leaf, "leaf_lr_source", None)
        if src is None or src in seen:
            continue
        seen.add(src)
        hojas = [l.node_id for l in model._get_leaves(model.root)
                 if getattr(l, "leaf_lr_source", None) == src]
        coefs = model.leaf_model_coefficients(leaf, top=top)
        for _, r in coefs.iterrows():
            rows.append({"modelo_de_nodo": src,
                         "usado_por_hojas": ", ".join(map(str, hojas)),
                         "variable": r["variable"],
                         "coeficiente": r["coeficiente"],
                         "odds_ratio": r["odds_ratio"]})
    return pd.DataFrame(rows)


def instance_table(model, ids, y_test, y_pred, p_test, X_test, cips):
    leaf_ids = model.apply(X_test)
    X_arr = model._to_array(X_test)
    kind = np.where(y_test == 1, np.where(y_pred == 1, "TP", "FN"),
                    np.where(y_pred == 1, "FP", "TN"))
    n_cov = [sum(model._match_path(x, c["path"]) for c in cips) for x in X_arr]
    return pd.DataFrame({"id_original": ids, "real": y_test, "prediccion": y_pred,
                         "probabilidad": p_test, "tipo": kind, "hoja": leaf_ids,
                         "patrones_que_la_cubren": n_cov})


# -------------------------------------------------------------------------
def save_excel(path, sheets, metrics, cm, config, partition):
    from openpyxl import Workbook
    from openpyxl.styles import Font

    wb = Workbook()
    ws = wb.active
    ws.title = "Resultado"
    ws["A1"] = f"Resultado final en TEST: {config['dataset']}"
    ws["A1"].font = Font(name=FONT, bold=True, size=14)
    ws["A2"] = (f"Estrategia de umbral: {config['strategy']} | umbral = "
                f"{config['umbral']:.4f} (elegido por CV interna) | remuestreo: "
                f"{config['remuestreo']}")
    ws["A2"].font = Font(name=FONT, italic=True, color="595959")

    # matriz de confusión
    ws["A4"] = "Matriz de confusión"
    ws["A4"].font = Font(name=FONT, bold=True, size=12)
    for j, h in enumerate(["", "Pred. 0", "Pred. 1"], start=1):
        _style_header(ws.cell(row=5, column=j, value=h or "Real"))
    for i, lab in enumerate(["Real 0", "Real 1"]):
        _style_header(ws.cell(row=6 + i, column=1, value=lab))
        for j in range(2):
            c = ws.cell(row=6 + i, column=2 + j, value=int(cm[i, j]))
            _style_body(c)
    # métricas: se calculan con fórmulas a partir de la matriz (B6:C7)
    ws["A10"] = "Métricas"
    ws["A10"].font = Font(name=FONT, bold=True, size=12)
    TN, FP, FN, TP = "$B$6", "$C$6", "$B$7", "$C$7"
    formulas = [
        ("Accuracy", f"=({TP}+{TN})/({TP}+{TN}+{FP}+{FN})"),
        ("Precision (ponderada)", metrics["precision"]),
        ("Recall (ponderado)", metrics["recall"]),
        ("F1-score (ponderado)", metrics["f1_score"]),
        ("MCC", f"=IFERROR(({TP}*{TN}-{FP}*{FN})/SQRT(({TP}+{FP})*({TP}+{FN})"
                f"*({TN}+{FP})*({TN}+{FN})),0)"),
        ("Error tipo I (FPR)", f"=IFERROR({FP}/({TN}+{FP}),0)"),
        ("Error tipo II (FNR)", f"=IFERROR({FN}/({TP}+{FN}),0)"),
        ("Balanced Error Rate", "=(B17+B18)/2"),
        ("Balanced accuracy", "=1-B19"),
        ("AUC", metrics["auc"]),
    ]
    _style_header(ws.cell(row=11, column=1, value="Métrica"))
    _style_header(ws.cell(row=11, column=2, value="Valor"))
    for i, (name, val) in enumerate(formulas, start=12):
        _style_body(ws.cell(row=i, column=1, value=name))
        _style_body(ws.cell(row=i, column=2, value=val), number=True)
    ws["A23"] = ("Precision/Recall/F1 y AUC se calcularon con ClassificationMetrics "
                 "(valores); el resto son fórmulas sobre la matriz de confusión.")
    ws["A23"].font = Font(name=FONT, italic=True, color="595959")
    ws.column_dimensions["A"].width = 26
    ws.column_dimensions["B"].width = 12
    ws.column_dimensions["C"].width = 12

    # partición
    wp = wb.create_sheet("Particion")
    _write_table(wp, partition, number_cols={"prop_positivos"})
    _autowidth(wp)

    num_cols_by_sheet = {
        "Seleccion_estrategia": {"error_tipo_I_cv", "error_tipo_II_cv", "ber_cv",
                                 "ber_cv_de", "max_error_cv", "max_error_cv_de"},
        "Umbrales_OOF": {"umbral", "error_tipo_I", "error_tipo_II", "ber"},
        "Compromiso_OOF": {"cota_error_tipo_II", "umbral", "error_tipo_I",
                           "error_tipo_II", "ber"},
        "Compromiso_OOF_errorI": {"cota_error_tipo_I", "umbral", "error_tipo_I",
                                  "error_tipo_II", "ber"},
        "Patrones": {"soporte_neg", "soporte_pos", "confianza_train",
                     "confianza_balanceada", "lift",
                     "confianza_test", "caida_confianza"},
        "Hojas": {"prop_pos_train", "prob_media_test"},
        "Modelos_hoja": {"coeficiente", "odds_ratio"},
        "Instancias_test": {"probabilidad"},
    }
    notes = {
        "Patrones": "clase = la que más se enriquece respecto a su proporción "
                    "global (no es remuestreo); "
                    "soporte_* = cobertura relativa a su clase en train; "
                    "lift = proporción de su clase / proporción global; "
                    "confianza_balanceada corrige por el prior; "
                    "confianza_test baja respecto a train = patrón inestable.",
        "Hojas": "Ordenadas por falsos positivos en test (error tipo I).",
        "Seleccion_estrategia": "Sólo con --strategy auto: cada estrategia elige su "
                                "umbral en 4/5 de las predicciones fuera de fold y "
                                "se evalúa en el 1/5 restante (10 repeticiones). "
                                "Se elige la de menor error medio. No usa el test.",
        "Compromiso_OOF_errorI": "Para cada cota de error tipo I, el menor error "
                                 "tipo II alcanzable (fuera de fold, sin test).",
        "Compromiso_OOF": "Para cada cota de error tipo II, el menor error tipo I "
                          "alcanzable con las predicciones fuera de fold del "
                          "entrenamiento (no usa el test).",
        "Modelos_hoja": "Coeficientes con variables numéricas estandarizadas; "
                        "odds_ratio > 1 aumenta la probabilidad de la clase 1.",
        "Umbrales_OOF": "Calculado con predicciones fuera de fold del 80 % de "
                        "entrenamiento (sin mirar el test). youden = mínimo BER.",
    }
    for name, df in sheets.items():
        w = wb.create_sheet(name)
        start = 1
        if name in notes:
            w["A1"] = notes[name]
            w["A1"].font = Font(name=FONT, italic=True, color="595959")
            start = 3
        if df is None or df.empty:
            w.cell(row=start, column=1, value="(sin datos)")
            continue
        _write_table(w, df, start_row=start,
                     number_cols=num_cols_by_sheet.get(name, set()))
        w.freeze_panes = w.cell(row=start + 1, column=1)
        _autowidth(w, max_width=70)

    wc = wb.create_sheet("Configuracion")
    dfc = pd.DataFrame({"parametro": list(config.keys()),
                        "valor": [str(v) for v in config.values()]})
    _write_table(wc, dfc)
    r = len(dfc) + 3
    dfd = pd.DataFrame({"metrica": list(METRIC_NOTES), "definicion":
                        list(METRIC_NOTES.values())})
    _write_table(wc, dfd, start_row=r)
    wc.column_dimensions["A"].width = 24
    wc.column_dimensions["B"].width = 80
    wb.save(path)


# -------------------------------------------------------------------------
def main(args):
    parts, cat_idx = split_data(args)
    X_tr, y_tr, id_tr = parts["train"]
    X_te, y_te, id_te = parts["test"]

    sampler = None
    resampling = "ninguno"
    if args.smote:
        sampler = ProccessData.build_sampler(cat_idx, args.seed)
        resampling = (f"SMOTENC ({len(cat_idx)} columnas categóricas)"
                      if cat_idx else "SMOTE (sin columnas categóricas)")

    args.tree_params = dict(
        max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf,
        criterion=args.criterion, alpha=args.alpha, beta=args.beta,
        random_state=args.seed, laplace=args.laplace, smoothing=args.smoothing,
        leaf_model=args.leaf_model, leaf_C=args.leaf_C,
        leaf_min_samples=args.leaf_min_samples,
        leaf_penalty=args.leaf_penalty,
        leaf_shrinkage=args.leaf_shrinkage)

    print(f"Partición: entrenamiento={len(y_tr)} ({np.mean(y_tr):.0%} positivos) | "
          f"test={len(y_te)} ({np.mean(y_te):.0%} positivos)")
    print(f"Remuestreo: {resampling}")

    # a) umbral con validación cruzada interna sobre el 80 %
    print(f"\nValidación cruzada interna ({args.inner_folds} folds) para el umbral:")
    oof = oof_probabilities(X_tr, y_tr, args, sampler)
    prior = float(np.mean(y_tr))
    fnr_cap = resolve_fnr_max(args.fnr_max, y_tr, oof)
    fpr_cap = resolve_fpr_max(args.fpr_max, y_tr, oof)
    print(f"\nCotas: error tipo II <= {fnr_cap:.4f}"
          f"{' (auto)' if str(args.fnr_max).lower() == 'auto' else ''} | "
          f"error tipo I <= {fpr_cap:.4f}"
          f"{' (auto)' if str(args.fpr_max).lower() == 'auto' else ''}"
          "  [sólo las usan np_fnr, np_fpr y constrained]")
    chosen = args.strategy
    df_sel = pd.DataFrame()
    if args.strategy == "auto":
        cands = [c.strip() for c in args.auto_candidates.split(",") if c.strip()]
        bad = [c for c in cands if c not in STRATEGIES or c == "auto"]
        if bad:
            raise ValueError(f"Candidatas no válidas: {bad}")
        chosen, df_sel = select_strategy_cv(y_tr, oof, prior, cands, fnr_cap,
                                            fpr_cap, metric=args.auto_metric,
                                            seed=args.seed)
        print(f"\nSelección automática de estrategia (sólo entrenamiento; "
              f"CV {5}x{10} sobre las predicciones fuera de fold, criterio "
              f"{args.auto_metric}):")
        print(df_sel.round(4).to_string(index=False))
        print(f"-> estrategia elegida: {chosen}")

    oof_rows = []
    for st in [x for x in STRATEGIES if x != "auto"]:
        thr_st = threshold_for(st, y_tr, oof, prior, fnr_cap, fpr_cap)
        pv = (oof >= thr_st).astype(int)
        tn, fp, fn, tp = CM.confusion_matrix(y_tr, pv).ravel()
        oof_rows.append({"estrategia": st, "umbral": thr_st,
                         "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
                         "error_tipo_I": CM.errorrateI(y_tr, pv),
                         "error_tipo_II": CM.errorrateII(y_tr, pv),
                         "ber": CM.balanced_error_rate(y_tr, pv),
                         "elegida": "sí" if st == chosen else ""})
    thr = threshold_for(chosen, y_tr, oof, prior, fnr_cap, fpr_cap)
    strategy_label = (f"auto -> {chosen}" if args.strategy == "auto" else chosen)
    if chosen == "constrained":
        feasible = constrained_threshold(y_tr, oof, fpr_cap, fnr_cap)[1]
        if not feasible:
            print("\nAVISO: ningún umbral cumple ambas cotas fuera de fold; se usa "
                  "el que menos se pasa. Revise las tablas de compromiso.")
    oof_auc = CM.auc(y_tr, oof)

    print(f"\nDecisión del umbral con el ENTRENAMIENTO ({len(y_tr)} instancias, "
          "predicciones fuera de fold; el test no se usa):")
    print(pd.DataFrame(oof_rows).round(4).to_string(index=False))

    targets = sorted(set([0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30]
                         + [round(fnr_cap, 4)]))
    df_trade = tradeoff_table(y_tr, oof, targets)
    print(f"\nCompromiso A: cota de error tipo II -> menor error tipo I "
          f"(fuera de fold, AUC={oof_auc:.4f}):")
    print(df_trade.round(4).to_string(index=False))

    targets_i = sorted(set([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
                           + [round(fpr_cap, 4)]))
    rows_i = []
    for t in targets_i:
        thr_i, info = select_threshold(y_tr, oof, fpr_max=t)
        rows_i.append({"cota_error_tipo_I": t, "umbral": thr_i,
                       "error_tipo_I": info["fpr_val"],
                       "error_tipo_II": info["fnr_val"], "ber": info["ber_val"]})
    df_trade_i = pd.DataFrame(rows_i)
    print("\nCompromiso B: cota de error tipo I -> menor error tipo II:")
    print(df_trade_i.round(4).to_string(index=False))

    # b) modelo final con todo el entrenamiento
    model, leaves_before = build_model(X_tr, y_tr, args, sampler, args.seed)
    model.threshold_ = thr
    n_leaves = len(model._get_leaves(model.root))

    # c) evaluación única en test
    p_te = model.predict_proba(X_te)
    y_pred = (p_te >= thr).astype(int)

    print(f"\nModelo final: hojas={n_leaves} (antes de podar: {leaves_before}) "
          f"| umbral={thr:.4f} ({strategy_label}) | AUC fuera de fold={oof_auc:.4f}")
    metrics, cm = evaluation_block("RESULTADOS FINALES: PBCRETree (test)",
                                   y_te, y_pred, p_te)

    # patrones (CIPs)
    cips = model.extract_cips(min_samples=args.min_samples_cip,
                              min_confidence=args.min_confidence,
                              leaves_only=args.leaves_only,
                              class_mode="balanced")
    pattern_note = ""
    if not cips and n_leaves > 1:
        # sin patrones con los filtros: se muestran los caminos de las hojas
        cips = model.extract_cips(min_samples=1, min_confidence=0.0,
                                  leaves_only=True, class_mode="balanced")
        pattern_note = (f"Ningún patrón superó min_confidence={args.min_confidence};"
                        " se muestran los caminos de todas las hojas sin filtrar.")
    df_pat = pattern_table(model, cips, X_te, y_te, y_tr)
    df_leaf = leaf_table(model, X_te, y_te, y_pred, p_te)
    df_lr = leaf_models_table(model, args.top_coef)
    df_inst = instance_table(model, id_te, y_te, y_pred, p_te, X_te, cips)

    print(f"\n{LINE}\nPATRONES: {len(df_pat)} CIPs "
          f"(clase del patrón = la que más se enriquece respecto a su proporción "
          f"global; no es remuestreo)\n{LINE}")
    if pattern_note:
        print(pattern_note)
    if len(df_pat):
        show = df_pat[["clase", "patron", "confianza_train", "lift",
                       "test_cubiertas", "confianza_test"]].head(args.show_patterns)
        with pd.option_context("display.max_colwidth", 90, "display.width", 200):
            print(show.round(3).to_string(index=False))
    else:
        print("El árbol quedó con una sola hoja: no hay patrones. Use --min_leaves "
              "mayor, --no_prune o --leaf_model constant.")

    print(f"\n{LINE}\nHOJAS CON MÁS FALSOS POSITIVOS EN TEST\n{LINE}")
    with pd.option_context("display.max_colwidth", 90, "display.width", 200):
        print(df_leaf[["hoja", "patron", "test_n", "FP", "FN",
                       "prob_media_test"]].head(5).round(3).to_string(index=False))

    # tabla de partición
    n_grow = int(round(len(y_tr) * (1 - args.prune_frac)))
    partition = pd.DataFrame([
        {"conjunto": "entrenamiento (80 %)", "instancias": len(y_tr),
         "negativos": int(np.sum(y_tr == 0)), "positivos": int(np.sum(y_tr == 1)),
         "prop_positivos": float(np.mean(y_tr)),
         "uso": f"umbral por CV interna ({args.inner_folds} folds); "
                "hojas finales reestimadas con todo este conjunto"},
        {"conjunto": "  crecimiento", "instancias": n_grow, "negativos": None,
         "positivos": None, "prop_positivos": None,
         "uso": f"{1 - args.prune_frac:.0%} del entrenamiento: crecer el árbol"},
        {"conjunto": "  poda", "instancias": len(y_tr) - n_grow, "negativos": None,
         "positivos": None, "prop_positivos": None,
         "uso": f"{args.prune_frac:.0%} del entrenamiento: podar"},
        {"conjunto": "test (20 %)", "instancias": len(y_te),
         "negativos": int(np.sum(y_te == 0)), "positivos": int(np.sum(y_te == 1)),
         "prop_positivos": float(np.mean(y_te)), "uso": "evaluación única"},
    ])

    os.makedirs(args.out, exist_ok=True)
    pdir = os.path.join(args.out, "particion")
    os.makedirs(pdir, exist_ok=True)
    for n, (X, y, ids) in parts.items():
        out = X.copy()
        out.insert(0, "id_original", ids)
        out["target"] = y
        out.to_csv(os.path.join(pdir, f"{n}.csv"), index=False)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    xlsx = os.path.join(args.out, f"{args.dataset}_final_{stamp}.xlsx")
    config = {"dataset": args.dataset, "strategy": strategy_label, "umbral": thr,
              "remuestreo": resampling, "auc_fuera_de_fold": oof_auc,
              "hojas": n_leaves, "hojas_antes_poda": leaves_before,
              "nota_patrones": pattern_note or "-"}
    config.update({k: v for k, v in vars(args).items() if k != "tree_params"})
    config["fecha"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sheets = {"Seleccion_estrategia": df_sel,
              "Umbrales_OOF": pd.DataFrame(oof_rows),
              "Compromiso_OOF": df_trade, "Compromiso_OOF_errorI": df_trade_i,
              "Patrones": df_pat,
              "Hojas": df_leaf, "Modelos_hoja": df_lr, "Instancias_test": df_inst}
    try:
        save_excel(xlsx, sheets, metrics, cm, config, partition)
        print(f"\nExcel: {os.path.abspath(xlsx)}")
    except ImportError:
        print("\nInstale openpyxl para generar el Excel (pip install openpyxl).")
    print(f"Partición: {os.path.abspath(pdir)}")

    # fila en la tabla comparativa de trabajos
    if not args.no_summary:
        if args.summary_file is None:
            args.summary_file = os.path.join(args.out, "comparacion_trabajos.xlsx")
        from sklearn.metrics import precision_score, recall_score, f1_score
        avg = "binary" if args.summary_average == "binary" else "weighted"
        kw = {"pos_label": 1} if avg == "binary" else {}
        row = {
            "trabajo": args.trabajo,
            "dataset": args.dataset,
            "Accuracy": metrics["accuracy"],
            "Precision": round(precision_score(y_te, y_pred, average=avg,
                                               zero_division=0, **kw), 4),
            "Recall": round(recall_score(y_te, y_pred, average=avg,
                                         zero_division=0, **kw), 4),
            "F1-Score": round(f1_score(y_te, y_pred, average=avg,
                                       zero_division=0, **kw), 4),
            "AUC": metrics["auc"],
            "MCC": metrics["mcc"],
            "ErrorTipoI": metrics["error_tipo_I"],
            "ErrorTipoII": metrics["error_tipo_II"],
            "BER": metrics["ber"],
            "TipoTrabajo": args.tipo_trabajo,
        }
        try:
            from srcc.pbcre.evaluation.excel_report import upsert_summary_row
            accion = upsert_summary_row(args.summary_file, row)
            print(f"Tabla comparativa ({accion}): {os.path.abspath(args.summary_file)}")
        except ImportError:
            print("Instale openpyxl para actualizar la tabla comparativa.")

    if args.print_tree:
        print(f"\n{LINE}\nÁRBOL\n{LINE}")
        model.print_tree()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--strategy", default="youden", choices=STRATEGIES,
                    help="youden = mínimo BER (automático); closest = mínima "
                         "distancia al clasificador perfecto (automático); "
                         "np_fnr = mínimo error I con error II <= --fnr_max")
    ap.add_argument("--fnr_max", default="auto",
                    help="cota de error tipo II para np_fnr/constrained: un número (p. ej. "
                         "0.157) o 'auto' (error II del umbral 0.5)")
    ap.add_argument("--auto_candidates", default="youden,minimax,np_fnr",
                    help="estrategias que compara --strategy auto")
    ap.add_argument("--auto_metric", default="ber", choices=["ber", "max_error"],
                    help="criterio de --strategy auto: BER o el mayor de los errores")
    ap.add_argument("--fpr_max", default="auto",
                    help="cota de error tipo I para np_fpr/constrained: un número "
                         "(p. ej. 0.30) o 'auto' (error I del umbral 0.5)")
    ap.add_argument("--trabajo", default="Our Proposal V2",
                    help="nombre del trabajo en la tabla comparativa")
    ap.add_argument("--tipo_trabajo", default="Propuesta")
    ap.add_argument("--summary_file", default=None,
                    help="por defecto <out>/comparacion_trabajos.xlsx")
    ap.add_argument("--summary_average", default="binary",
                    choices=["binary", "weighted"],
                    help="Precision/Recall/F1 de la clase positiva (binary) o "
                         "ponderados (weighted)")
    ap.add_argument("--no_summary", action="store_true")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--inner_folds", type=int, default=5,
                    help="folds de la CV interna para elegir el umbral")
    ap.add_argument("--prune_frac", type=float, default=0.2,
                    help="fracción del entrenamiento usada para podar")
    ap.add_argument("--min_leaves", type=int, default=6,
                    help="la poda no deja menos hojas que esto")
    ap.add_argument("--criterion", default="hybrid_fast")
    ap.add_argument("--max_depth", type=int, default=5)
    ap.add_argument("--min_samples_leaf", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--leaf_model", default="logistic",
                    choices=["constant", "logistic"])
    ap.add_argument("--leaf_C", type=float, default=0.1)
    ap.add_argument("--leaf_min_samples", type=int, default=30)
    ap.add_argument("--leaf_penalty", default="l2", choices=["l1", "l2"])
    ap.add_argument("--leaf_shrinkage", type=float, default=600.0,
                    help="k: encoge cada hoja hacia el modelo raíz; w = n/(n+k) (0 = apagado)")
    ap.add_argument("--smoothing", default="laplace", choices=["laplace", "prior"])
    ap.add_argument("--laplace", type=float, default=1.0)
    ap.add_argument("--smote", action="store_true",
                    help="SMOTENC/SMOTE sólo sobre los datos de entrenamiento")
    ap.add_argument("--no_prune", action="store_true")
    ap.add_argument("--prune_metric", default="balanced", choices=["auc", "balanced"])
    ap.add_argument("--min_samples_cip", type=int, default=5)
    ap.add_argument("--min_confidence", type=float, default=0.55,
                    help="confianza balanceada mínima de un patrón")
    ap.add_argument("--leaves_only", action="store_true")
    ap.add_argument("--top_coef", type=int, default=8)
    ap.add_argument("--show_patterns", type=int, default=15)
    ap.add_argument("--print_tree", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results_final")
    main(ap.parse_args())
