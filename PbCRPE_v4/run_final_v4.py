"""
PbCRPE v4 - Resultado FINAL con patrones minados de VARIOS árboles.

Protocolo (igual que la v3):
  1. Partición estratificada fija 80/20 (misma semilla -> mismo test que la v3).
  2. Dentro del 80 %: CV interna de 5 folds para obtener probabilidades
     fuera de fold y elegir el umbral (--strategy, incluida "auto").
  3. Modelo final con todo el 80 %; una única evaluación en test.

Bosque de minado (srcc/pbcre/ensemble/forest_miner.py):
  --n_trees árboles, cada uno con bootstrap estratificado de filas y un
  subconjunto aleatorio de variables (--feature_fraction), sin poda.
  Los patrones de todos los nodos se agrupan, se eliminan duplicados y se
  recalculan con todo el entrenamiento. Columna "arboles" = en cuántos
  árboles apareció (estabilidad).

Predicción (--predictor):
  pesos     cada patrón es una variable binaria de una logística L1 (por defecto)
  votacion  votación por soporte, estilo PBC4cip
  arbol_v3  predice con el árbol de la v3; el bosque sólo mina patrones

Uso:
  python run_final_v4.py --dataset german --strategy auto --fnr_max 0.157
  python run_final_v4.py --dataset german --predictor votacion
  python run_final_v4.py --dataset german --predictor arbol_v3
"""

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_final as v3
from srcc.pbcre.ensemble.forest_miner import PBCREPatternForest
from srcc.pbcre.evaluation.metrics import ClassificationMetrics as CM

LINE = "=" * 75


def make_forest(args, seed):
    return PBCREPatternForest(
        n_trees=args.n_trees, max_depth=args.mining_depth,
        min_samples_leaf=args.mining_min_leaf, criterion=args.mining_criterion,
        feature_fraction=args.feature_fraction, bootstrap=not args.no_bootstrap,
        min_samples_cip=args.min_samples_cip, min_confidence=args.min_confidence,
        min_lift=args.min_lift, max_q=args.max_q,
        predictor="votacion" if args.predictor == "arbol_v3" else args.predictor,
        random_state=seed, n_jobs=args.n_jobs)


def fit_predictor(X, y, args, seed):
    """Devuelve (objeto_que_predice, bosque)."""
    forest = make_forest(args, seed).fit(X, y)
    if args.predictor == "arbol_v3":
        model, _ = v3.build_model(X, y, args, None, seed)
        return model, forest
    return forest, forest


def main(args):
    t0 = time.time()
    parts, _ = v3.split_data(args)
    X_tr, y_tr, _ = parts["train"]
    X_te, y_te, id_te = parts["test"]
    args.tree_params = dict(
        max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf,
        criterion=args.criterion, alpha=args.alpha, beta=args.beta,
        random_state=args.seed, leaf_model="logistic", leaf_C=args.leaf_C,
        leaf_min_samples=30, leaf_shrinkage=args.leaf_shrinkage)

    print(f"Partición: entrenamiento={len(y_tr)} ({np.mean(y_tr):.0%} positivos) | "
          f"test={len(y_te)} ({np.mean(y_te):.0%} positivos)")
    print(f"Bosque de minado: {args.n_trees} árboles, criterio {args.mining_criterion}, "
          f"profundidad {args.mining_depth}, {args.feature_fraction:.0%} de variables "
          f"por árbol | predictor: {args.predictor}")

    # a) probabilidades fuera de fold para el umbral
    print(f"\nValidación cruzada interna ({args.inner_folds} folds):")
    oof = np.zeros(len(y_tr))
    skf = StratifiedKFold(args.inner_folds, shuffle=True, random_state=args.seed)
    for k, (tr, va) in enumerate(skf.split(X_tr, y_tr)):
        pred, _ = fit_predictor(X_tr.iloc[tr].reset_index(drop=True), y_tr[tr],
                                args, args.seed + k)
        oof[va] = pred.predict_proba(X_tr.iloc[va].reset_index(drop=True))
        print(f"  fold interno {k + 1}/{args.inner_folds} listo ({time.time() - t0:.0f}s)")

    prior = float(np.mean(y_tr))
    fnr_cap = v3.resolve_fnr_max(args.fnr_max, y_tr, oof)
    fpr_cap = v3.resolve_fpr_max(args.fpr_max, y_tr, oof)
    chosen, df_sel = args.strategy, pd.DataFrame()
    if args.strategy == "auto":
        cands = [c.strip() for c in args.auto_candidates.split(",") if c.strip()]
        chosen, df_sel = v3.select_strategy_cv(y_tr, oof, prior, cands, fnr_cap,
                                               fpr_cap, metric=args.auto_metric,
                                               seed=args.seed)
        print("\nSelección automática de estrategia (sólo entrenamiento):")
        print(df_sel.round(4).to_string(index=False))
        print(f"-> estrategia elegida: {chosen}")
    thr = v3.threshold_for(chosen, y_tr, oof, prior, fnr_cap, fpr_cap)
    label = f"auto -> {chosen}" if args.strategy == "auto" else chosen

    oof_rows = []
    for st in [s for s in v3.STRATEGIES if s != "auto"]:
        t = v3.threshold_for(st, y_tr, oof, prior, fnr_cap, fpr_cap)
        pv = (oof >= t).astype(int)
        tn, fp, fn, tp = CM.confusion_matrix(y_tr, pv).ravel()
        oof_rows.append({"estrategia": st, "umbral": round(t, 4), "TN": int(tn),
                         "FP": int(fp), "FN": int(fn), "TP": int(tp),
                         "error_tipo_I": CM.errorrateI(y_tr, pv),
                         "error_tipo_II": CM.errorrateII(y_tr, pv),
                         "ber": CM.balanced_error_rate(y_tr, pv),
                         "elegida": "sí" if st == chosen else ""})
    oof_auc = CM.auc(y_tr, oof)

    # b) modelo final con todo el entrenamiento
    pred, forest = fit_predictor(X_tr, y_tr, args, args.seed)
    p_te = pred.predict_proba(X_te)
    y_pred = (p_te >= thr).astype(int)

    print(f"\nModelo final: {forest.n_candidates_} patrones candidatos -> "
          f"{len(forest.patterns_)} tras filtros"
          + (f" -> {int((np.abs(forest.patterns_['peso']) > 1e-8).sum())} con peso"
             if "peso" in forest.patterns_ else "")
          + f" | umbral={thr:.4f} ({label}) | AUC fuera de fold={oof_auc:.4f}")
    metrics, cm = v3.evaluation_block("RESULTADOS FINALES: PbCRPE v4 (test)",
                                      y_te, y_pred, p_te)

    # patrones
    df_pat = forest.patterns_frame(X_te, y_te)
    if "peso" in df_pat:
        df_show = df_pat[np.abs(df_pat["peso"]) > 1e-8]
        cols = ["clase", "peso", "patron", "soporte", "lift", "arboles",
                "test_cubiertas", "confianza_test"]
        title = "PATRONES CON MAYOR PESO"
    else:
        df_show = df_pat.sort_values(["arboles", "soporte"], ascending=False)
        cols = ["clase", "patron", "soporte", "lift", "arboles",
                "test_cubiertas", "confianza_test"]
        title = "PATRONES MÁS ESTABLES (aparecen en más árboles)"
    print(f"\n{LINE}\n{title} ({len(df_pat)} patrones en total)\n{LINE}")
    with pd.option_context("display.max_colwidth", 90, "display.width", 220):
        print(df_show[cols].head(args.show_patterns).round(3).to_string(index=False))

    # salidas
    kind = np.where(y_te == 1, np.where(y_pred == 1, "TP", "FN"),
                    np.where(y_pred == 1, "FP", "TN"))
    M = forest.coverage_matrix(X_te)
    df_inst = pd.DataFrame({"id_original": id_te, "real": y_te, "prediccion": y_pred,
                            "probabilidad": np.round(p_te, 4), "tipo": kind,
                            "patrones_que_la_cubren": M.sum(1)})
    partition = pd.DataFrame([
        {"conjunto": "entrenamiento (80 %)", "instancias": len(y_tr),
         "negativos": int(np.sum(y_tr == 0)), "positivos": int(np.sum(y_tr == 1)),
         "prop_positivos": float(np.mean(y_tr)),
         "uso": f"bosque de minado y umbral por CV interna ({args.inner_folds} folds)"},
        {"conjunto": "test (20 %)", "instancias": len(y_te),
         "negativos": int(np.sum(y_te == 0)), "positivos": int(np.sum(y_te == 1)),
         "prop_positivos": float(np.mean(y_te)), "uso": "evaluación única"}])

    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    xlsx = os.path.join(args.out, f"{args.dataset}_v4_{stamp}.xlsx")
    config = {"dataset": args.dataset, "strategy": label, "umbral": thr,
              "remuestreo": "ninguno", "auc_fuera_de_fold": oof_auc,
              "patrones_candidatos": forest.n_candidates_,
              "patrones_filtrados": len(forest.patterns_)}
    config.update({k: v for k, v in vars(args).items() if k != "tree_params"})
    config["fecha"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sheets = {"Seleccion_estrategia": df_sel, "Umbrales_OOF": pd.DataFrame(oof_rows),
              "Patrones": df_pat.round(4), "Instancias_test": df_inst}
    try:
        v3.save_excel(xlsx, sheets, metrics, cm, config, partition)
        print(f"\nExcel: {os.path.abspath(xlsx)}")
    except ImportError:
        print("\nInstale openpyxl para generar el Excel.")

    if not args.no_summary:
        from sklearn.metrics import precision_score, recall_score, f1_score
        from srcc.pbcre.evaluation.excel_report import upsert_summary_row
        avg = args.summary_average
        kw = {"pos_label": 1} if avg == "binary" else {}
        row = {"trabajo": args.trabajo, "dataset": args.dataset,
               "Accuracy": metrics["accuracy"],
               "Precision": round(precision_score(y_te, y_pred, average=avg, zero_division=0, **kw), 4),
               "Recall": round(recall_score(y_te, y_pred, average=avg, zero_division=0, **kw), 4),
               "F1-Score": round(f1_score(y_te, y_pred, average=avg, zero_division=0, **kw), 4),
               "AUC": metrics["auc"], "MCC": metrics["mcc"],
               "ErrorTipoI": metrics["error_tipo_I"],
               "ErrorTipoII": metrics["error_tipo_II"], "BER": metrics["ber"],
               "TipoTrabajo": args.tipo_trabajo}
        path = args.summary_file or os.path.join(args.out, "comparacion_trabajos.xlsx")
        try:
            accion = upsert_summary_row(path, row)
            print(f"Tabla comparativa ({accion}): {os.path.abspath(path)}")
        except ImportError:
            print("Instale openpyxl para actualizar la tabla comparativa.")
    print(f"Tiempo total: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--predictor", default="pesos",
                    choices=["pesos", "votacion", "arbol_v3"])
    # bosque de minado
    ap.add_argument("--n_trees", type=int, default=30)
    ap.add_argument("--mining_depth", type=int, default=4)
    ap.add_argument("--mining_min_leaf", type=int, default=10)
    ap.add_argument("--mining_criterion", default="twoing",
                    choices=["twoing", "hybrid_fast"],
                    help="twoing: rápido (0.15 s/árbol); hybrid_fast: ~3 s/árbol")
    ap.add_argument("--feature_fraction", type=float, default=0.6)
    ap.add_argument("--no_bootstrap", action="store_true")
    ap.add_argument("--min_samples_cip", type=int, default=10)
    ap.add_argument("--min_confidence", type=float, default=0.55)
    ap.add_argument("--min_lift", type=float, default=1.1)
    ap.add_argument("--max_q", type=float, default=None,
                    help="q-valor máximo (Fisher + Benjamini-Hochberg); None = sin filtro")
    ap.add_argument("--n_jobs", type=int, default=1,
                    help="árboles en paralelo (-1 = todos los núcleos)")
    # umbral
    ap.add_argument("--strategy", default="auto", choices=v3.STRATEGIES)
    ap.add_argument("--auto_candidates", default="youden,minimax,np_fnr")
    ap.add_argument("--auto_metric", default="ber", choices=["ber", "max_error"])
    ap.add_argument("--fnr_max", default="auto")
    ap.add_argument("--fpr_max", default="auto")
    ap.add_argument("--inner_folds", type=int, default=5)
    ap.add_argument("--test_size", type=float, default=0.2)
    # árbol de la v3 (sólo --predictor arbol_v3)
    ap.add_argument("--criterion", default="hybrid_fast")
    ap.add_argument("--max_depth", type=int, default=5)
    ap.add_argument("--min_samples_leaf", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--leaf_C", type=float, default=0.1)
    ap.add_argument("--leaf_shrinkage", type=float, default=600.0)
    ap.add_argument("--prune_frac", type=float, default=0.2)
    ap.add_argument("--min_leaves", type=int, default=6)
    ap.add_argument("--prune_metric", default="balanced")
    ap.add_argument("--no_prune", action="store_true")
    # salidas
    ap.add_argument("--show_patterns", type=int, default=15)
    ap.add_argument("--trabajo", default=None,
                    help="por defecto 'Our Proposal V4 (<predictor>)'")
    ap.add_argument("--tipo_trabajo", default="Propuesta")
    ap.add_argument("--summary_file", default=None)
    ap.add_argument("--summary_average", default="binary", choices=["binary", "weighted"])
    ap.add_argument("--no_summary", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results_final_v4")
    a = ap.parse_args()
    if a.trabajo is None:
        a.trabajo = f"Our Proposal V4 ({a.predictor})"
    main(a)
