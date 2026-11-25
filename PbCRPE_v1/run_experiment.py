"""
PbCRPE_v1 - Experimento de control de errores tipo I y tipo II.

Para cada fold externo:
  train  -> fit (75 %) + validación interna (25 %)
  fit    -> (opcional) SMOTE sólo aquí -> entrenar árbol
  val    -> poda + selección de umbral
  test   -> evaluación (nunca se usa para decidir nada)

Variantes comparadas en test (todas con el MISMO árbol del fold):
  base_0.5     umbral 0.5 (modelo original)
  prior_thr    umbral = proporción de positivos en entrenamiento
  youden       umbral que minimiza BER en validación
  np_fpr       minimiza FNR con FPR <= FPR(base) en validación  (baja tipo II)
  np_fnr       minimiza FPR con FNR <= --fnr_max en validación   (baja tipo I)
  cips_youden  CIPs balanceados + umbral de mínimo BER
  cips_np_fnr  CIPs balanceados + misma restricción que np_fnr

La columna auc indica la capacidad de ordenamiento del modelo: el umbral
sólo se mueve sobre la curva ROC; para bajar AMBOS errores hay que subir el AUC.

Salidas en --out (por defecto results/):
  resultados_folds.csv, resumen.csv, wilcoxon.csv y
  <dataset>_<criterio>_<hojas>_<fecha>.xlsx con Accuracy, Precision, Recall,
  F1, MCC, AUC, error tipo I, error tipo II, BER y balanced accuracy
  (hojas Resumen, Folds, Wilcoxon y Configuracion).

Uso:
  python run_experiment.py --dataset NOMBRE          (lee ./dataset/NOMBRE.csv)
  python run_experiment.py --synthetic               (demo con datos simulados)
  python run_experiment.py --dataset german --fnr_max 0.25 --smoothing prior
  python run_experiment.py --dataset german --leaf_model logistic
"""

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from srcc.pbcre.tree.tree import PBCRETree
from srcc.pbcre.evaluation.metrics import error_report
from srcc.pbcre.evaluation.excel_report import fold_metrics, save_excel_report
from srcc.pbcre.utils.ProccessData import ProccessData


def synthetic_data(n=1200, pos_rate=0.12, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < pos_rate).astype(int)
    x1 = rng.normal(0, 1, n) + 1.2 * y
    x2 = rng.normal(0, 1, n) - 0.8 * y
    x3 = rng.normal(0, 1, n)
    cat = np.where(rng.random(n) < 0.3 + 0.3 * y, "A",
                   np.where(rng.random(n) < 0.5, "B", "C"))
    X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "cat": cat})
    return X, pd.Series(y)


def load_data(args):
    if args.synthetic:
        X, y = synthetic_data()
        cat_idx = [X.columns.get_loc("cat")]
        return X, y, cat_idx
    path = os.path.join("dataset", args.dataset + ".csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No se encontró {os.path.abspath(path)}. Ejecute el script desde "
            f"la carpeta que contiene 'dataset/' o copie ahí el CSV.")
    pdata = ProccessData(args.dataset)
    pdata.ReadDataset()
    X, y = pdata.X.reset_index(drop=True), pdata.y.reset_index(drop=True)
    cat_idx = [X.columns.get_loc(c) for c in ProccessData.categorical_columns(X)]
    return X, y, cat_idx


def run(args):
    X, y, cat_idx = load_data(args)
    y_arr = np.asarray(y).astype(int)
    print(f"Instancias: {len(y_arr)} | positivos: {y_arr.mean():.3f}")

    sampler = ProccessData.build_sampler(cat_idx, args.seed) if args.smote else None

    tree_params = dict(
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        criterion=args.criterion,
        alpha=args.alpha,
        beta=args.beta,
        random_state=args.seed,
        laplace=args.laplace,
        smoothing=args.smoothing,
        leaf_model=args.leaf_model,
        leaf_C=args.leaf_C,
        leaf_min_samples=args.leaf_min_samples,
        leaf_shrinkage=args.leaf_shrinkage,
        scoring=args.split_scoring, logistic_C=args.logistic_C,
        split_standardize=args.split_standardize,
    )
    if args.criterion == "hybrid_fast":
        tree_params.update(purity_mode=args.purity_mode,
                           class_quantiles=not args.no_class_quantiles)

    rskf = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats,
                                   random_state=args.seed)
    rows = []
    t0 = time.time()

    for k, (tr, te) in enumerate(rskf.split(X, y_arr)):
        X_tr, y_tr = X.iloc[tr].reset_index(drop=True), y_arr[tr]
        X_te, y_te = X.iloc[te].reset_index(drop=True), y_arr[te]

        X_fit, X_val, y_fit, y_val = train_test_split(
            X_tr, y_tr, test_size=0.25, stratify=y_tr, random_state=args.seed)
        X_fit, X_val = X_fit.reset_index(drop=True), X_val.reset_index(drop=True)

        if sampler is not None:
            X_fit, y_fit = sampler(X_fit, y_fit)

        model = PBCRETree(**tree_params).fit(X_fit, y_fit)
        if not args.no_prune:
            model.cost_complexity_pruning(X_val, y_val, metric=args.prune_metric,
                                          min_leaves=args.min_leaves)

        p_te = model.predict_proba(X_te)
        n_leaves = len(model._get_leaves(model.root))

        def record(name, y_pred, thr, scores):
            r = error_report(y_te, y_pred)
            r.update(fold_metrics(y_te, y_pred, scores))
            r.update(variant=name, fold=k, repeat=k // args.folds + 1,
                     threshold=thr, leaves=n_leaves)
            rows.append(r)

        def tree_variant(name, thr):
            model.threshold_ = thr
            record(name, (p_te >= thr).astype(int), thr, p_te)

        # A) base: umbral 0.5
        tree_variant("base_0.5", 0.5)

        # B) umbral = prior de entrenamiento (equivale a decidir por lift)
        tree_variant("prior_thr", float(np.mean(y_fit)))

        # C) Youden (mínimo BER en validación)
        model.fit_threshold(X_val, y_val)
        tree_variant("youden", model.threshold_)

        # D) baja tipo II sin subir tipo I (respecto a la base)
        model.threshold_ = 0.5
        model.fit_threshold(X_val, y_val, fpr_max="current")
        tree_variant("np_fpr", model.threshold_)

        # E) baja tipo I con tipo II <= fnr_max
        model.fit_threshold(X_val, y_val, fnr_max=args.fnr_max)
        tree_variant("np_fnr", model.threshold_)

        # F, G) CIPs balanceados
        if not args.no_cips:
            cips = model.extract_cips(min_samples=5, min_confidence=0.6,
                                      class_mode="balanced")
            if cips:
                kw = dict(support_mode="class")
                s_te = model.cip_scores(X_te, cips, **kw)

                model.fit_cip_threshold(X_val, y_val, cips, **kw)
                record("cips_youden", (s_te >= model.cip_threshold_).astype(int),
                       model.cip_threshold_, s_te)

                model.fit_cip_threshold(X_val, y_val, cips,
                                        fnr_max=args.fnr_max, **kw)
                record("cips_np_fnr", (s_te >= model.cip_threshold_).astype(int),
                       model.cip_threshold_, s_te)

        print(f"fold {k + 1}/{args.folds * args.repeats} listo "
              f"({time.time() - t0:.1f}s)")

    df = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "resultados_folds.csv"), index=False)

    summary = (df.groupby("variant")[["fpr", "fnr", "ber", "auc"]]
                 .agg(["mean", "std"]).round(4))
    order = [v for v in ["base_0.5", "prior_thr", "youden", "np_fpr", "np_fnr",
                         "cips_youden", "cips_np_fnr"]
             if v in summary.index]
    summary = summary.loc[order]
    summary.to_csv(os.path.join(args.out, "resumen.csv"))

    print("\n=== Resumen en test (media y desviación por fold) ===")
    print("fpr = error tipo I | fnr = error tipo II | "
          f"np_fnr usa FNR <= {args.fnr_max}")
    print(summary.to_string())
    print(f"\nHojas promedio del árbol: {df['leaves'].mean():.1f}")

    # métricas detalladas (sin matriz: en CV no existe un único test)
    if args.show_variants != "none":
        show = order if args.show_variants == "all" else [
            v for v in args.show_variants.split(",") if v in order]
        labels = [("Accuracy", "accuracy"), ("Precision", "precision"),
                  ("Recall", "recall"), ("F1-score", "f1_score"), ("MCC", "mcc"),
                  ("Error tipo I", "error_tipo_I"),
                  ("Error tipo II", "error_tipo_II"),
                  ("Balanced Error Rate", "ber"), ("AUC", "auc")]
        print("\n=== Métricas por variante (media ± DE entre folds) ===")
        tab = pd.DataFrame({v: [f"{df[df.variant == v][k].mean():.4f} ± "
                                f"{df[df.variant == v][k].std():.4f}"
                                for _, k in labels] for v in show},
                           index=[l for l, _ in labels])
        print(tab.to_string())

    # pruebas pareadas contra la base
    base = df[df.variant == "base_0.5"].set_index("fold")
    print("\n=== Wilcoxon pareado vs base_0.5 ===")
    tests = []
    for v in order[1:]:
        other = df[df.variant == v].set_index("fold")
        common = base.index.intersection(other.index)
        for m in ("fnr", "fpr", "ber", "accuracy", "f1_score", "mcc", "auc"):
            d = other.loc[common, m].values - base.loc[common, m].values
            if np.allclose(d, 0):
                p = 1.0
            else:
                p = wilcoxon(d, zero_method="zsplit").pvalue
            tests.append({"variant": v, "metric": m,
                          "mean_diff": d.mean(), "p_value": p})
    tests = pd.DataFrame(tests)
    tests.to_csv(os.path.join(args.out, "wilcoxon.csv"), index=False)
    print(tests[tests.metric.isin(["fnr", "fpr", "ber"])].round(4)
          .to_string(index=False))
    print("\nmean_diff < 0 = menos error que la base en esa métrica.")

    # ---------------- Excel con todas las métricas ----------------
    if not args.no_excel:
        name = args.dataset if args.dataset else "synthetic"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        xlsx_path = os.path.join(
            args.out, f"{name}_{args.criterion}_{args.leaf_model}_{stamp}.xlsx")
        df_x = df.rename(columns={"repeat": "repeticion", "variant": "variante",
                                  "threshold": "umbral", "leaves": "hojas"})
        tests_x = tests.copy()
        tests_x["metric"] = tests_x["metric"].replace(
            {"fpr": "error_tipo_I", "fnr": "error_tipo_II"})
        config = {"dataset": name, "instancias": len(y_arr),
                  "proporcion_positivos": round(float(y_arr.mean()), 4)}
        config.update({k: v for k, v in vars(args).items()
                       if k not in ("dataset", "synthetic")})
        config["fecha"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            save_excel_report(df_x, tests_x, config, xlsx_path,
                              variant_order=order)
            print(f"Excel guardado en {os.path.abspath(xlsx_path)}")
        except ImportError as exc:
            print(f"No se generó el Excel: {exc}")
    print("Con pocos folds la prueba tiene poca potencia: use --repeats 3 o más.")
    print(f"\nArchivos guardados en {os.path.abspath(args.out)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", help="nombre del CSV en ./dataset (sin .csv)")
    src.add_argument("--synthetic", action="store_true")
    ap.add_argument("--criterion", default="hybrid_fast")
    ap.add_argument("--max_depth", type=int, default=5)
    ap.add_argument("--min_samples_leaf", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--smote", action="store_true")
    ap.add_argument("--no_prune", action="store_true")
    ap.add_argument("--min_leaves", type=int, default=1,
                    help="la poda no deja menos hojas que esto")
    ap.add_argument("--prune_metric", default="balanced", choices=["auc", "balanced"])
    ap.add_argument("--fnr_max", type=float, default=0.20,
                    help="cota de error tipo II para np_fnr y cips_np_fnr")
    ap.add_argument("--smoothing", default="laplace", choices=["laplace", "prior"])
    ap.add_argument("--laplace", type=float, default=1.0)
    ap.add_argument("--purity_mode", default="balanced",
                    choices=["balanced", "majority"])
    ap.add_argument("--no_class_quantiles", action="store_true")
    ap.add_argument("--leaf_model", default="constant",
                    choices=["constant", "logistic"])
    ap.add_argument("--leaf_C", type=float, default=0.1)
    ap.add_argument("--leaf_min_samples", type=int, default=30)
    ap.add_argument("--split_scoring", default="balanced",
                    choices=["balanced", "f1", "f1_weighted", "auc"],
                    help="métrica de la logística que puntúa cada split (auc = probabilidades)")
    ap.add_argument("--logistic_C", type=float, default=0.005,
                    help="regularización de la logística del split")
    ap.add_argument("--split_standardize", action="store_true",
                    help="estandariza las numéricas antes de la logística del split")
    ap.add_argument("--leaf_shrinkage", type=float, default=0.0,
                    help="k: encoge cada hoja hacia el modelo raíz, w = n/(n+k); "
                         "0 = apagado (comportamiento original)")
    ap.add_argument("--no_cips", action="store_true")
    ap.add_argument("--show_variants", default="none",
                    help="variantes a detallar, p. ej. 'base_0.5,youden,np_fnr'")
    ap.add_argument("--no_excel", action="store_true",
                    help="no generar el reporte .xlsx")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results")
    run(ap.parse_args())
