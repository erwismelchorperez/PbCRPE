"""
Referencias por dataset: ¿cuánto margen tiene PbCRPE?

Para cada dataset aplica EXACTAMENTE el protocolo de run_final.py:
  partición estratificada 80/20 (misma semilla -> mismo test),
  probabilidades fuera de fold (CV de 5) sobre el 80 % para elegir el umbral,
  modelo final con todo el 80 % y una única evaluación en test.

Modelos: regresión logística, Random Forest (500 árboles) y Gradient Boosting.
Umbral: minimax (errores parejos) y youden (mínimo BER), elegidos sin mirar el test.

Columna clave: margen_auc = mejor AUC no lineal - AUC de la regresión logística.
  < 0.02  -> el problema es casi lineal; cambiar splits o patrones no moverá el AUC.
  > 0.05  -> hay interacciones; ahí la construcción del árbol puede mejorar.

Uso:
  python compare_baselines.py                                  # todos los CSV de ./dataset
  python compare_baselines.py --datasets german heloc gmsc
  python compare_baselines.py --datasets heloc --special_values heloc
"""

import argparse
import glob
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.simplefilter("ignore")

from srcc.pbcre.utils.ProccessData import ProccessData


def parse_special(text):
    if not text:
        return []
    if str(text).strip().lower() == "heloc":
        return [-7.0, -8.0, -9.0]
    return [float(v) for v in str(text).replace(" ", "").split(",") if v]


def load(name, special, seed):
    kwargs = {"random_state": seed}
    if special:
        try:
            kwargs.update(special_values=special)
            pdata = ProccessData(name, **kwargs)
        except TypeError:
            print("  (esta versión de ProccessData no admite valores especiales; se ignoran)")
            pdata = ProccessData(name, random_state=seed)
    else:
        pdata = ProccessData(name, **kwargs)
    pdata.ReadDataset()
    X = pdata.X.reset_index(drop=True)
    y = np.asarray(pdata.y).astype(int)
    return X, y


def preprocessor(X):
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])
           or pd.api.types.is_bool_dtype(X[c])]
    num = [c for c in X.columns if c not in cat]
    parts = []
    if num:
        parts.append((make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num))
    if cat:
        parts.append((make_pipeline(SimpleImputer(strategy="most_frequent"),
                                    OneHotEncoder(handle_unknown="ignore")), cat))
    return make_column_transformer(*parts)


def models(seed):
    return {
        "Regresión logística": LogisticRegression(C=0.3, max_iter=5000),
        "Random Forest": RandomForestClassifier(500, min_samples_leaf=3, n_jobs=-1,
                                                random_state=seed),
        "Gradient Boosting": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                            max_iter=300, random_state=seed),
    }


def threshold(y, p, rule):
    f, t, thr = roc_curve(y, p)
    ok = np.isfinite(thr)
    f, n, thr = f[ok], 1 - t[ok], thr[ok]
    if rule == "minimax":
        worst = np.maximum(f, n)
        idx = np.flatnonzero(np.isclose(worst, worst.min()))
        return thr[idx[np.argmin((f[idx] + n[idx]) / 2)]]
    j = 1 - (f + n)
    idx = np.flatnonzero(np.isclose(j, j.max()))
    return thr[idx[np.argmin(f[idx])]]


def errors(y, p, thr):
    pred = (p >= thr).astype(int)
    neg, pos = y == 0, y == 1
    e1 = float(np.mean(pred[neg] == 1)) if neg.any() else 0.0
    e2 = float(np.mean(pred[pos] == 0)) if pos.any() else 0.0
    return e1, e2, (e1 + e2) / 2


def evaluate(name, args):
    X, y = load(name, parse_special(args.special_values), args.seed)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=args.test_size,
                                              random_state=args.seed, stratify=y)
    if args.max_train and len(y_tr) > args.max_train:
        X_tr, _, y_tr, _ = train_test_split(X_tr, y_tr, train_size=args.max_train,
                                            random_state=args.seed, stratify=y_tr)
    cv = StratifiedKFold(5, shuffle=True, random_state=args.seed)
    rows = []
    for mname, model in models(args.seed).items():
        t0 = time.time()
        pipe = make_pipeline(preprocessor(X_tr), model)
        oof = cross_val_predict(pipe, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        p = pipe.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
        row = {"dataset": name, "modelo": mname, "n_train": len(y_tr),
               "prop_positivos": round(float(np.mean(y)), 3),
               "auc_oof": roc_auc_score(y_tr, oof), "auc_test": roc_auc_score(y_te, p)}
        for rule in ("minimax", "youden"):
            e1, e2, b = errors(y_te, p, threshold(y_tr, oof, rule))
            row.update({f"{rule}_error_I": e1, f"{rule}_error_II": e2, f"{rule}_ber": b})
        row["segundos"] = round(time.time() - t0, 1)
        rows.append(row)
        print(f"  {mname:20s} AUC oof={row['auc_oof']:.3f} test={row['auc_test']:.3f} | "
              f"minimax: I={row['minimax_error_I']:.3f} II={row['minimax_error_II']:.3f} "
              f"BER={row['minimax_ber']:.3f}")
    return rows


def main(args):
    names = args.datasets or sorted(os.path.splitext(os.path.basename(f))[0]
                                    for f in glob.glob(os.path.join("dataset", "*.csv")))
    all_rows = []
    for name in names:
        print(f"\n=== {name}")
        try:
            all_rows += evaluate(name, args)
        except Exception as exc:
            print(f"  ERROR: {exc}")
    if not all_rows:
        return
    df = pd.DataFrame(all_rows)
    piv = df.pivot(index="dataset", columns="modelo", values="auc_oof")
    lin = piv["Regresión logística"]
    nonlin = piv[["Random Forest", "Gradient Boosting"]].max(axis=1)
    summary = pd.DataFrame({"auc_logistica": lin, "auc_mejor_no_lineal": nonlin,
                            "margen_auc": nonlin - lin})
    summary["lectura"] = np.where(summary["margen_auc"] > 0.05, "hay interacciones: margen para el árbol",
                         np.where(summary["margen_auc"] < 0.02, "casi lineal: poco margen",
                                  "margen moderado"))
    summary = summary.sort_values("margen_auc", ascending=False)
    print("\n=== Resumen (AUC fuera de fold en el 80 % de entrenamiento)")
    print(summary.round(3).to_string())

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "referencias_modelos.xlsx")
    with pd.ExcelWriter(path) as w:
        summary.round(4).to_excel(w, sheet_name="Resumen")
        df.round(4).to_excel(w, sheet_name="Detalle", index=False)
    print(f"\nExcel: {os.path.abspath(path)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--special_values", default=None,
                    help="'heloc' o con '=': --special_values=96,98 (si tu versión lo admite)")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--max_train", type=int, default=None,
                    help="submuestra el entrenamiento en datasets grandes (p. ej. 20000)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results_referencias")
    main(ap.parse_args())
