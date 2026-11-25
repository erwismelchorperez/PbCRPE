# test_pbcre_performance.py

import os
import time
from copy import deepcopy

import numpy as np
import pandas as pd

from srcc.pbcre.tree.tree import PBCRETree
from srcc.pbcre.utils.ProccessData import ProccessData
from srcc.pbcre.evaluation.metrics import ClassificationMetrics

from srcc.pbcre.patterns.route_normalizer import (normalize_path,path_matches,RouteContradictionError)


# ============================================================
# CONFIGURACIÓN
# ============================================================

DATASET = "homecredit"# german

TREE_CRITERION = "hybrid_fast"
MAX_DEPTH = 100
MIN_SAMPLES_LEAF = 10
ALPHA = 0.5
BETA = 0.5

MIN_CIP_SAMPLES = 5
MIN_CIP_CONFIDENCE = 0.6

DEPTH_LAMBDA = 0.5

OUTPUT_TREE = "./rendimiento_ft4cip_cat.csv"
OUTPUT_CIPS = "./rendimiento_patterns_cat.csv"
OUTPUT_TIMES = "./execution_times_ft4cip.csv"


# ============================================================
# NORMALIZACIÓN DE CIPS
# ============================================================

def normalize_cips(tree, cips, X_train, y_train):
    """
    Normaliza los CIPs y recalcula sus estadísticas.

    Se conserva la cobertura de cada ruta original.
    """

    X_array = (
        X_train.to_numpy()
        if isinstance(X_train, pd.DataFrame)
        else np.asarray(X_train)
    )

    y_array = np.asarray(y_train).ravel()

    n_total = len(y_array)

    normalized_cips = []
    seen_paths = set()

    for cip in cips:

        original_path = cip["path"]

        try:
            normalized_path = normalize_path(original_path)

        except RouteContradictionError:
            continue

        original_coverage = np.array([
            path_matches(x, original_path)
            for x in X_array
        ], dtype=bool)

        normalized_coverage = np.array([
            path_matches(x, normalized_path)
            for x in X_array
        ], dtype=bool)

        if not np.array_equal(
            original_coverage,
            normalized_coverage
        ):
            raise RuntimeError(
                "La normalización modificó la cobertura de un CIP."
            )

        covered_indices = np.flatnonzero(normalized_coverage)

        if len(covered_indices) == 0:
            continue

        covered_classes = y_array[covered_indices]

        n_class_0 = int(np.sum(covered_classes == 0))
        n_class_1 = int(np.sum(covered_classes == 1))

        n_samples = len(covered_indices)

        confidence_0 = n_class_0 / n_samples
        confidence_1 = n_class_1 / n_samples

        support = n_samples / n_total

        support_0 = n_class_0 / n_total
        support_1 = n_class_1 / n_total

        confidence = max(confidence_0, confidence_1)

        class_counts = np.array([n_class_0, n_class_1])

        class_prob = np.array([
            confidence_0,
            confidence_1
        ])

        predicted_class = int(np.argmax(class_counts))

        path_key = tuple(
            (
                int(feature),
                str(operator).lower(),
                repr(threshold)
            )
            for feature, operator, threshold in normalized_path
        )

        if path_key in seen_paths:
            continue

        seen_paths.add(path_key)

        normalized_cip = deepcopy(cip)

        normalized_cip["original_path"] = deepcopy(original_path)
        normalized_cip["path"] = normalized_path

        normalized_cip["original_num_conditions"] = len(original_path)
        normalized_cip["num_conditions"] = len(normalized_path)

        normalized_cip["normalization_reduction"] = (
            len(original_path) - len(normalized_path)
        )

        normalized_cip["samples"] = n_samples

        normalized_cip["support"] = support
        normalized_cip["support_class_0"] = support_0
        normalized_cip["support_class_1"] = support_1

        normalized_cip["confidence"] = confidence

        normalized_cip["class"] = predicted_class

        normalized_cip["class_counts"] = class_counts
        normalized_cip["class_prob"] = class_prob

        normalized_cip["weight"] = support * confidence

        normalized_cips.append(normalized_cip)

    return normalized_cips


# ============================================================
# EVALUACIÓN
# ============================================================

def evaluate_predictor(name,y_true,y_pred,metrics,y_prob=None):
    """
    Calcula las métricas de clasificación de un predictor.
    """

    results = {
        "predictor": name,

        "accuracy": metrics.accuracy(y_true, y_pred),

        "precision": metrics.precision(y_true, y_pred),

        "recall": metrics.recall(y_true, y_pred),

        "f1_score": metrics.f1_score(y_true, y_pred),

        "matthews": metrics.matthews_corrcoef(y_true, y_pred),

        "error_rate_I": metrics.errorrateI(y_true, y_pred),

        "error_rate_II": metrics.errorrateII(y_true, y_pred),

        "balanced_error_rate": metrics.balanced_error_rate(
            y_true,
            y_pred
        ),

        "confusion_matrix": str(
            metrics.confusion_matrix(y_true, y_pred)
        )
    }

    if y_prob is not None:
        results["auc"] = metrics.auc(y_true, y_prob)

    print("\n" + "=" * 75)
    print(f"RESULTADOS: {name}")
    print("=" * 75)

    print("\nMatriz de confusión:")
    print(metrics.confusion_matrix(y_true, y_pred))

    print(f"\nAccuracy:             {results['accuracy']:.4f}")
    print(f"Precision:            {results['precision']:.4f}")
    print(f"Recall:               {results['recall']:.4f}")
    print(f"F1-score:             {results['f1_score']:.4f}")
    print(f"MCC:                  {results['matthews']:.4f}")
    print(f"Error tipo I:         {results['error_rate_I']:.4f}")
    print(f"Error tipo II:        {results['error_rate_II']:.4f}")
    print(f"Balanced Error Rate:  {results['balanced_error_rate']:.4f}")

    if y_prob is not None:
        print(f"AUC:                  {results['auc']:.4f}")

    return results


# ============================================================
# GUARDAR RESULTADOS
# ============================================================

def save_results(results, output_file, dataset):
    """
    Guarda o actualiza los resultados de un conjunto de datos.
    """

    new_results = pd.DataFrame([results])

    if os.path.isfile(output_file):

        previous_results = pd.read_csv(output_file)

        previous_results = previous_results[
            previous_results["dataset"] != dataset
        ]

        new_results = pd.concat(
            [previous_results, new_results],
            ignore_index=True
        )

    new_results.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# EXPERIMENTO PRINCIPAL
# ============================================================

if __name__ == "__main__":

    start_time = time.perf_counter()

    metrics = ClassificationMetrics()

    # --------------------------------------------------------
    # 1. CARGA Y PREPARACIÓN DE DATOS
    # --------------------------------------------------------

    print("\nCargando conjunto de datos:", DATASET)

    process_data = ProccessData(DATASET)

    process_data.ReadDataset()
    process_data.SplitDataset()
    process_data.BalancedDataset()

    X_train = process_data.getXtrain()
    y_train = process_data.getytrain()

    X_test = process_data.getXtest()
    y_test = process_data.getytest()

    # --------------------------------------------------------
    # 2. ENTRENAMIENTO DE PbCRPE
    # --------------------------------------------------------

    print("\nEntrenando PBCRETree...")

    tree = PBCRETree(criterion=TREE_CRITERION,max_depth=MAX_DEPTH,min_samples_leaf=MIN_SAMPLES_LEAF,alpha=ALPHA,beta=BETA)

    tree.fit(X_train, y_train)

    tree.cost_complexity_pruning(X_train,y_train)

    print("Nodos:", len(tree.nodes_))
    print("Hojas:", len(set(tree.apply(X_train))))

    # --------------------------------------------------------
    # 3. PREDICCIÓN DEL ÁRBOL
    # --------------------------------------------------------

    print("\nRealizando predicciones con PBCRETree...")

    y_pred_tree = tree.predict(X_test)
    y_prob_tree = tree.predict_proba(X_test)

    # --------------------------------------------------------
    # 4. EXTRACCIÓN DE CIPS
    # --------------------------------------------------------

    print("\nExtrayendo CIPs...")

    cips = tree.extract_cips(min_samples=MIN_CIP_SAMPLES,min_confidence=MIN_CIP_CONFIDENCE)

    print("CIPs originales:", len(cips))

    # --------------------------------------------------------
    # 5. NORMALIZACIÓN
    # --------------------------------------------------------

    print("\nNormalizando CIPs...")

    normalized_cips = normalize_cips(tree=tree,cips=cips,X_train=X_train,y_train=y_train)

    print("CIPs normalizados:", len(normalized_cips))

    # --------------------------------------------------------
    # 6. FILTRADO
    # --------------------------------------------------------

    print("\nFiltrando CIPs...")

    strong_cips = tree.filter_cips(normalized_cips)

    print("CIPs seleccionados:", len(strong_cips))

    # --------------------------------------------------------
    # 7. PREDICCIÓN CON FT4CIP
    # --------------------------------------------------------

    print("\nRealizando predicciones con FT4CIP...")

    y_pred_ft4cip = tree.predict_with_cips_weighted(X_test,strong_cips,use_support=True,use_confidence=True,depth_lambda=DEPTH_LAMBDA)

    diagnostics = tree._instance_score_diagnostics

    y_true = np.asarray(y_test)

    print("\nDISTRIBUCIÓN DE CIPS POR CLASE REAL")

    for cls in [0, 1]:
        indices = np.where(y_true == cls)[0]

        cips_0 = np.asarray([diagnostics[i]["cips_clase_0"] for i in indices])
        cips_1 = np.asarray([diagnostics[i]["cips_clase_1"] for i in indices])

        scores_0 = np.asarray([diagnostics[i]["score_0"] for i in indices])
        scores_1 = np.asarray([diagnostics[i]["score_1"] for i in indices])

        print(f"\nClase real {cls}:")
        print("Promedio CIPs clase 0:", cips_0.mean())
        print("Promedio CIPs clase 1:", cips_1.mean())
        print("Promedio score clase 0:", scores_0.mean())
        print("Promedio score clase 1:", scores_1.mean())
    # --------------------------------------------------------
    # 8. COBERTURA DE LOS CIPS
    # --------------------------------------------------------

    explanations = tree.explain_instances_with_cips(
        X_test.to_numpy()
        if isinstance(X_test, pd.DataFrame)
        else np.asarray(X_test),
        strong_cips
    )

    coverage = np.mean([
        explanation["num_cips"] > 0
        for explanation in explanations
    ])

    avg_cips = np.mean([
        explanation["num_cips"]
        for explanation in explanations
    ])

    print("\nCobertura FT4CIP:", round(coverage, 4))
    print("Promedio de CIPs por instancia:", round(avg_cips, 4))

    # --------------------------------------------------------
    # 9. EVALUACIÓN DEL ÁRBOL
    # --------------------------------------------------------

    tree_results = evaluate_predictor(
        name="PBCRETree",
        y_true=y_test,
        y_pred=y_pred_tree,
        metrics=metrics,
        y_prob=y_prob_tree
    )

    tree_results["dataset"] = DATASET

    save_results(
        tree_results,
        OUTPUT_TREE,
        DATASET
    )

    # --------------------------------------------------------
    # 10. EVALUACIÓN DE FT4CIP
    # --------------------------------------------------------

    cip_results = evaluate_predictor(
        name="FT4CIP",
        y_true=y_test,
        y_pred=y_pred_ft4cip,
        metrics=metrics
    )

    cip_results["dataset"] = DATASET
    cip_results["coverage"] = coverage
    cip_results["average_cips_per_instance"] = avg_cips
    cip_results["total_original_cips"] = len(cips)
    cip_results["total_normalized_cips"] = len(normalized_cips)
    cip_results["total_selected_cips"] = len(strong_cips)

    save_results(
        cip_results,
        OUTPUT_CIPS,
        DATASET
    )

    # --------------------------------------------------------
    # 11. TIEMPO TOTAL
    # --------------------------------------------------------

    execution_time = round(
        time.perf_counter() - start_time,
        4
    )

    time_results = {
        "dataset": DATASET,
        "execution_time_seconds": execution_time
    }

    save_results(
        time_results,
        OUTPUT_TIMES,
        DATASET
    )

    print("\n" + "=" * 75)
    print("EXPERIMENTO FINALIZADO")
    print("=" * 75)

    print(f"Dataset: {DATASET}")
    print(f"Tiempo total: {execution_time} segundos")
    print(f"Resultados del árbol: {OUTPUT_TREE}")
    print(f"Resultados de FT4CIP: {OUTPUT_CIPS}")
    print(f"Tiempos: {OUTPUT_TIMES}")
