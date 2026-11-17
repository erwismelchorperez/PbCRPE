"""
======================================================================
PbCRPE - AUDITORÍA DE HOJAS TERMINALES DE PBCRETree
======================================================================

OBJETIVO
--------
Analizar la estructura del árbol PBCRETree después del entrenamiento
y la poda, para comprender cómo las instancias de entrenamiento llegan
a cada hoja terminal.

MOTIVACIÓN
----------
En experimentos anteriores se observó una diferencia importante entre
la utilización de patrones extraídos de todos los nodos y aquellos
extraídos exclusivamente de las hojas.

Esta auditoría busca estudiar el comportamiento de las hojas antes
de modificar el algoritmo de extracción de CIPs.

PROCEDIMIENTO
-------------
1. Cargar el conjunto de datos German Credit.
2. Dividir los datos en entrenamiento y prueba.
3. Aplicar el procedimiento de balanceo existente.
4. Entrenar PBCRETree con hybrid_fast.
5. Aplicar cost-complexity pruning.
6. Identificar las hojas terminales.
7. Reconstruir las rutas desde la raíz.
8. Asignar las instancias de entrenamiento a sus hojas.
9. Calcular la distribución de clases por hoja.
10. Exportar los resultados a archivos CSV.

MÉTRICAS DESCRIPTIVAS
---------------------
- Número de instancias por hoja.
- Frecuencia de cada clase.
- Pureza mayoritaria.
- Proporción de cada clase dentro de la hoja.
- Soporte de cada clase respecto al entrenamiento.
- Diferencia absoluta entre las proporciones de clases.

IMPORTANTE
----------
Esta auditoría no modifica el árbol, no extrae CIPs y no modifica
las predicciones del predictor.

Los resultados se utilizan exclusivamente para el análisis
estructural y metodológico de PbCRPE.
======================================================================
"""
from srcc.pbcre.utils.ProccessData import ProccessData
from srcc.pbcre.tree.tree import PBCRETree
import os
import time

import numpy as np
import pandas as pd

# IMPORTACIONES DEL PROYECTO
# Utiliza aquí las mismas importaciones de PBCRETree y ProccessData
# que actualmente tienes en test_pbcre_cat.py.


# ==============================================================
# 1. CONFIGURACIÓN
# ==============================================================

DATASET = "german"

MAX_DEPTH = 100
MIN_SAMPLES_LEAF = 10

CRITERION = "hybrid_fast"
ALPHA = 0.5
BETA = 0.5

OUTPUT_DIR = "patterns/diagnostics"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================================================
# 2. RECONSTRUCCIÓN DE LAS RUTAS DEL ÁRBOL
# ==============================================================

def format_condition(tree, node, go_left):
    """
    Reconstruye la condición de una rama.

    Respeta la lógica utilizada en PBCRETree._build_tree():

    - Numéricas: col <= threshold
    - Categóricas: col == threshold
    - Tuplas: col.isin(threshold)
    """

    feature_idx = node.split_feature
    feature_name = tree.feature_names[feature_idx]

    threshold = node.threshold
    is_numeric = tree.feature_dtypes[feature_idx]

    if isinstance(threshold, tuple):

        values = ", ".join(map(str, threshold))

        if go_left:
            return f"{feature_name} IN ({values})"

        return f"{feature_name} NOT IN ({values})"

    if is_numeric:

        if go_left:
            return f"{feature_name} <= {threshold}"

        return f"{feature_name} > {threshold}"

    if go_left:
        return f"{feature_name} == '{threshold}'"

    return f"{feature_name} != '{threshold}'"


# ==============================================================
# 3. IDENTIFICACIÓN DE HOJAS Y RUTAS
# ==============================================================

def get_tree_leaves(tree):
    """
    Recorre el árbol y recupera todas sus hojas terminales.

    Devuelve un diccionario:
        node_id -> información de la hoja
    """

    leaves = {}

    def traverse(node, path):

        if node is None:
            return

        if node.is_leaf:

            counts = np.asarray(node.class_counts, dtype=int)

            leaves[node.node_id] = {
                "node": node,
                "node_id": node.node_id,
                "depth": node.depth,
                "predicted_class": node.predicted_class,
                "tree_samples": node.samples,
                "tree_class_0": int(counts[0]),
                "tree_class_1": int(counts[1]),
                "path": path.copy()
            }

            return

        left_condition = format_condition(
            tree, node, go_left=True
        )

        traverse(
            node.left,
            path + [left_condition]
        )

        right_condition = format_condition(
            tree, node, go_left=False
        )

        traverse(
            node.right,
            path + [right_condition]
        )

    traverse(tree.root, [])

    return leaves


# ==============================================================
# 4. AUDITORÍA DE INSTANCIAS POR HOJA
# ==============================================================

def audit_leaves(tree, X_train, y_train):
    """
    Asigna las instancias de entrenamiento a sus hojas y calcula
    estadísticas descriptivas para cada una.

    Utiliza tree.apply() para respetar las reglas reales de recorrido
    implementadas por PBCRETree.
    """

    X = X_train.copy()

    y = np.asarray(y_train).reshape(-1).astype(int)

    if len(X) != len(y):
        raise ValueError("X_train e y_train tienen tamaños diferentes.")

    original_indices = X.index.to_numpy()

    X = X.reset_index(drop=True)

    # Identificar la hoja que recibe cada instancia.
    leaf_ids = np.asarray(tree.apply(X)).reshape(-1)

    if len(leaf_ids) != len(y):
        raise ValueError(
            "El número de identificadores de hojas no coincide "
            "con el número de instancias."
        )

    leaves = get_tree_leaves(tree)

    # Comprobar que todas las hojas identificadas existen.
    unknown_ids = set(np.unique(leaf_ids)) - set(leaves.keys())

    if unknown_ids:
        raise ValueError(
            f"Se encontraron hojas no identificadas: {unknown_ids}"
        )

    # Totales de cada clase en entrenamiento.
    total_class_0 = int(np.sum(y == 0))
    total_class_1 = int(np.sum(y == 1))

    summary_records = []
    path_records = []
    instance_records = []

    # ----------------------------------------------------------
    # Recorrer todas las hojas
    # ----------------------------------------------------------

    for leaf_id, info in leaves.items():

        mask = leaf_ids == leaf_id

        indices = np.flatnonzero(mask)

        y_leaf = y[indices]

        n0 = int(np.sum(y_leaf == 0))
        n1 = int(np.sum(y_leaf == 1))

        total = len(indices)

        proportion_0 = n0 / total if total else 0
        proportion_1 = n1 / total if total else 0

        purity = max(proportion_0, proportion_1)

        # Diferencia entre proporciones de clases dentro de la hoja.
        discrimination = abs(proportion_0 - proportion_1)

        # Soporte relativo de cada clase respecto al entrenamiento.
        support_0 = (
            n0 / total_class_0 if total_class_0 else 0
        )

        support_1 = (
            n1 / total_class_1 if total_class_1 else 0
        )

        # ------------------------------------------------------
        # Registro de resumen
        # ------------------------------------------------------

        summary_records.append({

            "leaf_id": leaf_id,
            "depth": info["depth"],

            "predicted_class": info["predicted_class"],

            "n_instances": total,

            "class_0": n0,
            "class_1": n1,

            "proportion_class_0": proportion_0,
            "proportion_class_1": proportion_1,

            "purity": purity,
            "discrimination": discrimination,

            "support_class_0": support_0,
            "support_class_1": support_1,

            "tree_samples": info["tree_samples"],

            "tree_class_0": info["tree_class_0"],
            "tree_class_1": info["tree_class_1"],

            "path": " AND ".join(info["path"])

        })

        # ------------------------------------------------------
        # Registro de condiciones de la ruta
        # ------------------------------------------------------

        for position, condition in enumerate(info["path"], 1):

            path_records.append({

                "leaf_id": leaf_id,
                "depth": info["depth"],

                "condition_number": position,
                "condition": condition

            })

        # ------------------------------------------------------
        # Registro de instancias
        # ------------------------------------------------------

        for idx in indices:

            record = {

                "instance_id": int(idx),
                "original_index": original_indices[idx],

                "leaf_id": leaf_id,

                "true_class": int(y[idx]),

                "predicted_class": info["predicted_class"],

                "leaf_purity": purity

            }

            # Agregar las características originales.
            record.update(X.iloc[idx].to_dict())

            instance_records.append(record)

    # ==========================================================
    # 5. CONSTRUCCIÓN DE DATAFRAMES
    # ==========================================================

    df_summary = pd.DataFrame(summary_records)

    df_paths = pd.DataFrame(path_records)

    df_instances = pd.DataFrame(instance_records)

    # Ordenar por identificador de hoja.
    df_summary = df_summary.sort_values("leaf_id")

    df_paths = df_paths.sort_values(
        ["leaf_id", "condition_number"]
    )

    df_instances = df_instances.sort_values(
        ["leaf_id", "instance_id"]
    )

    return df_summary, df_paths, df_instances


# ==============================================================
# 6. IMPRESIÓN DEL REPORTE
# ==============================================================

def print_audit_report(df_summary, df_paths):

    print("\n")
    print("=" * 80)
    print("AUDITORÍA DE HOJAS TERMINALES")
    print("=" * 80)

    print(f"\nTotal de hojas: {len(df_summary)}")

    print(
        f"Total de instancias: "
        f"{df_summary['n_instances'].sum()}"
    )

    print("\n" + "=" * 80)

    for _, leaf in df_summary.iterrows():

        leaf_id = leaf["leaf_id"]

        print("\n" + "=" * 80)
        print(f"HOJA {leaf_id}")
        print("=" * 80)

        print(f"Profundidad: {leaf['depth']}")

        print(f"Clase predicha: {leaf['predicted_class']}")

        print(f"Instancias asignadas: {leaf['n_instances']}")

        print("\nDistribución de clases:")

        print(f"  Clase 0: {leaf['class_0']}")
        print(f"  Clase 1: {leaf['class_1']}")

        print("\nProporción dentro de la hoja:")

        print(
            f"  Clase 0: {leaf['proportion_class_0']:.4f}"
        )

        print(
            f"  Clase 1: {leaf['proportion_class_1']:.4f}"
        )

        print(f"\nPureza: {leaf['purity']:.4f}")

        print(
            f"Discriminación descriptiva: "
            f"{leaf['discrimination']:.4f}"
        )

        print("\nRUTA DESDE LA RAÍZ:")

        conditions = df_paths[
            df_paths["leaf_id"] == leaf_id
        ]

        for _, condition in conditions.iterrows():

            print(
                f"  {int(condition['condition_number'])}. "
                f"{condition['condition']}"
            )


# ==============================================================
# 7. RESUMEN GLOBAL
# ==============================================================

def print_global_summary(df_summary):

    print("\n")
    print("=" * 80)
    print("RESUMEN GLOBAL DE LAS HOJAS")
    print("=" * 80)

    print(
        "\nTotal de hojas:",
        len(df_summary)
    )

    print(
        "Hojas puras:",
        int((df_summary["purity"] == 1.0).sum())
    )

    print(
        "Hojas mixtas:",
        int((df_summary["class_0"] > 0).astype(int).mul(
            (df_summary["class_1"] > 0).astype(int)
        ).sum())
    )

    print(
        "\nPromedio de instancias por hoja:",
        round(df_summary["n_instances"].mean(), 2)
    )

    print(
        "Pureza promedio:",
        round(df_summary["purity"].mean(), 4)
    )

    print(
        "Discriminación promedio:",
        round(df_summary["discrimination"].mean(), 4)
    )

    print("\nDistribución de hojas por clase predicha:")

    print(
        df_summary["predicted_class"].value_counts().sort_index()
    )

    print("\nHojas ordenadas por discriminación:")

    columns = [
        "leaf_id",
        "depth",
        "n_instances",
        "class_0",
        "class_1",
        "purity",
        "discrimination"
    ]

    print(
        df_summary[columns]
        .sort_values("discrimination", ascending=False)
        .to_string(index=False)
    )

def summarize_leaf_distribution(leaf_ids, y, dataset_name):

    data = pd.DataFrame({
        "leaf_id": np.asarray(leaf_ids),
        "class": np.asarray(y).astype(int)
    })

    summary = (
        data.groupby(["leaf_id", "class"])
        .size()
        .unstack(fill_value=0)
    )

    # Garantizar que existan ambas columnas
    for class_label in [0, 1]:
        if class_label not in summary.columns:
            summary[class_label] = 0

    summary = summary[[0, 1]]

    summary.columns = [
        "class_0",
        "class_1"
    ]

    summary["total"] = (
        summary["class_0"] + summary["class_1"]
    )

    summary["proportion_class_0"] = (
        summary["class_0"] / summary["total"]
    )

    summary["proportion_class_1"] = (
        summary["class_1"] / summary["total"]
    )

    summary["purity"] = (
        summary[["class_0", "class_1"]].max(axis=1)
        / summary["total"]
    )

    summary["discrimination"] = (
        summary["proportion_class_0"]
        - summary["proportion_class_1"]
    ).abs()

    summary["dataset"] = dataset_name

    return summary.reset_index()

# ==============================================================
# 8. EJECUCIÓN PRINCIPAL
# ==============================================================

def main():

    start_time = time.time()

    print("\n")
    print("=" * 80)
    print("EXPERIMENTO: AUDITORÍA DE HOJAS PBCRETree")
    print("=" * 80)

    # ----------------------------------------------------------
    # Cargar y preparar datos
    # ----------------------------------------------------------

    print("\nPreparando conjunto de datos...")

    proccessdata = ProccessData(DATASET)

    proccessdata.ReadDataset()
    proccessdata.SplitDataset()
    proccessdata.BalancedDataset()

    # Datos balanceados
    X_train_balanced = proccessdata.getXtrain()
    y_train_balanced = proccessdata.getytrain()

    # Datos originales, antes de SMOTENC
    X_train_original = proccessdata.getXtrainOriginal()
    y_train_original = proccessdata.getytrainOriginal()

    # ----------------------------------------------------------
    # Entrenar predictor
    # ----------------------------------------------------------

    print("\nEntrenando PBCRETree...")

    tree = PBCRETree(
        criterion=CRITERION,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        alpha=ALPHA,
        beta=BETA
    )

    tree.fit(X_train_original, y_train_original)
    # Hojas correspondientes a las instancias originales
    leaf_ids_original = tree.apply(X_train_original)
    print("Entrenamiento finalizado.")

    # ----------------------------------------------------------
    # Aplicar poda
    # ----------------------------------------------------------

    print("\nAplicando cost-complexity pruning...")

    tree.cost_complexity_pruning(X_train_original, y_train_original)

    print("Poda finalizada.")

    print("\nNodos:", len(tree.nodes_))

    # ----------------------------------------------------------
    # Auditoría
    # ----------------------------------------------------------

    print("\nRealizando auditoría de hojas...")

    df_summary, df_paths, df_instances = audit_leaves(tree,X_train_original,y_train_original)

    # ----------------------------------------------------------
    # Imprimir resultados
    # ----------------------------------------------------------

    print_audit_report(df_summary, df_paths)

    print_global_summary(df_summary)

    # ----------------------------------------------------------
    # Exportar resultados
    # ----------------------------------------------------------

    df_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "tree_leaves_summary_german.csv"
        ),
        index=False
    )

    df_paths.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "tree_leaves_paths_german.csv"
        ),
        index=False
    )

    df_instances.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "tree_leaves_instances_german.csv"
        ),
        index=False
    )


    tree.fit(X_train_balanced, y_train_balanced)
    tree.cost_complexity_pruning(X_train_balanced,y_train_balanced)
    # Hojas correspondientes a las instancias balanceadas
    leaf_ids_balanced = tree.apply(X_train_balanced)
    

    # Distribución de las instancias balanceadas
    summary_balanced = summarize_leaf_distribution(
        leaf_ids_balanced,
        y_train_balanced,
        "balanced"
    )

    # Distribución de las instancias originales
    summary_original = summarize_leaf_distribution(
        leaf_ids_original,
        y_train_original,
        "original"
    )

    # Unir resultados
    comparison = pd.concat(
        [summary_balanced, summary_original],
        ignore_index=True
    )

    comparison.to_csv(
        "tree_leaves_distribution_comparison_german.csv",
        index=False
    )
    # ----------------------------------------------------------
    # Finalización
    # ----------------------------------------------------------

    elapsed = time.time() - start_time

    print("\n")
    print("=" * 80)
    print("AUDITORÍA FINALIZADA")
    print("=" * 80)

    print(f"\nTiempo de ejecución: {elapsed:.4f} segundos")

    print("\nArchivos generados:")

    print("1. tree_leaves_summary_german.csv")
    print("2. tree_leaves_paths_german.csv")
    print("3. tree_leaves_instances_german.csv")


if __name__ == "__main__":

    main()