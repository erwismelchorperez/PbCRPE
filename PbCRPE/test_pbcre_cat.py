from xml.parsers.expat import model

from srcc.pbcre.tree.tree import PBCRETree
from sklearn.datasets import make_classification
from srcc.pbcre.utils.ProccessData import ProccessData
from srcc.pbcre.evaluation.metrics import ClassificationMetrics
from test_normalize_german_routes import validate_leaf_normalization
import pandas as pd
import numpy as np
import os
from copy import deepcopy
import time
from numbers import Number
from srcc.pbcre.patterns.route_normalizer import normalize_path,path_matches,RouteContradictionError


def _get_node_feature_name(tree, feature_idx, X):
    """Obtiene el nombre de la característica de un nodo."""

    if isinstance(feature_idx, str):
        return feature_idx

    if hasattr(tree, "feature_names") and tree.feature_names is not None:
        return tree.feature_names[feature_idx]

    if isinstance(X, pd.DataFrame):
        return X.columns[feature_idx]

    return f"feature_{feature_idx}"
def print_leaf_instances(tree,X_train,y_train,show_features=True):
    """
    Imprime todas las hojas del árbol, sus rutas y las
    instancias de entrenamiento que llegan a cada hoja.
    """

    if not hasattr(tree, "root") or tree.root is None:
        raise ValueError("El árbol no tiene una raíz.")

    if len(X_train) != len(y_train):
        raise ValueError("X_train y y_train tienen tamaños diferentes.")

    # Conservar los identificadores originales.
    if isinstance(X_train, pd.DataFrame):
        X_display = X_train.copy()
        instance_ids = X_train.index.to_numpy()
        X_array = X_train.to_numpy()
    else:
        X_array = np.asarray(X_train)
        X_display = pd.DataFrame(X_array)
        instance_ids = np.arange(len(X_array))

    y_array = np.asarray(y_train).ravel()

    # Obtener la hoja asignada a cada instancia.
    leaf_assignments = np.asarray(tree.apply(X_array)).ravel()

    if len(leaf_assignments) != len(X_array):
        raise ValueError(
            "El número de asignaciones de hojas no coincide "
            "con el número de instancias."
        )

    # Agrupar instancias por identificador de hoja.
    assignments = pd.DataFrame({
        "instance_id": instance_ids,
        "leaf_id": leaf_assignments,
        "real_class": y_array
    })

    # Recorrer el árbol y recuperar las rutas.
    leaves = []

    def traverse(node, path):

        if node.is_leaf:
            leaves.append({
                "node": node,
                "node_id": node.node_id,
                "path": path.copy()
            })
            return

        left_condition = _format_leaf_condition(
            tree, node, "left", X_display
        )

        right_condition = _format_leaf_condition(
            tree, node, "right", X_display
        )

        traverse(
            node.left,
            path + [left_condition]
        )

        traverse(
            node.right,
            path + [right_condition]
        )

    traverse(tree.root, [])

    # Imprimir hojas ordenadas por identificador.
    leaves.sort(key=lambda item: item["node_id"])

    total_assigned = 0

    print("\n")
    print("=" * 80)
    print("ANÁLISIS COMPLETO DE HOJAS DEL ÁRBOL")
    print("=" * 80)

    print(f"Instancias de entrenamiento: {len(X_array)}")
    print(f"Número de hojas: {len(leaves)}")

    for leaf_info in leaves:

        node = leaf_info["node"]
        leaf_id = leaf_info["node_id"]
        path = leaf_info["path"]

        instances = assignments[
            assignments["leaf_id"] == leaf_id
        ].copy()

        n_instances = len(instances)
        total_assigned += n_instances

        class_counts = instances["real_class"].value_counts()

        n_class_0 = int(class_counts.get(0, 0))
        n_class_1 = int(class_counts.get(1, 0))

        predicted_class = int(node.predicted_class)

        purity = (
            max(n_class_0, n_class_1) / n_instances
            if n_instances > 0 else 0.0
        )

        predicted_class_count = (
            n_class_0 if predicted_class == 0 else n_class_1
        )

        predicted_purity = (
            predicted_class_count / n_instances
            if n_instances > 0 else 0.0
        )

        print("\n")
        print("=" * 80)
        print(f"HOJA {leaf_id}")
        print("=" * 80)

        print(f"Profundidad: {node.depth}")
        print(f"Clase predicha: {predicted_class}")
        print(f"Instancias asignadas: {n_instances}")

        print("\nDistribución de clases reales:")
        print(f"  Clase 0 (non-defaulters): {n_class_0}")
        print(f"  Clase 1 (defaulters):     {n_class_1}")

        print(f"\nPureza mayoritaria: {purity:.4f}")
        print(f"Precisión de la clase predicha: {predicted_purity:.4f}")

        print("\nRUTA DESDE LA RAÍZ:")

        if path:
            for level, condition in enumerate(path, start=1):
                print(f"  {level}. {condition}")
        else:
            print("  La raíz es una hoja.")

        print("\nINSTANCIAS QUE LLEGAN A ESTA HOJA:")

        if n_instances == 0:
            print("  No hay instancias asignadas.")
            continue

        print(
            instances[
                ["instance_id", "real_class"]
            ].to_string(index=False)
        )

        if show_features:

            row_positions = assignments.index[
                assignments["leaf_id"] == leaf_id
            ].to_numpy()

            print("\nVALORES DE LAS CARACTERÍSTICAS:")

            selected = X_display.iloc[row_positions].copy()
            selected.insert(
                0,
                "instance_id",
                instance_ids[row_positions]
            )
            selected.insert(
                1,
                "real_class",
                y_array[row_positions]
            )

            print(selected.to_string(index=False))

    print("\n")
    print("=" * 80)
    print("RESUMEN DE ASIGNACIÓN")
    print("=" * 80)

    print(f"Instancias originales: {len(X_array)}")
    print(f"Instancias asignadas:  {total_assigned}")
    print(f"Instancias sin asignar: {len(X_array) - total_assigned}")

    return assignments
def analyze_leaf_statistics(tree,X_train,y_train,output_path="./patterns/diagnostics/leaf_statistics.csv"):
    """
    Analiza estadísticamente todas las hojas del árbol.

    Calcula:
        - Número de instancias por hoja.
        - Distribución de clases.
        - Soporte por clase.
        - Confianza por clase.
        - Pureza.
        - Lift por clase.
        - Diferencia entre proporciones.
        - Ruta completa desde la raíz.

    El análisis utiliza exclusivamente los datos de entrenamiento.
    """

    if tree.root is None:
        raise ValueError("El árbol no tiene una raíz.")

    y_array = np.asarray(y_train).ravel()

    if isinstance(X_train, pd.DataFrame):
        X_array = X_train.to_numpy()
    else:
        X_array = np.asarray(X_train)

    if len(X_array) != len(y_array):
        raise ValueError("X_train y y_train tienen tamaños diferentes.")

    # --------------------------------------------------
    # 1. DISTRIBUCIÓN GLOBAL
    # --------------------------------------------------

    n_total = len(y_array)

    global_counts = pd.Series(y_array).value_counts()

    n_global_0 = int(global_counts.get(0, 0))
    n_global_1 = int(global_counts.get(1, 0))

    global_p0 = n_global_0 / n_total
    global_p1 = n_global_1 / n_total

    print("\n")
    print("=" * 90)
    print("DISTRIBUCIÓN GLOBAL DEL CONJUNTO DE ENTRENAMIENTO")
    print("=" * 90)

    print(f"Total de instancias: {n_total}")
    print(f"Clase 0: {n_global_0} ({global_p0:.4f})")
    print(f"Clase 1: {n_global_1} ({global_p1:.4f})")

    # --------------------------------------------------
    # 2. ASIGNACIÓN DE INSTANCIAS A LAS HOJAS
    # --------------------------------------------------

    leaf_assignments = np.asarray(
        tree.apply(X_array)
    ).ravel()

    if len(leaf_assignments) != n_total:
        raise ValueError(
            "El número de asignaciones no coincide "
            "con el número de instancias."
        )

    assignments = pd.DataFrame({
        "leaf_id": leaf_assignments,
        "real_class": y_array
    })

    # --------------------------------------------------
    # 3. RECORRER EL ÁRBOL Y OBTENER TODAS LAS HOJAS
    # --------------------------------------------------

    leaves = []

    def traverse(node, path):

        if node.is_leaf:

            leaves.append({
                "node": node,
                "leaf_id": node.node_id,
                "depth": node.depth,
                "path": path.copy()
            })

            return

        left_condition = _format_leaf_condition(
            tree, node, "left", X_train
        )

        right_condition = _format_leaf_condition(
            tree, node, "right", X_train
        )

        traverse(
            node.left,
            path + [left_condition]
        )

        traverse(
            node.right,
            path + [right_condition]
        )

    traverse(tree.root, [])

    leaves.sort(key=lambda item: item["leaf_id"])

    # --------------------------------------------------
    # 4. CALCULAR ESTADÍSTICAS POR HOJA
    # --------------------------------------------------

    results = []

    for leaf in leaves:

        leaf_id = leaf["leaf_id"]

        node = leaf["node"]

        instances = assignments[
            assignments["leaf_id"] == leaf_id
        ]

        n_leaf = len(instances)

        counts = instances["real_class"].value_counts()

        n0 = int(counts.get(0, 0))
        n1 = int(counts.get(1, 0))

        # Proporciones de clases dentro de la hoja.
        confidence_0 = n0 / n_leaf if n_leaf else 0.0
        confidence_1 = n1 / n_leaf if n_leaf else 0.0

        # Soporte respecto al conjunto completo.
        support_0 = n0 / n_total
        support_1 = n1 / n_total

        # Pureza de la hoja.
        purity = max(confidence_0, confidence_1)

        # Clase mayoritaria.
        if n0 > n1:
            majority_class = 0
        elif n1 > n0:
            majority_class = 1
        else:
            majority_class = -1

        # Precisión de la clase predicha por el árbol.
        predicted_class = int(node.predicted_class)

        predicted_purity = (
            confidence_0 if predicted_class == 0
            else confidence_1
        )

        # Lift: confianza / prevalencia global.
        lift_0 = (
            confidence_0 / global_p0
            if global_p0 > 0 else np.nan
        )

        lift_1 = (
            confidence_1 / global_p1
            if global_p1 > 0 else np.nan
        )

        # Diferencia absoluta entre las proporciones.
        discrimination = abs(
            confidence_0 - confidence_1
        )

        # Diferencia con signo:
        # positiva: favorece clase 0.
        # negativa: favorece clase 1.
        signed_discrimination = (
            confidence_0 - confidence_1
        )

        path_text = " AND ".join(leaf["path"])

        results.append({

            "leaf_id": leaf_id,
            "depth": leaf["depth"],

            "predicted_class": predicted_class,
            "majority_class": majority_class,

            "n_instances": n_leaf,
            "n_class_0": n0,
            "n_class_1": n1,

            "support_class_0": support_0,
            "support_class_1": support_1,

            "confidence_class_0": confidence_0,
            "confidence_class_1": confidence_1,

            "purity": purity,
            "predicted_class_purity": predicted_purity,

            "lift_class_0": lift_0,
            "lift_class_1": lift_1,

            "discrimination": discrimination,
            "signed_discrimination": signed_discrimination,

            "num_conditions": len(leaf["path"]),
            "path": path_text
        })

    # --------------------------------------------------
    # 5. GENERAR DATAFRAME
    # --------------------------------------------------

    df_results = pd.DataFrame(results)

    # Ordenar por discriminación descendente.
    df_results = df_results.sort_values(
        by="discrimination",
        ascending=False
    ).reset_index(drop=True)

    # --------------------------------------------------
    # 6. IMPRIMIR RESULTADOS
    # --------------------------------------------------

    print("\n")
    print("=" * 90)
    print("ANÁLISIS ESTADÍSTICO DE TODAS LAS HOJAS")
    print("=" * 90)

    columns_to_show = [
        "leaf_id",
        "depth",
        "n_instances",
        "n_class_0",
        "n_class_1",
        "confidence_class_0",
        "confidence_class_1",
        "purity",
        "lift_class_0",
        "lift_class_1",
        "discrimination"
    ]

    print(
        df_results[columns_to_show].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # --------------------------------------------------
    # 7. RESUMEN
    # --------------------------------------------------

    print("\n")
    print("=" * 90)
    print("RESUMEN DEL ANÁLISIS")
    print("=" * 90)

    print(f"Total de hojas: {len(df_results)}")

    print(
        "Hojas con instancias:",
        int((df_results["n_instances"] > 0).sum())
    )

    print(
        "Hojas sin instancias:",
        int((df_results["n_instances"] == 0).sum())
    )

    print(
        "Hojas que favorecen clase 0:",
        int((df_results["majority_class"] == 0).sum())
    )

    print(
        "Hojas que favorecen clase 1:",
        int((df_results["majority_class"] == 1).sum())
    )

    print(
        "Hojas empatadas:",
        int((df_results["majority_class"] == -1).sum())
    )

    # --------------------------------------------------
    # 8. GUARDAR RESULTADOS
    # --------------------------------------------------

    if output_path is not None:

        directory = os.path.dirname(output_path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        df_results.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig"
        )

        print("\nArchivo guardado en:")
        print(output_path)

    return df_results
def analyze_selected_leaf_routes(tree, X_train, y_train, selected_leaf_ids):
    """
    Analiza las rutas de un conjunto seleccionado de hojas terminales.

    Para cada hoja muestra:
    - Identificador.
    - Profundidad.
    - Número de condiciones.
    - Número de instancias por clase.
    - Pureza.
    - Discriminación.
    - Ruta completa desde la raíz.
    """

    X_array = (
        X_train.to_numpy()
        if isinstance(X_train, pd.DataFrame)
        else np.asarray(X_train)
    )

    y_array = np.asarray(y_train).ravel()

    # Obtener la hoja asignada a cada instancia.
    leaf_assignments = tree.apply(X_array)

    # Agrupar las etiquetas por identificador de hoja.
    leaf_data = {}

    for leaf_id, class_label in zip(leaf_assignments, y_array):
        leaf_data.setdefault(int(leaf_id), []).append(int(class_label))

    selected_leaf_ids = [int(x) for x in selected_leaf_ids]

    results = {}

    def traverse(node, path):
        if node is None:
            return

        if node.is_leaf:
            leaf_id = int(node.node_id)

            if leaf_id not in selected_leaf_ids:
                return

            labels = leaf_data.get(leaf_id, [])

            n_total = len(labels)
            n_class_0 = labels.count(0)
            n_class_1 = labels.count(1)

            if n_total == 0:
                return

            confidence_0 = n_class_0 / n_total
            confidence_1 = n_class_1 / n_total

            results[leaf_id] = {
                "leaf_id": leaf_id,
                "depth": node.depth,
                "n_conditions": len(path),
                "n_instances": n_total,
                "n_class_0": n_class_0,
                "n_class_1": n_class_1,
                "purity": max(confidence_0, confidence_1),
                "discrimination": abs(confidence_0 - confidence_1),
                "predicted_class": node.predicted_class,
                "path": " AND ".join(path)
            }

            return

        left_condition = _format_leaf_condition(
            tree, node, "left", X_train
        )

        right_condition = _format_leaf_condition(
            tree, node, "right", X_train
        )

        traverse(
            node.left,
            path + [left_condition]
        )

        traverse(
            node.right,
            path + [right_condition]
        )

    traverse(tree.root, [])

    # Mostrar resultados respetando el orden solicitado.
    print("\n" + "=" * 110)
    print("ANÁLISIS ESTRUCTURAL DE LAS RUTAS SELECCIONADAS")
    print("=" * 110)

    records = []

    for leaf_id in selected_leaf_ids:

        if leaf_id not in results:
            print(f"\nADVERTENCIA: No se encontró la hoja {leaf_id}.")
            continue

        result = results[leaf_id]
        records.append(result)

        print("\n" + "-" * 110)
        print(f"HOJA: {result['leaf_id']}")
        print("-" * 110)

        print(f"Profundidad:       {result['depth']}")
        print(f"Número condiciones: {result['n_conditions']}")
        print(f"Instancias:        {result['n_instances']}")
        print(f"Clase 0:           {result['n_class_0']}")
        print(f"Clase 1:           {result['n_class_1']}")
        print(f"Pureza:            {result['purity']:.4f}")
        print(f"Discriminación:    {result['discrimination']:.4f}")
        print(f"Clase predicha:    {result['predicted_class']}")

        print("\nRUTA COMPLETA:")

        for i, condition in enumerate(
            result["path"].split(" AND "), start=1
        ):
            print(f"  {i}. {condition}")

    return pd.DataFrame(records)
def _format_leaf_condition(tree, node, branch, X):
    """
    Construye la condición de una rama utilizando
    exactamente la semántica de división de PBCRETree.

    Numérica:
        izquierda: <=
        derecha: >

    Categórica individual:
        izquierda: ==
        derecha: !=

    Categórica agrupada:
        izquierda: IN
        derecha: NOT IN
    """

    feature_idx = node.split_feature
    threshold = node.threshold

    feature_name = _get_node_feature_name(
        tree, feature_idx, X
    )

    # División categórica agrupada.
    if isinstance(threshold, tuple):

        values = sorted(threshold, key=str)
        values_text = ", ".join(repr(v) for v in values)

        if branch == "left":
            return f"{feature_name} IN [{values_text}]"

        return f"{feature_name} NOT IN [{values_text}]"

    # División numérica.
    if tree.feature_dtypes[feature_idx]:

        if branch == "left":
            return f"{feature_name} <= {threshold}"

        return f"{feature_name} > {threshold}"

    # División categórica individual.
    if branch == "left":
        return f"{feature_name} == {threshold!r}"

    return f"{feature_name} != {threshold!r}"
def normalize_cips(tree, cips, X_train, y_train):
    """
    Normaliza las rutas de los CIPs y recalcula sus estadísticas.

    La normalización debe conservar exactamente la cobertura
    de cada ruta original.

    Parámetros
    ----------
    tree : PBCRETree
        Árbol entrenado.

    cips : list
        CIPs obtenidos mediante extract_cips().

    X_train : DataFrame o ndarray
        Características de entrenamiento.

    y_train : array-like
        Etiquetas reales de entrenamiento.

    Retorna
    -------
    normalized_cips : list
        CIPs normalizados.

    diagnostics : dict
        Diagnóstico de la normalización.
    """

    X_array = (
        X_train.to_numpy()
        if isinstance(X_train, pd.DataFrame)
        else np.asarray(X_train)
    )

    y_array = np.asarray(y_train).ravel()

    if len(X_array) != len(y_array):
        raise ValueError(
            "X_train y y_train tienen tamaños diferentes."
        )

    n_total = len(y_array)

    normalized_cips = []
    seen_paths = set()

    diagnostics = {
        "total_original": len(cips),
        "normalized": 0,
        "contradictory": 0,
        "without_coverage": 0,
        "duplicates": 0,
        "coverage_mismatches": 0,
        "total_original_conditions": 0,
        "total_normalized_conditions": 0
    }

    for cip in cips:

        original_path = cip["path"]

        diagnostics["total_original_conditions"] += len(
            original_path
        )

        # --------------------------------------------------
        # 1. NORMALIZACIÓN
        # --------------------------------------------------

        try:

            normalized_path = normalize_path(
                original_path
            )

        except RouteContradictionError:

            diagnostics["contradictory"] += 1
            continue

        # --------------------------------------------------
        # 2. COMPROBACIÓN DE COBERTURA
        # --------------------------------------------------

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

            diagnostics["coverage_mismatches"] += 1

            raise RuntimeError(
                "La normalización modificó la cobertura "
                "de un CIP.\n"
                f"Ruta original: {original_path}\n"
                f"Ruta normalizada: {normalized_path}"
            )

        # --------------------------------------------------
        # 3. INSTANCIAS CUBIERTAS
        # --------------------------------------------------

        covered_indices = np.flatnonzero(
            normalized_coverage
        )

        n_samples = len(covered_indices)

        if n_samples == 0:

            diagnostics["without_coverage"] += 1
            continue

        covered_classes = y_array[covered_indices]

        n_class_0 = int(
            np.sum(covered_classes == 0)
        )

        n_class_1 = int(
            np.sum(covered_classes == 1)
        )

        # --------------------------------------------------
        # 4. ESTADÍSTICAS
        # --------------------------------------------------

        confidence_0 = n_class_0 / n_samples
        confidence_1 = n_class_1 / n_samples

        support = n_samples / n_total

        support_0 = n_class_0 / n_total
        support_1 = n_class_1 / n_total

        confidence = max(
            confidence_0,
            confidence_1
        )

        class_counts = np.array(
            [n_class_0, n_class_1]
        )

        class_prob = np.array([
            confidence_0,
            confidence_1
        ])

        predicted_class = int(
            np.argmax(class_counts)
        )

        # --------------------------------------------------
        # 5. EVITAR RUTAS DUPLICADAS
        # --------------------------------------------------

        path_key = tuple(
            (
                int(feature),
                str(operator).lower(),
                repr(threshold)
            )
            for feature, operator, threshold
            in normalized_path
        )

        if path_key in seen_paths:

            diagnostics["duplicates"] += 1
            continue

        seen_paths.add(path_key)

        # --------------------------------------------------
        # 6. CONSTRUIR CIP NORMALIZADO
        # --------------------------------------------------

        normalized_cip = deepcopy(cip)

        normalized_cip["original_path"] = deepcopy(
            original_path
        )

        normalized_cip["path"] = normalized_path

        normalized_cip["original_num_conditions"] = len(
            original_path
        )

        normalized_cip["num_conditions"] = len(
            normalized_path
        )

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

        # Conservamos la profundidad original del árbol.
        # Esto permite mantener constante la ponderación
        # por profundidad en este primer experimento.

        normalized_cip["weight"] = (
            support * confidence
        )

        normalized_cips.append(normalized_cip)

        diagnostics["normalized"] += 1

        diagnostics["total_normalized_conditions"] += len(
            normalized_path
        )

    # ------------------------------------------------------
    # 7. DIAGNÓSTICO
    # ------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("DIAGNÓSTICO DE NORMALIZACIÓN DE CIPS")
    print("=" * 80)

    print(
        f"CIPs originales: "
        f"{diagnostics['total_original']}"
    )

    print(
        f"CIPs normalizados: "
        f"{diagnostics['normalized']}"
    )

    print(
        f"Rutas contradictorias: "
        f"{diagnostics['contradictory']}"
    )

    print(
        f"CIPs sin cobertura: "
        f"{diagnostics['without_coverage']}"
    )

    print(
        f"Duplicados eliminados: "
        f"{diagnostics['duplicates']}"
    )

    print(
        f"Discrepancias de cobertura: "
        f"{diagnostics['coverage_mismatches']}"
    )

    print("\nCONDICIONES")

    print(
        "Condiciones originales:",
        diagnostics["total_original_conditions"]
    )

    print(
        "Condiciones normalizadas:",
        diagnostics["total_normalized_conditions"]
    )

    print(
        "Condiciones eliminadas:",
        diagnostics["total_original_conditions"]
        - diagnostics["total_normalized_conditions"]
    )

    return normalized_cips, diagnostics
def save_normalization_diagnostics(
    original_cips,
    normalized_cips,
    output_path
):
    """
    Guarda las rutas originales y normalizadas
    para su análisis posterior.
    """

    normalized_by_original = {
        repr(cip["original_path"]): cip
        for cip in normalized_cips
    }

    results = []

    for cip in original_cips:

        original_path = cip["path"]

        normalized_cip = normalized_by_original.get(
            repr(original_path)
        )

        if normalized_cip is None:

            results.append({
                "class": cip.get("class"),
                "original_conditions": len(original_path),
                "normalized_conditions": np.nan,
                "reduction": np.nan,
                "original_path": str(original_path),
                "normalized_path": None,
                "status": "DISCARDED"
            })

            continue

        normalized_path = normalized_cip["path"]

        results.append({

            "class": normalized_cip["class"],

            "original_conditions": len(original_path),

            "normalized_conditions": len(normalized_path),

            "reduction": (
                len(original_path) - len(normalized_path)
            ),

            "original_path": str(original_path),

            "normalized_path": str(normalized_path),

            "samples": normalized_cip["samples"],

            "support": normalized_cip["support"],

            "confidence": normalized_cip["confidence"],

            "status": "OK"
        })

    df = pd.DataFrame(results)

    directory = os.path.dirname(output_path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nDiagnóstico guardado en:")
    print(output_path)

    return df
def analyze_cip_discrimination(cips, output_path=None):
    """
    Analiza la capacidad de discriminación de los CIPs.

    No modifica los patrones ni el predictor.
    """

    results = []

    for i, cip in enumerate(cips):

        class_prob = np.asarray(
            cip["class_prob"],
            dtype=float
        )

        p0 = float(class_prob[0])
        p1 = float(class_prob[1])

        discrimination = abs(p0 - p1)

        signed_discrimination = p0 - p1

        results.append({

            "cip_id": i,

            "class": cip["class"],

            "samples": cip["samples"],

            "support": cip["support"],

            "confidence": cip["confidence"],

            "prob_class_0": p0,

            "prob_class_1": p1,

            "discrimination": discrimination,

            "signed_discrimination": signed_discrimination,

            "depth": cip["depth"],

            "num_conditions": len(cip["path"]),

            "path": str(cip["path"])
        })

    df = pd.DataFrame(results)

    df = df.sort_values(
        by="discrimination",
        ascending=False
    ).reset_index(drop=True)

    print("\n")
    print("=" * 90)
    print("ANÁLISIS DE DISCRIMINACIÓN DE LOS CIPS")
    print("=" * 90)

    print("\nEstadísticas descriptivas:")

    print(
        df[
            [
                "confidence",
                "support",
                "discrimination",
                "samples",
                "num_conditions"
            ]
        ].describe().to_string()
    )

    print("\nDistribución por clase:")

    print(
        df["class"].value_counts().sort_index()
    )

    print("\nCIPs con menor discriminación:")

    print(
        df.sort_values(
            "discrimination",
            ascending=True
        )[
            [
                "cip_id",
                "class",
                "samples",
                "confidence",
                "discrimination",
                "support",
                "num_conditions"
            ]
        ].head(15).to_string(index=False)
    )

    print("\nCIPs con mayor discriminación:")

    print(
        df.sort_values(
            "discrimination",
            ascending=False
        )[
            [
                "cip_id",
                "class",
                "samples",
                "confidence",
                "discrimination",
                "support",
                "num_conditions"
            ]
        ].head(15).to_string(index=False)
    )

    if output_path is not None:

        directory = os.path.dirname(output_path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        df.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig"
        )

        print("\nArchivo guardado en:")
        print(output_path)

    return df
def analyze_discrimination_thresholds(cips,thresholds=None):
    """
    Analiza cuántos CIPs se conservan bajo distintos
    umbrales de discriminación.

    No modifica los CIPs ni realiza predicciones.
    """

    if thresholds is None:
        thresholds = [
            0.20,
            0.30,
            0.40,
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
            1.00
        ]

    results = []

    for threshold in thresholds:

        selected = []

        for cip in cips:

            probabilities = np.asarray(
                cip["class_prob"],
                dtype=float
            )

            discrimination = abs(
                probabilities[0] - probabilities[1]
            )

            if discrimination >= threshold:
                selected.append(cip)

        n_class_0 = sum(
            cip["class"] == 0
            for cip in selected
        )

        n_class_1 = sum(
            cip["class"] == 1
            for cip in selected
        )

        results.append({

            "min_discrimination": threshold,

            "total_cips": len(selected),

            "class_0": n_class_0,

            "class_1": n_class_1,

            "mean_support": np.mean([
                cip["support"]
                for cip in selected
            ]) if selected else 0,

            "mean_confidence": np.mean([
                cip["confidence"]
                for cip in selected
            ]) if selected else 0,

            "mean_conditions": np.mean([
                len(cip["path"])
                for cip in selected
            ]) if selected else 0

        })

    df = pd.DataFrame(results)

    print("\n")
    print("=" * 100)
    print("ANÁLISIS DE UMBRALES DE DISCRIMINACIÓN")
    print("=" * 100)

    print(
        df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    return df

if __name__ == "__main__":
    execution_times = []
    i = 0
    start_time = time.time()# inicio de ejecución


    cl_metrics = ClassificationMetrics()
    namedataset = 'heloc'# australian, aer, german, hmeq, crx, gmsc, heloc, loan_data_set, data-eiz-categorica, ppdaiData
    proccessdata = ProccessData(namedataset)
    proccessdata.ReadDataset()
    proccessdata.SplitDataset()
    proccessdata.BalancedDataset()
    
    print("Entrenando...")
    #twoing, functional, gini, hybrid
    """ Primera versión """
    #tree = PBCRETree(max_depth=100, min_samples_leaf=10, criterion="hybrid")
    """ Segunda versión """
    tree = PBCRETree(criterion="hybrid_fast",max_depth=100,min_samples_leaf=10,alpha=0.5,beta=0.5)#hybrid_fast,hybrid_stable

    tree.fit(proccessdata.getXtrain(), proccessdata.getytrain() )
    #tree.print_tree()
    # 🔥 FT4CIP PRUNING
    tree.cost_complexity_pruning(proccessdata.getXtrain(), proccessdata.getytrain())
    leaf_ids = tree.apply(proccessdata.getXtrain())

    print("Nodos:", len(tree.nodes_))
    print("Hojas únicas:", len(set(leaf_ids)))
    print("Predicciones:", tree.predict(proccessdata.getXtrain()[:5]))
    y_predict = tree.predict(proccessdata.getXtest())
    y_prob = tree.predict_proba(proccessdata.getXtest())

    #################################################
    cips = tree.extract_cips(min_samples=5,min_confidence=0.6)# estracción de patrones del árbol generado
    print("Total CIPs:", len(cips))

    confidences = [c["confidence"] for c in cips]
    print("Confianza min / mean / max:",
        min(confidences),
        np.mean(confidences),
        max(confidences))

    strong_cips = tree.filter_cips(cips)
    print("CIPs fuertes:", len(strong_cips))

    #y_pred_ft4cip = tree.predict_with_cips(X=proccessdata.getXtest(),cips=strong_cips)# primera versión de predicción
    """
        versión para probar la clasificación el peso
    """
    y_pred_ft4cip=tree.predict_with_cips_weighted(proccessdata.getXtest(), strong_cips, use_support=True, use_confidence=True, depth_lambda=0.5)

    """
        Pruebas con los patrones extraídos y la predicción de instancias con CIPs
    """
    # 4. Imprimir todas las hojas
    assignments = print_leaf_instances(tree=tree,X_train=proccessdata.getXtrain(),y_train=proccessdata.getytrain(),show_features=True)
    # ==========================================
    # ANÁLISIS ESTADÍSTICO DE LAS HOJAS
    # ==========================================
    df_leaf_statistics = analyze_leaf_statistics(tree=tree,X_train=proccessdata.getXtrain(),y_train=proccessdata.getytrain(),output_path=f"./patterns/diagnostics/leaf_statistics_{namedataset}.csv")
    # ==========================================
    # ANÁLISIS DE COBERTURA Y DISCRIMINACIÓN
    # ==========================================

    print("\nHOJAS CON MAYOR COBERTURA:")

    print(
        df_leaf_statistics.sort_values(
            by="n_instances",
            ascending=False
        )[
            [
                "leaf_id",
                "depth",
                "n_instances",
                "n_class_0",
                "n_class_1",
                "purity",
                "discrimination",
                "lift_class_0",
                "lift_class_1"
            ]
        ].head(15).to_string(index=False)
    )

    print("\nHOJAS CON MENOR DISCRIMINACIÓN:")

    print(
        df_leaf_statistics.sort_values(
            by="discrimination",
            ascending=True
        )[
            [
                "leaf_id",
                "depth",
                "n_instances",
                "n_class_0",
                "n_class_1",
                "purity",
                "discrimination"
            ]
        ].head(15).to_string(index=False)
    )

    selected_leaf_ids = [
        # Alta cobertura y alta discriminación
        16, 122,

        # Cobertura intermedia y alta discriminación
        20, 120,

        # Discriminación perfecta
        87, 43, 41,

        # Sin discriminación
        6, 50, 58, 35, 69
    ]

    df_selected_routes = analyze_selected_leaf_routes(
        tree=tree,
        X_train=proccessdata.getXtrain(),
        y_train=proccessdata.getytrain(),
        selected_leaf_ids=selected_leaf_ids
    )

    # Guardar resultados para su revisión.
    output_path = "./patterns/diagnostics/selected_leaf_routes_german.csv"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_selected_routes.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nArchivo guardado en:", output_path)
    """
        Pruebas con los patrones extraídos y la predicción de instancias con CIPs
    """

    tree.save_cips_json(strong_cips, proccessdata.getytrain(), "./patterns/json/cips_strong_"+namedataset+".json")
    tree.save_cips_csv(strong_cips, proccessdata.getytrain(), "./patterns/csv/cips_strong_"+namedataset+".csv")
    explanations = tree.explain_instances_with_cips(proccessdata.getXtest().values,strong_cips)

    tree.save_instance_explanations_json(explanations, "./patterns/explanations/instance_explanations_"+namedataset+".json")
    tree.save_instance_explanations_csv(explanations, "./patterns/explanations/instance_explanations_"+namedataset+".csv", proccessdata.getytest())


    end_time = time.time()# fin de ejecución
    execution_times.append({'Dataset': namedataset,'Execution_Time_Seconds': round(end_time - start_time, 4)})
    df_times = pd.DataFrame(execution_times)
    # Guardar en un archivo Excel independiente
    df_times.to_excel('./patterns/execution_times_'+namedataset+'.xlsx',index=False)
    df_times.to_csv('./patterns/execution_times_'+namedataset+'.csv',index=False)
    print("Archivo de tiempos guardado correctamente.")


    coverage = np.mean([e["num_cips"] > 0 for e in explanations])
    print("Coverage FT4CIP:", coverage)
    avg_cips = np.mean([e["num_cips"] for e in explanations])
    print("Avg CIPs per instance:", avg_cips)

    print("=== FT4CIP ===")
    print(cl_metrics.confusion_matrix(proccessdata.getytest(), y_pred_ft4cip))
    print("matriz confusión:            Predicción con FT4cip")
    print(cl_metrics.confusion_matrix(proccessdata.getytest(),y_pred_ft4cip))
    print("Accuracy:        ", cl_metrics.accuracy(proccessdata.getytest(),y_pred_ft4cip))
    print("Precision:       ", cl_metrics.precision(proccessdata.getytest(),y_pred_ft4cip))
    print("Recall:          ", cl_metrics.recall(proccessdata.getytest(),y_pred_ft4cip))
    print("F1-Score:        ", cl_metrics.f1_score(proccessdata.getytest(),y_pred_ft4cip))
    print("Matthews:        ", cl_metrics.matthews_corrcoef(proccessdata.getytest(),y_pred_ft4cip))
    print("ErrorRateI:      ", cl_metrics.errorrateI(proccessdata.getytest(),y_pred_ft4cip))
    print("ErrorRateII:     ", cl_metrics.errorrateII(proccessdata.getytest(),y_pred_ft4cip))
    #################################################
    ruta = "./rendimiento_patterns_cat.csv"
    col_rendi = ["dataset", "accuracy", "precision", "recall", "f1-score","matthews","errorratei", "errorrateii", "matriz_confusion"]
    if os.path.isfile(ruta):
        rendimiento = pd.read_csv("./rendimiento_patterns_cat.csv")
        if namedataset in rendimiento["dataset"].values:
            rendimiento.loc[rendimiento["dataset"] == namedataset] = [
                namedataset,
                cl_metrics.accuracy(proccessdata.getytest(),y_pred_ft4cip),
                cl_metrics.precision(proccessdata.getytest(),y_pred_ft4cip),
                cl_metrics.recall(proccessdata.getytest(),y_pred_ft4cip),
                cl_metrics.f1_score(proccessdata.getytest(),y_pred_ft4cip),
                cl_metrics.matthews_corrcoef(proccessdata.getytest(),y_pred_ft4cip),
                cl_metrics.errorrateI(proccessdata.getytest(),y_pred_ft4cip),
                cl_metrics.errorrateII(proccessdata.getytest(),y_pred_ft4cip),
                str(cl_metrics.confusion_matrix(proccessdata.getytest(), y_pred_ft4cip))
            ]
        else:
            nueva_fila = pd.DataFrame([[
                namedataset,
                cl_metrics.accuracy(proccessdata.getytest(),y_pred_ft4cip),
                cl_metrics.precision(proccessdata.getytest(),y_pred_ft4cip),
                cl_metrics.recall(proccessdata.getytest(),y_pred_ft4cip),
                cl_metrics.f1_score(proccessdata.getytest(),y_pred_ft4cip),
                cl_metrics.matthews_corrcoef(proccessdata.getytest(),y_pred_ft4cip),
                cl_metrics.errorrateI(proccessdata.getytest(),y_pred_ft4cip),
                cl_metrics.errorrateII(proccessdata.getytest(),y_pred_ft4cip),
                cl_metrics.confusion_matrix(proccessdata.getytest(), y_pred_ft4cip)
            ]], columns=col_rendi)

            rendimiento = pd.concat([rendimiento, nueva_fila], ignore_index=True)
    else:
        rendimiento = pd.DataFrame([[
                namedataset,
                cl_metrics.accuracy(proccessdata.getytest(),y_pred_ft4cip),
                cl_metrics.precision(proccessdata.getytest(),y_pred_ft4cip),
                cl_metrics.recall(proccessdata.getytest(),y_pred_ft4cip),
                cl_metrics.f1_score(proccessdata.getytest(),y_pred_ft4cip),
                cl_metrics.matthews_corrcoef(proccessdata.getytest(),y_pred_ft4cip),
                cl_metrics.errorrateI(proccessdata.getytest(),y_pred_ft4cip),
                cl_metrics.errorrateII(proccessdata.getytest(),y_pred_ft4cip),
                str(cl_metrics.confusion_matrix(proccessdata.getytest(), y_pred_ft4cip))
            ]], columns=col_rendi)
    rendimiento.to_csv("./rendimiento_patterns_cat.csv", index=False)

    print("matriz confusión:            Predicción con Patterns")
    print(cl_metrics.confusion_matrix(proccessdata.getytest(),y_predict))
    print("Accuracy:        ", cl_metrics.accuracy(proccessdata.getytest(),y_predict))
    print("Precision:       ", cl_metrics.precision(proccessdata.getytest(),y_predict))
    print("Recall:          ", cl_metrics.recall(proccessdata.getytest(),y_predict))
    print("F1-Score:        ", cl_metrics.f1_score(proccessdata.getytest(),y_predict))
    print("Matthews:        ", cl_metrics.matthews_corrcoef(proccessdata.getytest(),y_predict))
    print("AUC:             ", cl_metrics.auc(proccessdata.getytest(),y_prob))
    print("ErrorRateI:      ", cl_metrics.errorrateI(proccessdata.getytest(),y_predict))
    print("ErrorRateII:     ", cl_metrics.errorrateII(proccessdata.getytest(),y_predict))
    print("BalancedErrorRate: ", cl_metrics.balanced_error_rate(proccessdata.getytest(),y_predict))

    rendimiento = pd.DataFrame()
    ruta = "./rendimiento_ft4cip_cat.csv"
    col_rendi = ["dataset", "accuracy", "precision", "recall", "f1-score","matthews","auc","errorratei", "errorrateii", "balanced_error_rate","matriz_confusion"]
    if os.path.isfile(ruta):
        rendimiento = pd.read_csv("./rendimiento_ft4cip.csv")
        if namedataset in rendimiento["dataset"].values:
            rendimiento.loc[rendimiento["dataset"] == namedataset] = [
                namedataset,
                cl_metrics.accuracy(proccessdata.getytest(),y_predict),
                cl_metrics.precision(proccessdata.getytest(),y_predict),
                cl_metrics.recall(proccessdata.getytest(),y_predict),
                cl_metrics.f1_score(proccessdata.getytest(),y_predict),
                cl_metrics.matthews_corrcoef(proccessdata.getytest(),y_predict),
                cl_metrics.auc(proccessdata.getytest(),y_prob),
                cl_metrics.errorrateI(proccessdata.getytest(),y_predict),
                cl_metrics.errorrateII(proccessdata.getytest(),y_predict),
                cl_metrics.balanced_error_rate(proccessdata.getytest(),y_predict),
                str(cl_metrics.confusion_matrix(proccessdata.getytest(), y_predict))
            ]
        else:
            nueva_fila = pd.DataFrame([[
                namedataset,
                cl_metrics.accuracy(proccessdata.getytest(),y_pred_ft4cip),
                cl_metrics.precision(proccessdata.getytest(),y_pred_ft4cip),
                cl_metrics.recall(proccessdata.getytest(),y_pred_ft4cip),
                cl_metrics.f1_score(proccessdata.getytest(),y_pred_ft4cip),
                cl_metrics.matthews_corrcoef(proccessdata.getytest(),y_pred_ft4cip),
                cl_metrics.auc(proccessdata.getytest(),y_prob),
                cl_metrics.errorrateI(proccessdata.getytest(),y_predict),
                cl_metrics.errorrateII(proccessdata.getytest(),y_predict),
                cl_metrics.balanced_error_rate(proccessdata.getytest(),y_predict),
                str(cl_metrics.confusion_matrix(proccessdata.getytest(), y_predict))
            ]], columns=col_rendi)

            rendimiento = pd.concat([rendimiento, nueva_fila], ignore_index=True)
    else:
        rendimiento = pd.DataFrame([[
                namedataset,
                cl_metrics.accuracy(proccessdata.getytest(),y_predict),
                cl_metrics.precision(proccessdata.getytest(),y_predict),
                cl_metrics.recall(proccessdata.getytest(),y_predict),
                cl_metrics.f1_score(proccessdata.getytest(),y_predict),
                cl_metrics.matthews_corrcoef(proccessdata.getytest(),y_predict),
                cl_metrics.auc(proccessdata.getytest(),y_prob),
                cl_metrics.errorrateI(proccessdata.getytest(),y_predict),
                cl_metrics.errorrateII(proccessdata.getytest(),y_predict),
                cl_metrics.balanced_error_rate(proccessdata.getytest(),y_predict),
                str(cl_metrics.confusion_matrix(proccessdata.getytest(), y_predict))
            ]], columns=col_rendi)
    rendimiento.to_csv("./rendimiento_ft4cip_cat.csv", index=False)

    #df_normalization = validate_leaf_normalization(tree=tree,X_train=proccessdata.getXtrain(),output_path="./patterns/diagnostics/route_normalization_german.csv")
    """ Segundo apartado de normalización de rutas """
    execution_times = []

    start_time = time.time()

    # --------------------------------------------------
    # 1. CONFIGURACIÓN
    # --------------------------------------------------

    cl_metrics = ClassificationMetrics()

    namedataset = "german"

    # --------------------------------------------------
    # 2. CARGA Y PREPARACIÓN DE DATOS
    # --------------------------------------------------

    proccessdata = ProccessData(namedataset)

    proccessdata.ReadDataset()
    proccessdata.SplitDataset()
    proccessdata.BalancedDataset()

    Xtrain = proccessdata.getXtrain()
    ytrain = proccessdata.getytrain()

    Xtest = proccessdata.getXtest()
    ytest = proccessdata.getytest()

    # --------------------------------------------------
    # 3. ENTRENAMIENTO DEL ÁRBOL
    # --------------------------------------------------

    print("\nEntrenando PBCRETree...")

    tree = PBCRETree(
        criterion="hybrid_fast",
        max_depth=100,
        min_samples_leaf=10,
        alpha=0.5,
        beta=0.5
    )

    tree.fit(Xtrain, ytrain)

    # --------------------------------------------------
    # 4. PODA
    # --------------------------------------------------

    tree.cost_complexity_pruning(
        Xtrain,
        ytrain
    )

    leaf_ids = tree.apply(Xtrain)

    print("\nNodos:", len(tree.nodes_))
    print("Hojas únicas:", len(set(leaf_ids)))

    # --------------------------------------------------
    # 5. PREDICCIÓN ORIGINAL DEL ÁRBOL
    # --------------------------------------------------

    y_predict = tree.predict(Xtest)

    y_prob = tree.predict_proba(Xtest)

    # --------------------------------------------------
    # 6. EXTRACCIÓN DE CIPS
    # --------------------------------------------------

    print("\nExtrayendo CIPs...")

    cips = tree.extract_cips(
        min_samples=5,
        min_confidence=0.6
    )

    print("Total CIPs originales:", len(cips))

    if len(cips) == 0:
        raise RuntimeError(
            "No se extrajeron CIPs. "
            "Revisa los parámetros de extracción."
        )

    # --------------------------------------------------
    # 7. NORMALIZACIÓN DE CIPS
    # --------------------------------------------------

    print("\nNormalizando CIPs...")

    normalized_cips, normalization_diagnostics = normalize_cips(
        tree=tree,
        cips=cips,
        X_train=Xtrain,
        y_train=ytrain
    )

    if len(normalized_cips) == 0:
        raise RuntimeError(
            "No quedaron CIPs después de la normalización."
        )

    # --------------------------------------------------
    # 8. GUARDAR DIAGNÓSTICO
    # --------------------------------------------------

    df_normalization = save_normalization_diagnostics(
        original_cips=cips,
        normalized_cips=normalized_cips,
        output_path=(
            "./patterns/diagnostics/"
            f"cip_normalization_{namedataset}.csv"
        )
    )

    df_cip_analysis = analyze_cip_discrimination(normalized_cips,output_path=("./patterns/diagnostics/"f"cip_discrimination_{namedataset}.csv"))

    # --------------------------------------------------
    # 9. FILTRADO DE CIPS NORMALIZADOS
    # --------------------------------------------------
    df_thresholds = analyze_discrimination_thresholds(
        normalized_cips
    )

    df_thresholds.to_csv(
        "./patterns/diagnostics/"
        f"discrimination_thresholds_{namedataset}.csv",
        index=False,
        encoding="utf-8-sig"
    )
    print("\nFiltrando CIPs normalizados...")

    strong_cips = tree.filter_cips(
        normalized_cips
    )

    print("CIPs normalizados:", len(normalized_cips))
    print("CIPs filtrados:", len(strong_cips))

    if len(strong_cips) == 0:
        raise RuntimeError(
            "No quedaron CIPs después del filtrado."
        )

    # --------------------------------------------------
    # 10. PREDICCIÓN CON CIPS NORMALIZADOS
    # --------------------------------------------------

    print("\nRealizando predicciones con CIPs...")

    y_pred_ft4cip = tree.predict_with_cips_weighted(
        Xtest,
        strong_cips,
        use_support=True,
        use_confidence=True,
        depth_lambda=0.5
    )

    # --------------------------------------------------
    # 11. EXPLICACIONES
    # --------------------------------------------------

    explanations = tree.explain_instances_with_cips(
        Xtest.to_numpy()
        if isinstance(Xtest, pd.DataFrame)
        else np.asarray(Xtest),
        strong_cips
    )

    # --------------------------------------------------
    # 12. EXPORTACIÓN DE CIPS
    # --------------------------------------------------

    tree.save_cips_json(
        strong_cips,
        ytrain,
        f"./patterns/json/cips_strong_{namedataset}.json"
    )

    tree.save_cips_csv(
        strong_cips,
        ytrain,
        f"./patterns/csv/cips_strong_{namedataset}.csv"
    )

    # --------------------------------------------------
    # 13. EXPORTACIÓN DE EXPLICACIONES
    # --------------------------------------------------

    tree.save_instance_explanations_json(
        explanations,
        (
            "./patterns/explanations/"
            f"instance_explanations_{namedataset}.json"
        )
    )

    tree.save_instance_explanations_csv(
        explanations,
        (
            "./patterns/explanations/"
            f"instance_explanations_{namedataset}.csv"
        ),
        ytest
    )

    # --------------------------------------------------
    # 14. COBERTURA DE CIPS
    # --------------------------------------------------

    coverage = np.mean([
        e["num_cips"] > 0
        for e in explanations
    ])

    avg_cips = np.mean([
        e["num_cips"]
        for e in explanations
    ])

    print("\n")
    print("=" * 80)
    print("COBERTURA DE LOS CIPS")
    print("=" * 80)

    print("Coverage FT4CIP:", coverage)
    print("Average CIPs per instance:", avg_cips)

    # --------------------------------------------------
    # 15. EVALUACIÓN DE FT4CIP
    # --------------------------------------------------

    print("\n")
    print("=" * 80)
    print("RESULTADOS FT4CIP CON CIPS NORMALIZADOS")
    print("=" * 80)

    print(
        "Matriz de confusión:"
    )

    print(
        cl_metrics.confusion_matrix(
            ytest,
            y_pred_ft4cip
        )
    )

    print(
        "Accuracy:",
        cl_metrics.accuracy(ytest, y_pred_ft4cip)
    )

    print(
        "Precision:",
        cl_metrics.precision(ytest, y_pred_ft4cip)
    )

    print(
        "Recall:",
        cl_metrics.recall(ytest, y_pred_ft4cip)
    )

    print(
        "F1-Score:",
        cl_metrics.f1_score(ytest, y_pred_ft4cip)
    )

    print(
        "Matthews:",
        cl_metrics.matthews_corrcoef(ytest, y_pred_ft4cip)
    )

    print(
        "ErrorRateI:",
        cl_metrics.errorrateI(ytest, y_pred_ft4cip)
    )

    print(
        "ErrorRateII:",
        cl_metrics.errorrateII(ytest, y_pred_ft4cip)
    )

    # --------------------------------------------------
    # 16. EVALUACIÓN DEL ÁRBOL ORIGINAL
    # --------------------------------------------------

    print("\n")
    print("=" * 80)
    print("RESULTADOS DEL PREDICTOR PBCRETree")
    print("=" * 80)

    print(
        "Matriz de confusión:"
    )

    print(
        cl_metrics.confusion_matrix(
            ytest,
            y_predict
        )
    )

    print(
        "Accuracy:",
        cl_metrics.accuracy(ytest, y_predict)
    )

    print(
        "Precision:",
        cl_metrics.precision(ytest, y_predict)
    )

    print(
        "Recall:",
        cl_metrics.recall(ytest, y_predict)
    )

    print(
        "F1-Score:",
        cl_metrics.f1_score(ytest, y_predict)
    )

    print(
        "Matthews:",
        cl_metrics.matthews_corrcoef(ytest, y_predict)
    )

    print(
        "AUC:",
        cl_metrics.auc(ytest, y_prob)
    )

    print(
        "ErrorRateI:",
        cl_metrics.errorrateI(ytest, y_predict)
    )

    print(
        "ErrorRateII:",
        cl_metrics.errorrateII(ytest, y_predict)
    )

    print(
        "BalancedErrorRate:",
        cl_metrics.balanced_error_rate(ytest, y_predict)
    )

    # --------------------------------------------------
    # 17. GUARDAR RESULTADOS FT4CIP
    # --------------------------------------------------

    results_ft4cip = {

        "dataset": namedataset,

        "accuracy": cl_metrics.accuracy(
            ytest, y_pred_ft4cip
        ),

        "precision": cl_metrics.precision(
            ytest, y_pred_ft4cip
        ),

        "recall": cl_metrics.recall(
            ytest, y_pred_ft4cip
        ),

        "f1-score": cl_metrics.f1_score(
            ytest, y_pred_ft4cip
        ),

        "matthews": cl_metrics.matthews_corrcoef(
            ytest, y_pred_ft4cip
        ),

        "errorratei": cl_metrics.errorrateI(
            ytest, y_pred_ft4cip
        ),

        "errorrateii": cl_metrics.errorrateII(
            ytest, y_pred_ft4cip
        ),

        "matriz_confusion": str(
            cl_metrics.confusion_matrix(
                ytest, y_pred_ft4cip
            )
        )
    }

    output_file = "./rendimiento_patterns_cat.csv"

    df_results = pd.DataFrame([results_ft4cip])

    if os.path.isfile(output_file):

        previous_results = pd.read_csv(output_file)

        previous_results = previous_results[
            previous_results["dataset"] != namedataset
        ]

        df_results = pd.concat(
            [previous_results, df_results],
            ignore_index=True
        )

    df_results.to_csv(
        output_file,
        index=False
    )

    # --------------------------------------------------
    # 18. GUARDAR RESULTADOS DEL ÁRBOL
    # --------------------------------------------------

    results_tree = {

        "dataset": namedataset,

        "accuracy": cl_metrics.accuracy(
            ytest, y_predict
        ),

        "precision": cl_metrics.precision(
            ytest, y_predict
        ),

        "recall": cl_metrics.recall(
            ytest, y_predict
        ),

        "f1-score": cl_metrics.f1_score(
            ytest, y_predict
        ),

        "matthews": cl_metrics.matthews_corrcoef(
            ytest, y_predict
        ),

        "auc": cl_metrics.auc(
            ytest, y_prob
        ),

        "errorratei": cl_metrics.errorrateI(
            ytest, y_predict
        ),

        "errorrateii": cl_metrics.errorrateII(
            ytest, y_predict
        ),

        "balanced_error_rate": cl_metrics.balanced_error_rate(
            ytest, y_predict
        ),

        "matriz_confusion": str(
            cl_metrics.confusion_matrix(
                ytest, y_predict
            )
        )
    }

    output_file_tree = "./rendimiento_ft4cip_cat.csv"

    df_results_tree = pd.DataFrame([results_tree])

    if os.path.isfile(output_file_tree):

        previous_results = pd.read_csv(
            output_file_tree
        )

        previous_results = previous_results[
            previous_results["dataset"] != namedataset
        ]

        df_results_tree = pd.concat(
            [previous_results, df_results_tree],
            ignore_index=True
        )

    df_results_tree.to_csv(
        output_file_tree,
        index=False
    )

    # --------------------------------------------------
    # 19. TIEMPO DE EJECUCIÓN
    # --------------------------------------------------

    end_time = time.time()

    execution_time = round(
        end_time - start_time,
        4
    )

    execution_times.append({

        "Dataset": namedataset,

        "Execution_Time_Seconds": execution_time

    })

    df_times = pd.DataFrame(execution_times)

    df_times.to_excel(
        f"./patterns/execution_times_{namedataset}.xlsx",
        index=False
    )

    df_times.to_csv(
        f"./patterns/execution_times_{namedataset}.csv",
        index=False
    )

    print("\n")
    print("=" * 80)
    print("EXPERIMENTO FINALIZADO")
    print("=" * 80)

    print(
        "Tiempo de ejecución:",
        execution_time,
        "segundos"
    )



    """ Pruebas de script"""
    import numpy as np
    import pandas as pd

    from pathlib import Path
    from sklearn.metrics import (
        confusion_matrix,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        matthews_corrcoef
    )


    # ============================================================
    # EXPERIMENTO DE ABLACIÓN DE PONDERACIÓN DE CIPS
    # ============================================================

    def run_cip_ablation_experiment(
        tree,
        cips,
        Xtest,
        ytest,
        output_dir="./patterns/diagnostics"
    ):

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        ytest = np.asarray(ytest).ravel()

        # --------------------------------------------------------
        # 1. Configuraciones experimentales
        # --------------------------------------------------------

        experiments = [

            {
                "experiment": "E1",
                "name": "Baseline",
                "use_support": True,
                "use_confidence": True,
                "use_depth": True,
                "aggregation": "sum"
            },

            {
                "experiment": "E2",
                "name": "Without_Depth",
                "use_support": True,
                "use_confidence": True,
                "use_depth": False,
                "aggregation": "sum"
            },

            {
                "experiment": "E3",
                "name": "Support_Only",
                "use_support": True,
                "use_confidence": False,
                "use_depth": False,
                "aggregation": "sum"
            },

            {
                "experiment": "E4",
                "name": "Confidence_Only",
                "use_support": False,
                "use_confidence": True,
                "use_depth": False,
                "aggregation": "sum"
            },

            {
                "experiment": "E5",
                "name": "Max_With_Depth",
                "use_support": True,
                "use_confidence": True,
                "use_depth": True,
                "aggregation": "max"
            },

            {
                "experiment": "E6",
                "name": "Max_Without_Depth",
                "use_support": True,
                "use_confidence": True,
                "use_depth": False,
                "aggregation": "max"
            }

        ]

        results = []
        predictions_by_experiment = {}
        instance_diagnostics = []

        # --------------------------------------------------------
        # 2. Ejecutar experimentos
        # --------------------------------------------------------

        print("\n")
        print("=" * 70)
        print("EXPERIMENTO DE ABLACIÓN DE PONDERACIÓN DE CIPS")
        print("=" * 70)

        print(f"\nTotal de CIPs: {len(cips)}")
        print(f"Instancias de prueba: {len(Xtest)}")

        for config in experiments:

            experiment_id = config["experiment"]

            print("\n" + "-" * 70)
            print(
                f"{experiment_id}: {config['name']}"
            )
            print("-" * 70)

            # ----------------------------------------------------
            # Predicción
            # ----------------------------------------------------

            predictions = tree.predict_with_cips_weighted(
                Xtest,
                cips,

                use_support=config["use_support"],
                use_confidence=config["use_confidence"],
                use_depth=config["use_depth"],

                depth_lambda=0.5,

                default_strategy="tree",
                class_1_weight=1.0,

                aggregation=config["aggregation"]
            )

            predictions = np.asarray(predictions).ravel()

            predictions_by_experiment[experiment_id] = predictions.copy()

            # ----------------------------------------------------
            # Matriz de confusión
            # ----------------------------------------------------

            tn, fp, fn, tp = confusion_matrix(
                ytest,
                predictions,
                labels=[0, 1]
            ).ravel()

            # ----------------------------------------------------
            # Métricas
            # ----------------------------------------------------

            accuracy = accuracy_score(ytest, predictions)

            precision = precision_score(
                ytest,
                predictions,
                zero_division=0
            )

            recall = recall_score(
                ytest,
                predictions,
                zero_division=0
            )

            f1 = f1_score(
                ytest,
                predictions,
                zero_division=0
            )

            mcc = matthews_corrcoef(
                ytest,
                predictions
            )

            # Error tipo I:
            # Clase 0 real clasificada como clase 1.

            error_type_I = (
                fp / (tn + fp)
                if (tn + fp) > 0 else 0.0
            )

            # Error tipo II:
            # Clase 1 real clasificada como clase 0.

            error_type_II = (
                fn / (fn + tp)
                if (fn + tp) > 0 else 0.0
            )

            ber = (error_type_I + error_type_II) / 2

            # ----------------------------------------------------
            # Diagnósticos internos del predictor
            # ----------------------------------------------------

            diagnostics = tree._score_diagnostics.copy()

            # ----------------------------------------------------
            # Guardar resultados
            # ----------------------------------------------------

            result = {

                "experiment": experiment_id,
                "name": config["name"],

                "use_support": config["use_support"],
                "use_confidence": config["use_confidence"],
                "use_depth": config["use_depth"],
                "aggregation": config["aggregation"],

                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mcc": mcc,

                "error_type_I": error_type_I,
                "error_type_II": error_type_II,
                "BER": ber,

                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp,

                "covered": diagnostics["cubiertas"],
                "without_patterns": diagnostics["sin_patrones"],

                "both_classes": diagnostics["ambas_clases"],
                "only_class_0": diagnostics["solo_clase_0"],
                "only_class_1": diagnostics["solo_clase_1"],

                "ties": diagnostics["empates"],
                "exact_ties": diagnostics["empates_exactos"],
                "approximate_ties": diagnostics["empates_aproximados"],

                "tree_fallbacks": diagnostics["tree_fallbacks"],

                "total_cips_class_0": diagnostics["total_cips_clase_0"],
                "total_cips_class_1": diagnostics["total_cips_clase_1"]

            }

            results.append(result)

            # ----------------------------------------------------
            # Diagnósticos individuales
            # ----------------------------------------------------

            individual = pd.DataFrame(
                tree._instance_score_diagnostics
            )

            individual["experiment"] = experiment_id
            individual["name"] = config["name"]

            individual["true_label"] = ytest
            individual["predicted_label"] = predictions

            individual["correct"] = (
                individual["true_label"] ==
                individual["predicted_label"]
            )

            individual["error_type_I"] = (
                (individual["true_label"] == 0) &
                (individual["predicted_label"] == 1)
            )

            individual["error_type_II"] = (
                (individual["true_label"] == 1) &
                (individual["predicted_label"] == 0)
            )

            instance_diagnostics.append(individual)

            # ----------------------------------------------------
            # Mostrar resultados
            # ----------------------------------------------------

            print("\nMatriz de confusión:")
            print(np.array([
                [tn, fp],
                [fn, tp]
            ]))

            print("\nMétricas:")

            print(f"Accuracy:       {accuracy:.4f}")
            print(f"Precision:      {precision:.4f}")
            print(f"Recall:         {recall:.4f}")
            print(f"F1:             {f1:.4f}")
            print(f"MCC:            {mcc:.4f}")

            print(f"Error tipo I:   {error_type_I:.4f}")
            print(f"Error tipo II:  {error_type_II:.4f}")
            print(f"BER:            {ber:.4f}")

            print("\nDiagnósticos:")

            print(
                f"Instancias cubiertas: "
                f"{diagnostics['cubiertas']}"
            )

            print(
                f"Instancias sin patrones: "
                f"{diagnostics['sin_patrones']}"
            )

            print(
                f"Coincidencias con ambas clases: "
                f"{diagnostics['ambas_clases']}"
            )

            print(
                f"Empates: "
                f"{diagnostics['empates']}"
            )

            print(
                f"Fallbacks al árbol: "
                f"{diagnostics['tree_fallbacks']}"
            )

        # --------------------------------------------------------
        # 3. Comparación de cambios respecto a E1
        # --------------------------------------------------------

        baseline = predictions_by_experiment["E1"]

        for experiment_id, predictions in predictions_by_experiment.items():

            changed = np.sum(predictions != baseline)

            for result in results:

                if result["experiment"] == experiment_id:

                    result["changed_vs_E1"] = int(changed)

        # --------------------------------------------------------
        # 4. Exportar resultados
        # --------------------------------------------------------

        results_df = pd.DataFrame(results)

        results_path = (
            output_dir /
            "cip_ablation_summary_german.csv"
        )

        results_df.to_csv(
            results_path,
            index=False
        )

        # --------------------------------------------------------
        # 5. Exportar diagnósticos individuales
        # --------------------------------------------------------

        instances_df = pd.concat(
            instance_diagnostics,
            ignore_index=True
        )

        instances_path = (
            output_dir /
            "cip_ablation_instances_german.csv"
        )

        instances_df.to_csv(
            instances_path,
            index=False
        )

        # --------------------------------------------------------
        # 6. Mostrar tabla comparativa
        # --------------------------------------------------------

        print("\n")
        print("=" * 100)
        print("COMPARACIÓN FINAL DE EXPERIMENTOS")
        print("=" * 100)

        columns = [
            "experiment",
            "accuracy",
            "mcc",
            "error_type_I",
            "error_type_II",
            "BER",
            "ties",
            "tree_fallbacks",
            "changed_vs_E1"
        ]

        print(
            results_df[columns].to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}"
            )
        )

        print("\nArchivos generados:")

        print(results_path)
        print(instances_path)

        return results_df, instances_df


    # ============================================================
    # EJECUTAR EXPERIMENTO
    # ============================================================

    ablation_results, ablation_instances = run_cip_ablation_experiment(
        tree=tree,
        cips=normalized_cips,
        Xtest=Xtest,
        ytest=ytest
    )