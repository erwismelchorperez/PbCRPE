from pathlib import Path

import pandas as pd

from srcc.pbcre.patterns.route_normalizer import (
    normalize_path,
    path_matches,
    RouteContradictionError
)


def extract_leaf_paths(tree):
    """
    Recupera las rutas completas desde la raíz hasta cada hoja.

    Devuelve un diccionario:
        {
            leaf_id: [
                (feature_idx, operator, threshold),
                ...
            ]
        }
    """

    leaf_paths = {}

    def traverse(node, path):

        if node is None:
            return

        if node.is_leaf:
            leaf_paths[node.node_id] = list(path)
            return

        feature_idx = node.split_feature
        threshold = node.threshold

        # División categórica agrupada.
        if isinstance(threshold, tuple):

            left_condition = (
                feature_idx,
                "in",
                threshold
            )

            right_condition = (
                feature_idx,
                "not in",
                threshold
            )

        # División numérica.
        elif tree.feature_dtypes[feature_idx]:

            left_condition = (
                feature_idx,
                "<=",
                threshold
            )

            right_condition = (
                feature_idx,
                ">",
                threshold
            )

        # División categórica individual.
        else:

            left_condition = (
                feature_idx,
                "=",
                threshold
            )

            right_condition = (
                feature_idx,
                "!=",
                threshold
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

    return leaf_paths


def validate_leaf_normalization(
    tree,
    X_train,
    output_path="./patterns/diagnostics/route_normalization_german.csv"
):
    """
    Valida la equivalencia entre las rutas originales
    y las rutas normalizadas de todas las hojas.

    Compara:
        1. Asignación del árbol.
        2. Cobertura de la ruta original.
        3. Cobertura de la ruta normalizada.
        4. Número de condiciones antes y después.
    """

    print("\n" + "=" * 90)
    print("VALIDACIÓN DE NORMALIZACIÓN DE RUTAS")
    print("=" * 90)

    # Obtener las rutas originales.
    leaf_paths = extract_leaf_paths(tree)

    # Obtener las hojas asignadas por el árbol.
    tree_leaf_assignments = tree.apply(X_train)

    X_values = X_train.to_numpy()

    results = []

    total_original_conditions = 0
    total_normalized_conditions = 0

    total_original_coverage = 0
    total_normalized_coverage = 0

    total_mismatches = 0
    total_assignment_mismatches = 0

    for leaf_id, original_path in leaf_paths.items():

        print(f"\nAnalizando hoja {leaf_id}...")

        try:

            normalized_path = normalize_path(original_path)

        except RouteContradictionError as error:

            results.append({
                "leaf_id": leaf_id,
                "original_conditions": len(original_path),
                "normalized_conditions": None,
                "reduction": None,
                "original_coverage": None,
                "normalized_coverage": None,
                "mismatches": None,
                "assignment_mismatches": None,
                "status": "CONTRADICTION",
                "message": str(error)
            })

            print(f"  CONTRADICCIÓN: {error}")

            continue

        original_matches = []
        normalized_matches = []

        assignment_matches = []

        for index, instance in enumerate(X_values):

            original_result = path_matches(
                instance,
                original_path
            )

            normalized_result = path_matches(
                instance,
                normalized_path
            )

            expected_result = (
                tree_leaf_assignments[index] == leaf_id
            )

            if original_result:
                original_matches.append(index)

            if normalized_result:
                normalized_matches.append(index)

            if original_result != normalized_result:
                assignment_matches.append(index)

            # Verifica también que la ruta original
            # represente correctamente la hoja del árbol.
            if original_result != expected_result:
                total_assignment_mismatches += 1

        # Comparación de coberturas.
        original_set = set(original_matches)
        normalized_set = set(normalized_matches)

        mismatch_count = len(
            original_set.symmetric_difference(normalized_set)
        )

        assignment_mismatch_count = sum(
            1
            for index, instance in enumerate(X_values)
            if (
                path_matches(instance, original_path)
                != (tree_leaf_assignments[index] == leaf_id)
            )
        )

        original_conditions = len(original_path)
        normalized_conditions = len(normalized_path)

        reduction = (
            (
                original_conditions - normalized_conditions
            ) / original_conditions * 100
            if original_conditions > 0
            else 0.0
        )

        equivalent = mismatch_count == 0

        assignment_valid = assignment_mismatch_count == 0

        status = (
            "OK"
            if equivalent and assignment_valid
            else "MISMATCH"
        )

        total_original_conditions += original_conditions
        total_normalized_conditions += normalized_conditions

        total_original_coverage += len(original_matches)
        total_normalized_coverage += len(normalized_matches)

        total_mismatches += mismatch_count

        results.append({
            "leaf_id": leaf_id,
            "original_conditions": original_conditions,
            "normalized_conditions": normalized_conditions,
            "reduction": reduction,
            "original_coverage": len(original_matches),
            "normalized_coverage": len(normalized_matches),
            "mismatches": mismatch_count,
            "assignment_mismatches": assignment_mismatch_count,
            "status": status,
            "message": ""
        })

        print(f"  Condiciones originales:   {original_conditions}")
        print(f"  Condiciones normalizadas: {normalized_conditions}")
        print(f"  Reducción:                {reduction:.2f}%")
        print(f"  Cobertura original:       {len(original_matches)}")
        print(f"  Cobertura normalizada:    {len(normalized_matches)}")
        print(f"  Diferencias:              {mismatch_count}")
        print(f"  Validación árbol:         {assignment_mismatch_count}")
        print(f"  Estado:                   {status}")

    # Crear DataFrame.
    df_results = pd.DataFrame(results)

    # Crear directorio de salida.
    output_file = Path(output_path)
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df_results.to_csv(
        output_file,
        index=False
    )

    # Resumen.
    print("\n" + "=" * 90)
    print("RESUMEN DE VALIDACIÓN")
    print("=" * 90)

    print(f"Total de hojas: {len(leaf_paths)}")

    print(
        f"Hojas equivalentes: "
        f"{(df_results['status'] == 'OK').sum()}"
    )

    print(
        f"Hojas con problemas: "
        f"{(df_results['status'] != 'OK').sum()}"
    )

    print(
        f"Condiciones originales: "
        f"{total_original_conditions}"
    )

    print(
        f"Condiciones normalizadas: "
        f"{total_normalized_conditions}"
    )

    if total_original_conditions > 0:

        global_reduction = (
            (
                total_original_conditions
                - total_normalized_conditions
            )
            / total_original_conditions
        ) * 100

    else:
        global_reduction = 0.0

    print(
        f"Reducción global: "
        f"{global_reduction:.2f}%"
    )

    print(
        f"Cobertura original total: "
        f"{total_original_coverage}"
    )

    print(
        f"Cobertura normalizada total: "
        f"{total_normalized_coverage}"
    )

    print(
        f"Discrepancias entre rutas: "
        f"{total_mismatches}"
    )

    print(
        f"Discrepancias respecto al árbol: "
        f"{total_assignment_mismatches}"
    )

    print(f"\nArchivo guardado en: {output_file}")

    return df_results


if __name__ == "__main__":

    print(
        "Este script debe ejecutarse desde el experimento "
        "donde se entrena PBCRETree."
    )