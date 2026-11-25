from srcc.pbcre.utils.ProccessData import ProccessData
from srcc.pbcre.tree.tree import PBCRETree
import os
import time

import numpy as np
import pandas as pd

def main():
    # ============================================================
    # 1. CARGA Y PREPARACIÓN DE LOS DATOS
    # ============================================================

    dataset_name = "credit_risk_small"

    proccessdata = ProccessData(dataset_name)

    proccessdata.ReadDataset()
    proccessdata.SplitDataset()
    proccessdata.BalancedDataset()

    X_train = proccessdata.getXtrain()
    y_train = proccessdata.getytrain()

    X_original = proccessdata.getXtrainOriginal()
    y_original = proccessdata.getytrainOriginal()

    X_test = proccessdata.getXtest()
    y_test = proccessdata.getytest()

    print("\n========== INFORMACIÓN DE LOS DATOS ==========")

    print("Entrenamiento original:", X_original.shape)
    print("Entrenamiento balanceado:", X_train.shape)
    print("Prueba:", X_test.shape)

    print("\nDistribución original:")
    print(y_original.value_counts().sort_index())

    print("\nDistribución balanceada:")
    print(y_train.value_counts().sort_index())


    # ============================================================
    # 2. ENTRENAMIENTO DEL ÁRBOL
    # ============================================================

    print("\n========== ENTRENAMIENTO DE PbCRPE ==========")

    tree = PBCRETree(
        criterion="hybrid_fast",
        max_depth=5,
        min_samples_leaf=5,
        alpha=0.5,
        beta=0.5
    )

    tree.fit(X_train, y_train)

    print("Entrenamiento finalizado.")

    print("Nodos antes de poda:", len(tree.nodes_))

    leaves_before = tree.apply(X_train)

    print("Hojas antes de poda:", len(np.unique(leaves_before)))


    # ============================================================
    # 3. PODA DEL ÁRBOL
    # ============================================================

    print("\n========== PODA ==========")

    tree.cost_complexity_pruning(X_train, y_train)

    print("Nodos después de poda:", len(tree.nodes_))

    leaf_ids = tree.apply(X_train)

    print("Hojas después de poda:", len(np.unique(leaf_ids)))


    # ============================================================
    # 4. REVISIÓN DE LA ESTRUCTURA
    # ============================================================

    print("\n========== ESTRUCTURA DEL ÁRBOL ==========")

    nodes = tree.nodes_

    if isinstance(nodes, dict):
        nodes = list(nodes.values())

    internal_nodes = [node for node in nodes if not node.is_leaf]
    leaf_nodes = [node for node in nodes if node.is_leaf]

    print("Nodos internos:", len(internal_nodes))
    print("Hojas:", len(leaf_nodes))

    if internal_nodes:
        print(
            "Profundidad máxima:",
            max(node.depth for node in nodes)
        )

    print("\nDivisiones del árbol:")

    for node in internal_nodes:

        feature_idx = node.split_feature

        if isinstance(feature_idx, (int, np.integer)):
            feature_name = X_train.columns[feature_idx]
        else:
            feature_name = str(feature_idx)

        print(
            f"Nodo {node.node_id} | "
            f"Profundidad: {node.depth} | "
            f"Variable: {feature_name} | "
            f"Condición: {node.threshold} | "
            f"Muestras: {node.samples}"
        )


    # ============================================================
    # 5. DISTRIBUCIÓN DE CLASES EN LAS HOJAS
    # ============================================================

    print("\n========== DISTRIBUCIÓN DE LAS HOJAS ==========")

    leaf_data = pd.DataFrame({
        "leaf_id": leaf_ids,
        "class": np.asarray(y_train).astype(int)
    })

    leaf_summary = (
        leaf_data.groupby(["leaf_id", "class"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[0, 1], fill_value=0)
    )

    leaf_summary.columns = ["class_0", "class_1"]

    leaf_summary["total"] = (
        leaf_summary["class_0"] +
        leaf_summary["class_1"]
    )

    leaf_summary["purity"] = (
        leaf_summary[["class_0", "class_1"]].max(axis=1)
        / leaf_summary["total"]
    )

    leaf_summary["discrimination"] = (
        (
            leaf_summary["class_0"] -
            leaf_summary["class_1"]
        ).abs()
        / leaf_summary["total"]
    )

    leaf_summary["predicted_class"] = (
        leaf_summary[["class_0", "class_1"]].values.argmax(axis=1)
    )

    print(leaf_summary)

    leaf_summary.to_csv(
        "leaves_credit_risk_small.csv"
    )


    # ============================================================
    # 6. PREDICCIONES
    # ============================================================

    print("\n========== PREDICCIONES ==========")

    y_pred = tree.predict(X_test)
    y_prob = tree.predict_proba(X_test)

    print("Número de instancias de prueba:", len(y_test))
    print("Número de predicciones:", len(y_pred))
    print("Dimensiones de probabilidades:", np.asarray(y_prob).shape)

    print("\nPrimeras 10 predicciones:")

    for i in range(min(10, len(y_test))):

        print(
            f"Instancia {i} | "
            f"Real: {y_test.iloc[i]} | "
            f"Predicha: {y_pred[i]} | "
            f"Probabilidad: {y_prob[i]}"
        )


    # ============================================================
    # 7. VALIDACIONES BÁSICAS
    # ============================================================

    print("\n========== VALIDACIONES ==========")

    assert len(y_pred) == len(y_test), \
        "La cantidad de predicciones no coincide con el conjunto de prueba."

    assert len(np.unique(leaf_ids)) > 0, \
        "No se encontraron hojas."

    assert np.all(np.isin(np.asarray(y_pred), [0, 1])), \
        "Se encontraron etiquetas de predicción inválidas."

    assert np.isfinite(np.asarray(y_prob)).all(), \
        "Se encontraron probabilidades no finitas."

    assert np.all(np.asarray(y_prob) >= 0), \
        "Se encontraron probabilidades negativas."

    print("Todas las validaciones básicas fueron satisfactorias.")

    print("\nExperimento finalizado.")

namedataset = "credit_risk_small"
if __name__ == "__main__":
    main()
