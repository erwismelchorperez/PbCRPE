from src.pbcre.tree.tree import PBCRETree
from sklearn.datasets import make_classification
from src.pbcre.utils.ProccessData import ProccessData
from src.pbcre.evaluation.metrics import ClassificationMetrics
import pandas as pd
import numpy as np
import os

import time

if __name__ == "__main__":
    execution_times = []
    i = 0
    start_time = time.time()# inicio de ejecución


    cl_metrics = ClassificationMetrics()
    namedataset = 'default+of+credit+card+clients'# australian, aer, german, hmeq, crx, gmsc, heloc, loan_data_set, data-eiz-categorica
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
    tree.print_tree()
    # 🔥 FT4CIP PRUNING
    tree.cost_complexity_pruning(proccessdata.getXtrain(), proccessdata.getytrain())
    leaf_ids = tree.apply(proccessdata.getXtrain())

    print("Nodos:", len(tree.nodes_))
    print("Hojas únicas:", len(set(leaf_ids)))
    print("Predicciones:", tree.predict(proccessdata.getXtrain()[:5]))
    y_predict = tree.predict(proccessdata.getXtest())
    y_prob = tree.predict_proba(proccessdata.getXtest())

    #################################################
    cips = tree.extract_cips()
    #print(type(cips))
    #print(len(cips))
    #print(type(cips[0]))
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
    y_pred_ft4cip=tree.predict_with_cips_weighted(proccessdata.getXtest(), strong_cips, use_support=True, use_confidence=False)


    tree.save_cips_json(strong_cips, proccessdata.getytrain(), "./patterns/json/cips_strong_"+namedataset+".json")
    tree.save_cips_csv(strong_cips, proccessdata.getytrain(), "./patterns/csv/cips_strong_"+namedataset+".csv")
    explanations = tree.explain_instances_with_cips(
        proccessdata.getXtest().values,
        strong_cips
    )

    tree.save_instance_explanations_json(explanations, "./patterns/explanations/instance_explanations_"+namedataset+".json")
    tree.save_instance_explanations_csv(explanations, "./patterns/explanations/instance_explanations_"+namedataset+".csv")


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
    print("matriz confusión:            Predicción con patrones")
    print(cl_metrics.confusion_matrix(proccessdata.getytest(),y_pred_ft4cip))
    print("Accuracy:        ", cl_metrics.accuracy(proccessdata.getytest(),y_pred_ft4cip))
    print("Precision:       ", cl_metrics.precision(proccessdata.getytest(),y_pred_ft4cip))
    print("Recall:          ", cl_metrics.recall(proccessdata.getytest(),y_pred_ft4cip))
    print("F1-Score:        ", cl_metrics.f1_score(proccessdata.getytest(),y_pred_ft4cip))
    print("Matthews:        ", cl_metrics.matthews_corrcoef(proccessdata.getytest(),y_pred_ft4cip))
    print("ErrorRateI:      ", cl_metrics.errorrateI(proccessdata.getytest(),y_pred_ft4cip))
    print("ErrorRateII:     ", cl_metrics.errorrateII(proccessdata.getytest(),y_pred_ft4cip))
    #################################################
    ruta = "./rendimiento_patterns.csv"
    col_rendi = ["dataset", "accuracy", "precision", "recall", "f1-score","matthews","errorratei", "errorrateii", "matriz_confusion"]
    if os.path.isfile(ruta):
        rendimiento = pd.read_csv("./rendimiento_patterns.csv")
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
    rendimiento.to_csv("./rendimiento_patterns.csv", index=False)

    print("matriz confusión:            Predicción con FT4cip")
    print(cl_metrics.confusion_matrix(proccessdata.getytest(),y_predict))
    print("Accuracy:        ", cl_metrics.accuracy(proccessdata.getytest(),y_predict))
    print("Precision:       ", cl_metrics.precision(proccessdata.getytest(),y_predict))
    print("Recall:          ", cl_metrics.recall(proccessdata.getytest(),y_predict))
    print("F1-Score:        ", cl_metrics.f1_score(proccessdata.getytest(),y_predict))
    print("Matthews:        ", cl_metrics.matthews_corrcoef(proccessdata.getytest(),y_predict))
    print("AUC:             ", cl_metrics.auc(proccessdata.getytest(),y_prob))
    print("ErrorRateI:      ", cl_metrics.errorrateI(proccessdata.getytest(),y_predict))
    print("ErrorRateII:     ", cl_metrics.errorrateII(proccessdata.getytest(),y_predict))

    rendimiento = pd.DataFrame()
    ruta = "./rendimiento_ft4cip.csv"
    col_rendi = ["dataset", "accuracy", "precision", "recall", "f1-score","matthews","auc","errorratei", "errorrateii", "matriz_confusion"]
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
                str(cl_metrics.confusion_matrix(proccessdata.getytest(), y_predict))
            ]], columns=col_rendi)
    rendimiento.to_csv("./rendimiento_ft4cip.csv", index=False)
    """
    #Prueba
    print("\n=== CLASS DISTRIBUTION ===")
    print("y_train:")
    print(pd.Series(proccessdata.getytrain()).value_counts())

    print("y_test:")
    print(pd.Series(proccessdata.getytest()).value_counts())

    print("\n=== TREE ===")
    y_pred_tree = tree.predict(proccessdata.getXtest())
    print(np.unique(y_pred_tree, return_counts=True))

    print("\n=== CIPS ===")
    print("Total:", len(cips))
    print(
        pd.Series([c["class"] for c in cips])
        .value_counts()
    )

    print("\n=== CIPS PREDICTION ===")
    y_pred_cips = tree.predict_with_cips(
        proccessdata.getXtest(),
        cips
    )
    print(np.unique(y_pred_cips, return_counts=True))

    print("\n=== WEIGHTED CIPS ===")
    y_pred_weighted = tree.predict_with_cips_weighted(
        proccessdata.getXtest(),
        cips
    )
    print(np.unique(y_pred_weighted, return_counts=True))
    """
    """
    print("\n=== CIP COVERAGE ===")
    for i, x in enumerate(proccessdata.getXtest().values):

        matched = []

        for j, cip in enumerate(cips):
            if tree._match_path(x, cip["path"]):
                matched.append({
                    "cip": j,
                    "class": cip["class"],
                    "support": cip["support"],
                    "confidence": cip["confidence"],
                    "depth": cip["depth"],
                    "path": cip["path"]
                })

        print(f"\nInstance {i}")
        print(f"Actual class: {proccessdata.getytest().iloc[i]}")
        print(f"Number of CIPs: {len(matched)}")

        for m in matched:
            print(
                f"  CIP {m['cip']} | "
                f"class={m['class']} | "
                f"support={m['support']:.4f} | "
                f"confidence={m['confidence']:.4f} | "
                f"depth={m['depth']} | "
                f"path={m['path']}"
            )
    """
    """
    # esto es solo para la base de datos Alemana (German dataset)
    print("\n=== CIP COVERAGE: INSTANCE 199 ===")

    instance_idx = 199

    X_test = proccessdata.getXtest()
    y_test = proccessdata.getytest()

    # Obtener instancia y clase real
    x = X_test.iloc[instance_idx]
    y_true = y_test.iloc[instance_idx]

    print(f"\nInstance: {instance_idx}")
    print(f"Actual class: {y_true}")

    print("\n=== INSTANCE VALUES ===")
    for feature_idx, value in enumerate(x):
        print(f"Feature {feature_idx}: {value}")

    matched = []
    
    print("\n=== MATCHED CIPS ===")

    for j, cip in enumerate(cips):

        matched_cip = tree._match_path(
            x.values,
            cip["path"]
        )

        if matched_cip:

            matched.append({
                "cip": j,
                "class": cip["class"],
                "support": cip["support"],
                "confidence": cip["confidence"],
                "depth": cip["depth"],
                "path": cip["path"]
            })

            print(
                f"\nCIP {j}"
                f"\n  class      : {cip['class']}"
                f"\n  support    : {cip['support']:.4f}"
                f"\n  confidence : {cip['confidence']:.4f}"
                f"\n  depth      : {cip['depth']}"
                f"\n  path       : {cip['path']}"
            )

    print("\n=== SUMMARY ===")
    print(f"Actual class       : {y_true}")
    print(f"Number of CIPs     : {len(matched)}")

    if matched:
        class_0 = sum(cip["class"] == 0 for cip in matched)
        class_1 = sum(cip["class"] == 1 for cip in matched)

        print(f"CIPs class 0       : {class_0}")
        print(f"CIPs class 1       : {class_1}")
    else:
        print("No CIPs matched.")
    """