from src.pbcre.tree.tree import PBCRETree
from sklearn.datasets import make_classification
from src.pbcre.utils.ProccessData import ProccessData
from src.pbcre.evaluation.metrics import ClassificationMetrics
from src.pbcre.tuning.tunning_pyswarms import PBCREHyperparameterSearchPSO

import pandas as pd
import numpy as np
def metric(y_true, y_pred):
    return cl_metrics.accuracy(y_true, y_pred)
if __name__ == "__main__":
    cl_metrics = ClassificationMetrics()
    namedataset = 'australian'# australian, aer, german
    proccessdata = ProccessData(namedataset)
    proccessdata.ReadDataset()
    proccessdata.SplitDataset()
    proccessdata.BalancedDataset()

    metric = ClassificationMetrics().accuracy
    param_bounds = {
        "max_depth": (20, 120),
        "min_samples_leaf": (5, 30),
        "alpha": (0.0, 1.0),
        "beta": (0.0, 1.0)
    }
    search = PBCREHyperparameterSearchPSO(
        param_bounds=param_bounds,
        metric=metric,
        cv=5,
        n_particles=25,
        iters=40
    )
    search.fit(proccessdata.getXtrain(), proccessdata.getytrain())

    print("Best params:", search.best_params_)
    print("Best score:", search.best_score_)

    print("Entrenando...")
    #twoing, functional, gini, hybrid
    print("Entrenando árbol final con mejores hiperparámetros...")

    tree = PBCRETree(**search.best_params_)
    tree.fit(proccessdata.getXtrain(), proccessdata.getytrain())

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
    print(type(cips))
    print(len(cips))
    print(type(cips[0]))
    print("Total CIPs:", len(cips))

    confidences = [c["confidence"] for c in cips]
    print("Confianza min / mean / max:",
        min(confidences),
        np.mean(confidences),
        max(confidences))

    strong_cips = tree.filter_cips(cips)
    print("CIPs fuertes:", len(strong_cips))

    y_pred_ft4cip = tree.predict_with_cips(
        X=proccessdata.getXtest(),
        cips=strong_cips
    )
    tree.save_cips_json(strong_cips, proccessdata.getytrain(), "cips_strong_pyswarms.json")
    tree.save_cips_csv(strong_cips, proccessdata.getytrain(), "cips_strong_pyswarms.csv")
    explanations = tree.explain_instances_with_cips(
        proccessdata.getXtest().values,
        strong_cips
    )

    tree.save_instance_explanations_json(explanations, "instance_explanations_pyswarms.json")

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


    """

    spués de esto, lo natural es:

    ✅ SplitCriterion como clase abstracta

    ✅ GiniCriterion, TwoingCriterion

    🔜 FunctionalCriterion (regresión logística en nodo)

    🔜 Integrar SFS dentro del criterio funcional

    🔜 Evaluador unificado (confusión + métricas)
    """