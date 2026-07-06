from src.pbcre.tuning.tuning_pso import PBCREHyperparameterSearchPSO
from src.pbcre.utils.ProccessData import ProccessData
from src.pbcre.evaluation.metrics import ClassificationMetrics
from src.pbcre.tree.tree import PBCRETree
import pandas as pd

if __name__ == "__main__":
    metric = ClassificationMetrics()
    dataset = "australian"
    data = ProccessData(dataset)

    data.ReadDataset()
    data.SplitDataset()
    data.BalancedDataset()

    search = PBCREHyperparameterSearchPSO(
        param_bounds={
            "max_depth": (20,100),
            "min_samples_leaf": (5,30),
            "alpha": (0.0,1.0),
            "beta": (0.0,1.0),
            "logistic_C": (0.0001,10.0),
            "logistic_max_iter": (100,1000)
        },
        metric=metric.f1_score,
        criterion="hybrid",
        cv=5,
        pso_epochs=20,
        pso_pop_size=15
    )

    search.fit(data.getXtrain(),data.getytrain())

    print("Best params:")
    print(search.best_params_)
    print("Best score:")
    print(search.best_score_)
    tree = PBCRETree(**search.best_params_)
    tree.fit(data.getXtrain(),data.getytrain())

    y_pred = tree.predict(data.getXtest())

    print(metric.confusion_matrix(data.getytest(),y_pred))
    print(metric.accuracy(data.getytest(),y_pred))

    pd.DataFrame(search.history_).to_csv("pso_history.csv",index=False)