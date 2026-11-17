import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC, SMOTE


class ProccessData:

    def __init__(self, namedataset):

        self.namedataset = namedataset

        self.X = pd.DataFrame()
        self.y = pd.Series(dtype=int)

        self.X_train = pd.DataFrame()
        self.X_test = pd.DataFrame()

        self.y_train = pd.Series(dtype=int)
        self.y_test = pd.Series(dtype=int)

        # Conjuntos originales antes del balanceo
        self.X_train_original = pd.DataFrame()
        self.y_train_original = pd.Series(dtype=int)

        self.categorical_features = []

    def ReadDataset(self):

        self.dataset = pd.read_csv(
            "./dataset/" + self.namedataset + ".csv"
        )

        self.PreProccessDataset()

    def PreProccessDataset(self):

        self.dataset = self.dataset.dropna(axis=0)

        self.X = self.dataset.drop(columns="target")
        self.y = self.dataset["target"]

        self.y = self.y.replace({
            "positive": 1,
            "negative": 0
        })

    def SplitDataset(self):

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.3,
            random_state=42,
            stratify=self.y
        )

        self.categorical_features = [
            self.getXtrain().columns.get_loc(col)
            for col in self.getXtrain().select_dtypes(
                include=["object", "category", "bool"]
            ).columns
        ]

        print("Características          ", self.categorical_features)

        self.X_train = self.X_train.reset_index(drop=True)
        self.X_test = self.X_test.reset_index(drop=True)

        self.y_train = self.y_train.reset_index(drop=True)
        self.y_test = self.y_test.reset_index(drop=True)

        self.y_train = self.y_train.astype(int)
        self.y_test = self.y_test.astype(int)

        # Conservar los datos originales antes de SMOTENC/SMOTE
        self.X_train_original = self.X_train.copy(deep=True)
        self.y_train_original = self.y_train.copy(deep=True)

        print("\nDistribución original del entrenamiento:")
        print(self.y_train_original.value_counts().sort_index())

    def BalancedDataset(self):

        print("\nAplicando balanceo...")
        print("Instancias originales:", len(self.X_train))

        if len(self.categorical_features) == 0:

            smote = SMOTE(random_state=42)

            self.X_train, self.y_train = smote.fit_resample(
                self.X_train,
                self.y_train
            )

        else:

            smote_nc = SMOTENC(
                categorical_features=self.categorical_features,
                sampling_strategy="auto",
                random_state=42
            )

            self.X_train, self.y_train = smote_nc.fit_resample(
                self.X_train,
                self.y_train
            )

        # Mantener los índices consecutivos
        self.X_train = self.X_train.reset_index(drop=True)
        self.y_train = self.y_train.reset_index(drop=True)

        print("Instancias balanceadas:", len(self.X_train))

        print("\nDistribución después del balanceo:")
        print(self.y_train.value_counts().sort_index())

    # --------------------------------------------------
    # Métodos de acceso
    # --------------------------------------------------

    def getXtrain(self):
        return self.X_train

    def getytrain(self):
        return self.y_train

    def getXtest(self):
        return self.X_test

    def getytest(self):
        return self.y_test

    def getDataset(self):
        return self.dataset

    def getXtrainOriginal(self):
        return self.X_train_original

    def getytrainOriginal(self):
        return self.y_train_original