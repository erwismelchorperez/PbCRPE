import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class ProccessData:
    """
    Carga y preprocesamiento.

    Cambios en PbCRPE_v1:
    - imblearn se importa sólo cuando se usa (el paquete importa sin él).
    - SplitDataset(val_size=...) separa además un conjunto de VALIDACIÓN
      (para poda y umbral), estratificado, sin tocar el test.
    - make_sampler(): devuelve un remuestreador para aplicar DENTRO de cada
      fold del tuning. Aplicar SMOTE a todo X_train antes de la CV filtra
      información entre folds y hace que el BER de CV sea optimista.
    - BalancedDataset() se mantiene para el entrenamiento final.
    """

    def __init__(self, namedataset, random_state=42):
        self.namedataset = namedataset
        self.random_state = random_state
        self.X = pd.DataFrame()
        self.y = pd.Series(dtype=int)
        self.categorical_features = []
        self.X_val = None
        self.y_val = None

    def ReadDataset(self):
        self.dataset = pd.read_csv("./dataset/" + self.namedataset + ".csv")
        self.PreProccessDataset()

    # Etiquetas de texto reconocidas (sin distinguir mayúsculas/espacios)
    LABEL_MAP = {"positive": 1, "negative": 0, "pos": 1, "neg": 0,
                 "1": 1, "0": 0, "yes": 1, "no": 0, "true": 1, "false": 0}

    @staticmethod
    def categorical_columns(X):
        """
        Columnas no numéricas. Compatible con pandas 2 (dtype object) y
        pandas 3 (dtype str/StringDtype), donde select_dtypes(include="object")
        deja de incluir las columnas de texto.
        """
        return [c for c in X.columns
                if not pd.api.types.is_numeric_dtype(X[c])
                or pd.api.types.is_bool_dtype(X[c])]

    @classmethod
    def encode_target(cls, y):
        """Convierte la columna target a 0/1 (texto o numérica)."""
        if pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y):
            return y.astype(int)
        if pd.api.types.is_bool_dtype(y):
            return y.astype(int)

        normalized = y.astype(str).str.strip().str.lower()
        mapped = normalized.map(cls.LABEL_MAP)
        unknown = sorted(normalized[mapped.isna()].unique())
        if unknown:
            raise ValueError(
                f"Valores de 'target' no reconocidos: {unknown}. "
                f"Agréguelos a ProccessData.LABEL_MAP.")
        return mapped.astype(int)

    def PreProccessDataset(self):
        self.dataset = self.dataset.dropna(axis=0)
        self.X = self.dataset.drop(columns="target")
        self.y = self.encode_target(self.dataset["target"])

    def SplitDataset(self, test_size=0.2, val_size=None):
        """
        val_size: fracción del TRAIN que se reserva para validación
        (p. ej. 0.25). None conserva el comportamiento original.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size,
            random_state=self.random_state, stratify=self.y)

        if val_size:
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                self.X_train, self.y_train, test_size=val_size,
                random_state=self.random_state, stratify=self.y_train)
            self.X_val = self.X_val.reset_index(drop=True)
            self.y_val = self.y_val.reset_index(drop=True).astype(int)

        self.categorical_features = [
            self.X_train.columns.get_loc(col)
            for col in self.categorical_columns(self.X_train)
        ]
        print("Características          ", self.categorical_features)

        self.X_train = self.X_train.reset_index(drop=True)
        self.X_test = self.X_test.reset_index(drop=True)
        self.y_train = self.y_train.reset_index(drop=True).astype(int)
        self.y_test = self.y_test.reset_index(drop=True).astype(int)

    # -------------------------------------------------
    @staticmethod
    def build_sampler(categorical_features, random_state=42):
        """
        Devuelve sampler(X, y) -> (X_res, y_res), para usar dentro de cada
        fold sobre la parte de entrenamiento únicamente.
        """
        from imblearn.over_sampling import SMOTENC, SMOTE

        def sampler(X, y):
            y_arr = np.asarray(y)
            _, counts = np.unique(y_arr, return_counts=True)
            if len(counts) < 2:
                return X, y
            k = max(1, min(5, counts.min() - 1))
            if len(categorical_features) == 0:
                sm = SMOTE(random_state=random_state, k_neighbors=k)
            else:
                sm = SMOTENC(categorical_features=categorical_features,
                             sampling_strategy="auto",
                             random_state=random_state, k_neighbors=k)
            X_res, y_res = sm.fit_resample(X, y)
            return X_res, y_res

        return sampler

    def make_sampler(self):
        return self.build_sampler(self.categorical_features, self.random_state)

    def BalancedDataset(self):
        """Balancea X_train completo (usar para el modelo FINAL, no en la CV)."""
        print("Balanced", len(self.X_train))
        sampler = self.make_sampler()
        self.X_train, self.y_train = sampler(self.X_train, self.y_train)
        print("Balanced", len(self.X_train))

    def getXtrain(self):
        return self.X_train

    def getytrain(self):
        return self.y_train

    def getXval(self):
        return self.X_val

    def getyval(self):
        return self.y_val

    def getXtest(self):
        return self.X_test

    def getytest(self):
        return self.y_test

    def getDataset(self):
        return self.dataset
