# srcc/pbcre/tree/tree.py

import json
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, roc_curve

from .node import PBCRENode
from srcc.pbcre.split.twoing import TwoingCriterion
from srcc.pbcre.split.functional_criterion import FunctionalCriterion
from srcc.pbcre.split.hybrid_twoing_functional import HybridTwoingFunctionalCriterion
from srcc.pbcre.split.hybrid_twoing_functional_sfs import HybridTwoingFunctionalCriterionSFS
from srcc.pbcre.split.hybrid_fast import HybridTwoingFunctionalCriterionFast
from srcc.pbcre.split.hybrid_stable import HybridTwoingFunctionalCriterionStable
from srcc.pbcre.evaluation.metrics import ClassificationMetrics


class PBCRETree:

    def __init__(
        self,
        max_depth=5,
        min_samples_leaf=5,
        criterion="twoing",
        alpha=0.5,
        beta=0.5,
        logistic_C=0.005,
        logistic_solver="liblinear",
        logistic_max_iter=200
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.alpha = alpha
        self.beta = beta
        self.logistic_C = logistic_C
        self.logistic_solver = logistic_solver
        self.logistic_max_iter = logistic_max_iter

        print("Criterio seleccionado            " + criterion)

        if criterion == "twoing":
            self.splitter = TwoingCriterion()
        elif criterion == "functional":
            self.splitter = FunctionalCriterion()
        elif criterion == "hybrid":
            self.splitter = HybridTwoingFunctionalCriterion(
                alpha=self.alpha,
                beta=self.beta,
                logistic_C=logistic_C,
                logistic_solver=logistic_solver,
                logistic_max_iter=logistic_max_iter
            )
        elif criterion == "hybrid_sfs":
            self.splitter = HybridTwoingFunctionalCriterionSFS(
                alpha=self.alpha,
                beta=self.beta,
                logistic_C=logistic_C,
                logistic_solver=logistic_solver,
                logistic_max_iter=logistic_max_iter
            )
        elif criterion == "hybrid_fast":
            self.splitter = HybridTwoingFunctionalCriterionFast(
                alpha=self.alpha,
                beta=self.beta,
                logistic_C=self.logistic_C,
                logistic_solver=self.logistic_solver,
                logistic_max_iter=self.logistic_max_iter,
                n_quantiles=10
            )
        elif criterion == "hybrid_stable":
            self.splitter = HybridTwoingFunctionalCriterionStable(
                alpha=self.alpha,
                beta=self.beta,
                logistic_C=self.logistic_C
            )
        else:
            raise ValueError(f"Criterio desconocido: {criterion}")

        self.criterion = criterion
        self.node_count = 0
        self.root = None
        self.nodes_ = {}

    # ======================================================
    # HELPERS
    # ======================================================

    @staticmethod
    def _is_missing(value):
        """Comprueba si un valor escalar está ausente."""
        try:
            result = pd.isna(value)
            return bool(result) if np.ndim(result) == 0 else False
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _to_array(X):
        """Convierte Series, DataFrames o arrays a una matriz NumPy."""
        if isinstance(X, pd.Series):
            X = X.to_frame().T
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        return np.atleast_2d(np.asarray(X))

    def _go_left(self, value, threshold, feature_idx):
        """
        Numérico: value <= threshold.
        Categórico: value == threshold.
        Categórico agrupado: value pertenece a threshold.
        Los valores ausentes siguen la rama derecha.
        """
        if self._is_missing(value):
            return False

        if isinstance(threshold, tuple):
            try:
                return bool(value in threshold)
            except (TypeError, ValueError):
                return False

        if not self.feature_dtypes[feature_idx]:
            try:
                return bool(value == threshold)
            except (TypeError, ValueError):
                return False

        try:
            return bool(value <= threshold)
        except (TypeError, ValueError):
            return False

    def _get_value(self, x, feature_idx):
        return x[feature_idx]

    def _reachable_nodes(self, node=None):
        """Devuelve los nodos que pertenecen al árbol actual."""
        if node is None:
            node = self.root
        if node is None:
            return []

        result = [node]
        if not node.is_leaf:
            result.extend(self._reachable_nodes(node.left))
            result.extend(self._reachable_nodes(node.right))
        return result

    # ======================================================
    # TRAINING
    # ======================================================

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X = X.reset_index(drop=True)
        y = np.asarray(y).reshape(-1)

        if len(X) != len(y):
            raise ValueError("X e y deben contener el mismo número de instancias.")
        if len(y) == 0:
            raise ValueError("El conjunto de entrenamiento está vacío.")
        if len(np.unique(y)) > 2:
            raise ValueError("PBCRETree requiere clasificación binaria.")
        if not np.isin(y, [0, 1]).all():
            raise ValueError("Las etiquetas de y deben ser 0 o 1.")

        self.feature_names = list(X.columns)
        self.feature_dtypes = [
            pd.api.types.is_numeric_dtype(X.iloc[:, i])
            for i in range(X.shape[1])
        ]

        self.node_count = 0
        self.nodes_ = {}
        self.n_samples_ = len(y)
        self.root = self._build_tree(X, y.astype(int), depth=0)
        return self

    def _build_tree(self, X, y, depth):
        node = PBCRENode(node_id=self.node_count, depth=depth)
        self.nodes_[node.node_id] = node
        self.node_count += 1

        y = np.asarray(y, dtype=int)
        node.samples = len(y)
        node.class_counts = np.bincount(y, minlength=2)
        node.predicted_class = int(np.argmax(node.class_counts))

        if (
            depth >= self.max_depth
            or len(np.unique(y)) == 1
            or len(y) < 2 * self.min_samples_leaf
            or X.shape[1] == 0
        ):
            return node

        feat, thr, score, extra = self.splitter.best_split(
            X, y, self.min_samples_leaf
        )
        if feat is None:
            return node

        col = X.iloc[:, feat]
        if isinstance(thr, tuple):
            mask = col.isin(thr).to_numpy()
        elif self.feature_dtypes[feat]:
            mask = (col <= thr).fillna(False).to_numpy()
        else:
            mask = (col == thr).fillna(False).to_numpy()

        if (
            mask.sum() < self.min_samples_leaf
            or (~mask).sum() < self.min_samples_leaf
        ):
            return node

        left_idx = np.flatnonzero(mask)
        right_idx = np.flatnonzero(~mask)

        X_left = X.iloc[left_idx].reset_index(drop=True)
        X_right = X.iloc[right_idx].reset_index(drop=True)
        y_left = y[left_idx]
        y_right = y[right_idx]

        left = self._build_tree(X_left, y_left, depth + 1)
        right = self._build_tree(X_right, y_right, depth + 1)
        node.set_split(feat, thr, left, right, extra)
        return node

    # ======================================================
    # PREDICTION
    # ======================================================

    def predict_one(self, x, node=None):
        if node is None:
            node = self.root
        if node is None:
            raise RuntimeError("El árbol todavía no ha sido entrenado.")

        while not node.is_leaf:
            value = self._get_value(x, node.split_feature)
            if self._go_left(value, node.threshold, node.split_feature):
                node = node.left
            else:
                node = node.right

        total = int(node.class_counts.sum())
        if total == 0:
            return 0
        prob = node.class_counts[1] / total
        return int(prob >= node.leaf_threshold)

    def predict(self, X):
        X = self._to_array(X)
        return np.asarray([self.predict_one(row) for row in X], dtype=int)

    def predict_proba(self, X):
        """Devuelve la probabilidad suavizada de la clase positiva."""
        X = self._to_array(X)
        probs = np.zeros(X.shape[0], dtype=float)

        for i, x in enumerate(X):
            node = self.root
            while not node.is_leaf:
                value = self._get_value(x, node.split_feature)
                if self._go_left(value, node.threshold, node.split_feature):
                    node = node.left
                else:
                    node = node.right

            counts = node.class_counts
            smoothing = self.alpha
            denominator = counts.sum() + 2 * smoothing
            probs[i] = (
                (counts[1] + smoothing) / denominator
                if denominator > 0 else 0.0
            )
        return probs

    # ======================================================
    # AUC COST-COMPLEXITY PRUNING
    # ======================================================

    def _get_leaves(self, node):
        if node.is_leaf:
            return [node]
        return self._get_leaves(node.left) + self._get_leaves(node.right)

    def _predict_proba_pruned(self, X, pruned_node):
        X = self._to_array(X)
        probs = np.zeros(len(X), dtype=float)

        for i, x in enumerate(X):
            node = self.root
            while not node.is_leaf:
                if node is pruned_node:
                    break
                value = self._get_value(x, node.split_feature)
                if self._go_left(value, node.threshold, node.split_feature):
                    node = node.left
                else:
                    node = node.right

            counts = node.class_counts
            total = counts.sum()
            probs[i] = counts[1] / total if total > 0 else 0.0
        return probs

    def _compute_alpha(self, X_val, y_val, node, base_auc):
        if node.is_leaf:
            return -np.inf
        leaves = self._get_leaves(node)
        if len(leaves) <= 1:
            return -np.inf

        auc_pruned = roc_auc_score(
            y_val, self._predict_proba_pruned(X_val, node)
        )
        return (auc_pruned - base_auc) / (len(leaves) - 1)

    def cost_complexity_pruning(self, X_val, y_val):
        """Poda basada en AUC de validación."""
        y_val = np.asarray(y_val).reshape(-1)
        if len(np.unique(y_val)) < 2:
            raise ValueError("Se necesitan ambas clases en y_val para calcular AUC.")

        best_auc = roc_auc_score(y_val, self.predict_proba(X_val))
        #best_auc = ClassificationMetrics.balanced_error_rate(y_val, self.predict(X_val))

        while True:
            best_alpha = -np.inf
            best_node = None

            for node in self._reachable_nodes():
                if node.is_leaf:
                    continue
                alpha = self._compute_alpha(X_val, y_val, node, best_auc)
                if alpha > best_alpha:
                    best_alpha = alpha
                    best_node = node

            if best_node is None or best_alpha <= 0:
                break

            left, right = best_node.left, best_node.right
            best_node.left = None
            best_node.right = None
            best_node.is_leaf = True

            new_auc = roc_auc_score(y_val, self.predict_proba(X_val))
            #new_auc = ClassificationMetrics.balanced_error_rate(y_val, self.predict(X_val))

            if new_auc >= best_auc:
                best_auc = new_auc
            else:
                best_node.left = left
                best_node.right = right
                best_node.is_leaf = False
                break

    # ======================================================
    # APPLY
    # ======================================================

    def apply(self, X):
        """Devuelve el identificador de la hoja de cada instancia."""
        X = self._to_array(X)
        leaf_ids = np.zeros(X.shape[0], dtype=int)

        for i, x in enumerate(X):
            node = self.root
            while not node.is_leaf:
                value = self._get_value(x, node.split_feature)
                if self._go_left(value, node.threshold, node.split_feature):
                    node = node.left
                else:
                    node = node.right
            leaf_ids[i] = node.node_id
        return leaf_ids

    # ======================================================
    # LEAF THRESHOLDS
    # ======================================================

    def fit_leaf_thresholds(self, X, y):
        """
        Ajusta el umbral de clasificación de cada hoja mediante ROC.
        Debe ajustarse con entrenamiento/validación, nunca con el test final.
        """
        y = np.asarray(y).reshape(-1)
        leaf_ids = self.apply(X)

        if len(y) != len(leaf_ids):
            raise ValueError("X e y deben tener la misma longitud.")

        for node in self._get_leaves(self.root):
            idx = leaf_ids == node.node_id

            if idx.sum() < 5:
                continue

            y_leaf = y[idx]

            if len(np.unique(y_leaf)) < 2:
                continue

            prob = (
                node.class_counts[1] /
                max(node.class_counts.sum(), 1)
            )
            p_leaf = np.repeat(prob, idx.sum())

            fpr, tpr, thresholds = roc_curve(y_leaf, p_leaf)
            node.leaf_threshold = thresholds[np.argmax(tpr - fpr)]

    def extract_cips(
        self,
        min_samples=5,
        min_confidence=0.6,
        leaves_only=False
    ):
        """
        Extrae CIPs de los nodos del árbol.

        leaves_only=False conserva el comportamiento original y permite
        patrones de nodos internos y hojas.
        leaves_only=True extrae solamente caminos completos raíz-hoja.
        """
        if self.root is None:
            raise RuntimeError("El árbol todavía no ha sido entrenado.")
        if min_samples < 1:
            raise ValueError("min_samples debe ser mayor que cero.")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence debe estar entre 0 y 1.")

        cips = []
        root_total = int(self.root.class_counts.sum())
        if root_total == 0:
            return cips

        def dfs(node, path):
            total = int(node.class_counts.sum())
            if total == 0:
                return

            is_candidate = (not leaves_only) or node.is_leaf
            if is_candidate and path:
                neg, pos = map(int, node.class_counts)
                confidence = max(neg, pos) / total
                cip_class = int(np.argmax(node.class_counts))
                support = total / root_total

                if total >= min_samples and confidence >= min_confidence:
                    cips.append({
                        "path": path.copy(),
                        "samples": total,
                        "support": float(support),
                        "negative_support": float(neg / root_total),
                        "positive_support": float(pos / root_total),
                        "confidence": float(confidence),
                        "weight": float(support * confidence),
                        "class": cip_class,
                        "depth": len(path),
                        "class_prob": (node.class_counts / total).astype(float),
                        "class_counts": node.class_counts.copy()
                    })

            if node.is_leaf:
                return

            feat = node.split_feature
            thr = node.threshold

            if isinstance(thr, tuple):
                left_condition = (feat, "in", thr)
                right_condition = (feat, "not in", thr)
            elif self.feature_dtypes[feat]:
                left_condition = (feat, "<=", thr)
                right_condition = (feat, ">", thr)
            else:
                left_condition = (feat, "=", thr)
                right_condition = (feat, "!=", thr)

            dfs(node.left, path + [left_condition])
            dfs(node.right, path + [right_condition])

        dfs(self.root, [])
        return cips

    # ======================================================
    # CIP PREDICTION
    # ======================================================

    def predict_with_cips(self, X, cips, default_strategy="tree"):
        """Predicción por mayoría simple entre los CIPs que cubren X."""
        X = self._to_array(X)
        if default_strategy not in ("tree", "zero"):
            raise ValueError("default_strategy debe ser 'tree' o 'zero'.")

        predictions = []
        for x in X:
            votes = [
                int(cip["class"])
                for cip in cips
                if self._match_path(x, cip["path"])
            ]

            if votes:
                # En empate, se elige la clase 0 de manera determinista.
                pred = int(sum(votes) > len(votes) / 2)
            elif default_strategy == "tree":
                pred = self.predict_one(x)
            else:
                pred = 0

            predictions.append(pred)

        return np.asarray(predictions, dtype=int)
    def predict_with_cips_weighted(
        self,
        X,
        cips,
        use_confidence=True,
        use_support=True,
        use_depth=True,
        depth_lambda=0.5,
        default_strategy="tree",
        class_1_weight=1.0,
        aggregation="sum"):
        """
        Predicción ponderada por CIPs.

        aggregation='sum':
            Suma los pesos de todos los patrones de cada clase.

        aggregation='max':
            Utiliza el patrón de mayor peso por clase.

        class_1_weight:
            Factor de ponderación para la clase 1.

        Se conserva la estrategia original:
            - Sin patrones: utiliza el árbol o la clase 0.
            - Empate: utiliza la predicción del árbol.
            - Sin empate: selecciona la clase con mayor puntuación.

        Además, registra diagnósticos globales y por instancia.
        """

        if class_1_weight < 0:
            raise ValueError("class_1_weight no puede ser negativo.")

        if depth_lambda < 0:
            raise ValueError("depth_lambda no puede ser negativo.")

        if aggregation not in ("sum", "max"):
            raise ValueError("aggregation debe ser 'sum' o 'max'.")

        if default_strategy not in ("tree", "zero"):
            raise ValueError("default_strategy debe ser 'tree' o 'zero'.")

        X = self._to_array(X)

        max_depth = max(
            (cip.get("depth", 1) for cip in cips),
            default=1
        )

        # Diagnósticos globales
        self._score_diagnostics = {
            "n_instances": len(X),
            "cubiertas": 0,
            "sin_patrones": 0,
            "ambas_clases": 0,
            "solo_clase_0": 0,
            "solo_clase_1": 0,
            "empates": 0,
            "empates_exactos": 0,
            "empates_aproximados": 0,
            "tree_fallbacks": 0,
            "zero_fallbacks": 0,
            "total_patrones_cubiertos": 0,
            "total_cips_clase_0": 0,
            "total_cips_clase_1": 0
        }

        # Diagnósticos individuales
        self._instance_score_diagnostics = []

        predictions = []

        for instance_id, x in enumerate(X):

            scores = {
                0: [],
                1: []
            }

            matched_cip_indices = []

            # Evaluar todos los CIPs
            for cip_index, cip in enumerate(cips):

                if not self._match_path(x, cip["path"]):
                    continue

                cls = int(cip["class"])

                if cls not in (0, 1):
                    raise ValueError(
                        f"Clase no válida en CIP {cip_index}: {cls}"
                    )

                weight = 1.0

                if use_support:
                    weight *= float(cip.get("support", 1.0))

                if use_confidence:
                    weight *= float(cip.get("confidence", 1.0))

                if use_depth:
                    weight *= (
                        1.0 + depth_lambda *
                        (cip.get("depth", 1) / max(max_depth, 1))
                    )

                if cls == 1:
                    weight *= class_1_weight

                scores[cls].append(weight)
                matched_cip_indices.append(cip_index)

            # Número de CIPs que cubren la instancia
            n_cips_0 = len(scores[0])
            n_cips_1 = len(scores[1])

            matched_count = n_cips_0 + n_cips_1

            self._score_diagnostics[
                "total_patrones_cubiertos"
            ] += matched_count

            self._score_diagnostics[
                "total_cips_clase_0"
            ] += n_cips_0

            self._score_diagnostics[
                "total_cips_clase_1"
            ] += n_cips_1

            # Inicializar información de la instancia
            tree_prediction = None
            exact_tie = False
            near_tie = False
            decision = None

            # Caso 1: ningún CIP cubre la instancia
            if matched_count == 0:

                self._score_diagnostics["sin_patrones"] += 1

                if default_strategy == "tree":

                    tree_prediction = int(self.predict_one(x))
                    pred = tree_prediction

                    self._score_diagnostics[
                        "tree_fallbacks"
                    ] += 1

                    decision = "tree_no_patterns"

                else:

                    pred = 0

                    self._score_diagnostics[
                        "zero_fallbacks"
                    ] += 1

                    decision = "zero_no_patterns"

                score_0 = 0.0
                score_1 = 0.0

            else:

                self._score_diagnostics["cubiertas"] += 1

                # Diagnóstico de cobertura por clase
                if scores[0] and scores[1]:

                    self._score_diagnostics[
                        "ambas_clases"
                    ] += 1

                elif scores[0]:

                    self._score_diagnostics[
                        "solo_clase_0"
                    ] += 1

                else:

                    self._score_diagnostics[
                        "solo_clase_1"
                    ] += 1

                # Agregación de los pesos
                if aggregation == "sum":

                    score_0 = sum(scores[0])
                    score_1 = sum(scores[1])

                else:

                    score_0 = max(scores[0], default=0.0)
                    score_1 = max(scores[1], default=0.0)

                # Diagnóstico de empate
                near_tie = np.isclose(score_0, score_1)

                exact_tie = (
                    score_0 == score_1
                    and bool(scores[0])
                    and bool(scores[1])
                )

                if near_tie:

                    self._score_diagnostics["empates"] += 1

                    if exact_tie:
                        self._score_diagnostics[
                            "empates_exactos"
                        ] += 1
                    else:
                        self._score_diagnostics[
                            "empates_aproximados"
                        ] += 1

                    # Se conserva tu estrategia original
                    tree_prediction = int(self.predict_one(x))
                    pred = tree_prediction

                    self._score_diagnostics[
                        "tree_fallbacks"
                    ] += 1

                    decision = "tree_tie"

                else:

                    pred = int(score_1 > score_0)

                    decision = "patterns"

            # Guardar predicción
            predictions.append(pred)

            # Registrar información individual
            self._instance_score_diagnostics.append({

                "instance_id": instance_id,

                "cips_clase_0": n_cips_0,
                "cips_clase_1": n_cips_1,

                "total_cips": matched_count,

                "score_0": float(score_0),
                "score_1": float(score_1),

                # Positivo: favorece clase 1.
                # Negativo: favorece clase 0.
                "score_difference": float(score_1 - score_0),

                "exact_tie": exact_tie,
                "near_tie": near_tie,

                "tree_prediction": tree_prediction,

                "pattern_prediction": int(pred),

                "decision": decision,

                "matched_cip_indices": matched_cip_indices

            })

        return np.asarray(predictions, dtype=int)
    
    """
    def predict_with_cips_weighted(self,X,cips,use_confidence=True,use_support=True,use_depth=True,depth_lambda=0.5,default_strategy="tree",class_1_weight=1.0,aggregation="sum"):
        #Predicción ponderada por CIPs.

        #aggregation='sum': suma los pesos de todos los patrones de cada clase.
        #aggregation='max': utiliza el patrón de mayor peso por clase.
        #class_1_weight=1.0 evita favorecer o penalizar una clase a priori.
        
        if class_1_weight < 0:
            raise ValueError("class_1_weight no puede ser negativo.")
        if depth_lambda < 0:
            raise ValueError("depth_lambda no puede ser negativo.")
        if aggregation not in ("sum", "max"):
            raise ValueError("aggregation debe ser 'sum' o 'max'.")
        if default_strategy not in ("tree", "zero"):
            raise ValueError("default_strategy debe ser 'tree' o 'zero'.")

        X = self._to_array(X)
        max_depth = max((cip.get("depth", 1) for cip in cips), default=1)

        self._score_diagnostics = {
            "ambas_clases": 0,
            "solo_clase_0": 0,
            "solo_clase_1": 0,
            "sin_patrones": 0,
            "empates": 0,
            "total_patrones_cubiertos": 0
        }

        predictions = []
        for x in X:
            scores = {0: [], 1: []}

            for cip in cips:
                if not self._match_path(x, cip["path"]):
                    continue

                cls = int(cip["class"])
                weight = 1.0

                if use_support:
                    weight *= float(cip.get("support", 1.0))
                if use_confidence:
                    weight *= float(cip.get("confidence", 1.0))
                if use_depth:
                    weight *= (
                        1.0 + depth_lambda *
                        (cip.get("depth", 1) / max(max_depth, 1))
                    )
                if cls == 1:
                    weight *= class_1_weight

                scores[cls].append(weight)

            matched_count = len(scores[0]) + len(scores[1])
            self._score_diagnostics["total_patrones_cubiertos"] += matched_count

            if matched_count == 0:
                self._score_diagnostics["sin_patrones"] += 1
                pred = self.predict_one(x) if default_strategy == "tree" else 0
                predictions.append(int(pred))
                continue

            if scores[0] and scores[1]:
                self._score_diagnostics["ambas_clases"] += 1
            elif scores[0]:
                self._score_diagnostics["solo_clase_0"] += 1
            else:
                self._score_diagnostics["solo_clase_1"] += 1

            if aggregation == "sum":
                score_0 = sum(scores[0])
                score_1 = sum(scores[1])
            else:
                score_0 = max(scores[0], default=0.0)
                score_1 = max(scores[1], default=0.0)

            if np.isclose(score_0, score_1):
                self._score_diagnostics["empates"] += 1
                pred = self.predict_one(x)
            else:
                pred = int(score_1 > score_0)

            predictions.append(pred)

        return np.asarray(predictions, dtype=int)
    """
    def diagnose_cip_scores(
            self,
            X,
            cips,
            use_confidence=True,
            use_support=True,
            use_depth=True,
            depth_lambda=0.5,
            class_1_weight=1.0,
            max_instances=20):
        """Muestra las puntuaciones de los CIPs para diagnosticar sus predicciones."""
        X = self._to_array(X)
        max_depth = max(
            (cip.get("depth", 1) for cip in cips),
            default=1
        )

        records = []

        for i, x in enumerate(X):
            scores = {0: [], 1: []}
            matched = {0: [], 1: []}

            for j, cip in enumerate(cips):
                if not self._match_path(x, cip["path"]):
                    continue

                cls = int(cip["class"])
                weight = 1.0

                if use_support:
                    weight *= float(cip.get("support", 1.0))

                if use_confidence:
                    weight *= float(cip.get("confidence", 1.0))

                if use_depth:
                    weight *= (
                            1.0 + depth_lambda *
                            (cip.get("depth", 1) / max(max_depth, 1))
                    )

                if cls == 1:
                    weight *= class_1_weight

                scores[cls].append(weight)
                matched[cls].append({
                    "cip_id": j,
                    "weight": weight,
                    "depth": cip.get("depth", 1),
                    "support": cip.get("support", 1.0),
                    "confidence": cip.get("confidence", 1.0),
                    "path": cip["path"]
                })

            sum_0 = sum(scores[0])
            sum_1 = sum(scores[1])
            max_0 = max(scores[0], default=0.0)
            max_1 = max(scores[1], default=0.0)

            records.append({
                "instance": i,
                "num_cips_class_0": len(scores[0]),
                "num_cips_class_1": len(scores[1]),
                "sum_score_0": sum_0,
                "sum_score_1": sum_1,
                "max_score_0": max_0,
                "max_score_1": max_1,
                "pred_sum": (
                    self.predict_one(x)
                    if np.isclose(sum_0, sum_1)
                    else int(sum_1 > sum_0)
                ) if scores[0] or scores[1] else self.predict_one(x),
                "pred_max": (
                    self.predict_one(x)
                    if np.isclose(max_0, max_1)
                    else int(max_1 > max_0)
                ) if scores[0] or scores[1] else self.predict_one(x),
                "matched_class_0": matched[0],
                "matched_class_1": matched[1]
            })

        df = pd.DataFrame(records)

        both = df[
            (df["num_cips_class_0"] > 0) &
            (df["num_cips_class_1"] > 0)
            ].copy()

        print("\nRESUMEN DE PUNTUACIONES")
        print("Instancias analizadas:", len(df))
        print("Coincidencias con ambas clases:", len(both))

        if len(both):
            print("\nPrimeras instancias con ambas clases:")
            cols = [
                "instance",
                "num_cips_class_0",
                "num_cips_class_1",
                "sum_score_0",
                "sum_score_1",
                "max_score_0",
                "max_score_1",
                "pred_sum",
                "pred_max"
            ]
            print(both[cols].head(max_instances).to_string(index=False))

        print("\nPredicciones diferentes entre agregaciones:")
        different = df[df["pred_sum"] != df["pred_max"]]
        print("Cantidad:", len(different))

        return df
    # ======================================================
    # EXPLANATIONS
    # ======================================================

    def explain_instances_with_cips_all(self, X, cips):
        X = self._to_array(X)
        explanations = []

        for idx, x in enumerate(X):
            covered = []
            for i, cip in enumerate(cips):
                if self._match_path(x, cip["path"]):
                    covered.append({
                        "pattern_id": i,
                        "class": int(cip["class"]),
                        "support": cip.get("support", 0),
                        "confidence": cip.get("confidence", 0),
                        "depth": cip.get("depth", 0),
                        "path": cip["path"]
                    })

            explanations.append({
                "instance": idx,
                "num_patterns": len(covered),
                "patterns": covered
            })

        return explanations

    def explain_instances_with_cips(self, X, cips):
        X = self._to_array(X)
        explanations = []

        for i, x in enumerate(X):
            matched_cips = []
            for j, cip in enumerate(cips):
                if self._match_path(x, cip["path"]):
                    matched_cips.append({
                        "cip_id": j,
                        "pattern": cip,
                        "predicted_class": int(cip["class"]),
                        "confidence": cip["confidence"]
                    })

            explanations.append({
                "instance": x.tolist(),
                "instance_id": int(i),
                "matched_cips": self._json_serializable(matched_cips),
                "num_cips": len(matched_cips)
            })

        return explanations

    # ======================================================
    # CIP FILTERING AND EXPORT
    # ======================================================

    def filter_cips(self, cips, min_confidence=0.2):
        return [
            cip for cip in cips
            if cip.get("confidence", 0) >= min_confidence
        ]

    def save_cips_json(self, cips, y_train, filename):
        y_train = np.asarray(y_train)
        total_neg = int(np.sum(y_train == 0))
        total_pos = int(np.sum(y_train == 1))
        data = [self._cip_to_row(cip, total_neg, total_pos) for cip in cips]

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(
                self._json_serializable(data),
                f,
                indent=4,
                ensure_ascii=False
            )

    def _path_to_human(self, path):
        readable = []
        for feat, op, thr in path:
            name = self.feature_names[feat]
            if op == "in":
                condition = f"{name} IN {list(thr)}"
            elif op == "not in":
                condition = f"{name} NOT IN {list(thr)}"
            else:
                condition = f"{name} {op} {thr}"
            readable.append(f"[{condition}]")
        return " ∧ ".join(readable)

    def _cip_to_row(self, cip, total_neg, total_pos):
        neg_count = int(cip["class_counts"][0])
        pos_count = int(cip["class_counts"][1])
        return {
            "Pattern": self._path_to_human(cip["path"]),
            "negative Count": neg_count,
            "negative Support": neg_count / total_neg if total_neg else 0.0,
            "positive Count": pos_count,
            "positive Support": pos_count / total_pos if total_pos else 0.0
        }

    def save_cips_csv(self, cips, y_train, filename):
        y_train = np.asarray(y_train)
        total_neg = int(np.sum(y_train == 0))
        total_pos = int(np.sum(y_train == 1))
        rows = [self._cip_to_row(cip, total_neg, total_pos) for cip in cips]
        pd.DataFrame(rows).to_csv(filename, index=False)

    def save_instance_explanations_json(self, explanations, filepath):
        explanations = self._json_serializable(explanations)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(explanations, f, indent=4, ensure_ascii=False)
        print(f"Explicaciones guardadas en {filepath}")
    
    def save_instance_explanations_csv(self,explanations,filepath,y_true):
        """
        Guarda las explicaciones de las instancias y las estadísticas
        de los CIPs coincidentes.

        real_class: clase real de la instancia.
        predicted_class: clase asignada al CIP.
        covered_instances: total de instancias cubiertas por el CIP.
        negative_instances: instancias de clase 0 cubiertas.
        rinstances: instancias de clase 1 cubiertas.
        """

        rows = []
        y_true = np.asarray(y_true).reshape(-1)

        for exp in explanations:
            instance_id = exp["instance_id"]
            instance = " | ".join(map(str, exp["instance"]))

            if instance_id >= len(y_true):
                raise ValueError(
                    f"No existe una clase real para instance_id={instance_id}."
                )

            real_class = int(y_true[instance_id])

            if not exp["matched_cips"]:
                rows.append({
                    "instance_id": instance_id,
                    "instance": instance,
                    "real_class": real_class,
                    "cip_id": None,
                    "predicted_class": None,
                    "confidence": None,
                    "support": None,
                    "negative_support": None,
                    "positive_support": None,
                    "covered_instances": None,
                    "negative_instances": None,
                    "positive_instances": None,
                    "depth": None,
                    "num_conditions": 0,
                    "pattern": None
                })
                continue

            for cip in exp["matched_cips"]:
                pattern_data = cip["pattern"]
                class_counts = pattern_data["class_counts"]

                negative_instances = int(class_counts[0])
                positive_instances = int(class_counts[1])
                covered_instances = negative_instances + positive_instances

                pattern = [
                    f"[{self.feature_names[feat]}] {op} {thr}"
                    for feat, op, thr in pattern_data["path"]
                ]

                rows.append({
                    "instance_id": instance_id,
                    "instance": instance,
                    "real_class": real_class,
                    "cip_id": cip["cip_id"],
                    "predicted_class": cip["predicted_class"],
                    "confidence": cip["confidence"],
                    "support": pattern_data.get("support"),
                    "negative_support": pattern_data.get("negative_support"),
                    "positive_support": pattern_data.get("positive_support"),
                    "covered_instances": covered_instances,
                    "negative_instances": negative_instances,
                    "positive_instances": positive_instances,
                    "depth": pattern_data.get("depth"),
                    "num_conditions": len(pattern_data["path"]),
                    "pattern": " AND ".join(pattern)
                })

        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)

        return df

    """
    def save_instance_explanations_csv(self, explanations, filepath):
        rows = []
        for exp in explanations:
            instance_id = exp["instance_id"]
            instance = " | ".join(map(str, exp["instance"]))

            if not exp["matched_cips"]:
                rows.append({
                    "instance_id": instance_id,
                    "instance": instance,
                    "cip_id": None,
                    "predicted_class": None,
                    "confidence": None,
                    "support": None,
                    "depth": None,
                    "num_conditions": 0,
                    "pattern": None
                })
                continue

            for cip in exp["matched_cips"]:
                pattern = [
                    f"X[{feat}] {op} {thr}"
                    for feat, op, thr in cip["pattern"]["path"]
                ]
                rows.append({
                    "instance_id": instance_id,
                    "instance": instance,
                    "cip_id": cip["cip_id"],
                    "predicted_class": cip["predicted_class"],
                    "confidence": cip["confidence"],
                    "support": cip["pattern"].get("support"),
                    "depth": cip["pattern"].get("depth"),
                    "num_conditions": len(cip["pattern"]["path"]),
                    "pattern": " AND ".join(pattern)
                })

        pd.DataFrame(rows).to_csv(filepath, index=False)
    """
    # ======================================================
    # SERIALIZATION
    # ======================================================

    def _json_serializable(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {
                str(k): self._json_serializable(v)
                for k, v in obj.items()
            }
        if isinstance(obj, (list, tuple)):
            return [self._json_serializable(v) for v in obj]
        return obj

    # ======================================================
    # CIP PATH MATCHING
    # ======================================================

    def _match_path(self, x, path):
        """Comprueba si una instancia satisface todas las condiciones del CIP."""
        for feat, op, thr in path:
            value = x[feat]
            missing = self._is_missing(value)

            if op == "in":
                if missing:
                    return False
                try:
                    if value not in thr:
                        return False
                except (TypeError, ValueError):
                    return False
                continue

            if op == "not in":
                if missing:
                    # Los valores ausentes siguen la rama derecha.
                    continue
                try:
                    if value in thr:
                        return False
                except (TypeError, ValueError):
                    return False
                continue

            if not self.feature_dtypes[feat]:
                if op == "=":
                    if missing:
                        return False
                    try:
                        if not bool(value == thr):
                            return False
                    except (TypeError, ValueError):
                        return False
                elif op == "!=":
                    if missing:
                        continue
                    try:
                        if not bool(value != thr):
                            return False
                    except (TypeError, ValueError):
                        return False
                else:
                    raise ValueError(
                        f"Operador {op!r} no válido para una variable categórica."
                    )
                continue

            if op == "<=":
                if missing:
                    return False
                try:
                    if not float(value) <= float(thr):
                        return False
                except (TypeError, ValueError):
                    return False
            elif op == ">":
                if missing:
                    continue
                try:
                    if not float(value) > float(thr):
                        return False
                except (TypeError, ValueError):
                    return False
            else:
                raise ValueError(
                    f"Operador {op!r} no válido para una variable numérica."
                )

        return True

    # ======================================================
    # TREE PRINTING
    # ======================================================

    def print_tree(self, node=None, depth=0):
        if node is None:
            node = self.root
        if node is None:
            print("El árbol todavía no ha sido entrenado.")
            return

        indent = "  " * depth
        if node.is_leaf:
            total = int(node.class_counts.sum())
            neg, pos = node.class_counts
            pred_class = int(np.argmax(node.class_counts))
            confidence = max(neg, pos) / total if total else 0.0
            print(
                f"{indent}Leaf | samples={total} | class={pred_class} | "
                f"conf={confidence:.3f} | counts={node.class_counts}"
            )
            return

        feature = f"X[{node.split_feature}]"
        threshold = node.threshold
        feat_idx = node.split_feature

        if isinstance(threshold, tuple):
            condition = f"{feature} IN {list(threshold)}"
        elif self.feature_dtypes[feat_idx]:
            condition = f"{feature} <= {threshold:.4f}"
        else:
            condition = f"{feature} == {threshold}"

        print(f"{indent}Node | {condition}")
        self.print_tree(node.left, depth + 1)
        self.print_tree(node.right, depth + 1)

    def evaluate_cips_specificity(self, X, y, cips):
        """
        Evalúa la cobertura y especificidad de los CIPs
        respecto a las clases reales.
        """

        X = self._to_array(X)
        y = np.asarray(y).ravel()

        if len(X) != len(y):
            raise ValueError(
                "X e y deben contener el mismo número de instancias."
            )

        n_class_0 = np.sum(y == 0)
        n_class_1 = np.sum(y == 1)

        results = []

        for cip_id, cip in enumerate(cips):

            covered_indices = []

            for i, x in enumerate(X):

                if self._match_path(x, cip["path"]):
                    covered_indices.append(i)

            n_covered = len(covered_indices)

            if n_covered == 0:
                continue

            covered_y = y[covered_indices]

            n_real_0 = np.sum(covered_y == 0)
            n_real_1 = np.sum(covered_y == 1)

            precision_0 = (
                n_real_0 / n_covered
            )

            precision_1 = (
                n_real_1 / n_covered
            )

            coverage_0 = (
                n_real_0 / n_class_0
                if n_class_0 > 0 else 0.0
            )

            coverage_1 = (
                n_real_1 / n_class_1
                if n_class_1 > 0 else 0.0
            )

            associated_class = int(cip["class"])

            results.append({

                "cip_id": cip_id,

                "associated_class": associated_class,

                "support": cip.get("support", 0.0),

                "confidence_train": cip.get("confidence", 0.0),

                "depth": cip.get("depth", 0),

                "n_covered": n_covered,

                "real_class_0": int(n_real_0),

                "real_class_1": int(n_real_1),

                "precision_class_0": precision_0,

                "precision_class_1": precision_1,

                "coverage_class_0": coverage_0,

                "coverage_class_1": coverage_1,

                "specificity": (
                    precision_0
                    if associated_class == 0
                    else precision_1
                )

            })

        return pd.DataFrame(results)
