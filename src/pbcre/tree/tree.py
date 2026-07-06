# pbcre/tree/tree.py

import numpy as np
import pandas as pd
import json
from sklearn.metrics import roc_auc_score, roc_curve

from .node import PBCRENode
from src.pbcre.split.twoing import TwoingCriterion
from src.pbcre.split.functional_criterion import FunctionalCriterion
from src.pbcre.split.hybrid_twoing_functional import HybridTwoingFunctionalCriterion
from src.pbcre.split.hybrid_twoing_functional_sfs import HybridTwoingFunctionalCriterionSFS
from src.pbcre.split.hybrid_fast import HybridTwoingFunctionalCriterionFast
from src.pbcre.split.hybrid_stable import HybridTwoingFunctionalCriterionStable


class PBCRETree:
    def __init__(self, max_depth=5, min_samples_leaf=5, criterion="twoing", alpha=0.5, beta = 0.5, logistic_C=0.005, logistic_solver="liblinear", logistic_max_iter=200):
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
            self.splitter = HybridTwoingFunctionalCriterion(alpha=self.alpha, beta=self.beta, logistic_C=logistic_C, logistic_solver=logistic_solver, logistic_max_iter=logistic_max_iter)
        elif criterion == "hybrid_sfs":
            self.splitter = HybridTwoingFunctionalCriterionSFS(alpha=self.alpha,beta=self.beta, logistic_C=logistic_C, logistic_solver=logistic_solver, logistic_max_iter=logistic_max_iter)
        elif criterion == "hybrid_fast":
            self.splitter = HybridTwoingFunctionalCriterionFast(alpha=self.alpha,beta=self.beta,logistic_C=self.logistic_C, n_quantiles=10)
        elif criterion == "hybrid_stable":
            self.splitter = HybridTwoingFunctionalCriterionStable(alpha=self.alpha,beta=self.beta,logistic_C=self.logistic_C)
        else:
            raise ValueError("Unknown criterion")

        self.node_count = 0
        self.root = None
        self.nodes_ = {}
    # ======================================================
    # TRAINING
    # ======================================================
    def fit(self, X, y):
        self.feature_names = list(X.columns)
        self.node_count = 0
        self.nodes_ = {}
        self.root = self._build_tree(X, y, depth=0)
        self.n_samples_ = len(y)   # 🔥 agregar esto
    def _build_tree(self, X, y, depth):
        node = PBCRENode(node_id=self.node_count, depth=depth)
        self.nodes_[node.node_id] = node
        self.node_count += 1

        node.samples = len(y)
        node.class_counts = np.bincount(y, minlength=2)
        node.predicted_class = np.argmax(node.class_counts)

        # stopping conditions
        if (
            depth >= self.max_depth
            or len(np.unique(y)) == 1
            or len(y) < 2 * self.min_samples_leaf
        ):
            return node

        feat, thr, score, extra = self.splitter.best_split(
            X, y, self.min_samples_leaf
        )

        if feat is None:
            return node

        mask = X.iloc[:, feat] <= thr

        if mask.sum() < self.min_samples_leaf or (~mask).sum() < self.min_samples_leaf:
            return node

        left = self._build_tree(X[mask], y[mask], depth + 1)
        right = self._build_tree(X[~mask], y[~mask], depth + 1)

        node.set_split(feat, thr, left, right, extra)
        return node
    # ======================================================
    # PREDICTION
    # ======================================================
    def predict_one(self, x, node=None):
        if node is None:
            node = self.root

        if node.is_leaf:
            #return node.predicted_class
            prob = node.class_counts[1] / node.class_counts.sum()
            return int(prob >= node.leaf_threshold)

        if x[node.split_feature] <= node.threshold:
            return self.predict_one(x, node.left)
        else:
            return self.predict_one(x, node.right)
    def predict(self, X):
        if hasattr(X, "values"):
            X = X.values

        return np.array([self.predict_one(X[i]) for i in range(X.shape[0])])
    def predict_proba(self, X):
        """
        Probability of positive class for AUC optimization
        """
        if hasattr(X, "values"):
            X = X.values

        probs = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            node = self.root
            x = X[i]

            while not node.is_leaf:
                if x[node.split_feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right

            counts = node.class_counts
            #probs[i] = counts[1] / counts.sum()
            alpha = self.alpha#0.3
            probs[i] = (counts[1] + alpha) / (counts.sum() + 2 * alpha)

        return probs
    # ======================================================
    # AUC COST-COMPLEXITY PRUNING (FT4CIP)
    # ======================================================
    def _get_leaves(self, node):
        if node.is_leaf:
            return [node]
        return self._get_leaves(node.left) + self._get_leaves(node.right)
    def _predict_proba_pruned(self, X, pruned_node):
        if hasattr(X, "values"):
            X = X.values

        probs = np.zeros(len(X))

        for i in range(len(X)):
            node = self.root
            x = X[i]

            while not node.is_leaf:
                if node == pruned_node:
                    break
                if x[node.split_feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right

            counts = node.class_counts
            probs[i] = counts[1] / counts.sum()

        return probs
    def _compute_alpha(self, X_val, y_val, node, base_auc):
        if node.is_leaf:
            return -np.inf

        leaves = self._get_leaves(node)
        if len(leaves) <= 1:
            return -np.inf

        auc_pruned = roc_auc_score(
            y_val,
            self._predict_proba_pruned(X_val, node)
        )

        alpha = (auc_pruned - base_auc) / (len(leaves) - 1)
        return alpha
    def cost_complexity_pruning(self, X_val, y_val):
        """
        AUC-optimizing cost complexity pruning (FT4CIP)
        """
        best_auc = roc_auc_score(y_val, self.predict_proba(X_val))
        improved = True

        while improved:
            improved = False
            best_alpha = -np.inf
            best_node = None

            for node in self.nodes_.values():
                if not node.is_leaf:
                    alpha = self._compute_alpha(X_val, y_val, node, best_auc)
                    if alpha > best_alpha:
                        best_alpha = alpha
                        best_node = node

            if best_node is not None and best_alpha > 0:
                # Backup
                left, right = best_node.left, best_node.right

                # Apply pruning
                best_node.left = None
                best_node.right = None
                best_node.is_leaf = True

                new_auc = roc_auc_score(y_val, self.predict_proba(X_val))

                if new_auc >= best_auc:
                    best_auc = new_auc
                    improved = True
                else:
                    # Revert pruning
                    best_node.left = left
                    best_node.right = right
                    best_node.is_leaf = False
    def apply(self, X):
        """
        Devuelve el ID del nodo hoja para cada instancia.
        Usado para minería de CIPs (FT4CIP).
        """
        if hasattr(X, "values"):
            X = X.values

        leaf_ids = np.zeros(X.shape[0], dtype=int)

        for i in range(X.shape[0]):
            node = self.root
            x = X[i]

            while not node.is_leaf:
                if x[node.split_feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right

            leaf_ids[i] = node.node_id

        return leaf_ids
    def fit_leaf_thresholds(self, X, y):
        """
        Ajusta umbral óptimo por hoja usando ROC
        """
        leaf_ids = self.apply(X)

        for node in self._get_leaves(self.root):
            idx = leaf_ids == node.node_id

            if idx.sum() < 5:
                continue

            y_leaf = y[idx]
            p_leaf = np.repeat(
                node.class_counts[1] / node.class_counts.sum(),
                idx.sum()
            )

            if len(np.unique(y_leaf)) < 2:
                continue

            fpr, tpr, thr = roc_curve(y_leaf, p_leaf)
            node.leaf_threshold = thr[np.argmax(tpr - fpr)]
    def extract_cips(self, min_samples=5, min_confidence=0.6):
        cips = []

        def dfs(node, path):
            total = node.class_counts.sum()
            if total == 0:
                return

            neg, pos = node.class_counts
            confidence = max(neg, pos) / total
            cip_class = int(np.argmax(node.class_counts))
            #support = total / self.root.class_counts.sum()# esta linea considera todas las clases en el nodo
            """
                Esto consideramos solo los de la clase es decir clase 1 y clase 2
            
            if cip_class == 0:
                total_class = self.root.class_counts[0]
            else:
                total_class = self.root.class_counts[1]
            """
            support = total / self.root.class_counts.sum()

            if total >= min_samples and confidence >= min_confidence:
                cips.append({
                    "path": path.copy(),
                    "samples": int(total),
                    #"support": total / self.n_samples_,# esta es la primera versión
                    "support": support, # segunda versión
                    "confidence": float(confidence),
                    "weight": support * confidence,
                    "class": cip_class,
                    "depth": len(path),
                    "class_prob": (node.class_counts/total),
                    "class_counts": node.class_counts.copy()
                })

            if not node.is_leaf:
                dfs(node.left,  path + [(node.split_feature, "<=", node.threshold)])
                dfs(node.right, path + [(node.split_feature, ">",  node.threshold)])

        dfs(self.root, [])
        return cips
    """
    def extract_cips(self):
        #Extrae Class-Informative Patterns (CIPs)
        #Incluye clase dominante (positiva o negativa)
        
        cips = []

        def dfs(node, path):
            if node.is_leaf:
                total = node.class_counts.sum()

                if total == 0:
                    return

                neg, pos = node.class_counts
                confidence = max(neg, pos) / total
                cip_class = int(np.argmax(node.class_counts))  # 0 o 1

                cips.append({
                    "path": path.copy(),
                    "samples": int(total),
                    "confidence": float(confidence),
                    "class": cip_class,              # 🔥 CLAVE
                    "class_counts": node.class_counts.copy()  # opcional (para análisis)
                })
                return

            dfs(node.left,  path + [(node.split_feature, "<=", node.threshold)])
            dfs(node.right, path + [(node.split_feature, ">",  node.threshold)])

        dfs(self.root, [])
        return cips
    """
    def predict_with_cips(self, X, cips, default_strategy="tree"):
        if isinstance(X, pd.Series):
            X = X.to_frame().T
        elif isinstance(X, pd.DataFrame):
            X = X.values

        X = np.atleast_2d(X)
        predictions = []

        for x in X:
            votes = []

            for cip in cips:
                if self._match_path(x, cip["path"]):
                    votes.append(cip["class"])

            if votes:
                predictions.append(int(round(np.mean(votes))))
            else:
                predictions.append(self.predict_one(x))

        return np.array(predictions)
    def predict_with_cips_weighted(self,X,cips,use_confidence=True,use_support=True,use_depth=True,depth_lambda=0.5,default_strategy="tree"):

        if isinstance(X,pd.Series):
            X=X.to_frame().T
        elif isinstance(X,pd.DataFrame):
            X=X.values

        X=np.atleast_2d(X)
        predictions=[]

        max_depth=max(
            [c.get("depth",1) for c in cips],
            default=1
        )

        for x in X:

            scores={}
            matched=False

            for cip in cips:

                if self._match_path(x,cip["path"]):

                    matched=True
                    cls=cip["class"]

                    weight=1.0

                    support_alpha=0.3
                    confidence_beta=2
                    if use_support:
                        weight*=cip.get("support",1.0)# esta versión ya sale muy bien
                        #weight*=cip.get("support",1.0)**support_alpha# segunda versión ajuste

                    if use_confidence:
                        weight*=cip.get("confidence",1.0)
                        #weight*=cip.get("confidence",1.0)**confidence_beta

                    if use_depth:
                        weight*=( 1+ depth_lambda * ( cip.get("depth",1) / max_depth ))

                    positive_boost=0.6
                    if cls==1:
                        weight*=positive_boost

                    scores[cls]=scores.get(cls,0)+weight
                    #print(x,"             ", "  ", weight," \n ", cip)
                    """probs=cip["class_prob"]

                    scores[0]=(scores.get(0,0) + weight*probs[0] )
                    scores[1]=(scores.get(1,0) + weight*probs[1])"""

            if matched:
                pred=max(scores,key=scores.get)

            elif default_strategy=="tree":
                pred=self.predict_one(x)

            else:
                pred=0

            predictions.append(pred)

        return np.array(predictions)
    def explain_instances_with_cips_all(self,X,cips):
        if isinstance(X,pd.DataFrame):
            X=X.values

        explanations=[]

        for idx,x in enumerate(X):

            covered=[]

            for i,cip in enumerate(cips):

                if self._match_path(x,cip["path"]):

                    covered.append({

                        "pattern_id":i,

                        "class":cip["class"],

                        "support":cip.get(
                            "support",
                            0
                        ),

                        "confidence":cip.get(
                            "confidence",
                            0
                        ),

                        "depth":cip.get(
                            "depth",
                            0
                        ),

                        "path":cip["path"]

                    })

            explanations.append({

                "instance":idx,

                "num_patterns":len(
                    covered
                ),

                "patterns":covered

            })

        return explanations
    """
    def predict_with_cips_weighted(self,X,cips,use_confidence=True,use_support=True,default_strategy="tree"):

        if isinstance(X, pd.Series):
            X = X.to_frame().T

        elif isinstance(X, pd.DataFrame):
            X = X.values

        X = np.atleast_2d(X)

        predictions = []

        for x in X:

            scores = {}

            matched = False

            for cip in cips:

                if self._match_path(x, cip["path"]):

                    matched = True

                    cls = cip["class"]

                    weight = 1.0

                    if use_support:
                        weight *= cip.get("support", 1.0)
                    if use_confidence:
                        weight *= cip.get("confidence", 1.0)

                    scores[cls] = scores.get(cls, 0) + weight

            if matched:

                pred = max(scores, key=scores.get)

            else:

                if default_strategy == "tree":
                    pred = self.predict_one(x)

                else:
                    pred = 0

            predictions.append(pred)

        return np.array(predictions)
    """
    def filter_cips(self, cips, min_confidence=0.2):
        """
        Docstring para filter_cips
        
        :param self: Descripción
        :param cips: patrones
        :param min_confidence: soporte de los patrones se usa el 0.6
        """
        return [
            cip for cip in cips
            if cip.get("confidence", 0) >= min_confidence
        ]
    def save_cips_json(self, cips, y_train, filename):
        total_neg = int(np.sum(y_train == 0))
        total_pos = int(np.sum(y_train == 1))

        data = [
            self._cip_to_row(cip, total_neg, total_pos)
            for cip in cips
        ]

        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
    def _path_to_human(self, path):
        readable = []
        for feat, op, thr in path:
            name = self.feature_names[feat]

            if isinstance(thr, str):
                readable.append(f"[{name} = {thr}]")
            else:
                readable.append(f"[{name} {op} {thr}]")

        return " ∧ ".join(readable)
    def _cip_to_row(self, cip, total_neg, total_pos):
        neg_count = int(cip["class_counts"][0])
        pos_count = int(cip["class_counts"][1])

        return {
            "Pattern": self._path_to_human(cip["path"]),
            "negative Count": neg_count,
            "negative Support": neg_count / total_neg if total_neg > 0 else 0.0,
            "positive Count": pos_count,
            "positive Support": pos_count / total_pos if total_pos > 0 else 0.0,
        }
    def save_cips_csv(self, cips, y_train, filename):
        total_neg = int(np.sum(y_train == 0))
        total_pos = int(np.sum(y_train == 1))

        rows = [
            self._cip_to_row(cip, total_neg, total_pos)
            for cip in cips
        ]

        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
    def explain_instances_with_cips(self,X, cips):
        X = np.atleast_2d(X)
        explanations = []

        for i in range(X.shape[0]):
            x = X[i]
            matched_cips = []

            for j, cip in enumerate(cips):
                satisfied = True

                for feat, op, thr in cip["path"]:
                    if op == "<=" and not x[feat] <= thr:
                        satisfied = False
                        break
                    if op == ">" and not x[feat] > thr:
                        satisfied = False
                        break

                if satisfied:
                    matched_cips.append({
                        "cip_id": j,
                        "pattern":cip,
                        "predicted_class": int(np.argmax(cip["class_counts"])),
                        "confidence": cip["confidence"]
                    })

            explanations.append({
                "instance":x.tolist(),
                "instance_id": int(i),
                "matched_cips": self._json_serializable(matched_cips),
                "num_cips": int(len(matched_cips))
            })

        return explanations
    def save_instance_explanations_json(self,explanations, filepath):
        explanations = self._json_serializable(explanations)
        with open(filepath, "w") as f:
            json.dump(explanations, f, indent=4)

        print(f"✔ Explicaciones guardadas en {filepath}")
    def save_instance_explanations_csv(self,explanations,filepath):
        rows=[]

        for exp in explanations:

            instance_id=exp["instance_id"]

            instance=" | ".join(
                map(
                    str,
                    exp["instance"]
                )
            )

            if len(exp["matched_cips"])==0:

                rows.append({

                    "instance_id":instance_id,
                    "instance":instance,
                    "cip_id":None,
                    "predicted_class":None,
                    "confidence":None,
                    "support":None,
                    "depth":None,
                    "num_conditions":0,
                    "pattern":None

                })

            else:

                for cip in exp["matched_cips"]:

                    pattern=[]

                    for feat,op,thr in cip["pattern"]["path"]:

                        pattern.append(
                            f"X[{feat}] {op} {thr}"
                        )

                    rows.append({

                        "instance_id":instance_id,

                        "instance":instance,

                        "cip_id":cip["cip_id"],

                        "predicted_class":
                        cip["predicted_class"],

                        "confidence":
                        cip["confidence"],

                        "support":
                        cip["pattern"].get(
                            "support",
                            None
                        ),

                        "depth":
                        cip["pattern"].get(
                            "depth",
                            None
                        ),

                        "num_conditions":
                        len(
                            cip["pattern"]["path"]
                        ),

                        "pattern":
                        " AND ".join(
                            pattern
                        )

                    })

        pd.DataFrame(
            rows
        ).to_csv(
            filepath,
            index=False
        )
    def _json_serializable(self,obj):
        import numpy as np
        if isinstance(
            obj,
            (
                np.integer,
                np.int32,
                np.int64
            )
        ):
            return int(obj)

        elif isinstance(
            obj,
            (
                np.floating,
                np.float32,
                np.float64
            )
        ):
            return float(obj)

        elif isinstance(
            obj,
            np.ndarray
        ):
            return obj.tolist()

        elif isinstance(
            obj,
            dict
        ):
            return {
                str(k):
                self._json_serializable(v)
                for k,v in obj.items()
            }

        elif isinstance(
            obj,
            list
        ):
            return [
                self._json_serializable(v)
                for v in obj
            ]

        elif isinstance(
            obj,
            tuple
        ):
            return [
                self._json_serializable(v)
                for v in obj
            ]

        return obj
    def _match_path(self, x, path):
        for feat, op, thr in path:
            x_val = x[feat]

            # CATEGÓRICO
            if isinstance(thr, str):
                if op == "=" and x_val != thr:
                    return False
                if op == "!=" and x_val == thr:
                    return False

            # NUMÉRICO
            else:
                try:
                    x_val = float(x_val)
                except:
                    return False

                if op == "<=" and not (x_val <= thr):
                    return False
                if op == ">" and not (x_val > thr):
                    return False

        return True
    def print_tree(self, node=None, depth=0):
        if node is None:
            node = self.root

        indent = "  " * depth

        if node.is_leaf:
            total = node.class_counts.sum()
            neg, pos = node.class_counts
            pred_class = np.argmax(node.class_counts)
            confidence = max(neg, pos) / total if total > 0 else 0

            print(f"{indent}Leaf | samples={int(total)} | "
                f"class={pred_class} | conf={confidence:.3f} | "
                f"counts={node.class_counts}")
            return

        feature = f"X[{node.split_feature}]"

        # 🔹 Detectar tipo de split
        if isinstance(node.threshold, (int, float, np.number)):
            condition = f"{feature} <= {node.threshold:.4f}"
        else:
            condition = f"{feature} == {node.threshold}"

        print(f"{indent}Node | {condition}")

        self.print_tree(node.left, depth + 1)
        self.print_tree(node.right, depth + 1)



