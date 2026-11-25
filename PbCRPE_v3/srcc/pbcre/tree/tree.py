# srcc/pbcre/tree/tree.py

import json
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .node import PBCRENode
from srcc.pbcre.split.twoing import TwoingCriterion
from srcc.pbcre.split.functional_criterion import FunctionalCriterion
from srcc.pbcre.split.hybrid_twoing_functional import HybridTwoingFunctionalCriterion
from srcc.pbcre.split.hybrid_twoing_functional_sfs import HybridTwoingFunctionalCriterionSFS
from srcc.pbcre.split.hybrid_fast import HybridTwoingFunctionalCriterionFast
from srcc.pbcre.split.hybrid_stable import HybridTwoingFunctionalCriterionStable
from srcc.pbcre.evaluation.metrics import ClassificationMetrics, balanced_accuracy
from srcc.pbcre.evaluation.thresholds import select_threshold, resolve_current


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
        logistic_max_iter=200,
        laplace=1.0,
        smoothing="laplace",
        leaf_model="constant",
        leaf_C=0.1,
        leaf_min_samples=30,
        leaf_penalty="l2",
        scoring="balanced",
        threshold=0.5,
        random_state=42,
        verbose=False,
        **splitter_kwargs
    ):
        """
        Parámetros nuevos en PbCRPE_v1
        ------------------------------
        smoothing : "laplace" -> (pos + m) / (n + 2m), encoge hacia 0.5.
            "prior"   -> (pos + m * prior) / (n + m) (m-estimate), encoge
            hacia la proporción de positivos del entrenamiento. Útil cuando
            el prior está lejos de 0.5 (p. ej. German, 70 % positivos).
            En ambos casos m = laplace.
        laplace : fuerza del suavizado de las probabilidades de hoja.
        leaf_model : "constant" -> probabilidad = proporción (suavizada) de
            la hoja (comportamiento original).
            "logistic" -> cada nodo con al menos leaf_min_samples instancias
            y >= 5 de cada clase ajusta una regresión logística; la hoja usa
            la de su ancestro más cercano que la tenga (estilo LMT). Da
            probabilidades continuas dentro de cada hoja, lo que mejora el
            AUC y permite mover el umbral con más precisión.
        leaf_C : regularización de las logísticas de hoja (variables
            estandarizadas y one-hot).
        leaf_min_samples : tamaño mínimo de un nodo para tener modelo propio.
        leaf_penalty : "l2" (por defecto) o "l1". Con "l1" cada hoja usa
            pocas variables (coeficientes exactamente cero), lo que hace más
            legible el modelo de cada patrón.
            Antes se usaba `alpha`, que también es el peso del criterio
            híbrido; tunear alpha cambiaba las probabilidades sin querer.
        scoring : métrica del modelo funcional local en los splits
            ("balanced" = 1 - BER, "f1", "f1_weighted").
        threshold : umbral global de decisión sobre predict_proba.
            Se ajusta con fit_threshold(X_val, y_val, fpr_max=...).
        random_state : semilla para el submuestreo y los splits internos.
        verbose : imprime el criterio elegido (desactivado para no
            saturar la salida durante el tuning).
        **splitter_kwargs : parámetros extra del criterio, p. ej.
            n_quantiles, purity_mode, class_quantiles, twoing_threshold.
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.alpha = alpha
        self.beta = beta
        self.logistic_C = logistic_C
        self.logistic_solver = logistic_solver
        self.logistic_max_iter = logistic_max_iter
        self.laplace = laplace
        if smoothing not in ("laplace", "prior"):
            raise ValueError("smoothing debe ser 'laplace' o 'prior'.")
        self.smoothing = smoothing
        if leaf_model not in ("constant", "logistic"):
            raise ValueError("leaf_model debe ser 'constant' o 'logistic'.")
        self.leaf_model = leaf_model
        self.leaf_C = leaf_C
        self.leaf_min_samples = leaf_min_samples
        if leaf_penalty not in ("l1", "l2"):
            raise ValueError("leaf_penalty debe ser 'l1' o 'l2'.")
        self.leaf_penalty = leaf_penalty
        self.scoring = scoring
        self.threshold_ = float(threshold)
        self.random_state = random_state
        self.verbose = verbose

        if verbose:
            print("Criterio seleccionado            " + criterion)

        common = dict(
            alpha=self.alpha,
            beta=self.beta,
            logistic_C=self.logistic_C,
            logistic_solver=self.logistic_solver,
            logistic_max_iter=self.logistic_max_iter,
            scoring=self.scoring,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        if criterion == "twoing":
            self.splitter = TwoingCriterion()
        elif criterion == "functional":
            self.splitter = FunctionalCriterion()
        elif criterion == "hybrid":
            self.splitter = HybridTwoingFunctionalCriterion(**common)
        elif criterion == "hybrid_sfs":
            self.splitter = HybridTwoingFunctionalCriterionSFS(
                **common, **splitter_kwargs)
        elif criterion == "hybrid_fast":
            fast_kwargs = {"n_quantiles": 10}
            fast_kwargs.update(splitter_kwargs)
            self.splitter = HybridTwoingFunctionalCriterionFast(
                **common, **fast_kwargs)
        elif criterion == "hybrid_stable":
            self.splitter = HybridTwoingFunctionalCriterionStable(
                **common, **splitter_kwargs)
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

        y = y.astype(int)
        self.class_totals_ = np.bincount(y, minlength=2)
        self.class_prior_ = self.class_totals_ / self.class_totals_.sum()

        # el criterio conoce el prior de la raíz (pureza balanceada) y
        # reinicia su generador para que cada fit sea reproducible
        if hasattr(self.splitter, "set_class_prior"):
            self.splitter.set_class_prior(self.class_prior_[1])
        if hasattr(self.splitter, "reset_rng"):
            self.splitter.reset_rng()

        self.root = self._build_tree(X, y, depth=0)

        if self.leaf_model == "logistic":
            self._fit_node_models(X, y)
        return self

    # ======================================================
    # FUNCTIONAL LEAVES (leaf_model="logistic")
    # ======================================================

    def _as_frame(self, X):
        """Convierte a DataFrame con los nombres y tipos del entrenamiento."""
        if isinstance(X, pd.DataFrame):
            df = X.reset_index(drop=True).copy()
            df.columns = self.feature_names
        else:
            df = pd.DataFrame(self._to_array(X), columns=self.feature_names)
        for name, is_num in zip(self.feature_names, self.feature_dtypes):
            if is_num:
                df[name] = pd.to_numeric(df[name], errors="coerce").astype(float)
            else:
                df[name] = df[name].astype(str)
        return df

    def _fit_node_models(self, X, y):
        num = [n for n, d in zip(self.feature_names, self.feature_dtypes) if d]
        cat = [n for n, d in zip(self.feature_names, self.feature_dtypes) if not d]
        transformers = []
        if num:
            transformers.append(("num", StandardScaler(), num))
        if cat:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore",
                                                      sparse_output=False), cat))
        self.encoder_ = ColumnTransformer(transformers)
        X_df = self._as_frame(X)
        X_enc = self.encoder_.fit_transform(X_df)

        X_arr = self._to_array(X)
        members = {}
        for i, x in enumerate(X_arr):
            node = self.root
            members.setdefault(node.node_id, []).append(i)
            while not node.is_leaf:
                value = self._get_value(x, node.split_feature)
                node = (node.left if self._go_left(value, node.threshold,
                                                   node.split_feature)
                        else node.right)
                members.setdefault(node.node_id, []).append(i)

        def fit_rec(node, inherited, source):
            model = inherited
            idx = np.asarray(members.get(node.node_id, []), dtype=int)
            if len(idx) >= self.leaf_min_samples:
                counts = np.bincount(y[idx], minlength=2)
                if counts.min() >= 5:
                    if self.leaf_penalty == "l1":
                        model = LogisticRegression(
                            C=self.leaf_C, penalty="l1", solver="liblinear",
                            max_iter=2000).fit(X_enc[idx], y[idx])
                    else:
                        model = LogisticRegression(
                            C=self.leaf_C, max_iter=2000,
                            solver="lbfgs").fit(X_enc[idx], y[idx])
                    source = node.node_id
            node.leaf_lr = model
            # nodo dueño del modelo (puede ser un ancestro)
            node.leaf_lr_source = source if model is not None else None
            if not node.is_leaf:
                fit_rec(node.left, model, source)
                fit_rec(node.right, model, source)

        fit_rec(self.root, None, None)

    def leaf_model_coefficients(self, node, top=None):
        """
        Coeficientes de la logística que usa un nodo (leaf_model="logistic").
        Devuelve DataFrame (variable, coeficiente, odds_ratio) ordenado por
        |coeficiente|; las numéricas están estandarizadas.
        """
        lr = getattr(node, "leaf_lr", None)
        if lr is None:
            return pd.DataFrame(columns=["variable", "coeficiente", "odds_ratio"])
        names = [n.split("__", 1)[-1]
                 for n in self.encoder_.get_feature_names_out()]
        coef = lr.coef_[0]
        df = pd.DataFrame({"variable": names, "coeficiente": coef,
                           "odds_ratio": np.exp(coef)})
        df = df.reindex(df["coeficiente"].abs().sort_values(ascending=False).index)
        return df.head(top) if top else df

    def refit_leaves(self, X, y):
        """
        Mantiene la ESTRUCTURA (splits) y reestima con (X, y) los conteos de
        clase de todos los nodos y, si leaf_model="logistic", sus logísticas.

        Uso típico: crecer y podar con una parte del entrenamiento y después
        aprovechar el entrenamiento completo para las probabilidades.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        X = X.reset_index(drop=True)
        y = np.asarray(y).reshape(-1).astype(int)

        for node in self._reachable_nodes():
            node.class_counts = np.zeros(2, dtype=int)

        for x, label in zip(self._to_array(X), y):
            node = self.root
            node.class_counts[label] += 1
            while not node.is_leaf:
                value = self._get_value(x, node.split_feature)
                node = (node.left if self._go_left(value, node.threshold,
                                                   node.split_feature)
                        else node.right)
                node.class_counts[label] += 1

        for node in self._reachable_nodes():
            node.samples = int(node.class_counts.sum())
            node.predicted_class = int(np.argmax(node.class_counts))

        self.class_totals_ = np.bincount(y, minlength=2)
        self.class_prior_ = self.class_totals_ / self.class_totals_.sum()
        self.n_samples_ = len(y)

        if self.leaf_model == "logistic":
            self._fit_node_models(X, y)
        return self

    def leaf_paths(self):
        """{node_id de hoja: lista de condiciones raíz-hoja}."""
        paths = {}

        def dfs(node, path):
            if node.is_leaf:
                paths[node.node_id] = path
                return
            feat, thr = node.split_feature, node.threshold
            if isinstance(thr, tuple):
                lc, rc = (feat, "in", thr), (feat, "not in", thr)
            elif self.feature_dtypes[feat]:
                lc, rc = (feat, "<=", thr), (feat, ">", thr)
            else:
                lc, rc = (feat, "=", thr), (feat, "!=", thr)
            dfs(node.left, path + [lc])
            dfs(node.right, path + [rc])

        dfs(self.root, [])
        return paths

    def _proba_for_nodes(self, X, end_nodes):
        """Probabilidad de la clase 1 dado el nodo terminal de cada fila."""
        if self.leaf_model != "logistic":
            return np.asarray([self._leaf_proba(n) for n in end_nodes],
                              dtype=float)

        out = np.zeros(len(end_nodes), dtype=float)
        X_enc = None
        groups = {}
        for i, node in enumerate(end_nodes):
            lr = getattr(node, "leaf_lr", None)
            if lr is None:
                out[i] = self._leaf_proba(node)
            else:
                groups.setdefault(id(lr), (lr, []))[1].append(i)
        if groups:
            X_enc = self.encoder_.transform(self._as_frame(X))
            for lr, rows in groups.values():
                rows = np.asarray(rows)
                out[rows] = lr.predict_proba(X_enc[rows])[:, 1]
        return out

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

    def _leaf_proba(self, node):
        """Probabilidad suavizada de la clase positiva en un nodo."""
        counts = node.class_counts
        m = self.laplace
        if self.smoothing == "prior":
            prior = float(self.class_prior_[1])
            denominator = counts.sum() + m
            return (counts[1] + m * prior) / denominator if denominator > 0 else prior
        denominator = counts.sum() + 2 * m
        return (counts[1] + m) / denominator if denominator > 0 else 0.0

    def _find_leaf(self, x, stop_at=None):
        node = self.root
        while not node.is_leaf:
            if node is stop_at:
                break
            value = self._get_value(x, node.split_feature)
            if self._go_left(value, node.threshold, node.split_feature):
                node = node.left
            else:
                node = node.right
        return node

    def predict_one(self, x, node=None, threshold=None):
        """
        PbCRPE_v1: la decisión usa un umbral GLOBAL sobre la probabilidad
        suavizada (antes: umbral por hoja, que fit_leaf_thresholds dejaba
        en +inf y anulaba la clase positiva).
        """
        if self.root is None:
            raise RuntimeError("El árbol todavía no ha sido entrenado.")
        t = self.threshold_ if threshold is None else threshold
        leaf = self._find_leaf(x) if node is None else node
        while not leaf.is_leaf:
            value = self._get_value(x, leaf.split_feature)
            leaf = leaf.left if self._go_left(value, leaf.threshold, leaf.split_feature) else leaf.right
        x_row = np.asarray(x, dtype=object).reshape(1, -1)
        return int(self._proba_for_nodes(x_row, [leaf])[0] >= t)

    def predict(self, X, threshold=None):
        """Predice 1 si predict_proba(X) >= threshold (por defecto threshold_)."""
        t = self.threshold_ if threshold is None else threshold
        return (self.predict_proba(X) >= t).astype(int)

    def predict_proba(self, X):
        """Devuelve la probabilidad suavizada (Laplace) de la clase positiva."""
        if self.root is None:
            raise RuntimeError("El árbol todavía no ha sido entrenado.")
        X_arr = self._to_array(X)
        leaves = [self._find_leaf(x) for x in X_arr]
        return self._proba_for_nodes(X_arr, leaves)

    # ======================================================
    # GLOBAL DECISION THRESHOLD
    # ======================================================

    def fit_threshold(self, X_val, y_val, fpr_max=None, fnr_max=None):
        """
        Ajusta el umbral global de decisión en VALIDACIÓN.

        Sin restricciones   -> minimiza el BER (equivale a maximizar Youden).
        fpr_max=float       -> minimiza error tipo II con error tipo I <= fpr_max.
        fnr_max=float       -> minimiza error tipo I con error tipo II <= fnr_max.
        "current" en cualquiera de los dos usa como cota el error del umbral
        actual (mejorar uno de los errores sin empeorar el otro).

        Devuelve un dict con los errores esperados en validación.
        """
        y_val = np.asarray(y_val).reshape(-1).astype(int)
        probs = self.predict_proba(X_val)
        fpr_max, fnr_max = resolve_current(y_val, probs, self.threshold_,
                                           fpr_max, fnr_max)
        threshold, info = select_threshold(y_val, probs,
                                           fpr_max=fpr_max, fnr_max=fnr_max)
        self.threshold_ = threshold
        return info

    # ======================================================
    # AUC COST-COMPLEXITY PRUNING
    # ======================================================

    def _get_leaves(self, node):
        if node.is_leaf:
            return [node]
        return self._get_leaves(node.left) + self._get_leaves(node.right)

    def _predict_proba_pruned(self, X, pruned_node):
        X_arr = self._to_array(X)
        nodes = [self._find_leaf(x, stop_at=pruned_node) for x in X_arr]
        return self._proba_for_nodes(X_arr, nodes)

    @staticmethod
    def _pruning_score(y_val, probs, metric):
        """
        auc      -> AUC ROC (independiente del umbral).
        balanced -> máximo de 1 - BER alcanzable con algún umbral
                    (= (1 + Youden J) / 2). También independiente del
                    umbral, pero alineado con el BER.
        """
        if metric == "auc":
            return roc_auc_score(y_val, probs)
        fpr, tpr, _ = roc_curve(y_val, probs)
        return 0.5 * (1.0 + np.max(tpr - fpr))

    def cost_complexity_pruning(self, X_val, y_val, metric="auc", tol=0.0,
                                min_leaves=1):
        """
        Poda en validación.

        PbCRPE_v1:
        - metric="auc" (original) o "balanced" (alineada con el BER).
        - En cada ronda se evalúan TODAS las podas y se aplica la de mayor
          ganancia por hoja eliminada; se acepta si la métrica no baja más
          que `tol` (a igualdad se prefiere el árbol más simple).
          Antes la poda se detenía en el primer rechazo.
        - Las probabilidades usan el mismo suavizado de Laplace que predict.
        - min_leaves: no se aplica una poda que deje menos hojas. Evita que el
          árbol colapse a la raíz (sin patrones), algo frecuente con
          leaf_model="logistic", donde la logística de la raíz ya es fuerte.
        Devuelve el número de podas aplicadas.
        """
        if metric not in ("auc", "balanced"):
            raise ValueError("metric debe ser 'auc' o 'balanced'.")
        y_val = np.asarray(y_val).reshape(-1)
        if len(np.unique(y_val)) < 2:
            raise ValueError("Se necesitan ambas clases en y_val para podar.")

        X_val = self._to_array(X_val)
        base = self._pruning_score(y_val, self.predict_proba(X_val), metric)
        n_pruned = 0

        while True:
            best_gain = -np.inf
            best_node = None
            best_value = None

            total_leaves = len(self._get_leaves(self.root))
            for node in self._reachable_nodes():
                if node.is_leaf:
                    continue
                n_leaves = len(self._get_leaves(node))
                if total_leaves - (n_leaves - 1) < min_leaves:
                    continue
                value = self._pruning_score(
                    y_val, self._predict_proba_pruned(X_val, node), metric)
                gain = (value - base) / max(n_leaves - 1, 1)
                if gain > best_gain:
                    best_gain = gain
                    best_node = node
                    best_value = value

            if best_node is None or best_value < base - tol:
                break

            best_node.left = None
            best_node.right = None
            best_node.is_leaf = True
            base = best_value
            n_pruned += 1

        return n_pruned

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

    def fit_leaf_thresholds(self, X, y, fpr_max=None):
        """
        OBSOLETO en PbCRPE_v1.

        La versión original elegía un umbral por hoja con roc_curve sobre una
        puntuación constante; el umbral resultante era +inf y la hoja nunca
        predecía la clase positiva (aumentaba el error tipo II).
        Ahora delega en fit_threshold (umbral global).
        """
        import warnings
        warnings.warn(
            "fit_leaf_thresholds está obsoleto; use fit_threshold.",
            DeprecationWarning, stacklevel=2)
        return self.fit_threshold(X, y, fpr_max=fpr_max)

    def extract_cips(
        self,
        min_samples=5,
        min_confidence=0.6,
        leaves_only=False,
        class_mode="majority",
        min_lift=None
    ):
        """
        Extrae CIPs de los nodos del árbol.

        leaves_only=False conserva el comportamiento original y permite
        patrones de nodos internos y hojas.
        leaves_only=True extrae solamente caminos completos raíz-hoja.

        PbCRPE_v1
        ---------
        class_mode="majority" (original): clase = mayoría del nodo y
            confianza = proporción mayoritaria.
        class_mode="balanced": clase = la que más se enriquece respecto al
            prior (lift = p_c / prior_c). Un nodo con 40 % de positivos cuando
            el prior es 10 % se etiqueta como positivo. En este modo
            min_confidence se aplica a la confianza balanceada (proporción
            reponderada por el prior, rango [0.5, 1]).
        min_lift : si se indica, exige lift >= min_lift para la clase del CIP.

        Cada CIP incluye además:
            class_support : count_c / N_c (soporte relativo a su clase).
            lift          : p_c / prior_c.
            balanced_confidence : confianza con clases reponderadas.
        """
        if class_mode not in ("majority", "balanced"):
            raise ValueError("class_mode debe ser 'majority' o 'balanced'.")
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
        totals = self.root.class_counts.astype(float)
        prior = totals / totals.sum()

        def dfs(node, path):
            total = int(node.class_counts.sum())
            if total == 0:
                return

            is_candidate = (not leaves_only) or node.is_leaf
            if is_candidate and path:
                neg, pos = map(int, node.class_counts)
                support = total / root_total
                p_node = node.class_counts / total
                lifts = np.divide(p_node, prior,
                                  out=np.zeros(2), where=prior > 0)
                balanced_conf_vec = (lifts / lifts.sum()
                                     if lifts.sum() > 0 else np.array([0.5, 0.5]))

                if class_mode == "majority":
                    cip_class = int(np.argmax(node.class_counts))
                    confidence = max(neg, pos) / total
                    filter_conf = confidence
                else:
                    cip_class = int(np.argmax(lifts))
                    confidence = float(p_node[cip_class])
                    filter_conf = float(balanced_conf_vec[cip_class])

                class_support = (node.class_counts[cip_class] /
                                 totals[cip_class] if totals[cip_class] else 0.0)
                lift_ok = min_lift is None or lifts[cip_class] >= min_lift

                if total >= min_samples and filter_conf >= min_confidence and lift_ok:
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
                        "class_counts": node.class_counts.copy(),
                        "class_support": float(class_support),
                        "lift": float(lifts[cip_class]),
                        "balanced_confidence": float(balanced_conf_vec[cip_class])
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
                n_pos = sum(votes)
                if 2 * n_pos == len(votes):
                    # PbCRPE_v1: el empate lo decide el árbol (antes: clase 0,
                    # lo que sesgaba hacia el error tipo II).
                    pred = self.predict_one(x)
                else:
                    pred = int(n_pos > len(votes) / 2)
            elif default_strategy == "tree":
                pred = self.predict_one(x)
            else:
                pred = 0

            predictions.append(pred)

        return np.asarray(predictions, dtype=int)

    @staticmethod
    def _cip_weight(cip, max_depth, use_confidence=True, use_support=True,
                    use_depth=True, depth_lambda=0.5, class_1_weight=1.0,
                    support_mode="global"):
        """
        Peso de un CIP.
        support_mode="global" (original): support = n_nodo / N.
            Favorece a la clase mayoritaria.
        support_mode="class": class_support = count_c / N_c.
            Cada patrón se mide respecto al tamaño de su propia clase.
        """
        weight = 1.0
        if use_support:
            key = "class_support" if support_mode == "class" else "support"
            weight *= float(cip.get(key, cip.get("support", 1.0)))
        if use_confidence:
            weight *= float(cip.get("confidence", 1.0))
        if use_depth:
            weight *= 1.0 + depth_lambda * (cip.get("depth", 1) / max(max_depth, 1))
        if int(cip["class"]) == 1:
            weight *= class_1_weight
        return weight

    def cip_scores(
        self,
        X,
        cips,
        use_confidence=True,
        use_support=True,
        use_depth=True,
        depth_lambda=0.5,
        class_1_weight=1.0,
        aggregation="sum",
        support_mode="class",
        tree_smoothing=0.1
    ):
        """
        Puntuación continua de la clase positiva a partir de los CIPs:

            s = (score_1 + k * p_arbol) / (score_0 + score_1 + k)

        con k = tree_smoothing. El término del árbol evita que muchas
        instancias cubiertas sólo por CIPs positivos queden empatadas en
        s = 1, lo que impediría elegir un umbral que respete la cota de
        error tipo I. Si ningún CIP cubre la instancia, s = p_arbol.
        Sobre esta puntuación se puede elegir un umbral con restricción
        de error tipo I (ver fit_cip_threshold).
        """
        if aggregation not in ("sum", "max"):
            raise ValueError("aggregation debe ser 'sum' o 'max'.")
        X = self._to_array(X)
        max_depth = max((cip.get("depth", 1) for cip in cips), default=1)
        out = np.zeros(len(X), dtype=float)
        covered = np.zeros(len(X), dtype=bool)
        p_tree_all = self.predict_proba(X)

        for i, x in enumerate(X):
            scores = {0: [], 1: []}
            for cip in cips:
                if self._match_path(x, cip["path"]):
                    scores[int(cip["class"])].append(self._cip_weight(
                        cip, max_depth, use_confidence, use_support,
                        use_depth, depth_lambda, class_1_weight, support_mode))

            p_tree = p_tree_all[i]
            if not scores[0] and not scores[1]:
                out[i] = p_tree
                continue

            covered[i] = True
            agg = sum if aggregation == "sum" else (lambda v: max(v, default=0.0))
            s0, s1 = agg(scores[0]), agg(scores[1])
            k = tree_smoothing
            denom = s0 + s1 + k
            out[i] = (s1 + k * p_tree) / denom if denom > 0 else p_tree

        self._cip_coverage = covered
        return out

    def fit_cip_threshold(self, X_val, y_val, cips, fpr_max=None,
                          fnr_max=None, **score_kwargs):
        """
        Umbral para cip_scores, elegido en VALIDACIÓN.
        fpr_max / fnr_max igual que en fit_threshold ("current" usa el
        umbral actual cip_threshold_, por defecto 0.5).
        """
        y_val = np.asarray(y_val).reshape(-1).astype(int)
        s = self.cip_scores(X_val, cips, **score_kwargs)
        current = getattr(self, "cip_threshold_", 0.5)
        fpr_max, fnr_max = resolve_current(y_val, s, current, fpr_max, fnr_max)
        threshold, info = select_threshold(y_val, s, fpr_max=fpr_max,
                                           fnr_max=fnr_max)
        self.cip_threshold_ = threshold
        self.cip_score_kwargs_ = dict(score_kwargs)
        return info

    def predict_with_cips_scored(self, X, cips, threshold=None, **score_kwargs):
        """Predicción con cip_scores y umbral (por defecto cip_threshold_)."""
        if threshold is None:
            threshold = getattr(self, "cip_threshold_", 0.5)
        if not score_kwargs:
            score_kwargs = getattr(self, "cip_score_kwargs_", {})
        return (self.cip_scores(X, cips, **score_kwargs) >= threshold).astype(int)

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
        aggregation="sum",
        support_mode="global"
    ):
        """
        Predicción ponderada por CIPs (votación dura, API original).

        aggregation='sum': suma los pesos de todos los patrones de cada clase.
        aggregation='max': utiliza el patrón de mayor peso por clase.
        support_mode='class' (PbCRPE_v1) usa el soporte relativo a cada clase.
        Para controlar el error tipo I/II use cip_scores + fit_cip_threshold.
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
                scores[int(cip["class"])].append(self._cip_weight(
                    cip, max_depth, use_confidence, use_support, use_depth,
                    depth_lambda, class_1_weight, support_mode))

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

    def diagnose_cip_scores(
            self,
            X,
            cips,
            use_confidence=True,
            use_support=True,
            use_depth=True,
            depth_lambda=0.5,
            class_1_weight=1.0,
            max_instances=20,
            support_mode="global"
    ):
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
                weight = self._cip_weight(
                    cip, max_depth, use_confidence, use_support, use_depth,
                    depth_lambda, class_1_weight, support_mode)

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

    @staticmethod
    def _fmt_value(v):
        if isinstance(v, (float, np.floating)):
            return f"{float(v):g}"
        return str(v)

    def simplify_path(self, path):
        """
        Fusiona condiciones redundantes de un camino (sólo para mostrarlo):
          A > 3 ∧ A > 7      -> A > 7
          A > 3 ∧ A <= 9     -> 3 < A <= 9
          B != x ∧ B != y    -> B NOT IN [x, y]
        Devuelve una lista de textos, uno por variable, en orden de aparición.
        """
        order, info = [], {}
        for feat, op, thr in path:
            if feat not in info:
                order.append(feat)
                info[feat] = {"lo": None, "hi": None, "eq": None,
                              "in": None, "not_in": []}
            d = info[feat]
            if op == ">":
                d["lo"] = thr if d["lo"] is None else max(d["lo"], thr)
            elif op == "<=":
                d["hi"] = thr if d["hi"] is None else min(d["hi"], thr)
            elif op == "=":
                d["eq"] = thr
            elif op == "!=":
                d["not_in"].append(thr)
            elif op == "in":
                d["in"] = list(thr) if d["in"] is None else [
                    v for v in d["in"] if v in thr]
            elif op == "not in":
                d["not_in"].extend(list(thr))

        f = self._fmt_value
        parts = []
        for feat in order:
            name = self.feature_names[feat]
            d = info[feat]
            if d["eq"] is not None:
                parts.append(f"{name} = {f(d['eq'])}")
                continue
            if d["in"] is not None:
                vals = [v for v in d["in"] if v not in d["not_in"]]
                parts.append(f"{name} IN [{', '.join(map(f, vals))}]")
            elif d["not_in"]:
                vals = list(dict.fromkeys(d["not_in"]))
                if len(vals) == 1:
                    parts.append(f"{name} != {f(vals[0])}")
                else:
                    parts.append(f"{name} NOT IN [{', '.join(map(f, vals))}]")
            if d["lo"] is not None and d["hi"] is not None:
                parts.append(f"{f(d['lo'])} < {name} <= {f(d['hi'])}")
            elif d["lo"] is not None:
                parts.append(f"{name} > {f(d['lo'])}")
            elif d["hi"] is not None:
                parts.append(f"{name} <= {f(d['hi'])}")
        return parts

    def _path_to_human(self, path):
        return " ∧ ".join(f"[{c}]" for c in self.simplify_path(path))

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
        positive_instances: instancias de clase 1 cubiertas.
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
