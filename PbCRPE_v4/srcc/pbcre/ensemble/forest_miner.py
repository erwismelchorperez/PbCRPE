# srcc/pbcre/ensemble/forest_miner.py
"""
PbCRPE v4: minado de patrones de VARIOS árboles y clasificación por patrones.

1. Bosque de minado: n_trees árboles PBCRETree, cada uno con
   - bootstrap de filas (estratificado, conserva la proporción de clases),
   - subconjunto aleatorio de variables (feature_fraction),
   - sin poda, para obtener patrones diversos.
2. Cada nodo (excepto la raíz) de cada árbol produce un patrón candidato.
   Los patrones se normalizan (condiciones redundantes fusionadas), se
   eliminan duplicados y sus estadísticas se RECALCULAN con todo el
   conjunto de entrenamiento (no con el bootstrap del árbol que lo generó).
3. Filtros: mínimo de instancias, clase por lift (la que más se enriquece
   respecto a su proporción global), lift mínimo, confianza balanceada
   mínima y, opcionalmente, significancia (Fisher exacta + Benjamini-
   Hochberg).
4. Clasificación, dos modos (predictor):
   "pesos": cada patrón es una variable binaria (cubre / no cubre) de una
            regresión logística con L1 (C elegido por CV interna, AUC). Cada
            patrón recibe un peso; muchos quedan en cero (selección).
   "votacion" (estilo PBC4cip, Loyola-González et al.):
       votos_c(x) = sum(soporte_c(p) : p de clase c cubre x)
                    / sum(soporte_c(p) : p de clase c)
       puntuación(x) = votos_1 / (votos_0 + votos_1)
   soporte_c(p) = instancias de clase c cubiertas por p / total de clase c,
   lo que hace la votación insensible al desbalance.
"""

import numpy as np
import pandas as pd

from srcc.pbcre.tree.tree import PBCRETree


class PBCREPatternForest:

    def __init__(self, n_trees=30, max_depth=4, min_samples_leaf=10,
                 criterion="twoing", feature_fraction=0.6, bootstrap=True,
                 min_samples_cip=10, min_confidence=0.55, min_lift=1.1,
                 max_q=None, predictor="pesos", random_state=42, n_jobs=1,
                 verbose=False):
        if predictor not in ("pesos", "votacion"):
            raise ValueError("predictor debe ser 'pesos' o 'votacion'.")
        self.predictor = predictor
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.feature_fraction = feature_fraction
        self.bootstrap = bootstrap
        self.min_samples_cip = min_samples_cip
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.max_q = max_q
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    # ------------------------------------------------------------------
    # representación normalizada de un patrón
    # ------------------------------------------------------------------
    def _normalize(self, conditions):
        """
        conditions: lista de (columna_global, op, valor).
        Devuelve un dict {col: restricción} con:
          numéricas: {"lo": >, "hi": <=}
          categóricas: {"eq": v} o {"in": set} y/o {"not_in": set}
        """
        norm = {}
        for col, op, thr in conditions:
            d = norm.setdefault(col, {})
            if op == ">":
                d["lo"] = thr if "lo" not in d else max(d["lo"], thr)
            elif op == "<=":
                d["hi"] = thr if "hi" not in d else min(d["hi"], thr)
            elif op == "=":
                d["eq"] = thr
            elif op == "!=":
                d.setdefault("not_in", set()).add(thr)
            elif op == "in":
                s = set(thr)
                d["in"] = s if "in" not in d else d["in"] & s
            elif op == "not in":
                d.setdefault("not_in", set()).update(thr)
        for d in norm.values():
            if "eq" in d:
                d.pop("in", None)
                d.pop("not_in", None)
            elif "in" in d and "not_in" in d:
                d["in"] = d["in"] - d.pop("not_in")
        return norm

    @staticmethod
    def _key(norm):
        items = []
        for col in sorted(norm):
            d = norm[col]
            items.append((col,
                          float(d["lo"]) if "lo" in d else None,
                          float(d["hi"]) if "hi" in d else None,
                          str(d["eq"]) if "eq" in d else None,
                          tuple(sorted(map(str, d["in"]))) if "in" in d else None,
                          tuple(sorted(map(str, d["not_in"]))) if "not_in" in d else None))
        return tuple(items)

    def _mask(self, X, norm):
        m = np.ones(len(X), dtype=bool)
        for col, d in norm.items():
            v = X.iloc[:, col]
            if "lo" in d or "hi" in d:
                vals = pd.to_numeric(v, errors="coerce").to_numpy(dtype=float)
                if "lo" in d:
                    m &= ~(vals <= d["lo"])          # NaN va a la derecha (>)
                if "hi" in d:
                    m &= vals <= d["hi"]
            else:
                vals = v.astype(str).to_numpy()
                if "eq" in d:
                    m &= vals == str(d["eq"])
                if "in" in d:
                    m &= np.isin(vals, [str(x) for x in d["in"]])
                if "not_in" in d:
                    m &= ~np.isin(vals, [str(x) for x in d["not_in"]])
        return m

    def _text(self, norm):
        f = lambda v: f"{float(v):g}" if isinstance(v, (int, float, np.floating, np.integer)) else str(v)
        parts = []
        for col in norm:                       # orden de aparición
            name = self.feature_names_[col]
            d = norm[col]
            if "eq" in d:
                parts.append(f"[{name} = {f(d['eq'])}]")
                continue
            if "in" in d:
                parts.append(f"[{name} IN [{', '.join(sorted(map(f, d['in'])))}]]")
            if d.get("not_in"):
                vals = sorted(map(f, d["not_in"]))
                parts.append(f"[{name} != {vals[0]}]" if len(vals) == 1
                             else f"[{name} NOT IN [{', '.join(vals)}]]")
            if "lo" in d and "hi" in d:
                parts.append(f"[{f(d['lo'])} < {name} <= {f(d['hi'])}]")
            elif "lo" in d:
                parts.append(f"[{name} > {f(d['lo'])}]")
            elif "hi" in d:
                parts.append(f"[{name} <= {f(d['hi'])}]")
        return " ∧ ".join(parts)

    # ------------------------------------------------------------------
    def _grow_one(self, t, X, y):
        rng = np.random.default_rng(self.random_state + 1000 * t)
        n, p = X.shape
        k = max(1, int(round(self.feature_fraction * p)))
        cols = np.sort(rng.choice(p, size=k, replace=False))
        if self.bootstrap:
            rows = np.concatenate([rng.choice(np.flatnonzero(y == c),
                                              size=int(np.sum(y == c)), replace=True)
                                   for c in (0, 1)])
        else:
            rows = np.arange(n)
        Xt = X.iloc[rows, cols].reset_index(drop=True)
        yt = y[rows]
        tree = PBCRETree(max_depth=self.max_depth,
                         min_samples_leaf=self.min_samples_leaf,
                         criterion=self.criterion, leaf_model="constant",
                         random_state=self.random_state + t).fit(Xt, yt)
        cips = tree.extract_cips(min_samples=self.min_samples_leaf,
                                 min_confidence=0.0, leaves_only=False,
                                 class_mode="balanced")
        return [[(int(cols[f]), op, thr) for f, op, thr in c["path"]] for c in cips]

    def fit(self, X, y):
        X = X.reset_index(drop=True)
        y = np.asarray(y).astype(int)
        self.feature_names_ = list(X.columns)
        if self.n_jobs != 1:
            from joblib import Parallel, delayed
            paths = Parallel(n_jobs=self.n_jobs)(
                delayed(self._grow_one)(t, X, y) for t in range(self.n_trees))
        else:
            paths = []
            for t in range(self.n_trees):
                paths.append(self._grow_one(t, X, y))
                if self.verbose:
                    print(f"    árbol {t + 1}/{self.n_trees}")

        # agrupar patrones idénticos y contar en cuántos árboles aparecen
        pool = {}
        for t, tree_paths in enumerate(paths):
            for cond in tree_paths:
                norm = self._normalize(cond)
                key = self._key(norm)
                if key not in pool:
                    pool[key] = {"norm": norm, "trees": set()}
                pool[key]["trees"].add(t)

        N = np.bincount(y, minlength=2).astype(float)
        prior = N / N.sum()
        rows = []
        for key, item in pool.items():
            m = self._mask(X, item["norm"])
            cnt = np.array([np.sum(m & (y == 0)), np.sum(m & (y == 1))], float)
            n_cov = cnt.sum()
            if n_cov < self.min_samples_cip:
                continue
            p_node = cnt / n_cov
            lift = p_node / prior
            cls = int(np.argmax(lift))
            bal = lift / lift.sum()
            if lift[cls] < self.min_lift or bal[cls] < self.min_confidence:
                continue
            rows.append({"key": key, "norm": item["norm"], "clase": cls,
                         "train_neg": int(cnt[0]), "train_pos": int(cnt[1]),
                         "soporte": float(cnt[cls] / N[cls]),
                         "confianza": float(p_node[cls]),
                         "confianza_balanceada": float(bal[cls]),
                         "lift": float(lift[cls]),
                         "arboles": len(item["trees"]),
                         "condiciones": sum(len(d) for d in item["norm"].values())})

        pats = pd.DataFrame(rows)
        if len(pats):
            pats = self._significance(pats, N)
            if self.max_q is not None:
                pats = pats[pats["q_valor"] <= self.max_q]
        self.patterns_ = pats.reset_index(drop=True)
        self.n_candidates_ = len(pool)
        self.prior_ = prior
        self.class_totals_ = N
        if len(self.patterns_):
            tot = self.patterns_.groupby("clase")["soporte"].sum()
            self.support_totals_ = np.array([tot.get(0, 0.0), tot.get(1, 0.0)])
        else:
            self.support_totals_ = np.zeros(2)

        self.weights_model_ = None
        if self.predictor == "pesos" and len(self.patterns_):
            from sklearn.linear_model import LogisticRegressionCV
            M = self.coverage_matrix(X).astype(float)
            self.weights_model_ = LogisticRegressionCV(
                Cs=[0.01, 0.03, 0.1, 0.3, 1.0], penalty="l1", solver="liblinear",
                cv=5, scoring="roc_auc", max_iter=2000,
                random_state=self.random_state).fit(M, y)
            self.patterns_["peso"] = self.weights_model_.coef_[0]
        return self

    @staticmethod
    def _significance(pats, N):
        """Fisher exacta (unilateral, enriquecimiento) + Benjamini-Hochberg."""
        from scipy.stats import fisher_exact
        p = []
        for _, r in pats.iterrows():
            c = r["clase"]
            a = r["train_pos"] if c == 1 else r["train_neg"]
            b = (r["train_neg"] + r["train_pos"]) - a
            cc = N[c] - a
            dd = (N.sum() - N[c]) - b
            p.append(fisher_exact([[a, b], [cc, dd]], alternative="greater")[1])
        p = np.asarray(p)
        order = np.argsort(p)
        q = np.empty_like(p)
        m = len(p)
        prev = 1.0
        for rank, i in reversed(list(enumerate(order, start=1))):
            prev = min(prev, p[i] * m / rank)
            q[i] = prev
        pats = pats.copy()
        pats["p_valor"] = p
        pats["q_valor"] = q
        return pats

    # ------------------------------------------------------------------
    def coverage_matrix(self, X):
        X = X.reset_index(drop=True)
        if not len(self.patterns_):
            return np.zeros((len(X), 0), dtype=bool)
        return np.column_stack([self._mask(X, n) for n in self.patterns_["norm"]])

    def predict_proba(self, X):
        """Puntuación de la clase positiva (según self.predictor)."""
        M = self.coverage_matrix(X)
        if M.shape[1] == 0:
            return np.full(len(X), self.prior_[1])
        self.uncovered_ = int((M.sum(1) == 0).sum())
        if self.predictor == "pesos" and self.weights_model_ is not None:
            return self.weights_model_.predict_proba(M.astype(float))[:, 1]
        cls = self.patterns_["clase"].to_numpy()
        sup = self.patterns_["soporte"].to_numpy()
        votes = np.zeros((len(X), 2))
        for c in (0, 1):
            sel = cls == c
            if sel.any() and self.support_totals_[c] > 0:
                votes[:, c] = (M[:, sel] @ sup[sel]) / self.support_totals_[c]
        total = votes.sum(1)
        out = np.full(len(X), self.prior_[1])
        ok = total > 0
        out[ok] = votes[ok, 1] / total[ok]
        return out

    def patterns_frame(self, X_test=None, y_test=None):
        """Tabla legible de patrones; con test añade cobertura y confianza en test."""
        df = self.patterns_.copy()
        if not len(df):
            return df
        df.insert(0, "patron", [self._text(n) for n in df["norm"]])
        if X_test is not None:
            M = self.coverage_matrix(X_test)
            yt = np.asarray(y_test).astype(int)
            df["test_cubiertas"] = M.sum(0)
            df["confianza_test"] = [float(np.mean(yt[M[:, j]] == c)) if M[:, j].any()
                                    else np.nan for j, c in enumerate(df["clase"])]
        df = df.drop(columns=["key", "norm"])
        if "peso" in df:
            df["peso_abs"] = df["peso"].abs()
            df = df.sort_values(["peso_abs", "soporte"], ascending=[False, False])
            return df.drop(columns="peso_abs").reset_index(drop=True)
        return df.sort_values(["clase", "soporte"], ascending=[True, False]).reset_index(drop=True)
