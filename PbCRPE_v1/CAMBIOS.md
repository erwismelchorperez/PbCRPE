# PbCRPE_v1 — Registro de cambios

Objetivo: reducir el **error tipo II (FNR)** sin sacrificar el **error tipo I (FPR)**,
usando el **Balanced Error Rate** (BER = (FPR + FNR) / 2) como guía.

Relación usada en todo el proyecto: `Youden J = TPR − FPR = 1 − 2·BER`.
Maximizar Youden es minimizar el BER.

## Estructura

```
PbCRPE_v1/
├── run_experiment.py        # NUEVO: comparación de estrategias (CV)
├── run_final.py             # NUEVO: resultado final único + patrones
├── CAMBIOS.md
└── srcc/pbcre/
    ├── evaluation/metrics.py      # reescrito
    ├── evaluation/thresholds.py   # NUEVO: select_threshold
    ├── tree/tree.py               # modificado
    ├── split/*.py                 # modificados
    ├── tuning/cv_utils.py         # NUEVO: CV compartida
    ├── tuning/*.py                # modificados
    └── utils/ProccessData.py      # modificado
```

Las importaciones siguen siendo `srcc.pbcre...`; ejecute los scripts desde `PbCRPE_v1/`.

## 1. Errores corregidos

| Archivo | Problema | Corrección |
|---|---|---|
| `tree.py` `fit_leaf_thresholds` | `roc_curve` sobre una puntuación constante devolvía umbral `+inf`: la hoja nunca predecía 1 | Obsoleto; delega en `fit_threshold` (umbral global) |
| `tree.py` `predict_proba` | El suavizado de Laplace usaba `alpha`, el mismo peso del criterio híbrido | Nuevo parámetro `laplace=1.0` |
| `metrics.py` | `confusion_matrix(...).ravel()` fallaba con una sola clase | `labels=[0, 1]` en todas las métricas |
| `metrics.py` / poda / split | El BER (menor es mejor) se usaba donde el código maximiza | `balanced_accuracy = 1 − BER`; `greater_is_better` en el tuning |
| `hybrid_stable.py` | La caché usaba `(n_muestras, clases)`: subconjuntos distintos compartían score | Clave = hash del contenido `(X_enc, y)` |
| `hybrid_stable.py` | `len(np.unique(y)) < cv_folds` siempre verdadero: la CV nunca corría | Se usa el tamaño de la clase minoritaria |
| `functional_criterion.py` | `print` de depuración con `y_tr` sin definir (`NameError`) | Eliminados |
| `hybrid_twoing_functional.py` | `alpha` se aplicaba dos veces (`alpha²`) | Se aplica una vez (`_combine`) |
| `hybrid_fast.py` | `np.random.choice` sin semilla: árboles distintos en cada corrida | Generador con `random_state`, reiniciado en cada `fit` |
| `gini.py` | Devolvía 3 valores en vez de 4 | Devuelve `(feat, thr, score, None)` |

## 2. Sesgos hacia la clase mayoritaria reducidos

- **Score funcional**: `scoring="balanced"` (1 − BER) por defecto en todos los criterios
  híbridos. Las opciones `"f1"` y `"f1_weighted"` siguen disponibles.
- **Pureza** (`hybrid_fast`): `purity_mode="balanced"` repondera por el prior de la raíz;
  vale 0.5 si el nodo reproduce el prior. `"majority"` = original.
- **Umbrales numéricos** (`hybrid_fast`): `class_quantiles=True` añade cuantiles de la clase
  positiva (incluyendo las colas 0.05–0.95).
- **Logística local**: `class_weight="balanced"` también en el criterio `hybrid`.
- **Split híbrido**: el score funcional es un promedio ponderado por tamaño en vez de una suma.

## 3. Umbral de decisión global

```python
model.fit_threshold(X_val, y_val, fpr_max=None)       # mínimo BER (Youden)
model.fit_threshold(X_val, y_val, fpr_max=0.05)       # mínimo FNR con FPR <= 0.05
model.fit_threshold(X_val, y_val, fpr_max="current")  # mínimo FNR sin superar el FPR actual
model.predict(X_test)                                  # usa model.threshold_
```

Siempre con validación, nunca con el test final.

## 4. Poda

`cost_complexity_pruning(X_val, y_val, metric="auc" | "balanced", tol=0.0)`.
En cada ronda evalúa todas las podas y aplica la de mayor ganancia por hoja; acepta
empates (prefiere el árbol más simple). `"balanced"` usa el máximo 1 − BER alcanzable
con algún umbral, así que es independiente del umbral y está alineada con el BER.

## 5. CIPs

- `extract_cips(class_mode="balanced", min_lift=None)`: la clase del CIP es la que más se
  enriquece respecto al prior. Cada CIP incluye `class_support`, `lift` y
  `balanced_confidence`.
- `predict_with_cips`: los empates los decide el árbol (antes: clase 0).
- `predict_with_cips_weighted(..., support_mode="class")`: soporte relativo a cada clase.
- NUEVO: `cip_scores`, `fit_cip_threshold`, `predict_with_cips_scored`: puntuación continua
  más umbral con restricción de FPR. `tree_smoothing` desempata con la probabilidad del árbol.

## 6. Tuning

- `sampler`: SMOTE/SMOTENC aplicado **dentro** de cada fold (`ProccessData.build_sampler`).
- `metric=ConstrainedObjective(fpr_ref, lam)`: `−(BER + lam·max(0, FPR − fpr_ref))`.
- `greater_is_better=False` para pasar métricas de error (BER) sin invertir la optimización.
- `threshold_strategy`: `None`, `"youden"`, `"fpr_current"` o un float (validación interna).
- `history_` registra FPR, FNR y BER por evaluación; `history_frame()` lo devuelve como DataFrame.
- `tuning_pso.py` acepta el `Agent` que devuelve mealpy ≥ 3.

Nota: en los criterios híbridos solo importa la proporción `beta : (1 − beta)·alpha`,
así que basta tunear uno de los dos.

## 7. Cambios de comportamiento a tener en cuenta

- `predict` usa la probabilidad **suavizada** (`laplace=1.0`) contra `threshold_=0.5`.
  Con `laplace=0` se recupera la regla original de proporción en la hoja.
- `PBCRETree(verbose=False)` por defecto, para no saturar la salida en el tuning.
- `scoring="balanced"` cambia los splits respecto a la versión anterior.
  Para reproducir resultados previos: `scoring="f1_weighted"` (hybrid_fast) o `"f1"` (hybrid).

## 8. Experimento

```bash
python run_experiment.py --dataset NOMBRE --criterion hybrid_fast --repeats 3
python run_experiment.py --dataset NOMBRE --smote
python run_experiment.py --synthetic
```

Genera `results/resultados_folds.csv`, `results/resumen.csv` y `results/wilcoxon.csv`.

La restricción de FPR se cumple en **validación**. En test puede variar un poco,
sobre todo con validaciones pequeñas y árboles de pocas hojas (probabilidades discretas).
Si necesita una garantía más estricta, use una cota algo menor que el FPR base
(p. ej. `fpr_max = 0.8 * fpr_base`) o aumente la validación.

## 9. Compatibilidad con pandas 3 (corrección posterior)

En pandas 3 las columnas de texto tienen dtype `str` en lugar de `object`, así que:
- `target` con valores `positive`/`negative` no se convertía a 0/1 (`ValueError: invalid literal for int()`).
  Ahora `ProccessData.encode_target` normaliza el texto y usa `LABEL_MAP`; si aparece un valor
  desconocido, el error lo indica explícitamente.
- La detección de columnas categóricas usaba `select_dtypes(include="object")`.
  Ahora `ProccessData.categorical_columns` toma todas las columnas no numéricas (pandas 2 y 3).
- `run_experiment.py` indica la ruta absoluta esperada si no encuentra `dataset/NOMBRE.csv`.

## 10. Control del error tipo I y hojas funcionales

- `select_threshold(..., fnr_max=...)`, `fit_threshold(..., fnr_max=...)` y
  `fit_cip_threshold(..., fnr_max=...)`: minimizan el **error tipo I** con el
  error tipo II acotado (`"current"` también es válido). En el tuning:
  `threshold_strategy="fnr_current"` o `("fnr", 0.2)`.
- `smoothing="prior"`: m-estimate que encoge las hojas hacia el prior de
  entrenamiento en vez de hacia 0.5.
- `class_quantiles` ahora usa la clase **minoritaria** (antes, siempre la clase 1).
- `leaf_model="logistic"` (`leaf_C`, `leaf_min_samples`): cada nodo con datos
  suficientes ajusta una regresión logística y las hojas usan la de su ancestro
  más cercano (estilo LMT). Las probabilidades son continuas dentro de la hoja.
- `run_experiment.py`: variantes `prior_thr`, `np_fpr` (antes `np_current`),
  `np_fnr`, `cips_youden`, `cips_np_fnr`; columna `auc`, número de hojas y
  opciones `--fnr_max --smoothing --laplace --purity_mode --no_class_quantiles
  --leaf_model --leaf_C --leaf_min_samples --no_cips`.

Referencia en German Credit (5 folds, umbral `prior_thr`, criterio `hybrid_fast`):

| Configuración | Hojas | AUC | FPR | FNR | BER |
|---|---|---|---|---|---|
| hojas constantes | 8.4 | 0.70 | 0.31 | 0.35 | 0.33 |
| `--leaf_model logistic` | 4.4 | 0.76 | 0.31 | 0.29 | 0.30 |
| `--leaf_model logistic --leaf_C 0.3 --leaf_min_samples 50` | 1.6 | 0.78 | 0.28 | 0.29 | 0.28 |

La última configuración casi no deja patrones (equivale a una regresión logística global).

## 11. Reporte en Excel

`run_experiment.py` genera `<dataset>_<criterio>_<hojas>_<fecha>.xlsx` en `--out`
(desactivar con `--no_excel`). Las métricas por fold se calculan con
`ClassificationMetrics` (`srcc/pbcre/evaluation/excel_report.py`):

- **Resumen**: media y desviación estándar por variante de Accuracy, Precision,
  Recall, F1 (ponderados), MCC, AUC, error tipo I, error tipo II, BER y balanced
  accuracy. Son fórmulas (`AVERAGEIFS`, `SUMPRODUCT`) sobre la hoja Folds.
  La columna Folds indica cuántos folds tiene cada variante (las variantes de CIPs
  se omiten en los folds donde la poda deja el árbol sin patrones).
- **Folds**: una fila por fold y variante, con la matriz de confusión, el umbral y las hojas.
- **Wilcoxon**: pruebas pareadas contra `base_0.5` para errores, accuracy, F1, MCC y AUC.
- **Configuracion**: parámetros del experimento y definición de cada métrica.

Requiere `openpyxl` (`pip install openpyxl`).

## 12. Resultado final único y patrones (`run_final.py`)

`run_experiment.py` sirve para COMPARAR estrategias (validación cruzada).
`run_final.py` produce EL resultado final: un modelo, una evaluación en test.

Protocolo 80/20 (objetivo por defecto: mínimo BER, `--strategy youden`):
1. Partición estratificada fija: entrenamiento 80 % | test 20 % (`particion/train.csv`, `test.csv`).
2. Dentro del 80 %:
   - **Umbral**: CV interna de 5 folds (`--inner_folds`); el umbral se elige con las
     probabilidades fuera de fold de las 800 instancias de entrenamiento.
   - **Modelo final**: crece con el 80 % del entrenamiento, se poda con el 20 % restante
     (`--prune_frac`) y luego `refit_leaves` reestima las hojas con todo el entrenamiento.
   - La poda no deja menos de `--min_leaves` hojas (6 por defecto), para que el
     árbol no colapse y se pierdan los patrones.
3. Test: evaluación única.

Remuestreo sólo con `--smote` (SMOTENC si hay categóricas). "Clase del patrón por lift"
NO es remuestreo: indica que la clase de cada patrón es la que más se enriquece respecto
a su proporción global.

Si ningún patrón supera `--min_confidence` (0.55, confianza balanceada), se muestran los
caminos de todas las hojas y se deja constancia en la hoja Configuracion.

Excel `<dataset>_final_<fecha>.xlsx`: Resultado, Particion, Umbrales_OOF, Patrones, Hojas,
Modelos_hoja, Instancias_test, Configuracion.

Nuevos métodos del árbol: `refit_leaves(X, y)`, `leaf_paths()`,
`leaf_model_coefficients(node, top)`, `simplify_path(path)`;
`cost_complexity_pruning(..., min_leaves=)`.

Ejemplos:
```bash
python run_final.py --dataset german
python run_final.py --dataset german --smote
python run_final.py --dataset german --min_leaves 4 --print_tree
```

German (mi copia pública, semilla 42): 8 hojas, 11 patrones, AUC test 0.786,
error tipo I 0.217, error tipo II 0.300, BER 0.258. Con `--min_leaves 3`: 3 hojas,
BER 0.300; el AUC fuera de fold fue prácticamente igual (0.779) en ambos casos.
Un único split tiene mucha varianza: repórtelo junto con la media ± DE de la CV.

## 13. Tabla de compromiso entre errores

`run_final.py` imprime y guarda (hoja **Compromiso_OOF**) el menor error tipo I alcanzable
para cada cota de error tipo II (0.10 … 0.30 y la de `--fnr_max`), calculado con las
predicciones fuera de fold del entrenamiento. Sirve para elegir `--fnr_max` sin mirar el test.

```bash
python run_final.py --dataset german --strategy np_fnr --fnr_max 0.157
```

## 14. Matrices de confusión en `run_experiment.py`

Tras el resumen se imprime, para cada variante, el bloque habitual: matriz de confusión
acumulada (los folds de una repetición suman cada instancia una vez; se promedia entre
repeticiones) y Accuracy, Precision, Recall, F1, MCC, errores tipo I y II, BER y AUC
como media ± DE por fold. `--show_variants base_0.5,youden,np_fnr` limita las variantes.
La hoja Resumen del Excel incluye la misma matriz (TN, FP, FN, TP) con fórmulas.
También se acepta `--min_leaves`.

## 15. Una sola matriz de confusión

- `run_experiment.py` (validación cruzada) ya no imprime matrices: en CV no existe un único
  test, y la matriz promediada daba valores decimales. Muestra la tabla de errores
  (media ± DE) y, con `--show_variants`, las demás métricas.
- `run_final.py` es el resultado oficial: una matriz de confusión entera sobre el test (20 %).
  Antes imprime la tabla de decisión del umbral calculada sólo con el entrenamiento
  (TN, FP, FN, TP enteros sobre las predicciones fuera de fold).

## 16. Corrección en `twoing.py`

Para variables categóricas el criterio `twoing` evaluaba `col <= thr` (orden alfabético),
pero el árbol aplica `col == thr` al dividir y predecir: el split elegido no era el aplicado.
Ahora usa igualdad para categóricas. No afecta a `hybrid_fast` (criterio por defecto).

## Selección automática de estrategia (`--strategy auto`), agregada a la versión de trabajo

Nuevas estrategias en `run_final.py`:
- `minimax`: minimiza el mayor de los dos errores (los deja parejos).
- `closest`: mínima distancia al clasificador perfecto, sqrt(error_I² + error_II²).
- `auto`: elige la estrategia de cada dataset usando sólo el entrenamiento. Sobre las
  predicciones fuera de fold repite 10 veces una CV de 5 partes; cada candidata elige su umbral
  con 4/5 y se evalúa en el 1/5 restante. Gana la de menor BER medio.

Opciones: `--auto_candidates youden,minimax,np_fnr` (por defecto; `np_fnr` usa `--fnr_max`) y
`--auto_metric max_error` (elige por el mayor de los dos errores en lugar del BER).
La tabla se imprime y se guarda en la hoja `Seleccion_estrategia`; la hoja Configuracion
registra la elección (p. ej. `auto -> minimax`).

Sin `--strategy auto` los resultados no cambian (verificado en German con
`--strategy np_fnr --fnr_max 0.157`: misma matriz de confusión). La tabla de decisión del
umbral muestra ahora también las filas `closest` y `minimax`.

Uso:
python run_final.py --dataset NOMBRE --strategy auto --fnr_max 0.157

## Puntaje funcional del split: AUC, estandarización y C (`--split_scoring`, `--split_standardize`, `--logistic_C`)

Problemas del puntaje funcional original de `hybrid_fast`:
- la logística que puntúa cada split usaba las variables numéricas sin estandarizar y una
  regularización muy fuerte (C = 0.005): las variables de escala grande dominaban;
- el puntaje era la balanced accuracy de predicciones 0/1 en ~150 instancias: muy ruidoso.

Opciones nuevas (por defecto todo queda igual):
- `--split_scoring auc`: puntúa con el AUC de las probabilidades.
- `--split_standardize`: estandariza las numéricas dentro de cada nodo antes de la logística.
- `--logistic_C 0.1`: regularización de esa logística (por defecto 0.005).

Resultados (CV 5 folds, youden, dos semillas):

| Dataset | Configuración | AUC 42 / 7 | BER 42 / 7 |
|---|---|---|---|
| German | k = 600 | 0.778 / 0.788 | 0.299 / 0.264 |
| German | k = 600 + split nuevo | 0.779 / 0.785 | 0.290 / 0.273 |
| Simulado con interacciones | split actual | 0.884 / 0.901 | 0.192 / 0.176 |
| Simulado con interacciones | split nuevo | 0.899 / 0.905 | 0.179 / 0.161 |

En German no cambia: una regresión logística global ya captura casi toda la información
(LR 0.79 ≈ Random Forest 0.79) y la forma de los splits apenas importa. Con interacciones
(LR 0.81, Random Forest 0.94) el split nuevo mejora en ambas semillas. Más profundidad
(`--max_depth 7`) no mejoró. Sin las opciones, los resultados son idénticos a los anteriores.

Comando sugerido para probar en cada dataset:
python run_experiment.py --dataset NOMBRE --leaf_model logistic --min_leaves 6 --no_cips --repeats 2 --split_scoring auc --split_standardize --logistic_C 0.1

## Script de referencias por dataset (`compare_baselines.py`)

Con la misma partición 80/20 y la misma elección de umbral sin mirar el test (minimax y youden
sobre predicciones fuera de fold), entrena regresión logística, Random Forest y Gradient Boosting
en cada dataset y calcula `margen_auc` = mejor AUC no lineal − AUC de la logística:
- < 0.02: problema casi lineal; cambios en splits o patrones no moverán el AUC.
- > 0.05: hay interacciones; ahí la construcción del árbol puede mejorar.

python compare_baselines.py                          # todos los CSV de ./dataset
python compare_baselines.py --datasets german gmsc newhmeq --max_train 20000

Guarda `results_referencias/referencias_modelos.xlsx` (hojas Resumen y Detalle).
Nota: usa ProccessData, que elimina las filas con faltantes (igual que PbCRPE).
