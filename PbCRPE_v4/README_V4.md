# PbCRPE v4: minado de patrones de varios árboles

## Qué cambia respecto a la v3

| | v3 (`run_final.py`) | v4 (`run_final_v4.py`) |
|---|---|---|
| Árboles para patrones | 1 (el árbol final) | `--n_trees` (30 por defecto) |
| Diversidad | — | bootstrap estratificado + 60 % de variables por árbol |
| Criterio de los árboles de minado | hybrid_fast | twoing (rápido) o hybrid_fast |
| Patrones | nodos del árbol podado | nodos de todos los árboles, sin duplicados, recalculados con todo el entrenamiento |
| Predicción | árbol con hojas logísticas | `pesos`, `votacion` o `arbol_v3` |

El protocolo (partición 80/20, CV interna, umbral `auto`, Excel y tabla comparativa) es el
mismo, y con la misma semilla el test es idéntico al de la v3.

## Modos de predicción

- `pesos` (por defecto): cada patrón es una variable binaria (cubre / no cubre) de una regresión
  logística con L1; el peso positivo empuja hacia la clase 1 y el negativo hacia la 0. Muchos
  patrones quedan con peso cero. Con patrones correlacionados, un patrón puede tener un peso de
  signo contrario a su clase: lo que decide es el peso.
- `votacion`: estilo PBC4cip; cada clase suma el soporte de sus patrones que cubren la
  instancia, normalizado por el soporte total de la clase.
- `arbol_v3`: predice con el árbol de la v3 y usa el bosque sólo para minar patrones.

## Columnas de la hoja Patrones

`patron`, `clase` (la que más se enriquece por lift), `train_neg`, `train_pos`, `soporte`
(de su clase), `confianza`, `confianza_balanceada`, `lift`, `arboles` (en cuántos árboles
apareció), `condiciones`, `p_valor` y `q_valor` (Fisher exacta + Benjamini-Hochberg), `peso`
(modo `pesos`), `test_cubiertas`, `confianza_test`.

## Resultados en German (mi copia pública; CV de 5 folds, dos semillas)

| Predictor | AUC semilla 42 | AUC semilla 7 | Promedio |
|---|---|---|---|
| v4 votación | 0.733 | 0.754 | 0.744 |
| v4 pesos L1 | 0.745 | 0.761 | 0.753 |
| v3 / v4 arbol_v3 | 0.778 | 0.788 | 0.783 |

Filtrar por significancia (`--max_q 0.05`) no cambió el AUC.

## Uso

```bash
python run_final_v4.py --dataset german --fnr_max 0.157                        # pesos
python run_final_v4.py --dataset german --fnr_max 0.157 --predictor votacion
python run_final_v4.py --dataset german --fnr_max 0.157 --predictor arbol_v3
python run_final_v4.py --dataset default+of+credit+card+clients --n_jobs -1    # en paralelo
```

Por defecto `--trabajo` es `Our Proposal V4 (<predictor>)`, así cada modo queda como una fila
distinta en `results_final_v4/comparacion_trabajos.xlsx`.

Tiempo en German (1 núcleo): ~20 s con `pesos` o `votacion`, ~60 s con `arbol_v3`.
Con `--mining_criterion hybrid_fast` cada árbol tarda ~20 veces más.
