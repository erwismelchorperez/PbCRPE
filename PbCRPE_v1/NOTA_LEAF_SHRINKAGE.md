# Cambio sobre tu versión: `--leaf_shrinkage`

Único cambio respecto al código que enviaste (srcc_1.zip):

- `srcc/pbcre/tree/tree.py`: parámetro `leaf_shrinkage` (por defecto 0). Cada hoja combina su
  regresión logística con la de la raíz en escala logit: z = w·z_hoja + (1 − w)·z_raíz,
  con w = n / (n + k) y n = instancias con que se ajustó el modelo de la hoja.
- `run_final.py` y `run_experiment.py`: opción `--leaf_shrinkage` (por defecto 0).

Con el valor por defecto los resultados son idénticos a los de tu versión
(verificado en German con `--strategy np_fnr --fnr_max 0.157`: misma matriz de confusión).

## Para decidir en Australian (validación cruzada, dos semillas)

```bash
python run_experiment.py --dataset australian --leaf_model logistic --min_leaves 6 --no_cips --repeats 2 --seed 42
python run_experiment.py --dataset australian --leaf_model logistic --min_leaves 6 --no_cips --repeats 2 --seed 42 --leaf_shrinkage 150
python run_experiment.py --dataset australian --leaf_model logistic --min_leaves 6 --no_cips --repeats 2 --seed 42 --leaf_shrinkage 600
```

Repetir con `--seed 7`. Adoptar el valor de k sólo si mejora el AUC (y el error tipo I de
`np_fnr`) con ambas semillas. Después:

```bash
python run_final.py --dataset australian --strategy np_fnr --fnr_max 0.127 --leaf_shrinkage K
```
