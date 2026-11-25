# PbCRPE v1: configuración de referencia

Resultado obtenido en German (test de 200 instancias, partición 80/20, semilla 42):

```
Matriz de confusión:
[[ 37  23]
 [ 23 117]]
Error tipo I:  0.3833   Error tipo II: 0.1643   BER: 0.2738   AUC: 0.7769
```

Comando:

```bash
python run_final.py --dataset german --strategy np_fnr --fnr_max 0.157 \
    --criterion hybrid_fast --max_depth 5 --min_samples_leaf 10 --alpha 0.5 --beta 0.3 \
    --leaf_model logistic --leaf_C 0.1 --leaf_min_samples 30 \
    --prune_metric balanced --min_leaves 6 --prune_frac 0.2 \
    --inner_folds 5 --test_size 0.2 --seed 42
```

Para la variante de mínimo BER: `--strategy youden`.
La cota de error tipo II se cumple en las predicciones fuera de fold del entrenamiento;
en test puede variar ±0.03 (140 positivos).

Nota: desde la sección 18 de CAMBIOS.md el umbral por defecto es `youden` (automático);
esta configuración requiere indicar `--strategy np_fnr --fnr_max 0.157`.
