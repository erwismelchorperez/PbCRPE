# srcc/pbcre/evaluation/excel_report.py
"""
Reporte en Excel de un experimento.

Hojas:
  Resumen        media y desviación estándar por variante. Son FÓRMULAS sobre
                 la hoja Folds, así que se recalculan si se editan o filtran
                 los datos (Excel/LibreOffice las calculan al abrir).
  Folds          una fila por fold y variante con todas las métricas
                 (calculadas con ClassificationMetrics).
  Wilcoxon       pruebas pareadas contra la variante base (calculadas en
                 Python con scipy; son valores, no fórmulas).
  Configuracion  parámetros del experimento y definición de las métricas.
"""

from datetime import datetime

import numpy as np
import pandas as pd
try:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
    _OPENPYXL_ERROR = None
except ImportError as exc:          # el resto del experimento sigue funcionando
    _OPENPYXL_ERROR = exc

from srcc.pbcre.evaluation.metrics import ClassificationMetrics as CM


# (columna en Folds, encabezado, "mayor es mejor")
METRICS = [
    ("accuracy", "Accuracy", True),
    ("precision", "Precision", True),
    ("recall", "Recall", True),
    ("f1_score", "F1-score", True),
    ("mcc", "MCC", True),
    ("auc", "AUC", True),
    ("error_tipo_I", "Error tipo I (FPR)", False),
    ("error_tipo_II", "Error tipo II (FNR)", False),
    ("ber", "BER", False),
    ("balanced_accuracy", "Balanced accuracy", True),
]

METRIC_NOTES = {
    "Accuracy": "(TP + TN) / N",
    "Precision": "Ponderada por soporte de clase (average='weighted').",
    "Recall": "Ponderado por soporte de clase (average='weighted').",
    "F1-score": "Ponderado por soporte de clase (average='weighted').",
    "MCC": "Coeficiente de correlación de Matthews.",
    "AUC": "Área bajo la curva ROC de la puntuación de la variante "
           "(probabilidad del árbol o puntuación de los CIPs).",
    "Error tipo I (FPR)": "FP / (TN + FP)",
    "Error tipo II (FNR)": "FN / (TP + FN)",
    "BER": "(FPR + FNR) / 2",
    "Balanced accuracy": "1 - BER",
}

FONT = "Arial"
if _OPENPYXL_ERROR is not None:
    PatternFill = Side = lambda *a, **k: None
    Border = lambda *a, **k: None
HEADER_FILL = PatternFill("solid", start_color="1F4E78")
SUBHEADER_FILL = PatternFill("solid", start_color="D9E1F2")
BASE_FILL = PatternFill("solid", start_color="F2F2F2")
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
NUM_FMT = "0.0000"


def fold_metrics(y_true, y_pred, scores):
    """Métricas comunes de un fold con la clase ClassificationMetrics."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = CM.confusion_matrix(y_true, y_pred).ravel()
    try:
        auc = CM.auc(y_true, np.asarray(scores, dtype=float))
    except ValueError:
        auc = np.nan
    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": CM.accuracy(y_true, y_pred),
        "precision": CM.precision(y_true, y_pred),
        "recall": CM.recall(y_true, y_pred),
        "f1_score": CM.f1_score(y_true, y_pred),
        "mcc": CM.matthews_corrcoef(y_true, y_pred),
        "auc": auc,
        "error_tipo_I": CM.errorrateI(y_true, y_pred),
        "error_tipo_II": CM.errorrateII(y_true, y_pred),
        "ber": CM.balanced_error_rate(y_true, y_pred),
        "balanced_accuracy": CM.balanced_accuracy(y_true, y_pred),
    }


# -------------------------------------------------------------------------
def _style_header(cell, fill=HEADER_FILL, color="FFFFFF"):
    cell.font = Font(name=FONT, bold=True, color=color)
    cell.fill = fill
    cell.alignment = Alignment(horizontal="center", vertical="center",
                               wrap_text=True)
    cell.border = BORDER


def _style_body(cell, number=False):
    cell.font = Font(name=FONT)
    cell.border = BORDER
    if number:
        cell.number_format = NUM_FMT
        cell.alignment = Alignment(horizontal="right")


def _write_table(ws, df, start_row=1, number_cols=()):
    for j, col in enumerate(df.columns, start=1):
        _style_header(ws.cell(row=start_row, column=j, value=col))
    for i, row in enumerate(df.itertuples(index=False), start=start_row + 1):
        for j, value in enumerate(row, start=1):
            if isinstance(value, (np.floating, float)) and np.isnan(value):
                value = None
            elif isinstance(value, np.generic):
                value = value.item()
            cell = ws.cell(row=i, column=j, value=value)
            _style_body(cell, number=df.columns[j - 1] in number_cols)


def _autowidth(ws, min_width=9, max_width=45):
    for col_cells in ws.columns:
        letter = get_column_letter(col_cells[0].column)
        longest = max((len(str(c.value)) for c in col_cells
                       if c.value is not None and not str(c.value).startswith("=")),
                      default=0)
        ws.column_dimensions[letter].width = max(min_width,
                                                 min(max_width, longest + 2))


# -------------------------------------------------------------------------
def save_excel_report(df_folds, df_tests, config, path, variant_order,
                      base_variant="base_0.5", n_splits=None):
    """
    df_folds : una fila por (fold, variante) con las columnas de METRICS.
    df_tests : resultados de Wilcoxon (variant, metric, mean_diff, p_value).
    config   : dict con los parámetros del experimento.
    """
    if _OPENPYXL_ERROR is not None:
        raise ImportError("Para generar el Excel instale openpyxl: "
                          "pip install openpyxl") from _OPENPYXL_ERROR
    wb = Workbook()

    # ---------------- Folds ----------------
    ws_f = wb.active
    ws_f.title = "Folds"
    fold_cols = (["repeticion", "fold", "variante", "umbral", "hojas",
                  "tn", "fp", "fn", "tp"] + [m[0] for m in METRICS])
    df_f = df_folds[fold_cols].copy()
    number_cols = {"umbral"} | {m[0] for m in METRICS}
    _write_table(ws_f, df_f, number_cols=number_cols)
    ws_f.freeze_panes = "D2"
    ws_f.auto_filter.ref = ws_f.dimensions
    _autowidth(ws_f)

    n_rows = len(df_f) + 1
    col_letter = {c: get_column_letter(i) for i, c in enumerate(fold_cols, 1)}
    var_rng = f"Folds!${col_letter['variante']}$2:${col_letter['variante']}${n_rows}"

    # ---------------- Resumen ----------------
    ws_r = wb.create_sheet("Resumen", 0)
    ws_r["A1"] = f"Resumen: {config.get('dataset', '')}"
    ws_r["A1"].font = Font(name=FONT, bold=True, size=14)
    ws_r["A2"] = ("Media y desviación estándar por fold (fórmulas sobre la hoja "
                  "Folds). Fila gris = variante base.")
    ws_r["A2"].font = Font(name=FONT, italic=True, color="595959")

    def block(title, start_row, kind):
        ws_r.cell(row=start_row, column=1, value=title).font = Font(
            name=FONT, bold=True, size=12)
        hdr = start_row + 1
        _style_header(ws_r.cell(row=hdr, column=1, value="Variante"))
        _style_header(ws_r.cell(row=hdr, column=2, value="Folds"))
        for j, (_, label, better) in enumerate(METRICS, start=3):
            arrow = "↑" if better else "↓"
            _style_header(ws_r.cell(row=hdr, column=j, value=f"{label} {arrow}"))

        for i, variant in enumerate(variant_order, start=hdr + 1):
            name_cell = ws_r.cell(row=i, column=1, value=variant)
            _style_body(name_cell)
            name_cell.font = Font(name=FONT, bold=True)
            a = f"$A{i}"
            cnt = ws_r.cell(row=i, column=2,
                            value=f"=COUNTIF({var_rng},{a})")
            _style_body(cnt)
            for j, (col, _, _) in enumerate(METRICS, start=3):
                L = col_letter[col]
                rng = f"Folds!${L}$2:${L}${n_rows}"
                if kind == "mean":
                    f = f"=IFERROR(AVERAGEIFS({rng},{var_rng},{a}),\"\")"
                else:
                    mean = f"AVERAGEIFS({rng},{var_rng},{a})"
                    f = (f"=IFERROR(SQRT(SUMPRODUCT(({var_rng}={a})*({rng}-{mean})^2)"
                         f"/(COUNTIF({var_rng},{a})-1)),\"\")")
                c = ws_r.cell(row=i, column=j, value=f)
                _style_body(c, number=True)
            if variant == base_variant:
                for j in range(1, len(METRICS) + 3):
                    ws_r.cell(row=i, column=j).fill = BASE_FILL
        return hdr + len(variant_order) + 2

    nxt = block("Media", 4, "mean")
    nxt = block("Desviación estándar", nxt, "std")

    ws_r.column_dimensions["A"].width = 16
    ws_r.column_dimensions["B"].width = 8
    for j in range(3, len(METRICS) + 3):
        ws_r.column_dimensions[get_column_letter(j)].width = 14
    ws_r.row_dimensions[5].height = 32
    ws_r.row_dimensions[nxt + 1].height = 32
    ws_r.freeze_panes = "B6"

    # ---------------- Wilcoxon ----------------
    ws_w = wb.create_sheet("Wilcoxon")
    ws_w["A1"] = f"Prueba de Wilcoxon pareada contra {base_variant}"
    ws_w["A1"].font = Font(name=FONT, bold=True, size=12)
    ws_w["A2"] = ("Calculada en Python (scipy.stats.wilcoxon, zero_method='zsplit'). "
                  "dif_media = variante - base; en métricas de error, negativo = mejora.")
    ws_w["A2"].font = Font(name=FONT, italic=True, color="595959")
    if df_tests is not None and len(df_tests):
        t = df_tests.rename(columns={"variant": "variante", "metric": "metrica",
                                     "mean_diff": "dif_media",
                                     "p_value": "p_valor"}).copy()
        t["significativo_0.05"] = np.where(t["p_valor"] < 0.05, "sí", "no")
        _write_table(ws_w, t, start_row=4, number_cols={"dif_media", "p_valor"})
        ws_w.auto_filter.ref = f"A4:{get_column_letter(t.shape[1])}{len(t) + 4}"
    _autowidth(ws_w, max_width=22)
    ws_w.column_dimensions["A"].width = 16

    # ---------------- Configuracion ----------------
    ws_c = wb.create_sheet("Configuracion")
    ws_c["A1"] = "Configuración del experimento"
    ws_c["A1"].font = Font(name=FONT, bold=True, size=12)
    conf = dict(config)
    conf.setdefault("fecha", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    df_conf = pd.DataFrame({"parametro": list(conf.keys()),
                            "valor": [str(v) for v in conf.values()]})
    _write_table(ws_c, df_conf, start_row=3)
    r = len(df_conf) + 6
    ws_c.cell(row=r - 1, column=1, value="Definición de métricas").font = Font(
        name=FONT, bold=True, size=12)
    df_def = pd.DataFrame({"metrica": list(METRIC_NOTES.keys()),
                           "definicion": list(METRIC_NOTES.values())})
    _write_table(ws_c, df_def, start_row=r)
    ws_c.column_dimensions["A"].width = 24
    ws_c.column_dimensions["B"].width = 80

    wb.save(path)
    return path


# -------------------------------------------------------------------------
SUMMARY_COLUMNS = ["trabajo", "dataset", "Accuracy", "Precision", "Recall",
                   "F1-Score", "AUC", "MCC", "ErrorTipoI", "ErrorTipoII", "BER",
                   "TipoTrabajo"]


def upsert_summary_row(path, row, sheet="Resultados"):
    """
    Tabla comparativa de trabajos (una fila por trabajo y dataset).
    Si ya existe la fila (mismo trabajo y dataset) se reemplaza; si no, se
    agrega al final. Las filas de otros trabajos que el usuario haya
    escrito a mano se conservan.
    Devuelve "agregada" o "actualizada".
    """
    if _OPENPYXL_ERROR is not None:
        raise ImportError("Instale openpyxl") from _OPENPYXL_ERROR
    import os
    from openpyxl import load_workbook

    if os.path.exists(path):
        wb = load_workbook(path)
        ws = wb[sheet] if sheet in wb.sheetnames else wb.active
        header = [c.value for c in ws[1]]
    else:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        wb = Workbook()
        ws = wb.active
        ws.title = sheet
        header = list(SUMMARY_COLUMNS)
        for j, h in enumerate(header, start=1):
            _style_header(ws.cell(row=1, column=j, value=h))
        ws.freeze_panes = "A2"

    target = None
    for r in range(2, ws.max_row + 1):
        if (ws.cell(row=r, column=header.index("trabajo") + 1).value == row["trabajo"]
                and ws.cell(row=r, column=header.index("dataset") + 1).value
                == row["dataset"]):
            target = r
            break
    action = "actualizada" if target else "agregada"
    if target is None:
        target = ws.max_row + 1

    for j, col in enumerate(header, start=1):
        if col not in row:
            continue
        value = row[col]
        if isinstance(value, np.generic):
            value = value.item()
        cell = ws.cell(row=target, column=j, value=value)
        _style_body(cell, number=isinstance(value, float))

    widths = {"trabajo": 26, "dataset": 30, "TipoTrabajo": 14}
    for j, col in enumerate(header, start=1):
        ws.column_dimensions[get_column_letter(j)].width = widths.get(col, 12)
    ws.auto_filter.ref = f"A1:{get_column_letter(len(header))}{ws.max_row}"
    wb.save(path)
    return action
