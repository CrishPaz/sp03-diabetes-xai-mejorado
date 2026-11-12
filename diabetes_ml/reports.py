"""Fase 5: figuras a 600 DPI (PNG/TIFF + PDF vectorial) y reportes PDF/Word/Excel con paridad de cifras."""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from diabetes_ml.config import ARTIFACTS, DISCLAIMER, ROOT, load  # noqa: E402

OKABE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
NAMES = {"lr": "Reg. logística", "rf": "Random Forest", "xgb": "XGBoost", "tabnet": "TabNet", "stack": "Stacking"}


def out_dir(key: str) -> Path:
    d = ROOT / load("reports")["salida"] / key
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(fig, path: Path) -> List[str]:
    dpi = load("reports")["dpi"]
    files = []
    for ext, kw in (("png", {"dpi": dpi}), ("tiff", {"dpi": dpi, "pil_kwargs": {"compression": "tiff_lzw"}}), ("pdf", {})):
        p = path.with_suffix("." + ext)
        fig.savefig(p, bbox_inches="tight", **kw)
        files.append(p.relative_to(ROOT).as_posix())
    plt.close(fig)
    return files


def figures(res: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    cfg = load("reports")
    w1 = cfg["columna_mm"] / 25.4
    plt.rcParams.update({"font.size": cfg["fuente_pt"], "axes.prop_cycle": plt.cycler(color=OKABE)})
    pm = res["evaluacion_prueba"]["por_modelo"]
    d = out_dir(key)
    figs = []

    fig, ax = plt.subplots(figsize=(w1, w1))
    for m, v in pm.items():
        ax.plot(v["curvas"]["roc"]["fpr"], v["curvas"]["roc"]["tpr"], label=f"{NAMES.get(m, m)} ({v['metricas']['roc_auc']:.3f})", lw=1)
    ax.plot([0, 1], [0, 1], ls=":", c="grey", lw=0.8)
    ax.set_xlabel("1 - especificidad"); ax.set_ylabel("Sensibilidad"); ax.legend(fontsize=6, frameon=False)
    best = max(pm, key=lambda m: pm[m]["metricas"]["roc_auc"])
    figs.append({"id": "roc", "archivos": _save(fig, d / "fig_roc"),
                 "leyenda": f"Curvas ROC en la prueba aislada. AUC más alto: {NAMES.get(best, best)} ({pm[best]['metricas']['roc_auc']:.3f}). La diagonal punteada es el azar."})

    fig, ax = plt.subplots(figsize=(w1, w1))
    for m, v in pm.items():
        ax.plot(v["curvas"]["pr"]["recall"], v["curvas"]["pr"]["precision"], label=NAMES.get(m, m), lw=1)
    prev = next(iter(pm.values()))["curvas"]["pr"]["prevalencia"]
    ax.axhline(prev, ls=":", c="grey", lw=0.8)
    ax.set_xlabel("Sensibilidad (recall)"); ax.set_ylabel("VPP (precisión)"); ax.legend(fontsize=6, frameon=False)
    figs.append({"id": "pr", "archivos": _save(fig, d / "fig_pr"),
                 "leyenda": f"Curvas precisión-sensibilidad. La línea punteada es la prevalencia ({prev:.3f}), el desempeño de un clasificador sin información."})

    fig, ax = plt.subplots(figsize=(w1, w1))
    for m, v in pm.items():
        c = v["curvas"]["calibracion"]
        ax.plot([r["pred"] for r in c], [r["obs"] for r in c], marker="o", ms=2, lw=1, label=f"{NAMES.get(m, m)} (pend. {v['metricas']['pendiente']:.2f})")
    ax.plot([0, 1], [0, 1], ls=":", c="grey", lw=0.8)
    ax.set_xlabel("Probabilidad predicha"); ax.set_ylabel("Frecuencia observada"); ax.legend(fontsize=6, frameon=False)
    figs.append({"id": "calibracion", "archivos": _save(fig, d / "fig_calibracion"),
                 "leyenda": "Diagrama de confiabilidad (10 bins por cuantiles) tras calibrar en Validación; la pendiente ideal es 1."})

    fig, ax = plt.subplots(figsize=(w1, w1))
    for m, v in pm.items():
        dc = v["curva_decision"]
        ax.plot([r["umbral"] for r in dc], [r["modelo"] for r in dc], lw=1, label=NAMES.get(m, m))
    dc = next(iter(pm.values()))["curva_decision"]
    ax.plot([r["umbral"] for r in dc], [r["tratar_todos"] for r in dc], ls="--", c="grey", lw=0.8, label="Tratar a todos")
    ax.axhline(0, c="black", lw=0.6)
    ax.set_ylim(-0.05, max(0.05, prev * 1.1)); ax.set_xlabel("Umbral de probabilidad"); ax.set_ylabel("Beneficio neto"); ax.legend(fontsize=6, frameon=False)
    figs.append({"id": "curva_decision", "archivos": _save(fig, d / "fig_curva_decision"),
                 "leyenda": "Curva de decisión: un modelo aporta utilidad clínica en los umbrales donde supera a «tratar a todos» y a «ninguno» (0)."})
    return figs


def tables(res: Dict[str, Any]) -> List[Tuple[str, List[str], List[List[Any]], Dict[str, str]]]:
    pm = res["evaluacion_prueba"]["por_modelo"]
    keys = ["roc_auc", "pr_auc", "sensibilidad", "especificidad", "vpp", "vpn", "f1", "brier", "pendiente"]
    t1 = [[NAMES.get(m, m)] + [v["metricas"][k] for k in keys] + [round(v["umbral"], 4)] for m, v in pm.items()]
    nested = res["cv_anidada"]["resumen"]
    t2 = [[NAMES.get(m, m), s["media"], s["de"], s["ic95_t"][0], s["ic95_t"][1], s["brecha_optimismo"]] for m, s in nested.items()]
    pr = res["evaluacion_prueba"]["pruebas"]["pares"]
    t3 = [[f"{NAMES.get(r['modelo_a'])} vs {NAMES.get(r['modelo_b'])}", r["dif_auc"], round(r["delong_p"], 4), round(r["delong_p_holm"], 4), round(r["mcnemar_p_holm"], 4)] for r in pr]
    part = res["particion"]["tabla"]
    t0 = [[r["clase"], r["desarrollo"], r["validacion"], r["prueba"], r["total"]] for r in part]
    return [
        ("Tabla 1. Partición 70/15/15 por clase", ["Clase", "Desarrollo", "Validación", "Prueba", "Total"], t0, res["particion"]["interpretabilidad"]),
        ("Tabla 2. CV anidada (AUC externo)", ["Modelo", "Media", "DE", "IC95 inf", "IC95 sup", "Brecha optimismo"], t2, res["cv_anidada"]["interpretabilidad"]),
        ("Tabla 3. Métricas en prueba aislada", ["Modelo"] + keys + ["umbral"], t1, res["evaluacion_prueba"]["interpretabilidad"]),
        ("Tabla 4. Pruebas pareadas en prueba", ["Par", "Dif. AUC", "DeLong p", "DeLong p Holm", "McNemar p Holm"], t3, res["evaluacion_prueba"]["pruebas"]["interpretabilidad"]),
    ]


def metadata(res: Dict[str, Any]) -> List[Tuple[str, str]]:
    import sklearn, xgboost, optuna
    return [("Cohorte", res["cohorte"]), ("Escenario", f"{res['escenario']}: {res['auditoria']['descripcion_escenario']}"),
            ("n oficial", str(res["auditoria"]["n_oficial"])), ("SHA-256 datos", res["auditoria"]["sha256_datos"]),
            ("Hash índices de prueba", res["particion"]["hash_prueba"]), ("Semilla", str(res["semilla"])),
            ("Python", platform.python_version()), ("scikit-learn", sklearn.__version__), ("xgboost", xgboost.__version__),
            ("optuna", optuna.__version__), ("Run MLflow", str(res.get("mlflow_run"))),
            ("Generado (UTC)", datetime.now(timezone.utc).isoformat(timespec="seconds"))]


def _interp_lines(i: Dict[str, str]) -> List[str]:
    return [f"Qué muestra: {i['que_muestra']}", f"Cómo leerla: {i['como_leerla']}", f"Interpretación: {i['interpretacion']}", f"Limitaciones: {i['limitaciones']}"]


def generate(key: str, formats=("pdf", "docx", "xlsx"), tipo: str = "cientifico") -> Dict[str, Any]:
    res = json.loads((ARTIFACTS / key / "results.json").read_text(encoding="utf-8"))
    figs = figures(res, key)
    tabs = tables(res) if tipo != "gerencial" else tables(res)[2:3]
    meta = metadata(res)
    d = out_dir(key)
    files = {}
    title = f"SP-03 · Reporte {tipo} · {res['cohorte']} · escenario {res['escenario']}"
    if "xlsx" in formats:
        from openpyxl import Workbook
        from openpyxl.formatting.rule import ColorScaleRule
        wb = Workbook()
        ws = wb.active
        ws.title = "Léeme"
        for row in ([title], [DISCLAIMER], ["Cada hoja contiene una tabla y su bloque de interpretabilidad."]):
            ws.append(row)
        for i, (t, head, rows, it) in enumerate(tabs, 1):
            sh = wb.create_sheet(f"Tabla{i}")
            sh.append([t]); sh.append(head)
            for r in rows:
                sh.append(r)
            sh.freeze_panes = "A3"
            if len(rows) and len(head) > 2:
                sh.conditional_formatting.add(f"B3:{chr(64 + min(len(head), 26))}{2 + len(rows)}",
                                              ColorScaleRule(start_type="min", start_color="F8696B", mid_type="percentile", mid_value=50, mid_color="FFEB84", end_type="max", end_color="63BE7B"))
            sh.append([])
            for line in _interp_lines(it):
                sh.append([line])
        mt = wb.create_sheet("Metadatos")
        for k, v in meta:
            mt.append([k, v])
        p = d / f"reporte_{tipo}.xlsx"
        wb.save(p)
        files["xlsx"] = p.relative_to(ROOT).as_posix()
    if "docx" in formats:
        from docx import Document
        from docx.shared import Mm
        doc = Document()
        doc.add_heading(title, 0)
        doc.add_paragraph(DISCLAIMER).italic = True
        for t, head, rows, it in tabs:
            doc.add_heading(t, level=2)
            tb = doc.add_table(rows=1, cols=len(head))
            tb.style = "Light Grid Accent 1"
            for c, h in zip(tb.rows[0].cells, head):
                c.text = str(h)
            for r in rows:
                for c, v in zip(tb.add_row().cells, r):
                    c.text = str(v)
            for line in _interp_lines(it):
                doc.add_paragraph(line, style="List Bullet")
        for i, f in enumerate(figs, 1):
            doc.add_heading(f"Figura {i}", level=2)
            doc.add_picture(str(ROOT / f["archivos"][0]), width=Mm(load("reports")["columna_mm"]))
            doc.add_paragraph(f["leyenda"])
        doc.add_heading("Metadatos de reproducibilidad", level=2)
        for k, v in meta:
            doc.add_paragraph(f"{k}: {v}")
        p = d / f"reporte_{tipo}.docx"
        doc.save(p)
        files["docx"] = p.relative_to(ROOT).as_posix()
    if "pdf" in formats:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
        st = getSampleStyleSheet()
        story = [Paragraph(title, st["Title"]), Paragraph(DISCLAIMER, st["Italic"]), Spacer(1, 6)]
        for t, head, rows, it in tabs:
            story += [Paragraph(t, st["Heading2"])]
            tbl = Table([head] + [[str(v) for v in r] for r in rows], repeatRows=1)
            tbl.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6F1")), ("FONTSIZE", (0, 0), (-1, -1), 6.5)]))
            story += [tbl] + [Paragraph(line, st["BodyText"]) for line in _interp_lines(it)] + [Spacer(1, 6)]
        for i, f in enumerate(figs, 1):
            story += [Paragraph(f"Figura {i}", st["Heading2"]), Image(str(ROOT / f["archivos"][0]), width=85 * mm, height=85 * mm), Paragraph(f["leyenda"], st["BodyText"])]
        story += [Paragraph("Metadatos de reproducibilidad", st["Heading2"])] + [Paragraph(f"{k}: {v}", st["BodyText"]) for k, v in meta]

        def footer(canvas, doc_):
            canvas.setFont("Helvetica", 7)
            canvas.drawString(15 * mm, 8 * mm, f"SP-03 · uso investigativo, no clínico · página {doc_.page}")

        p = d / f"reporte_{tipo}.pdf"
        SimpleDocTemplate(str(p), pagesize=A4).build(story, onFirstPage=footer, onLaterPages=footer)
        files["pdf"] = p.relative_to(ROOT).as_posix()
    return {"archivos": files, "figuras": figs, "tablas": [{"titulo": t, "columnas": h, "filas": r, "interpretabilidad": i} for t, h, r, i in tabs]}
