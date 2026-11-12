"""Genera las figuras de explicabilidad (600 DPI) y compila todas las figuras v3 en
docs/FIGURAS_V3.docx y docs/FIGURAS_V3.md (índice con rutas para Claude Code)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from diabetes_ml.config import load  # noqa: E402
from diabetes_ml.reports import _save  # noqa: E402

KEYS = ["nhanes_2017_2018__A", "nhanes_2017_2018__B", "pima_indians__A", "pima_indians__B"]
TITLES = {"nhanes_2017_2018__A": "NHANES · escenario A (con HbA1c y glucosa)", "nhanes_2017_2018__B": "NHANES · escenario B (sin HbA1c ni glucosa)",
          "pima_indians__A": "Pima · escenario A (con glucosa OGTT)", "pima_indians__B": "Pima · escenario B (sin glucosa ni insulina)"}
BASE = [("roc", "Curvas ROC"), ("pr", "Curvas precisión-sensibilidad"), ("calibracion", "Calibración"), ("curva_decision", "Curva de decisión")]


def explain_figures(key):
    exp = json.loads((ROOT / "models/v3" / key / "explain.json").read_text(encoding="utf-8"))
    res = json.loads((ROOT / "models/v3" / key / "results.json").read_text(encoding="utf-8"))
    feats = res["predictores"]
    out_dir = ROOT / load("reports")["salida"] / key
    w = load("reports")["columna_mm"] / 25.4
    plt.rcParams.update({"font.size": load("reports")["fuente_pt"]})
    figs = []
    shap_key = "shap_xgb" if "shap_xgb" in exp["global"] else next(k for k in exp["global"] if k.startswith("shap"))
    imp = sorted(exp["global"][shap_key].items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(w, w * 0.9))
    ax.barh([k for k, _ in imp], [v for _, v in imp], color="#0072B2")
    ax.set_xlabel("Media |SHAP| (log-odds)")
    top = imp[-1][0]
    figs.append(("shap_global", "Importancia SHAP global (XGBoost)", _save(fig, out_dir / "fig_shap_global"),
                 f"Importancia global por media del valor SHAP absoluto en la prueba aislada. La variable con mayor peso es {top}. "
                 "Describe asociaciones aprendidas por el modelo, no causalidad."))
    if "tabnet" in exp:
        M = np.array(exp["tabnet"]["mapa_calor"]["matriz"])
        fig, ax = plt.subplots(figsize=(w, w))
        im = ax.imshow(M, aspect="auto", cmap="viridis", interpolation="nearest")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Peso de atención", fontsize=6)
        ax.set_xticks(range(len(feats)))
        ax.set_xticklabels(feats, rotation=60, ha="right", fontsize=6)
        ax.set_ylabel("Pacientes de prueba (mayor a menor riesgo)")
        figs.append(("tabnet_atencion", "Mapa de atención de TabNet", _save(fig, out_dir / "fig_tabnet_atencion"),
                     "Máscara de atención agregada de TabNet en 200 pacientes de prueba estratificados, ordenados por probabilidad. "
                     "Los colores claros indican las variables que el modelo selecciona en cada paciente."))
    conc = exp.get("concordancia", [])
    med = float(np.median([c["spearman"] for c in conc])) if conc else float("nan")
    return figs, med


def main():
    from docx import Document
    from docx.shared import Mm, Pt
    doc = Document()
    doc.styles["Normal"].font.size = Pt(10)
    doc.add_heading("Figuras del pipeline v3 — SP-03 Diabetes XAI", 0)
    doc.add_paragraph("Figuras a 600 DPI generadas desde los artefactos reales de models/v3 (partición 70/15/15, CV anidada 5×3, "
                      "prueba aislada leída una sola vez). Cada figura existe en PNG, TIFF (LZW) y PDF vectorial en la ruta indicada. "
                      "Paleta Okabe-Ito, apta para daltonismo. Uso investigativo, no clínico.")
    md = ["# Figuras del pipeline v3", "", "Índice para actualizar el artículo. Cada figura tiene versión PNG/TIFF a 600 DPI y PDF vectorial "
          "(misma ruta, cambia la extensión). Cifras de origen: `docs/RESULTADOS_V3.md`.", "",
          "| N.º | Cohorte · escenario | Figura | Archivo PNG | Pie sugerido |", "| --- | --- | --- | --- | --- |"]
    n = 0
    for key in KEYS:
        res = json.loads((ROOT / "models/v3" / key / "results.json").read_text(encoding="utf-8"))
        pm = res["evaluacion_prueba"]["por_modelo"]
        best = max(pm, key=lambda m: pm[m]["metricas"]["roc_auc"])
        extra, med = explain_figures(key)
        doc.add_heading(TITLES[key], 1)
        doc.add_paragraph(f"n = {res['auditoria']['n_oficial']}, eventos = {res['auditoria']['eventos']}. "
                          f"Mayor ROC-AUC en prueba: {best} ({pm[best]['metricas']['roc_auc']:.3f}). "
                          f"Spearman mediano entre métodos de explicabilidad: {med:.2f}.")
        captions = {
            "roc": "Curvas ROC en la prueba aislada para los cinco modelos; entre paréntesis, el ROC-AUC. La diagonal es el azar.",
            "pr": f"Curvas precisión-sensibilidad; la línea punteada es la prevalencia ({res['auditoria']['prevalencia']:.3f}).",
            "calibracion": "Diagrama de confiabilidad (10 bins por cuantiles) tras calibración sigmoide en Validación; la pendiente ideal es 1.",
            "curva_decision": "Curva de decisión: beneficio neto frente a «tratar a todos» y «ninguno» para umbrales de 5–50 %.",
        }
        items = [(fid, title, [f"reports/v3/{key}/fig_{fid}.png"], captions[fid]) for fid, title in BASE] + extra
        for fid, title, files, cap in items:
            n += 1
            png = files[0] if isinstance(files, list) else files
            doc.add_heading(f"Figura {n}. {title}", 2)
            doc.add_picture(str(ROOT / png), width=Mm(110))
            doc.add_paragraph(cap)
            doc.add_paragraph(f"Archivo: {png} (también .tiff y .pdf)").runs[0].font.size = Pt(8)
            md.append(f"| {n} | {TITLES[key]} | {title} | `{png}` | {cap} |")
    md += ["", "## Correspondencia sugerida con la plantilla del artículo", "",
           "- Figuras 2–5 del artículo: usar ROC, PR, calibración y curva de decisión de **NHANES · B** (resultado principal) y las de NHANES · A como comparación.",
           "- Figura 6: SHAP global de NHANES · B. Figura 7: mapa de atención TabNet de NHANES · B.",
           "- Pima: material suplementario.",
           "", "Regenerar: `python -m scripts.run_pipeline_v3 --all` y luego `python -m scripts.figures_v3_doc`."]
    doc.save(ROOT / "docs/FIGURAS_V3.docx")
    (ROOT / "docs/FIGURAS_V3.md").write_text("\n".join(md), encoding="utf-8")
    print(f"{n} figuras -> docs/FIGURAS_V3.docx y docs/FIGURAS_V3.md")


if __name__ == "__main__":
    main()
