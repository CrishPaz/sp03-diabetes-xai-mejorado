"""
Módulo de generación de reportes multiformato:
- PDF (ReportLab con gráficos, tablas y semáforo de riesgo)
- Word .docx (python-docx con tablas académicas)
- Excel .xlsx (openpyxl con formato condicional para lotes)
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any, Dict, List

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

import docx
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.shared import Cm, Pt, RGBColor


def generate_patient_pdf(
    patient_info: Dict[str, Any],
    pred_data: Dict[str, Any],
    explanation_text: str = "",
) -> bytes:
    """Genera un reporte clínico formal en PDF con ReportLab."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=17,
        textColor=colors.HexColor("#0f172a"),
        alignment=1,
    )
    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#475569"),
        alignment=1,
    )
    h2_style = ParagraphStyle(
        "H2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#0f766e"),
        spaceBefore=8,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#1e293b"),
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#64748b"),
        alignment=4,
    )

    elements = []

    # Encabezado
    elements.append(Paragraph("PROYECTO SP-03: SISTEMA INTELIGENTE DE VIGILANCIA Y PREDICCIÓN DE DIABETES", title_style))
    elements.append(Paragraph("Reporte de Estratificación Cardiometabólica e Inteligencia Artificial Explicable", subtitle_style))
    elements.append(Spacer(1, 6))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0f766e"), spaceAfter=10))

    # Semáforo de riesgo
    prob = pred_data.get("probability", 0.0)
    stratum = pred_data.get("risk_stratum", "Bajo")
    badge_color = "#15803d" if prob < 0.25 else ("#d97706" if prob < 0.50 else ("#ea580c" if prob < 0.75 else "#dc2626"))

    risk_table_data = [
        [
            Paragraph(f"<b>Fecha de Emisión:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}", body_style),
            Paragraph(f"<b>Modelo Predictivo:</b> {pred_data.get('model_name', 'XGBoost Classifier')}", body_style),
        ],
        [
            Paragraph(f"<b>Probabilidad Estimada:</b> <font size='12' color='{badge_color}'><b>{prob * 100:.1f}%</b></font>", body_style),
            Paragraph(f"<b>Estrato de Riesgo:</b> <font size='11' color='{badge_color}'><b>Riesgo {stratum}</b></font>", body_style),
        ],
    ]
    t_risk = Table(risk_table_data, colWidths=[270, 270])
    t_risk.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    elements.append(t_risk)
    elements.append(Spacer(1, 8))

    # Consenso entre los 5 modelos
    elements.append(Paragraph("1. Consenso entre los 5 Algoritmos Entrenados (3 Individuales + 2 Híbridos)", h2_style))
    all_models = pred_data.get("all_models_comparison", [])
    if all_models:
        m_table_data = [["Algoritmo / Arquitectura", "Probabilidad", "Estrato Asignado"]]
        for m in all_models:
            m_table_data.append([m["model_name"], f"{m['percentage']:.2f}%", m["risk_stratum"]])
        t_models = Table(m_table_data, colWidths=[280, 130, 130])
        t_models.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ]
            )
        )
        elements.append(t_models)
        elements.append(Spacer(1, 8))

    # Desglose SHAP
    elements.append(Paragraph("2. Factores Clínicos Determinantes (Descomposición de Shapley - TreeSHAP)", h2_style))
    shap_factors = pred_data.get("shap_contributions", [])[:6]
    if shap_factors:
        shap_table_data = [["Factor Clínico", "Valor Medido", "Contribución SHAP", "Efecto sobre el Riesgo"]]
        for sf in shap_factors:
            val_txt = f"{sf['value']} {sf.get('unit', '')}"
            sval = sf["shap_value"]
            effect = "Incrementa Riesgo" if sval > 0.005 else ("Factor Protector" if sval < -0.005 else "Neutro")
            shap_table_data.append([sf["label"], val_txt, f"{sval:+.4f}", effect])
        t_shap = Table(shap_table_data, colWidths=[180, 120, 110, 130])
        t_shap.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ]
            )
        )
        elements.append(t_shap)
        elements.append(Spacer(1, 8))

    # Explicación RAG
    elements.append(Paragraph("3. Fundamentación Médica RAG y Recomendaciones (Guías ADA 2024 / OMS)", h2_style))
    exp_clean = explanation_text or pred_data.get("risk_recommendation", "Se sugiere valoración médica preventiva y pruebas de laboratorio.")
    elements.append(Paragraph(exp_clean.replace("\n", "<br/>"), body_style))
    elements.append(Spacer(1, 10))

    # Advertencia regulatoria
    elements.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#e2e8f0"), spaceAfter=6))
    elements.append(
        Paragraph(
            "<b>ADVERTENCIA REGULATORIA Y ÉTICA:</b> Este documento es un reporte generado por un prototipo de "
            "inteligencia artificial explicable con fines de investigación y apoyo al tamizaje. No constituye un diagnóstico médico "
            "formal (SaMD) ni sustituye el juicio clínico presencial ni las pruebas de confirmación diagnóstica (ADA/OMS).",
            disclaimer_style,
        )
    )

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


def generate_patient_docx(
    patient_info: Dict[str, Any],
    pred_data: Dict[str, Any],
    explanation_text: str = "",
) -> bytes:
    """Genera reporte clínico en formato Microsoft Word (.docx)."""
    doc = docx.Document()
    for s in doc.sections:
        s.top_margin = Cm(2.2)
        s.bottom_margin = Cm(2.2)
        s.left_margin = Cm(2.2)
        s.right_margin = Cm(2.2)

    # Título
    p_title = doc.add_paragraph()
    r = p_title.add_run("PROYECTO SP-03: SISTEMA INTELIGENTE DE PREDICCIÓN DE DIABETES MELLITUS\n")
    r.bold = True
    r.font.size = Pt(13)
    r.font.color.rgb = RGBColor(15, 23, 42)

    r_sub = p_title.add_run("Reporte de Estratificación de Riesgo e Inteligencia Artificial Explicable (XAI)")
    r_sub.font.size = Pt(10)
    r_sub.font.color.rgb = RGBColor(71, 85, 105)

    doc.add_heading("1. Resumen de Inferencia y Estratificación", level=2)
    p_res = doc.add_paragraph()
    p_res.add_run(f"Modelo Evaluado: {pred_data.get('model_name', 'XGBoost')}\n").bold = True
    p_res.add_run(f"Probabilidad de Riesgo Estimada: {pred_data.get('probability', 0.0) * 100:.1f}%\n")
    p_res.add_run(f"Estrato de Riesgo Clínico: Riesgo {pred_data.get('risk_stratum', 'Bajo')}\n")
    p_res.add_run(f"Fecha de Emisión: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

    # Tabla de 5 modelos
    doc.add_heading("2. Consenso de Algoritmos (3 Individuales + 2 Híbridos)", level=2)
    all_models = pred_data.get("all_models_comparison", [])
    if all_models:
        tbl = doc.add_table(rows=1 + len(all_models), cols=3)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        hdr = tbl.rows[0].cells
        hdr[0].text = "Algoritmo"
        hdr[1].text = "Probabilidad"
        hdr[2].text = "Estrato"
        for i, m in enumerate(all_models):
            r_cells = tbl.rows[i + 1].cells
            r_cells[0].text = m["model_name"]
            r_cells[1].text = f"{m['percentage']:.2f}%"
            r_cells[2].text = m["risk_stratum"]

    # Explicación RAG
    doc.add_heading("3. Justificación Clínica y Evidencia Médica (RAG ADA 2024 / OMS)", level=2)
    doc.add_paragraph(explanation_text or pred_data.get("risk_recommendation", "Sin observaciones adicionales."))

    # Advertencia
    p_adv = doc.add_paragraph()
    r_adv = p_adv.add_run(
        "Aviso Legal: Reporte investigativo generado por IA. No es un dispositivo médico certificado. "
        "Requiere evaluación médica profesional."
    )
    r_adv.italic = True
    r_adv.font.size = Pt(8.5)
    r_adv.font.color.rgb = RGBColor(100, 116, 139)

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()


def generate_batch_excel(batch_predictions: List[Dict[str, Any]]) -> bytes:
    """Genera hoja de cálculo Excel (.xlsx) con formato condicional de colores por estrato de riesgo."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Predicciones_Lote"

    # Encabezado
    headers = ["Fila #", "Probabilidad (%)", "Predicción Binaria", "Estrato de Riesgo", "Factor Principal SHAP"]
    ws.append(headers)

    header_fill = PatternFill(start_color="0F172A", end_color="0F172A", fill_type="solid")
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    for col_idx in range(1, len(headers) + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center")

    # Rellenos de colores para estratos
    fill_bajo = PatternFill(start_color="DCFCE7", end_color="DCFCE7", fill_type="solid")
    fill_moderado = PatternFill(start_color="FEF3C7", end_color="FEF3C7", fill_type="solid")
    fill_alto = PatternFill(start_color="FEE2E2", end_color="FEE2E2", fill_type="solid")

    for r_idx, item in enumerate(batch_predictions, start=2):
        row_vals = [
            item.get("row_index", r_idx - 1),
            round(item.get("percentage", item.get("probability", 0.0) * 100), 2),
            item.get("prediction", 0),
            item.get("risk_stratum", "Bajo"),
            item.get("top_risk_factor", "N/A"),
        ]
        ws.append(row_vals)

        # Aplicar formato condicional al estrato
        stratum_cell = ws.cell(row=r_idx, column=4)
        stratum_val = str(stratum_cell.value).lower()
        if "bajo" in stratum_val:
            stratum_cell.fill = fill_bajo
        elif "moderado" in stratum_val:
            stratum_cell.fill = fill_moderado
        else:
            stratum_cell.fill = fill_alto

    # Ajustar ancho de columnas
    for col in ws.columns:
        max_len = max(len(str(cell.value or "")) for cell in col)
        col_letter = openpyxl.utils.get_column_letter(col[0].column)
        ws.column_dimensions[col_letter].width = max(max_len + 4, 14)

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.getvalue()


class ClinicalReportGenerator:
    """Clase wrapper para la generación de reportes clínicos y gerenciales multiformato."""

    def __init__(self) -> None:
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent.parent
        self.reports_dir = root / "data" / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_pdf_report(
        self,
        patient_name: str,
        patient_id: str,
        features: Dict[str, float],
        prediction_result: Dict[str, Any],
    ) -> Any:
        patient_info = {
            "name": patient_name,
            "id": patient_id,
            "features": features,
        }
        pdf_bytes = generate_patient_pdf(
            patient_info=patient_info,
            pred_data=prediction_result,
            explanation_text=prediction_result.get("risk_recommendation", ""),
        )
        ts = int(datetime.now(timezone.utc).timestamp())
        out_path = self.reports_dir / f"reporte_{patient_id}_{ts}.pdf"
        with open(out_path, "wb") as f:
            f.write(pdf_bytes)
        return out_path

    def generate_docx_report(
        self,
        patient_name: str,
        patient_id: str,
        features: Dict[str, float],
        prediction_result: Dict[str, Any],
    ) -> Any:
        patient_info = {
            "name": patient_name,
            "id": patient_id,
            "features": features,
        }
        docx_bytes = generate_patient_docx(
            patient_info=patient_info,
            pred_data=prediction_result,
            explanation_text=prediction_result.get("risk_recommendation", ""),
        )
        ts = int(datetime.now(timezone.utc).timestamp())
        out_path = self.reports_dir / f"reporte_{patient_id}_{ts}.docx"
        with open(out_path, "wb") as f:
            f.write(docx_bytes)
        return out_path

    def generate_excel_report(
        self,
        patient_name: str,
        patient_id: str,
        features: Dict[str, float],
        prediction_result: Dict[str, Any],
    ) -> Any:
        batch_rows = [
            {
                "row_index": 1,
                "percentage": prediction_result.get("percentage", 75.0),
                "prediction": prediction_result.get("prediction", 1),
                "risk_stratum": prediction_result.get("risk_stratum", "Alto"),
                "top_risk_factor": "Glucosa sérica / HbA1c",
            }
        ]
        if "all_models_comparison" in prediction_result:
            for idx, m in enumerate(prediction_result["all_models_comparison"], start=2):
                batch_rows.append(
                    {
                        "row_index": idx,
                        "percentage": m.get("percentage", 0.0),
                        "prediction": int(m.get("probability", 0.0) >= 0.5),
                        "risk_stratum": m.get("risk_stratum", "Moderado"),
                        "top_risk_factor": m.get("model_name", "Modelo"),
                    }
                )
        excel_bytes = generate_batch_excel(batch_rows)
        ts = int(datetime.now(timezone.utc).timestamp())
        out_path = self.reports_dir / f"reporte_{patient_id}_{ts}.xlsx"
        with open(out_path, "wb") as f:
            f.write(excel_bytes)
        return out_path
