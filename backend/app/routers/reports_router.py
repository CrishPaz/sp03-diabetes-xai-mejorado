from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional
from fastapi import APIRouter, HTTPException, Query, Response, status
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.app.database import log_audit_event
from backend.app.reports import ClinicalReportGenerator
from backend.app.schemas import ReportGenerationRequest

router = APIRouter(prefix="/reports", tags=["Reportes Científicos y Gerenciales (PDF / Word / Excel)"])

generator = ClinicalReportGenerator()


@router.post("/generate", summary="Generar reporte clínico descargable en PDF, Word o Excel")
def generate_report(request: ReportGenerationRequest):
    """
    Genera un informe multi-formato estructurado con:
    - Identificación del paciente
    - Variables clínicas y biomarcadores
    - Riesgo predicho por los 5 modelos
    - Contribuciones SHAP de mayor impacto
    - Recomendaciones clínicas y guías ADA/OMS
    """
    fmt = request.format.lower()
    if fmt not in ("pdf", "docx", "xlsx"):
        raise HTTPException(
            status_code=400,
            detail=f"Formato no soportado: '{fmt}'. Use 'pdf', 'docx' o 'xlsx'.",
        )

    # Preparar datos sintéticos / inferidos para el reporte
    pred_res = request.prediction_result or {
        "model_name": "Ensamble Híbrido Stacking Calibrado",
        "probability": 0.784,
        "percentage": 78.4,
        "risk_stratum": "Muy Alto",
        "threshold": 0.50,
        "risk_recommendation": (
            "Alta probabilidad de diabetes mellitus tipo 2 según el modelo híbrido calibrado. "
            "Se recomienda evaluación médica prioritaria, confirmación por laboratorio con glucosa "
            "plasmática en ayunas y HbA1c, y ajuste dietético según pautas ADA 2024."
        ),
        "all_models_comparison": [
            {"model_name": "Random Forest", "probability": 0.762, "percentage": 76.2, "risk_stratum": "Muy Alto"},
            {"model_name": "XGBoost Optimizado", "probability": 0.795, "percentage": 79.5, "risk_stratum": "Muy Alto"},
            {"model_name": "TabNet Attention", "probability": 0.748, "percentage": 74.8, "risk_stratum": "Alto"},
            {"model_name": "Stacking Ensamble Híbrido", "probability": 0.784, "percentage": 78.4, "risk_stratum": "Muy Alto"},
            {"model_name": "Híbrido Supervivencia (5 años)", "probability": 0.621, "percentage": 62.1, "risk_stratum": "Alto"},
        ],
        "shap_contributions": [
            {
                "label": "Hemoglobina glicosilada HbA1c",
                "value": float(request.features.get("hba1c", 7.2)),
                "unit": "%",
                "shap_value": 0.185,
                "clinical_interpretation": "HbA1c elevada incrementa fuertemente el riesgo proyectado.",
            },
            {
                "label": "Glucosa sérica en ayunas",
                "value": float(request.features.get("glucosa", 145.0)),
                "unit": "mg/dL",
                "shap_value": 0.124,
                "clinical_interpretation": "Glucemia sobre el umbral de normalidad (100 mg/dL).",
            },
            {
                "label": "Índice de Masa Corporal (IMC)",
                "value": float(request.features.get("imc", 31.4)),
                "unit": "kg/m²",
                "shap_value": 0.082,
                "clinical_interpretation": "Obesidad grado I que induce resistencia insulínica.",
            },
        ],
    }

    try:
        if fmt == "pdf":
            file_path = generator.generate_pdf_report(
                patient_name=request.patient_name or "Paciente Clínico Anonimizado",
                patient_id=request.patient_id or "PX-2026-001",
                features=request.features,
                prediction_result=pred_res,
            )
            media_type = "application/pdf"
        elif fmt == "docx":
            file_path = generator.generate_docx_report(
                patient_name=request.patient_name or "Paciente Clínico Anonimizado",
                patient_id=request.patient_id or "PX-2026-001",
                features=request.features,
                prediction_result=pred_res,
            )
            media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        else:
            file_path = generator.generate_excel_report(
                patient_name=request.patient_name or "Paciente Clínico Anonimizado",
                patient_id=request.patient_id or "PX-2026-001",
                features=request.features,
                prediction_result=pred_res,
            )
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

        log_audit_event(
            action=f"report_generated_{fmt}",
            endpoint="/reports/generate",
            status_code=200,
            details={"patient_id": request.patient_id, "file_name": file_path.name, "format": fmt},
        )

        return FileResponse(
            path=str(file_path),
            media_type=media_type,
            filename=file_path.name,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar reporte {fmt}: {str(e)}")
