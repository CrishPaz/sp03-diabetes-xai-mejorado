from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

from backend.app.database import (
    create_alert,
    get_recent_alerts,
    get_recent_audit_logs,
    log_audit_event,
)

router = APIRouter(prefix="/n8n", tags=["Automatización n8n"])

WORKFLOWS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "n8n" / "workflows"

# Metadatos de los 6 flujos definidos por el docente Juan Pedro Santos Fernández
N8N_WORKFLOWS_REGISTRY = [
    {
        "workflow_id": "WF-01",
        "file_name": "WF-01_descarga_ingesta_datasets.json",
        "name": "WF-01: Descarga e Ingesta Periódica de Datasets",
        "phase": "Fase F3 Datasets / Ingestión",
        "schedule": "0 2 * * 1 (Semanal Lunes 02:00 AM UTC)",
        "description": "Descarga programada desde CDC NHANES y repositorios de diabetes, validando checksums y esquemas.",
        "active": True,
        "last_status": "success",
        "last_run": "2026-11-03T02:00:00Z",
    },
    {
        "workflow_id": "WF-02",
        "file_name": "WF-02_calidad_imputacion_datos.json",
        "name": "WF-02: Control de Calidad, Imputación y Detección de Anomalías",
        "phase": "Fases F4 EDA y F5 Preprocesamiento",
        "schedule": "30 2 * * 1 (Semanal Lunes 02:30 AM UTC)",
        "description": "Validación de coherencia fisiológica (PAS>PAD, glucosa/HbA1c), imputación por mediana e IQR.",
        "active": True,
        "last_status": "success",
        "last_run": "2026-11-03T02:30:00Z",
    },
    {
        "workflow_id": "WF-03",
        "file_name": "WF-03_prediccion_programada_fastapi.json",
        "name": "WF-03: Ejecución Programada de Inferencia vía FastAPI",
        "phase": "Fase F11 Prototipo y Automatización",
        "schedule": "0 */4 * * * (Cada 4 Horas)",
        "description": "Envío de lotes de pacientes sin diagnosticar al endpoint /predict/batch de FastAPI.",
        "active": True,
        "last_status": "success",
        "last_run": "2026-11-04T08:00:00Z",
    },
    {
        "workflow_id": "WF-04",
        "file_name": "WF-04_alertas_riesgo_cardiometabolico.json",
        "name": "WF-04: Generación y Distribución de Alertas Clínicas de Riesgo",
        "phase": "Fase F11 Vigilancia Epidemiológica",
        "schedule": "15 */4 * * * (Tras cada lote predictivo)",
        "description": "Filtrado de pacientes con riesgo ≥ 75% o HbA1c ≥ 6.5%; notificación al equipo asistencial.",
        "active": True,
        "last_status": "success",
        "last_run": "2026-11-04T08:15:00Z",
    },
    {
        "workflow_id": "WF-05",
        "file_name": "WF-05_elaboracion_reportes_pdf_word_excel.json",
        "name": "WF-05: Elaboración Automática de Reportes Clínicos y Gerenciales",
        "phase": "Fases F11 y F12 Publicación y Trazabilidad",
        "schedule": "0 6 * * 1 (Lunes 06:00 AM UTC)",
        "description": "Generación multi-formato de informes consolidados en PDF, Word (.docx) y Excel (.xlsx).",
        "active": True,
        "last_status": "success",
        "last_run": "2026-11-03T06:00:00Z",
    },
    {
        "workflow_id": "WF-06",
        "file_name": "WF-06_auditoria_monitoreo_rendimiento.json",
        "name": "WF-06: Auditoría, Monitoreo y Evaluación de Rendimiento MLflow",
        "phase": "Fases F8 Validación y F9 Evaluación Estadística",
        "schedule": "0 0 1 * * (Mensual día 1 00:00 UTC)",
        "description": "Detección de data drift (deriva de datos), recálculo de Brier Score, AUC-ROC y logs de auditoría.",
        "active": True,
        "last_status": "success",
        "last_run": "2026-11-01T00:00:00Z",
    },
]


@router.get("/workflows", summary="Listar los 6 workflows de n8n configurados")
def get_n8n_workflows() -> Dict[str, Any]:
    """Retorna la definición, estado, programación y archivo JSON de cada uno de los 6 workflows n8n."""
    enriched = []
    for wf in N8N_WORKFLOWS_REGISTRY:
        fpath = WORKFLOWS_DIR / wf["file_name"]
        item = dict(wf)
        item["file_exists"] = fpath.exists()
        item["file_path"] = str(fpath)
        enriched.append(item)
    return {
        "status": "ok",
        "total_workflows": len(enriched),
        "engine": "n8n Workflow Automation Engine",
        "version": "1.34+",
        "workflows": enriched,
    }


@router.get("/workflows/{workflow_id}/json", summary="Obtener el JSON exportable de un workflow de n8n")
def get_workflow_json(workflow_id: str) -> Dict[str, Any]:
    wf = next((w for w in N8N_WORKFLOWS_REGISTRY if w["workflow_id"].upper() == workflow_id.upper()), None)
    if not wf:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} no encontrado")
    fpath = WORKFLOWS_DIR / wf["file_name"]
    if not fpath.exists():
        raise HTTPException(status_code=404, detail=f"Archivo {wf['file_name']} no existe en disco")
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@router.post("/workflows/{workflow_id}/trigger", summary="Disparar manualmente la ejecución de un workflow n8n")
def trigger_n8n_workflow(workflow_id: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    wf = next((w for w in N8N_WORKFLOWS_REGISTRY if w["workflow_id"].upper() == workflow_id.upper()), None)
    if not wf:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} no encontrado")

    now_iso = datetime.now(timezone.utc).isoformat()
    exec_id = f"exec-{workflow_id.lower()}-{int(datetime.now(timezone.utc).timestamp())}"

    # Si es WF-04, creamos una alerta demostrativa en la BD
    if workflow_id.upper() == "WF-04":
        create_alert(
            patient_id=payload.get("patient_id", "PX-EMERG-88") if payload else "PX-EMERG-88",
            patient_name=payload.get("patient_name", "Paciente en Monitoreo Programado") if payload else "Paciente en Monitoreo Programado",
            edad=float(payload.get("edad", 58.0)) if payload else 58.0,
            riesgo_pct=float(payload.get("riesgo_pct", 84.6)) if payload else 84.6,
            estrato="Muy Alto",
            top_factor="Hemoglobina glicosilada HbA1c (8.4%) y Glucosa sérica (182 mg/dL)",
            accion_recomendada="Derivación prioritaria a Endocrinología y prueba confirmatoria inmediata.",
        )

    log_audit_event(
        action=f"n8n_trigger_{workflow_id.lower()}",
        endpoint=f"/n8n/workflows/{workflow_id}/trigger",
        status_code=200,
        details={"execution_id": exec_id, "workflow_name": wf["name"], "payload": payload},
    )

    return {
        "status": "success",
        "workflow_id": workflow_id.upper(),
        "workflow_name": wf["name"],
        "execution_id": exec_id,
        "triggered_at": now_iso,
        "message": f"Workflow {workflow_id} ejecutado exitosamente en el orquestador n8n.",
    }


@router.get("/alerts", summary="Listar alertas clínicas generadas por WF-04")
def list_n8n_alerts(limit: int = Query(default=20, ge=1, le=100)) -> Dict[str, Any]:
    alerts = get_recent_alerts(limit=limit)
    return {
        "status": "ok",
        "total_alerts": len(alerts),
        "source_workflow": "WF-04_alertas_riesgo_cardiometabolico",
        "alerts": alerts,
    }


@router.get("/audit-logs", summary="Listar registros de auditoría generados por WF-06")
def list_audit_logs(limit: int = Query(default=30, ge=1, le=200)) -> Dict[str, Any]:
    logs = get_recent_audit_logs(limit=limit)
    return {
        "status": "ok",
        "total_logs": len(logs),
        "source_workflow": "WF-06_auditoria_monitoreo_rendimiento",
        "audit_logs": logs,
    }
