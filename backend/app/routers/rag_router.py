from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from backend.app.rag.agent import DiabetesClinicalRAGAgent
from backend.app.rag.knowledge_base import CLINICAL_GUIDELINES_CORPUS
from backend.app.schemas import RagQueryRequest, RagQueryResponse

router = APIRouter(prefix="/rag", tags=["Asistente RAG Clínico y LangChain / LangFlow"])

LANGFLOW_DIR = Path(__file__).resolve().parent.parent.parent.parent / "langflow"
agent = DiabetesClinicalRAGAgent()


@router.post("/query", response_model=RagQueryResponse, summary="Consultar al Agente Clínico RAG con Guías ADA/OMS/CDC")
def query_clinical_rag(request: RagQueryRequest) -> RagQueryResponse:
    """
    Ejecuta el pipeline RAG (Retrieval-Augmented Generation) sobre el corpus clínico oficial:
    1. Recupera las directrices más afines (ADA Standards of Care 2024, OMS, CDC).
    2. Aplica restricciones clínicas y guardrails (el LLM explica y orienta, no emite diagnóstico final).
    3. Contextualiza con los parámetros del paciente (si se proporcionan).
    4. Genera una síntesis con evidencia, grado de recomendación y DOI verificables.
    """
    result = agent.answer_query(
        query=request.query,
        patient_context=request.patient_context,
        top_k=request.max_sources,
    )
    return RagQueryResponse(**result)


@router.get("/guidelines", summary="Catálogo de Guías Clínicas Oficiales Indexadas")
def get_guidelines_catalog() -> Dict[str, Any]:
    """Retorna las directrices clínicas oficiales indexadas en la base vectorial/léxica RAG."""
    return {
        "status": "ok",
        "total_guidelines": len(CLINICAL_GUIDELINES_CORPUS),
        "organizations": ["American Diabetes Association (ADA 2024)", "World Health Organization (OMS)", "CDC"],
        "guidelines": CLINICAL_GUIDELINES_CORPUS,
    }


@router.get("/langflow-schema", summary="Exportar flujo visual de LangFlow para el Agente RAG")
def get_langflow_schema() -> Dict[str, Any]:
    """Retorna el flujo JSON compatible con la interfaz visual de LangFlow."""
    flow_path = LANGFLOW_DIR / "diabetes_clinical_rag_flow.json"
    if not flow_path.exists():
        raise HTTPException(status_code=404, detail="Archivo de flujo LangFlow no encontrado")
    with open(flow_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "status": "ok",
        "tool": "LangFlow Visual Pipeline",
        "flow": data,
    }
