"""DeepSeek como proveedor LLM (API compatible con OpenAI) con guardrails clínicos y respaldo determinista.
La clave se lee del entorno y nunca se registra."""
from __future__ import annotations

import os
import re
from typing import Any, Callable, Dict, List, Optional

import httpx

from diabetes_ml.config import DISCLAIMER, load

SYSTEM_PROMPT = ("Eres un asistente de apoyo investigativo sobre diabetes. Nunca emites diagnósticos, dosis ni prescripciones. "
                 "Usas lenguaje de riesgo estimado ('rango compatible con criterios ADA'), citas la guía y recomiendas confirmación clínica.")
FORBIDDEN = [r"\btiene diabetes\b", r"\busted es diab[eé]tic", r"\bdiagn[oó]stico confirmado\b", r"\btome \d", r"\b\d+\s?(mg|ui|unidades)\b.*\b(al d[ií]a|cada)\b",
             r"\bprescrib", r"\bdosis de\b"]
REPLACEMENT = "[contenido retirado por guardrail: el sistema no diagnostica ni prescribe]"
_TRANSPORT: Optional[Callable[[List[Dict[str, str]]], str]] = None   # inyectable en tests


def _cfg() -> Dict[str, Any]:
    c = load("llm")
    return {"key": os.getenv(c["api_key_env"], ""), "base": os.getenv(c["base_url_env"], "https://api.deepseek.com"),
            "model": os.getenv(c["modelo_env"], c["modelo_por_defecto"]), "temp": float(os.getenv("LLM_TEMPERATURE", c["temperatura"])),
            "timeout": float(os.getenv("LLM_TIMEOUT_S", c["timeout_s"])), "retries": int(os.getenv("LLM_MAX_RETRIES", c["max_reintentos"]))}


def status() -> str:
    if _TRANSPORT is not None:
        return "ok (simulado)"
    return "ok" if _cfg()["key"] else "sin clave"


def chat(messages: List[Dict[str, str]]) -> Optional[str]:
    if _TRANSPORT is not None:
        return _TRANSPORT(messages)
    c = _cfg()
    if not c["key"]:
        return None
    for _ in range(c["retries"] + 1):
        try:
            r = httpx.post(f"{c['base'].rstrip('/')}/chat/completions", timeout=c["timeout"],
                           headers={"Authorization": f"Bearer {c['key']}"},
                           json={"model": c["model"], "temperature": c["temp"], "messages": messages})
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception:
            continue
    return None


def enforce_guardrails(text: str, citation: str) -> str:
    for pat in FORBIDDEN:
        text = re.sub(pat + r"[^.]*\.?", REPLACEMENT, text, flags=re.IGNORECASE)
    if citation not in text:
        text += f"\n\nFuente: {citation}."
    if DISCLAIMER not in text:
        text += f"\n\n{DISCLAIMER}"
    return text


def guarded_answer(question: str) -> Dict[str, Any]:
    from backend.app.rag.agent import DiabetesClinicalRAGAgent
    rag = DiabetesClinicalRAGAgent().answer_query(question, top_k=2)
    edition = load("llm")["edicion_guia_ada"]
    cite = rag["citations"][0]["guideline"] if rag["citations"] else edition
    context = "\n".join(c["recommendation"] for c in rag["citations"])
    llm = chat([{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Contexto de guías:\n{context}\n\nPregunta: {question}"}])
    text, source = (llm, "deepseek") if llm else (rag["synthesized_answer"], "plantilla determinista (LLM no disponible)")
    return {"respuesta": enforce_guardrails(text, cite), "fuente_generacion": source, "edicion_guia": edition, "cita": cite}


def guarded_narrative(pred: Dict[str, Any]) -> Dict[str, str]:
    top = "; ".join(e["frase"] for e in pred.get("top5_shap", [])[:3])
    base = (f"Riesgo estimado por el modelo: {pred['probabilidad']:.0%} (nivel {pred['severidad_modelo'].lower()}). "
            f"Factores que más pesan en esta estimación: {top or 'no disponibles'}. "
            "Según ADA Standards of Care, un riesgo elevado justifica tamizaje con HbA1c o glucosa en ayunas y confirmación clínica.")
    llm = chat([{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Reformula en español claro, sin cambiar cifras ni añadir diagnósticos: " + base}])
    return {"texto": enforce_guardrails(llm or base, "ADA Standards of Care in Diabetes"),
            "fuente": "deepseek" if llm else "plantilla determinista"}
