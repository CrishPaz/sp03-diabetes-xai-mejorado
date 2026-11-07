"""
Módulo de persistencia relacional y auditoría (SQLAlchemy).
Soporta SQLite local para ejecución sin fricción y PostgreSQL para producción.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_DB_URL = f"sqlite:///{DATA_DIR / 'diabetes_app.db'}"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DB_URL)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class PatientRecord(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    patient_code = Column(String(64), unique=True, index=True)
    dataset_id = Column(String(32), default="nhanes_2017_2018")
    features_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    patient_code = Column(String(64), index=True, nullable=True)
    dataset_id = Column(String(32), default="nhanes_2017_2018")
    selected_model_id = Column(String(32), default="xgboost")
    probability = Column(Float, nullable=False)
    risk_stratum = Column(String(32), nullable=False)
    all_models_json = Column(Text, nullable=False)
    shap_factors_json = Column(Text, nullable=True)
    latency_ms = Column(Float, default=0.0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class AlertRecord(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    workflow_id = Column(String(32), default="WF-04")
    event_type = Column(String(64), nullable=False)
    severity = Column(String(16), default="CRITICAL")
    message = Column(Text, nullable=False)
    payload_json = Column(Text, nullable=True)
    status = Column(String(16), default="PENDING")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    event_name = Column(String(64), nullable=False, index=True)
    details = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def init_db() -> None:
    """Inicializa todas las tablas del esquema en SQLite o PostgreSQL."""
    Base.metadata.create_all(bind=engine)


init_db()


def log_audit_event(
    action: str = "action",
    details: Optional[Dict[str, Any]] = None,
    endpoint: Optional[str] = None,
    status_code: int = 200,
    event_name: Optional[str] = None,
) -> None:
    session = SessionLocal()
    try:
        ename = event_name or action
        det_dict = details or {}
        if endpoint:
            det_dict["endpoint"] = endpoint
        if status_code:
            det_dict["status_code"] = status_code
        log = AuditLog(event_name=ename, details=json.dumps(det_dict, ensure_ascii=False))
        session.add(log)
        session.commit()
    except Exception:
        pass
    finally:
        session.close()


def save_patient_and_prediction(
    patient_id: str,
    features: Dict[str, float],
    dataset_id: str,
    model_id: str,
    probability: float,
    prediction: int,
    risk_stratum: str,
    shap_top_factor: Optional[str] = None,
) -> int:
    session = SessionLocal()
    try:
        # Registrar paciente si no existe
        pat = session.query(PatientRecord).filter_by(patient_code=patient_id).first()
        if not pat:
            pat = PatientRecord(
                patient_code=patient_id,
                dataset_id=dataset_id,
                features_json=json.dumps(features, ensure_ascii=False),
            )
            session.add(pat)
            session.commit()

        # Registrar predicción
        rec = PredictionRecord(
            patient_code=patient_id,
            dataset_id=dataset_id,
            selected_model_id=model_id,
            probability=probability,
            risk_stratum=risk_stratum,
            all_models_json=json.dumps({"prediction": prediction}, ensure_ascii=False),
            shap_factors_json=json.dumps({"top_factor": shap_top_factor}, ensure_ascii=False) if shap_top_factor else None,
        )
        session.add(rec)
        session.commit()
        session.refresh(rec)
        return rec.id
    except Exception:
        return 0
    finally:
        session.close()


def create_alert(
    patient_id: str,
    patient_name: str,
    edad: float,
    riesgo_pct: float,
    estrato: str,
    top_factor: str,
    accion_recomendada: str,
    workflow_id: str = "WF-04",
) -> int:
    payload = {
        "patient_id": patient_id,
        "patient_name": patient_name,
        "edad": edad,
        "riesgo_pct": riesgo_pct,
        "estrato": estrato,
        "top_factor": top_factor,
        "accion_recomendada": accion_recomendada,
    }
    return save_alert(
        workflow_id=workflow_id,
        event_type="ALERTA_RIESGO_CRITICO",
        message=f"Alerta de riesgo para {patient_name} ({patient_id}): {riesgo_pct:.1f}% ({estrato})",
        payload=payload,
        severity="CRITICAL",
    )


def get_recent_audit_logs(limit: int = 30) -> List[Dict[str, Any]]:
    session = SessionLocal()
    try:
        rows = (
            session.query(AuditLog)
            .order_by(AuditLog.created_at.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": r.id,
                "event_name": r.event_name,
                "details": json.loads(r.details or "{}"),
                "created_at": r.created_at.isoformat() if r.created_at else "",
            }
            for r in rows
        ]
    except Exception:
        return []
    finally:
        session.close()


def save_prediction(
    dataset_id: str,
    selected_model_id: str,
    probability: float,
    risk_stratum: str,
    all_models: List[Dict[str, Any]],
    shap_factors: List[Dict[str, Any]],
    patient_code: Optional[str] = None,
    latency_ms: float = 0.0,
) -> int:
    session = SessionLocal()
    try:
        rec = PredictionRecord(
            patient_code=patient_code,
            dataset_id=dataset_id,
            selected_model_id=selected_model_id,
            probability=probability,
            risk_stratum=risk_stratum,
            all_models_json=json.dumps(all_models, ensure_ascii=False),
            shap_factors_json=json.dumps(shap_factors, ensure_ascii=False),
            latency_ms=latency_ms,
        )
        session.add(rec)
        session.commit()
        session.refresh(rec)
        return rec.id
    finally:
        session.close()


def save_alert(
    workflow_id: str,
    event_type: str,
    message: str,
    payload: Dict[str, Any],
    severity: str = "CRITICAL",
) -> int:
    session = SessionLocal()
    try:
        alert = AlertRecord(
            workflow_id=workflow_id,
            event_type=event_type,
            severity=severity,
            message=message,
            payload_json=json.dumps(payload, ensure_ascii=False),
        )
        session.add(alert)
        session.commit()
        session.refresh(alert)
        return alert.id
    finally:
        session.close()


def get_recent_alerts(limit: int = 15) -> List[Dict[str, Any]]:
    session = SessionLocal()
    try:
        rows = (
            session.query(AlertRecord)
            .order_by(AlertRecord.created_at.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": r.id,
                "workflow_id": r.workflow_id,
                "event_type": r.event_type,
                "severity": r.severity,
                "message": r.message,
                "payload": json.loads(r.payload_json or "{}"),
                "status": r.status,
                "created_at": r.created_at.isoformat() if r.created_at else "",
            }
            for r in rows
        ]
    finally:
        session.close()
