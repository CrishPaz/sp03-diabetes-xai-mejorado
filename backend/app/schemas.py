from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, model_validator


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    service: str
    version: str
    models_loaded: bool
    available_cohorts: List[str]
    available_models: List[str]
    random_seed: int
    disclaimer: str
    versiones: Dict[str, str] = {}
    llm: str = "sin clave"
    hash_datasets: Dict[str, str] = {}
    modelos_v3: List[str] = []


class NhanesPatientInput(BaseModel):
    edad: float = Field(..., ge=18.0, le=110.0, description="Edad en años cumplidos (≥18)")
    imc: float = Field(..., ge=12.0, le=80.0, description="Índice de Masa Corporal (kg/m²)")
    glucosa: float = Field(..., ge=35.0, le=600.0, description="Glucosa sérica (mg/dL)")
    hba1c: float = Field(..., ge=3.0, le=20.0, description="Hemoglobina glicosilada HbA1c (%)")
    presion_sistolica: float = Field(..., ge=65.0, le=260.0, description="Presión arterial sistólica (mmHg)")
    presion_diastolica: float = Field(..., ge=35.0, le=160.0, description="Presión arterial diastólica (mmHg)")
    colesterol_total: float = Field(..., ge=70.0, le=600.0, description="Colesterol total sérico (mg/dL)")
    colesterol_hdl: float = Field(..., ge=10.0, le=160.0, description="Colesterol HDL sérico (mg/dL)")
    trigliceridos: float = Field(..., ge=20.0, le=1500.0, description="Triglicéridos séricos (mg/dL)")
    antecedente_familiar_diabetes: int = Field(
        ..., ge=0, le=1, description="Antecedente familiar de diabetes en primer grado (0=No, 1=Sí)"
    )
    antecedente_hipertension: int = Field(
        ..., ge=0, le=1, description="Diagnóstico previo de hipertensión arterial (0=No, 1=Sí)"
    )
    tratamiento_antihipertensivo: int = Field(
        ..., ge=0, le=1, description="Uso actual de medicación antihipertensiva (0=No, 1=Sí)"
    )

    @model_validator(mode="after")
    def validate_clinical_coherence(self) -> "NhanesPatientInput":
        if self.presion_diastolica >= self.presion_sistolica:
            raise ValueError(
                "La presión arterial diastólica debe ser estrictamente menor que la presión arterial sistólica."
            )
        if self.colesterol_hdl >= self.colesterol_total:
            raise ValueError(
                "El colesterol HDL no puede ser mayor o igual que el colesterol total."
            )
        return self


class PimaPatientInput(BaseModel):
    embarazos: float = Field(..., ge=0.0, le=25.0, description="Número de embarazos previos")
    glucosa: float = Field(..., ge=35.0, le=400.0, description="Glucosa plasmática a las 2h OGTT (mg/dL)")
    presion_diastolica: float = Field(..., ge=30.0, le=150.0, description="Presión arterial diastólica (mmHg)")
    pliegue_cutaneo: float = Field(..., ge=5.0, le=110.0, description="Pliegue cutáneo tricipital (mm)")
    insulina: float = Field(..., ge=10.0, le=950.0, description="Insulina sérica a las 2h (μU/mL)")
    imc: float = Field(..., ge=12.0, le=80.0, description="Índice de Masa Corporal (kg/m²)")
    pedigri_diabetes: float = Field(..., ge=0.02, le=3.5, description="Función de pedigrí de diabetes")
    edad: float = Field(..., ge=18.0, le=110.0, description="Edad en años cumplidos")


class PredictRequest(BaseModel):
    dataset_id: Literal["nhanes_2017_2018", "pima_indians"] = Field(
        default="nhanes_2017_2018",
        description="Cohorte clínica de referencia para el modelo predictivo",
    )
    model_id: Literal[
        "xgboost",
        "random_forest",
        "tabnet",
        "stacking_ensemble",
        "hybrid_survival",
        "logistic_regression",
    ] = Field(
        default="stacking_ensemble",
        description="Algoritmo predictivo seleccionado (individual o híbrido)",
    )
    features: Dict[str, float] = Field(
        ..., description="Diccionario de variables clínicas requeridas por la cohorte seleccionada"
    )
    patient_id: Optional[str] = Field(
        default=None, description="Identificador único anonimizado del paciente (opcional)"
    )


class ShapFeatureContribution(BaseModel):
    feature: str
    label: str
    value: float
    unit: str
    shap_value: float
    abs_importance: float
    direction: Literal["incrementa_riesgo", "reduce_riesgo", "neutro"]
    clinical_interpretation: str


class ModelProbabilityComparison(BaseModel):
    model_id: str
    model_name: str
    probability: float
    percentage: float
    risk_stratum: str


class SurvivalProjection(BaseModel):
    horizon_years: int
    complication_risk_probability: float
    complication_risk_percentage: float
    projected_event: str
    hazard_ratio_estimate: float


class PredictResponse(BaseModel):
    dataset_id: str
    dataset_name: str
    model_id: str
    model_name: str
    outcome_evaluated: str
    complications_note: str
    probability: float
    percentage: float
    prediction: int
    threshold: float
    risk_stratum: Literal["Bajo", "Moderado", "Alto", "Muy Alto"]
    risk_recommendation: str
    base_value: float
    shap_contributions: List[ShapFeatureContribution]
    all_models_comparison: List[ModelProbabilityComparison]
    survival_projections: Optional[List[SurvivalProjection]] = None
    clinical_evidence_summary: Optional[str] = None
    disclaimer: str


# ==========================================
# Esquemas RAG (LangChain / ADA / OMS / CDC)
# ==========================================


class RagQueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Pregunta clínica o epidemiológica a resolver")
    patient_context: Optional[Dict[str, Any]] = Field(
        default=None, description="Variables clínicas y riesgo del paciente para contextualizar la respuesta"
    )
    max_sources: int = Field(default=3, ge=1, le=8, description="Número máximo de fuentes oficiales a citar")


class RagSourceCitation(BaseModel):
    source_id: str
    guideline: str
    section: str
    topic: str
    recommendation: str
    evidence_level: str
    doi_or_url: str
    similarity_score: float


class RagQueryResponse(BaseModel):
    query: str
    synthesized_answer: str
    citations: List[RagSourceCitation]
    clinical_guardrails: List[str]
    disclaimer: str
    response_latency_ms: float


# ==========================================
# Esquemas de Automatización n8n
# ==========================================


class N8nWorkflowStatus(BaseModel):
    workflow_id: str
    name: str
    phase: str
    description: str
    active: bool
    schedule: str
    last_run: Optional[str] = None
    last_status: Literal["success", "running", "failed", "pending"] = "pending"


class N8nTriggerRequest(BaseModel):
    workflow_id: str
    payload: Optional[Dict[str, Any]] = None


class N8nTriggerResponse(BaseModel):
    workflow_id: str
    status: str
    execution_id: str
    message: str
    timestamp: str


class AlertItem(BaseModel):
    id: int
    patient_id: str
    patient_name: str
    edad: float
    riesgo_pct: float
    estrato: str
    top_factor: str
    accion_recomendada: str
    generada_en: str
    estado: str


# ==========================================
# Esquemas de Reportes Multiformato
# ==========================================


class ReportGenerationRequest(BaseModel):
    format: Literal["pdf", "docx", "xlsx"] = Field(..., description="Formato del reporte generado")
    patient_name: Optional[str] = "Paciente Clínico Anonimizado"
    patient_id: Optional[str] = "PX-2026-001"
    model_id: str = "stacking_ensemble"
    features: Dict[str, float]
    prediction_result: Optional[Dict[str, Any]] = None


# ==========================================
# Esquemas de Evaluación Estadística
# ==========================================


class StatisticalEvaluationResponse(BaseModel):
    status: str
    friedman_test: Dict[str, Any]
    wilcoxon_pairwise_holm: List[Dict[str, Any]]
    mcnemar_pairwise: List[Dict[str, Any]]
    delong_roc_comparison: Optional[Dict[str, Any]] = None
    bootstrap_confidence_intervals: Dict[str, Any]
    models_evaluated: List[str]
    sample_size_test: int
    methodology_notes: str


class BatchRowPrediction(BaseModel):
    row_index: int
    probability: float
    percentage: float
    prediction: int
    risk_stratum: str
    top_risk_factor: str


class BatchPredictResponse(BaseModel):
    dataset_id: str
    model_id: str
    is_synthetic_demo: bool
    demo_warning: Optional[str]
    total_rows_processed: int
    high_risk_count: int
    mean_probability: float
    predictions: List[BatchRowPrediction]
    disclaimer: str
