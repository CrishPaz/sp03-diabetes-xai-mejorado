from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.app.database import (
    init_db,
    log_audit_event,
    save_patient_and_prediction,
)
from backend.app.predictor import DiabetesPredictorService, ROOT_DIR
from backend.app.routers.n8n_router import router as n8n_router
from backend.app.routers.rag_router import router as rag_router
from backend.app.routers.reports_router import router as reports_router
from backend.app.routers.stats_router import router as stats_router
from backend.app.schemas import (
    BatchPredictResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

# Inicializar tablas de SQLite / PostgreSQL
init_db()

app = FastAPI(
    title="SP-03 | API de Predicción Temprana de Diabetes Mellitus Tipo 2 con IA Explicable",
    description=(
        "Backend científico reproducible para el Caso SP-03: Multimodal Explainable Artificial "
        "Intelligence for Early Prediction of Type 2 Diabetes and Its Long-Term Complications. "
        "Arquitectura recomendada: 5 Modelos (3 individuales + 2 híbridos) + Optuna + SHAP + "
        "Batería Estadística (Friedman, Wilcoxon-Holm, McNemar, Bootstrap) + n8n + LangChain/RAG + Reportes."
    ),
    version="2.0.0",
)

cors_origins_env = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173",
)
allowed_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registrar Routers Modulares
app.include_router(rag_router)
app.include_router(n8n_router)
app.include_router(reports_router)
app.include_router(stats_router)

predictor_service = DiabetesPredictorService()


@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
def get_health() -> HealthResponse:
    """Verifica disponibilidad del servicio y de los artefactos de modelos entrenados."""
    return HealthResponse(
        status="ok" if predictor_service.is_ready else "degraded",
        service="sp03-diabetes-xai-api",
        version="2.0.0",
        models_loaded=predictor_service.is_ready,
        available_cohorts=list(predictor_service.bundles.keys()),
        available_models=[
            "random_forest",
            "xgboost",
            "tabnet",
            "stacking_ensemble",
            "hybrid_survival",
            "logistic_regression",
        ],
        random_seed=int(predictor_service.metadata.get("random_seed", 42)),
        disclaimer=predictor_service.metadata.get("clinical_disclaimer", ""),
    )


@app.get("/model-info", tags=["Modelos y Auditoría"])
def get_model_info():
    """
    Devuelve las métricas verificadas en conjunto de prueba aislado y validación cruzada
    de 5 pliegues, parámetros, calibración, análisis de equidad por subgrupos (edad y sexo),
    importancia global SHAP y matriz de trazabilidad con Caso03.docx.
    """
    if not predictor_service.is_ready:
        raise HTTPException(status_code=503, detail="Los modelos aún no están cargados.")
    return predictor_service.metadata


@app.post("/predict", response_model=PredictResponse, tags=["Inferencia Explicable"])
def post_predict(payload: PredictRequest) -> PredictResponse:
    """
    Ejecuta predicción probabilística individual para aparición/presencia de Diabetes Mellitus Tipo 2
    y calcula las contribuciones individuales SHAP para cada variable clínica.
    """
    try:
        response = predictor_service.predict_single(
            dataset_id=payload.dataset_id,
            model_id=payload.model_id,
            raw_features=payload.features,
        )
        # Persistencia en base de datos para trazabilidad y auditoría
        try:
            save_patient_and_prediction(
                patient_id=payload.patient_id or "PX-ANON",
                features=payload.features,
                dataset_id=payload.dataset_id,
                model_id=payload.model_id,
                probability=response.probability,
                prediction=response.prediction,
                risk_stratum=response.risk_stratum,
                shap_top_factor=response.shap_contributions[0].feature if response.shap_contributions else None,
            )
        except Exception:
            pass
        return response
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Error al procesar la predicción: {exc}"
        ) from exc


@app.post("/predict-batch", response_model=BatchPredictResponse, tags=["Inferencia por Lotes CSV"])
async def post_predict_batch(
    file: UploadFile = File(...),
    dataset_id: Literal["nhanes_2017_2018", "pima_indians"] = Form("nhanes_2017_2018"),
    model_id: Literal["xgboost", "random_forest", "tabnet", "logistic_regression"] = Form("xgboost"),
) -> BatchPredictResponse:
    """
    Importa un archivo CSV (de datos clínicos o el archivo sintético rotulado como DEMO)
    y devuelve la estratificación probabilística por fila.
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Por favor adjunte un archivo con extensión .csv")
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        return predictor_service.predict_batch_df(df=df, dataset_id=dataset_id, model_id=model_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"No se pudo procesar el archivo CSV: {exc}"
        ) from exc


@app.get("/sample-csv/{sample_type}", tags=["Inferencia por Lotes CSV"])
def download_sample_csv(
    sample_type: Literal[
        "demo_sintetico",
        "nhanes_test",
        "pima_test",
        "nhanes_completo",
        "pima_completo",
        "nhanes_excel",
        "pima_excel",
        "brfss_completo",
        "brfss_excel",
        "hospitals_completo",
        "hospitals_excel",
        "early_stage_completo",
        "early_stage_excel",
    ]
) -> FileResponse:
    """Descarga plantillas CSV verificadas, datasets completos o el archivo DEMO sintético rotulado."""
    mapping = {
        "demo_sintetico": (
            ROOT_DIR / "data" / "demo" / "DEMO_SINTETICO_NO_CLINICO.csv",
            "DEMO_SINTETICO_NO_CLINICO.csv",
            "text/csv",
        ),
        "nhanes_test": (
            ROOT_DIR / "data" / "processed" / "nhanes_2017_2018_test_sample.csv",
            "nhanes_2017_2018_test_sample.csv",
            "text/csv",
        ),
        "pima_test": (
            ROOT_DIR / "data" / "processed" / "pima_indians_test_sample.csv",
            "pima_indians_test_sample.csv",
            "text/csv",
        ),
        "nhanes_completo": (
            ROOT_DIR / "data" / "processed" / "nhanes_2017_2018_dataset_completo.csv",
            "nhanes_2017_2018_dataset_completo.csv",
            "text/csv",
        ),
        "pima_completo": (
            ROOT_DIR / "data" / "processed" / "pima_indians_dataset_completo.csv",
            "pima_indians_dataset_completo.csv",
            "text/csv",
        ),
        "nhanes_excel": (
            ROOT_DIR / "data" / "processed" / "nhanes_2017_2018_dataset_completo.xlsx",
            "nhanes_2017_2018_dataset_completo.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ),
        "pima_excel": (
            ROOT_DIR / "data" / "processed" / "pima_indians_dataset_completo.xlsx",
            "pima_indians_dataset_completo.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ),
        "brfss_completo": (
            ROOT_DIR / "data" / "processed" / "cdc_brfss_5050_clean.csv",
            "cdc_brfss_5050_clean.csv",
            "text/csv",
        ),
        "brfss_excel": (
            ROOT_DIR / "data" / "processed" / "cdc_brfss_5050_sample_5k.xlsx",
            "cdc_brfss_5050_sample_5k.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ),
        "hospitals_completo": (
            ROOT_DIR / "data" / "processed" / "diabetes_130_hospitals_clean.csv",
            "diabetes_130_hospitals_clean.csv",
            "text/csv",
        ),
        "hospitals_excel": (
            ROOT_DIR / "data" / "processed" / "diabetes_130_hospitals_sample_5k.xlsx",
            "diabetes_130_hospitals_sample_5k.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ),
        "early_stage_completo": (
            ROOT_DIR / "data" / "processed" / "early_stage_diabetes_sylhet_clean.csv",
            "early_stage_diabetes_sylhet_clean.csv",
            "text/csv",
        ),
        "early_stage_excel": (
            ROOT_DIR / "data" / "processed" / "early_stage_diabetes_sylhet_clean.xlsx",
            "early_stage_diabetes_sylhet_clean.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ),
    }
    path, filename, media_type = mapping[sample_type]
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Archivo no encontrado: {filename}")
    return FileResponse(path=path, filename=filename, media_type=media_type)


@app.get("/datasets/info", tags=["Datasets Clínicos Reales"])
def get_datasets_info():
    """Retorna los metadatos y enlaces de descarga de los 5 datasets clínicos reales consolidados."""
    return {
        "status": "ok",
        "description": "Cohortes reales procesadas, auditadas y estandarizadas para investigación biomédica y comparación de modelos.",
        "datasets": [
            {
                "id": "nhanes_2017_2018",
                "name": "CDC NHANES 2017–2018 (Cohorte Primaria de Laboratorio)",
                "category": "Biomarcadores de Laboratorio HPLC",
                "total_rows": 4829,
                "total_columns": 16,
                "positive_diabetes_cases": 772,
                "prevalence": "15.99%",
                "license": "Dominio Público EE. UU. (CC0 1.0)",
                "source": "CDC / National Center for Health Statistics (NCHS)",
                "csv_download_url": "/sample-csv/nhanes_completo",
                "excel_download_url": "/sample-csv/nhanes_excel",
                "file_size_csv": "415 KB",
                "file_size_excel": "330 KB",
            },
            {
                "id": "pima_indians",
                "name": "NIDDK Pima Indians Diabetes Dataset (Benchmark Secundario)",
                "category": "Benchmark Histórico",
                "total_rows": 768,
                "total_columns": 12,
                "positive_diabetes_cases": 268,
                "prevalence": "34.90%",
                "license": "Open Access / CC-BY 4.0",
                "source": "National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)",
                "csv_download_url": "/sample-csv/pima_completo",
                "excel_download_url": "/sample-csv/pima_excel",
                "file_size_csv": "62 KB",
                "file_size_excel": "47 KB",
            },
            {
                "id": "cdc_brfss_2015",
                "name": "CDC BRFSS 2015 (Diabetes Health Indicators)",
                "category": "Salud Poblacional y Estilo de Vida",
                "total_rows": 70692,
                "total_columns": 22,
                "positive_diabetes_cases": 35346,
                "prevalence": "50.00% (Balanceado 50/50)",
                "license": "Dominio Público EE. UU. / CDC",
                "source": "Centers for Disease Control and Prevention (CDC BRFSS)",
                "csv_download_url": "/sample-csv/brfss_completo",
                "excel_download_url": "/sample-csv/brfss_excel",
                "file_size_csv": "6.1 MB",
                "file_size_excel": "240 KB (Muestra 5k)",
            },
            {
                "id": "diabetes_130_hospitals",
                "name": "Diabetes 130-US Hospitals (UCI 296: Pacientes Únicos)",
                "category": "Complicaciones y Reingreso Hospitalario",
                "total_rows": 71515,
                "total_columns": 18,
                "positive_diabetes_cases": 6293,
                "prevalence": "8.80% (Reingreso <30 días)",
                "license": "Open Data / UCI Machine Learning Repository",
                "source": "Strack et al. / Center for Clinical and Translational Research",
                "csv_download_url": "/sample-csv/hospitals_completo",
                "excel_download_url": "/sample-csv/hospitals_excel",
                "file_size_csv": "4.8 MB",
                "file_size_excel": "380 KB (Muestra 5k)",
            },
            {
                "id": "early_stage_sylhet",
                "name": "Early Stage Diabetes Risk Prediction (UCI 529 - Sylhet)",
                "category": "Sintomatología Prodrómica",
                "total_rows": 520,
                "total_columns": 17,
                "positive_diabetes_cases": 320,
                "prevalence": "61.54%",
                "license": "Open Data / UCI Machine Learning Repository",
                "source": "Sylhet Diabetes Hospital / Islam et al.",
                "csv_download_url": "/sample-csv/early_stage_completo",
                "excel_download_url": "/sample-csv/early_stage_excel",
                "file_size_csv": "34 KB",
                "file_size_excel": "28 KB",
            },
        ],
    }


@app.get("/datasets/comparative-benchmark", tags=["Datasets Clínicos Reales"])
def get_datasets_comparative_benchmark():
    """Retorna los resultados científicos del benchmark de modelos (RF, XGBoost, LogReg, Stacking) sobre las 5 cohortes reales."""
    benchmark_file = ROOT_DIR / "models" / "benchmark_cross_dataset_results.json"
    if not benchmark_file.exists():
        raise HTTPException(status_code=404, detail="El archivo de benchmark de datasets aún no ha sido generado.")
    with open(benchmark_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "status": "ok",
        "description": "Evaluación con 5-Fold Stratified Cross-Validation sobre 5 cohortes reales independientes.",
        "benchmarks": data,
    }
