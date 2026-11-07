from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/statistics", tags=["Evaluación Estadística Formal (Fase F9)"])

MODELS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models"


@router.get("/evaluation", summary="Resultados de la batería de pruebas estadísticas (Friedman, Wilcoxon-Holm, McNemar, Bootstrap)")
def get_statistical_evaluation() -> Dict[str, Any]:
    """
    Retorna la evaluación estadística formal de los 5 modelos:
    - Prueba de Friedman (chi-cuadrado sobre pliegues de validación cruzada)
    - Prueba de rangos con signo de Wilcoxon con ajuste de Holm-Bonferroni
    - Prueba pareada de McNemar con corrección por continuidad de Edwards
    - Intervalos de confianza al 95% obtenidos mediante remuestreo Bootstrap (B=1000)
    """
    stats_path = MODELS_DIR / "statistical_evaluation.json"
    if stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Si aún no existe el archivo generado por el pipeline v2, buscar metrics_and_metadata.json
    meta_path = MODELS_DIR / "metrics_and_metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            if "statistical_evaluation" in meta:
                return meta["statistical_evaluation"]

    # Retorno provisional estructurado conforme al protocolo del docente
    return {
        "status": "pending_or_computing",
        "message": "La batería de pruebas estadísticas se está calculando o aún no se ha consolidado en disco.",
        "models_evaluated": [
            "Random Forest",
            "XGBoost",
            "TabNet",
            "Stacking Calibrado (Híbrido 1)",
            "Weibull Hazard (Híbrido 2)",
        ],
        "battery_components": [
            "Prueba no paramétrica de Friedman para k=5 modelos dependientes",
            "Pruebas pareadas post-hoc de Wilcoxon con corrección estricta de Holm",
            "Prueba de McNemar para concordancia de clasificaciones erróneas",
            "Intervalos de confianza Bootstrap al 95% con B=1000 réplicas",
        ],
    }


@router.get("/cross-cohorts-benchmark", summary="Benchmark comparativo de algoritmos en 5 cohortes reales")
def get_cross_cohorts_benchmark() -> Dict[str, Any]:
    """
    Retorna el benchmark comparativo de algoritmos (Random Forest, XGBoost, Regresión Logística, Stacking Híbrido)
    evaluados mediante 5-Fold Stratified Cross-Validation en 5 cohortes reales:
    - CDC NHANES 2017-2018 (Biomarcadores de Laboratorio)
    - NIDDK Pima Indians (Benchmark Histórico)
    - CDC BRFSS 2015 (Salud Poblacional y Estilo de Vida, N=70,692)
    - Diabetes 130-US Hospitals (Complicaciones y Reingreso UCI 296, N=71,515)
    - Early Stage Diabetes Sylhet (Sintomatología Prodrómica, N=520)
    """
    bench_path = MODELS_DIR / "benchmark_cross_dataset_results.json"
    if not bench_path.exists():
        raise HTTPException(status_code=404, detail="Benchmark aún no disponible.")
    with open(bench_path, "r", encoding="utf-8") as f:
        return json.load(f)

