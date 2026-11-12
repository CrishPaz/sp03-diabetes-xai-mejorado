from __future__ import annotations

from pathlib import Path
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)
ROOT_DIR = Path(__file__).resolve().parent.parent


def test_health_endpoint() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["models_loaded"] is True
    assert "nhanes_2017_2018" in data["available_cohorts"]
    assert "pima_indians" in data["available_cohorts"]
    # Los 5 modelos exigidos por la arquitectura recomendada
    assert "xgboost" in data["available_models"]
    assert "random_forest" in data["available_models"]
    assert "tabnet" in data["available_models"]
    assert "stacking_ensemble" in data["available_models"]
    assert "hybrid_survival" not in data["available_models"]  # H2: supervivencia deshabilitada
    assert data["version"] == "3.0.0" and "llm" in data and data["hash_datasets"]


def test_model_info_endpoint() -> None:
    resp = client.get("/model-info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["project_code"] == "SP-03"
    assert "nhanes_2017_2018" in data["cohorts"]
    assert "pima_indians" in data["cohorts"]
    assert len(data.get("traceability_matrix", [])) >= 8

    nhanes = data["cohorts"]["nhanes_2017_2018"]
    assert nhanes["split_summary"]["patient_overlap_count"] == 0
    for mid in ["xgboost", "random_forest", "tabnet", "stacking_hybrid", "survival_risk_hybrid"]:
        if mid in nhanes["models"]:
            metrics = nhanes["models"][mid]["test_metrics"]
            assert 0.80 <= metrics["roc_auc"] <= 1.0
            assert "confusion_matrix" in metrics
            assert "calibration_curve" in metrics


def test_predict_nhanes_all_models() -> None:
    payload = {
        "dataset_id": "nhanes_2017_2018",
        "model_id": "xgboost",
        "features": {
            "edad": 58.0,
            "imc": 33.4,
            "glucosa": 146.0,
            "hba1c": 6.6,
            "presion_sistolica": 142.0,
            "presion_diastolica": 86.0,
            "colesterol_total": 218.0,
            "colesterol_hdl": 41.0,
            "trigliceridos": 210.0,
            "antecedente_familiar_diabetes": 1,
            "antecedente_hipertension": 1,
            "tratamiento_antihipertensivo": 1,
        },
    }
    for model_id in ["xgboost", "random_forest", "tabnet", "stacking_ensemble"]:
        payload["model_id"] = model_id
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["dataset_id"] == "nhanes_2017_2018"
        assert body["model_id"] == model_id
        assert 0.0 < body["probability"] < 1.0
        assert len(body["shap_contributions"]) == 12
        assert len(body["all_models_comparison"]) >= 4
        assert body["survival_projections"] is None
    payload["model_id"] = "hybrid_survival"
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_pima_cohort() -> None:
    payload = {
        "dataset_id": "pima_indians",
        "model_id": "random_forest",
        "features": {
            "embarazos": 4,
            "glucosa": 158,
            "presion_diastolica": 78,
            "pliegue_cutaneo": 32,
            "insulina": 165,
            "imc": 34.8,
            "pedigri_diabetes": 0.72,
            "edad": 41,
        },
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["dataset_id"] == "pima_indians"
    assert len(body["shap_contributions"]) == 8


def test_predict_validation_error() -> None:
    bad_payload = {
        "dataset_id": "nhanes_2017_2018",
        "model_id": "xgboost",
        "features": {
            "edad": 50.0,
            "imc": 28.0,
            "glucosa": 100.0,
            "hba1c": 5.5,
            "presion_sistolica": 110.0,
            "presion_diastolica": 120.0,  # Diastólica >= Sistólica
            "colesterol_total": 190.0,
            "colesterol_hdl": 50.0,
            "trigliceridos": 130.0,
            "antecedente_familiar_diabetes": 0,
            "antecedente_hipertension": 0,
            "tratamiento_antihipertensivo": 0,
        },
    }
    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422


def test_batch_csv_demo_detection() -> None:
    demo_csv = ROOT_DIR / "data" / "demo" / "DEMO_SINTETICO_NO_CLINICO.csv"
    with open(demo_csv, "rb") as f:
        resp = client.post(
            "/predict-batch",
            files={"file": ("DEMO_SINTETICO_NO_CLINICO.csv", f, "text/csv")},
            data={"dataset_id": "nhanes_2017_2018", "model_id": "xgboost"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_synthetic_demo"] is True
    assert body["demo_warning"] is not None
    assert body["total_rows_processed"] == 25


def test_rag_clinical_agent() -> None:
    payload = {
        "query": "¿Cuáles son los criterios de diagnóstico de diabetes según ADA 2024?",
        "max_sources": 3,
    }
    resp = client.post("/rag/query", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "ADA" in body["synthesized_answer"] or "criterios" in body["synthesized_answer"].lower()
    assert len(body["citations"]) >= 1
    assert len(body["clinical_guardrails"]) >= 1


def test_n8n_workflows_list_and_trigger() -> None:
    resp = client.get("/n8n/workflows")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_workflows"] == 6

    # Disparar WF-04
    trig_resp = client.post(
        "/n8n/workflows/WF-04/trigger",
        json={"patient_id": "PX-TEST-99", "riesgo_pct": 87.5},
    )
    assert trig_resp.status_code == 200
    assert trig_resp.json()["status"] == "success"

    # Verificar alertas generadas
    alerts_resp = client.get("/n8n/alerts")
    assert alerts_resp.status_code == 200
    assert len(alerts_resp.json()["alerts"]) >= 1


def test_statistical_evaluation_endpoint() -> None:
    resp = client.get("/statistics/evaluation")
    assert resp.status_code == 200
    data = resp.json()
    assert "nhanes_2017_2018" in data or "friedman_test" in data or "battery_components" in data


def test_report_generation_endpoints() -> None:
    features = {
        "edad": 52.0,
        "imc": 29.5,
        "glucosa": 130.0,
        "hba1c": 6.8,
        "presion_sistolica": 135.0,
        "presion_diastolica": 82.0,
        "colesterol_total": 210.0,
        "colesterol_hdl": 44.0,
        "trigliceridos": 180.0,
        "antecedente_familiar_diabetes": 1,
        "antecedente_hipertension": 1,
        "tratamiento_antihipertensivo": 0,
    }
    for fmt in ["pdf", "docx", "xlsx"]:
        payload = {
            "format": fmt,
            "patient_name": "Paciente de Prueba Automática",
            "patient_id": f"PX-TEST-{fmt.upper()}",
            "features": features,
        }
        resp = client.post("/reports/generate", json=payload)
        assert resp.status_code == 200
        assert len(resp.content) > 100


def test_real_datasets_download() -> None:
    # 1. Metadatos de datasets reales (5 cohortes preparadas)
    info_resp = client.get("/datasets/info")
    assert info_resp.status_code == 200
    info_data = info_resp.json()
    assert len(info_data["datasets"]) >= 5
    assert info_data["datasets"][0]["total_rows"] == 4829

    # 2. Descarga del dataset completo de NHANES
    nhanes_csv = client.get("/sample-csv/nhanes_completo")
    assert nhanes_csv.status_code == 200
    assert len(nhanes_csv.content) > 100000  # Más de 100 KB reales

    # 3. Descarga del dataset completo de Pima
    pima_csv = client.get("/sample-csv/pima_completo")
    assert pima_csv.status_code == 200
    assert len(pima_csv.content) > 30000

    # 4. Descarga de BRFSS y Hospitals
    brfss_csv = client.get("/sample-csv/brfss_completo")
    assert brfss_csv.status_code == 200
    assert len(brfss_csv.content) > 1000000  # Más de 1 MB real

    hosp_csv = client.get("/sample-csv/hospitals_completo")
    assert hosp_csv.status_code == 200
    assert len(hosp_csv.content) > 1000000

    early_csv = client.get("/sample-csv/early_stage_completo")
    assert early_csv.status_code == 200
    assert len(early_csv.content) > 10000

    # 5. Benchmark comparativo de algoritmos sobre las 5 cohortes
    bench_resp = client.get("/datasets/comparative-benchmark")
    assert bench_resp.status_code == 200
    bench_data = bench_resp.json()
    assert "benchmarks" in bench_data
    assert len(bench_data["benchmarks"]) >= 5
