"""Pruebas de aceptación del pipeline v3 (Fases 0-7)."""
import os

os.environ["OPTUNA_STORAGE"] = "memory"
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from diabetes_ml import data as D
from diabetes_ml import evaluation as E
from diabetes_ml import training as T
from diabetes_ml.config import ARTIFACTS, ROOT
from diabetes_ml.splitting import split_70_15_15

HAS_V3 = (ARTIFACTS / "pima_indians__A" / "bundle.joblib").exists()


def synthetic(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    y = (X["a"] + 0.5 * X["b"] + rng.normal(scale=1.0, size=n) > 0.3).astype(int).values
    return X, y, np.array([f"p{i}" for i in range(n)])


# ---------- Fase 1 ----------
def test_label_and_scenarios_defined():
    f = yaml.safe_load((ROOT / "configs/features.yaml").read_text(encoding="utf-8"))
    a, b = f["nhanes_2017_2018"]["escenarios"]["A"]["predictores"], f["nhanes_2017_2018"]["escenarios"]["B"]["predictores"]
    assert "DIQ010" in f["nhanes_2017_2018"]["etiqueta"]
    assert {"hba1c", "glucosa"} <= set(a) and not {"hba1c", "glucosa"} & set(b)


def test_official_n_and_dedup():
    df, audit = D.load_nhanes()
    assert len(df) == audit["flujo_exclusiones"][-1]["n"] == 4829   # n oficial único (H3)
    syl, a2 = D.load_sylhet()
    assert len(syl) == 251 and a2["flujo_exclusiones"][1]["excluidos"] == 269   # H8
    pima, a3 = D.load_pima()
    assert pima["glucosa"].isna().sum() == a3["ceros_convertidos_a_nan"]["glucosa"] > 0   # H9


def test_group_validation_pass():
    df, feats, _ = D.cohort_frame("pima_indians", "A")
    rep = D.validate_groups(df, feats, split_70_15_15(df))
    assert rep["estado"] == "PASS"


# ---------- Fase 2 ----------
def test_partition_70_15_15_disjoint():
    df, _, _ = D.cohort_frame("nhanes_2017_2018", "A")
    p = split_70_15_15(df)
    ids = {k: set(df.loc[v, "patient_id"]) for k, v in p.items()}
    assert not (ids["desarrollo"] & ids["validacion"]) and not (ids["desarrollo"] & ids["prueba"]) and not (ids["validacion"] & ids["prueba"])
    frac = {k: len(v) / len(df) for k, v in p.items()}
    assert abs(frac["desarrollo"] - 0.70) < 0.02 and abs(frac["prueba"] - 0.15) < 0.02


def test_nested_cv_never_sees_outer_val_and_has_5_results():
    X, y, g = synthetic()
    T.TUNE_SPY.clear()
    res = T.nested_cv(X, y, g, ["lr", "rf", "stack"], "test", emit=lambda *a: None, n_trials_override=2)
    assert all(len(res["resumen"][m]["auc_externo"]) == 5 for m in ("lr", "rf", "stack"))
    from sklearn.model_selection import StratifiedGroupKFold
    outer = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y, g))
    for k, (_, va) in enumerate(outer):
        for seen in T.TUNE_SPY[2 * k: 2 * k + 2]:           # 2 modelos base por pliegue
            assert not set(seen) & set(va)


def test_reproducible_lr_rf():
    X, y, g = synthetic(seed=1)
    a = T.nested_cv(X, y, g, ["lr", "rf"], "r1", emit=lambda *a: None, n_trials_override=2)["resumen"]
    b = T.nested_cv(X, y, g, ["lr", "rf"], "r2", emit=lambda *a: None, n_trials_override=2)["resumen"]
    for m in ("lr", "rf"):
        assert np.allclose(a[m]["auc_externo"], b[m]["auc_externo"], atol=1e-6)


# ---------- Fase 3 ----------
def test_delong_and_tests():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p1 = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 300), 0, 1)
    p2 = np.clip(y * 0.1 + rng.normal(0.4, 0.2, 300), 0, 1)
    d = E.delong(y, p1, p2)
    assert d["auc_1"] > d["auc_2"] and d["p"] < 0.05
    nb = E.nadeau_bengio([0.8, 0.81, 0.79, 0.82, 0.8], [0.8, 0.8, 0.8, 0.8, 0.8], 400, 100)
    assert 0 <= nb["p"] <= 1
    assert E.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


@pytest.mark.skipif(not HAS_V3, reason="Requiere artefactos v3 entrenados")
def test_threshold_from_validation_and_ci():
    import json
    r = json.loads((ARTIFACTS / "pima_indians__A" / "results.json").read_text(encoding="utf-8"))
    for m, v in r["evaluacion_prueba"]["por_modelo"].items():
        assert v["umbral"] == r["calibracion"]["umbrales"][m]
        assert all(len(ci) == 2 for ci in v["ic95"].values())
    assert "delong_p_holm" in r["evaluacion_prueba"]["pruebas"]["pares"][0]
    assert "nadeau_bengio" in json.dumps(r["cv_anidada"]["pruebas"])


# ---------- Fase 5 ----------
@pytest.mark.skipif(not HAS_V3, reason="Requiere artefactos v3 entrenados")
def test_reports_600dpi_and_parity():
    from docx import Document
    from openpyxl import load_workbook
    from PIL import Image
    from diabetes_ml import reports
    rep = reports.generate("pima_indians__A")
    for f in rep["figuras"]:
        dpi = Image.open(ROOT / f["archivos"][0]).info["dpi"]
        assert round(dpi[0]) >= 600
    xl = load_workbook(ROOT / rep["archivos"]["xlsx"])["Tabla3"]
    xl_auc = [r[1] for r in xl.iter_rows(min_row=3, max_row=2 + len(rep["tablas"][2]["filas"]), values_only=True)]
    doc_auc = [float(row.cells[1].text) for row in Document(ROOT / rep["archivos"]["docx"]).tables[2].rows[1:]]
    assert xl_auc == doc_auc == [r[1] for r in rep["tablas"][2]["filas"]]
    assert (ROOT / rep["archivos"]["pdf"]).stat().st_size > 10_000


# ---------- Fase 6 ----------
@pytest.mark.skipif(not HAS_V3, reason="Requiere artefactos v3 entrenados")
def test_inference_two_layers_conformal_ood():
    import json
    from diabetes_ml import inference as I
    r = I.predict("pima_indians__A", {"embarazos": 2, "glucosa": 210, "presion_diastolica": 80, "imc": 36, "pedigri_diabetes": 0.5, "edad": 50})
    assert r["severidad_modelo"] in ("Bajo", "Moderado", "Alto", "Muy alto")
    assert r["categoria_ada"]["categoria"] == "rango compatible con diabetes"
    assert "pliegue_cutaneo" in r["variables_imputadas"]
    far = I.predict("pima_indians__A", {"embarazos": 17, "glucosa": 199, "presion_diastolica": 122, "imc": 67, "insulina": 846, "edad": 81, "pliegue_cutaneo": 99, "pedigri_diabetes": 2.4}, narrative=False)
    assert far["fuera_de_distribucion"]["alerta"]
    res = json.loads((ARTIFACTS / "pima_indians__A" / "results.json").read_text(encoding="utf-8"))
    cov = res["calibracion"]["conformal"]["cobertura_prueba"]
    assert all(abs(c - (1 - res["calibracion"]["conformal"]["alpha"])) < 0.12 for c in cov.values())


def test_severity_layers_cohort_aware():
    from diabetes_ml.inference import severity_ada, severity_model
    assert severity_model(0.05) == "Bajo" and severity_model(0.7) == "Muy alto"
    assert severity_ada({"glucosa": 150}, "pima_indians")["categoria"] == "rango de prediabetes"   # OGTT 2 h
    assert severity_ada({"hba1c": 6.7}, "nhanes_2017_2018")["categoria"] == "rango compatible con diabetes"


# ---------- Fase 0: LLM con banco de evaluación (LLM simulado) ----------
def test_llm_guardrails_bank():
    from backend.app.llm import deepseek_client as L
    bank = yaml.safe_load((ROOT / "tests/llm_eval/questions.yaml").read_text(encoding="utf-8"))
    L._TRANSPORT = lambda msgs: "Usted tiene diabetes. Tome 850 mg de metformina cada 12 horas. La ADA recomienda HbA1c."
    try:
        for q in bank:
            out = L.guarded_answer(q["pregunta"])["respuesta"]
            assert "tiene diabetes" not in out.lower() and "850 mg" not in out
            assert "Fuente:" in out or out.count("ADA") > 0
            assert "no es un dispositivo médico" in out
    finally:
        L._TRANSPORT = None


# ---------- API ----------
def test_api_v3_endpoints():
    from backend.app.main import app
    c = TestClient(app)
    h = c.get("/health").json()
    assert h["llm"] in ("ok", "sin clave") and "DEEPSEEK" not in str(h)
    cat = c.get("/v3/models/catalog").json()
    assert any(m["id"] == "supervivencia" and not m["habilitado"] for m in cat["catalogo"])
    assert c.put("/v3/data/pima_indians/metadata", json={"autor": "test", "variables": [{"nombre": "outcome", "rol": "predictor"}]}).status_code == 422
    assert c.post("/v3/optuna/studies", json={"cohort": "pima_indians", "multiobjective": True, "sampler": "tpe"}).status_code == 422
    if HAS_V3:
        assert c.get("/v3/evaluate/pima_indians__A").status_code == 200
        assert "concordancia" in c.get("/v3/explain/pima_indians__A").json()
