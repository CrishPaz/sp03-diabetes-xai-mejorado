"""API v3 (Fases 1-7): datos, entrenamiento por jobs, evaluación, explicabilidad, inferencia, reportes y Optuna."""
from __future__ import annotations

import asyncio
import io
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from diabetes_ml import data as D
from diabetes_ml import jobs
from diabetes_ml.config import ARTIFACTS, DISCLAIMER, ROOT, load

router = APIRouter(prefix="/v3", tags=["Pipeline v3 (Fases 0-7)"])
Cohort = Literal["nhanes_2017_2018", "pima_indians"]
Scenario = Literal["A", "B"]


def _results(key: str) -> Dict[str, Any]:
    p = ARTIFACTS / key / "results.json"
    if not p.exists():
        raise HTTPException(404, f"Sin resultados para {key}. Lance un entrenamiento en /v3/train/jobs.")
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------- Fase 1: datos ----------------
@router.get("/data/registry")
def data_registry():
    return {"cohortes": D.registry(), "aviso": DISCLAIMER}


@router.get("/data/{cohort}/eda")
def data_eda(cohort: Cohort, scenario: Scenario = "A"):
    return D.eda(cohort, scenario)


@router.post("/data/{cohort}/validate-groups")
def data_validate(cohort: Cohort, scenario: Scenario = "A"):
    from diabetes_ml.splitting import partition_table, split_70_15_15
    df, feats, _ = D.cohort_frame(cohort, scenario)
    parts = split_70_15_15(df)
    return {**D.validate_groups(df, feats, parts), "tabla_particion": partition_table(df, parts)}


class VariableMeta(BaseModel):
    nombre: str
    etiqueta: str = ""
    codigo_origen: str = ""
    unidad: str = ""
    tipo: Literal["numerica", "binaria", "categorica", "texto"] = "numerica"
    rango_min: Optional[float] = None
    rango_max: Optional[float] = None
    codigos_faltante: List[str] = []
    imputacion: Literal["mediana", "ninguna", "moda"] = "mediana"
    rol: Literal["predictor", "objetivo", "grupo", "sensible_solo_auditoria", "excluida"] = "predictor"
    justificacion: str = ""


class MetadataIn(BaseModel):
    autor: str = Field(..., min_length=2)
    variables: List[VariableMeta]


def _meta_versions(cohort: str):
    return sorted((ROOT / "data/metadata").glob(f"{cohort}_v*.json"), key=lambda p: int(p.stem.split("_v")[-1]))


@router.get("/data/{cohort}/metadata")
def get_metadata(cohort: Cohort):
    vs = _meta_versions(cohort)
    if vs:
        return json.loads(vs[-1].read_text(encoding="utf-8"))
    from diabetes_ml.inference import LABELS, UNITS
    feats = set(load("features")[cohort]["escenarios"]["A"]["predictores"])
    rows = [VariableMeta(nombre=f, etiqueta=LABELS.get(f, f), unidad=UNITS.get(f, "")).model_dump() for f in sorted(feats)]
    rows += [VariableMeta(nombre="outcome", rol="objetivo", tipo="binaria").model_dump(),
             VariableMeta(nombre="patient_id", rol="grupo", tipo="texto", imputacion="ninguna").model_dump(),
             VariableMeta(nombre="sexo", rol="sensible_solo_auditoria", tipo="categorica", imputacion="ninguna").model_dump()]
    return {"version": 0, "autor": "sistema", "variables": rows}


@router.put("/data/{cohort}/metadata")
def put_metadata(cohort: Cohort, body: MetadataIn):
    errors = []
    for v in body.variables:
        if v.nombre in ("patient_id", "SEQN", "patient_nbr") and v.rol == "predictor":
            errors.append(f"{v.nombre}: el identificador de grupo no puede ser predictor")
        if v.nombre == "outcome" and v.rol == "predictor":
            errors.append("outcome: la etiqueta no puede ser predictor")
        if v.nombre in ("sexo", "RIAGENDR") and v.rol == "predictor" and len(v.justificacion) < 20:
            errors.append(f"{v.nombre}: usar sexo como predictor requiere justificación escrita (>= 20 caracteres)")
    if errors:
        raise HTTPException(422, {"errores": errors})
    vs = _meta_versions(cohort)
    prev = json.loads(vs[-1].read_text(encoding="utf-8")) if vs else get_metadata(cohort)
    old = {v["nombre"]: v for v in prev["variables"]}
    diff = [{"variable": v.nombre, "cambios": {k: [old.get(v.nombre, {}).get(k), val] for k, val in v.model_dump().items() if old.get(v.nombre, {}).get(k) != val}}
            for v in body.variables if old.get(v.nombre) != v.model_dump()]
    n = (int(vs[-1].stem.split("_v")[-1]) if vs else 0) + 1
    rec = {"version": n, "autor": body.autor, "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "diff": diff, "variables": [v.model_dump() for v in body.variables]}
    (ROOT / "data/metadata").mkdir(parents=True, exist_ok=True)
    (ROOT / f"data/metadata/{cohort}_v{n}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    try:
        from backend.app.database import log_audit_event
        log_audit_event("metadata_update", {"cohorte": cohort, "version": n, "autor": body.autor, "cambios": len(diff)}, endpoint=f"/v3/data/{cohort}/metadata")
    except Exception:
        pass
    return rec


@router.post("/data/upload")
async def data_upload(cohort: Cohort = Form(...), file: UploadFile = File(...)):
    raw = await file.read()
    df = pd.read_excel(io.BytesIO(raw)) if file.filename.lower().endswith((".xlsx", ".xls")) else pd.read_csv(io.BytesIO(raw))
    meta = {v["nombre"]: v for v in get_metadata(cohort)["variables"]}
    feats = load("features")[cohort]["escenarios"]["A"]["predictores"]
    errors = [f"Falta la columna {f}" for f in feats if f not in df.columns]
    for f in feats:
        if f in df.columns:
            m = meta.get(f, {})
            col = pd.to_numeric(df[f], errors="coerce")
            if col.isna().sum() > df[f].isna().sum():
                errors.append(f"{f}: valores no numéricos")
            lo, hi = m.get("rango_min"), m.get("rango_max")
            if lo is not None and (col < lo).any() or hi is not None and (col > hi).any():
                errors.append(f"{f}: valores fuera del rango fisiológico [{lo}, {hi}]")
    if errors:
        raise HTTPException(422, {"errores": errors, "filas": len(df)})
    return {"aceptado": True, "filas": len(df), "columnas": list(df.columns)}


# ---------------- Fase 2: entrenamiento ----------------
class TrainJobIn(BaseModel):
    cohort: Cohort
    scenario: Scenario = "A"
    models: List[Literal["lr", "rf", "xgb", "tabnet", "stack"]] = ["lr", "rf", "xgb", "tabnet", "stack"]
    n_trials: Optional[int] = Field(None, ge=0, le=500)


@router.get("/models/catalog")
def model_catalog():
    m = load("models")
    return {"catalogo": [{"id": k, "nombre": m["nombres"][k], "habilitado": True} for k in m["catalogo"]] +
            [{"id": "supervivencia", "nombre": "Híbrido de supervivencia", "habilitado": False, "motivo": m["supervivencia"]["motivo"]}],
            "espacios_busqueda": load("search_spaces"), "cv": load("cv")}


@router.post("/train/jobs")
def train_start(body: TrainJobIn):
    args = ["--cohort", body.cohort, "--scenario", body.scenario, "--models", ",".join(body.models)]
    if body.n_trials is not None:
        args += ["--n-trials", str(body.n_trials)]
    try:
        return jobs.start("train", args)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))


@router.get("/train/jobs")
def train_list():
    return jobs.list_jobs()


@router.get("/train/jobs/{job_id}")
def train_get(job_id: str):
    j = jobs.get(job_id)
    if not j:
        raise HTTPException(404, "Trabajo no encontrado")
    return j


@router.post("/train/jobs/{job_id}/cancel")
def train_cancel(job_id: str):
    return jobs.cancel(job_id)


@router.get("/train/jobs/{job_id}/events")
def train_events(job_id: str, offset: int = 0):
    return {**jobs.events(job_id, offset), "estado": (jobs.get(job_id) or {}).get("estado")}


@router.get("/train/jobs/{job_id}/stream")
async def train_stream(job_id: str, offset: int = 0):
    """SSE: eventos tipados (fold_started, trial_completed, epoch, fold_completed, resource, completed)."""
    async def gen():
        off = offset
        while True:
            chunk = jobs.events(job_id, off)
            off = chunk["offset"]
            for ev in chunk["eventos"]:
                yield f"data: {json.dumps(ev, ensure_ascii=False, default=float)}\n\n"
            state = (jobs.get(job_id) or {}).get("estado")
            if state in ("completed", "failed", "cancelled", None):
                yield f"data: {json.dumps({'event': 'job_' + str(state)})}\n\n"
                return
            await asyncio.sleep(1.5)
    return StreamingResponse(gen(), media_type="text/event-stream")


# ---------------- Fase 3: evaluación ----------------
@router.get("/results")
def results_available():
    from diabetes_ml.inference import available
    out = []
    for k in available():
        r = _results(k)
        out.append({"clave": k, "cohorte": r["cohorte"], "escenario": r["escenario"], "descripcion": r["auditoria"]["descripcion_escenario"],
                    "n": r["auditoria"]["n_oficial"], "modelo_recomendado": r["modelo_recomendado"],
                    "auc_prueba": {m: v["metricas"]["roc_auc"] for m, v in r["evaluacion_prueba"]["por_modelo"].items()}})
    return out


@router.get("/evaluate/{key}")
def evaluate(key: str):
    r = _results(key)
    return {k: r[k] for k in ("cohorte", "escenario", "auditoria", "particion", "cv_anidada", "calibracion", "evaluacion_prueba",
                              "modelo_final", "modelo_recomendado", "supervivencia", "mlflow_run", "segundos", "aviso")}


# ---------------- Fase 4: explicabilidad ----------------
@router.get("/explain/{key}")
def explain(key: str):
    p = ARTIFACTS / key / "explain.json"
    if not p.exists():
        raise HTTPException(404, "Explicaciones no generadas")
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------- Fase 6: inferencia ----------------
class PredictIn(BaseModel):
    key: str = Field(..., examples=["nhanes_2017_2018__B"])
    features: Dict[str, Optional[float]]
    model: Optional[str] = None
    glucosa_mmol: bool = False


@router.post("/predict")
def predict(body: PredictIn):
    from diabetes_ml import inference as I
    feats = {k: v for k, v in body.features.items() if v is not None}
    if body.glucosa_mmol and "glucosa" in feats:
        feats["glucosa"] = I.mmol_to_mgdl(feats["glucosa"])
    if body.key not in I.available():
        raise HTTPException(404, f"Modelo {body.key} no entrenado")
    return I.predict(body.key, feats, body.model)


class WhatIfIn(BaseModel):
    key: str
    features: Dict[str, float]
    variable: str
    values: List[float]


@router.post("/what-if")
def what_if(body: WhatIfIn):
    from diabetes_ml import inference as I
    try:
        return I.what_if(body.key, body.features, body.variable, body.values)
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@router.post("/drift")
async def drift(key: str = Form(...), file: UploadFile = File(...)):
    from diabetes_ml import inference as I
    df = pd.read_csv(io.BytesIO(await file.read()))
    return I.drift(key, df)


# ---------------- Fase 5: reportes ----------------
class ReportIn(BaseModel):
    key: str
    tipo: Literal["cientifico", "gerencial"] = "cientifico"
    formatos: List[Literal["pdf", "docx", "xlsx"]] = ["pdf", "docx", "xlsx"]


@router.post("/reports/generate")
def reports_generate(body: ReportIn):
    from diabetes_ml import reports
    _results(body.key)
    return reports.generate(body.key, tuple(body.formatos), body.tipo)


@router.get("/reports/{key}/download")
def reports_download(key: str, format: Literal["pdf", "docx", "xlsx"] = "pdf", tipo: str = "cientifico"):
    p = ROOT / load("reports")["salida"] / key / f"reporte_{tipo}.{format}"
    if not p.exists():
        raise HTTPException(404, "Reporte no generado")
    return FileResponse(p, filename=p.name)


@router.get("/figures/{key}/{name}")
def figure(key: str, name: str):
    p = ROOT / load("reports")["salida"] / key / f"{name}.png"
    if not p.exists() or ".." in name:
        raise HTTPException(404, "Figura no encontrada")
    return FileResponse(p, media_type="image/png")


# ---------------- Fase 7: Optuna ----------------
class StudyIn(BaseModel):
    cohort: Cohort
    scenario: Scenario = "A"
    model: Literal["lr", "rf", "xgb", "tabnet"] = "xgb"
    n_trials: int = Field(20, ge=1, le=500)
    sampler: Literal["tpe", "random", "cmaes", "nsga2"] = "tpe"
    pruner: Literal["median", "hyperband", "none"] = "median"
    multiobjective: bool = False
    space: Optional[Dict[str, Dict[str, Any]]] = None


@router.get("/optuna/studies")
def optuna_studies():
    from diabetes_ml.optuna_ui import list_studies
    return list_studies()


@router.get("/optuna/studies/{name}")
def optuna_study(name: str):
    from diabetes_ml.optuna_ui import study_detail
    try:
        return study_detail(name)
    except KeyError:
        raise HTTPException(404, "Estudio no encontrado")


@router.post("/optuna/studies")
def optuna_launch(body: StudyIn):
    if body.sampler == "cmaes" and body.space and any(s["type"] == "cat" for s in body.space.values()):
        raise HTTPException(422, "CMA-ES solo admite parámetros continuos o enteros")
    if body.multiobjective and body.sampler != "nsga2":
        raise HTTPException(422, "El modo multiobjetivo requiere NSGA-II")
    if body.space:
        for n, s in body.space.items():
            if s.get("type") in ("int", "float") and s["low"] >= s["high"]:
                raise HTTPException(422, f"{n}: low debe ser menor que high")
    args = ["--explore", "--cohort", body.cohort, "--scenario", body.scenario, "--model", body.model, "--n-trials", str(body.n_trials),
            "--sampler", body.sampler, "--pruner", body.pruner]
    if body.multiobjective:
        args.append("--multiobjective")
    if body.space:
        args += ["--space", json.dumps(body.space)]
    try:
        j = jobs.start("optuna", args)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
    return {**j, "estudio": f"explor__{body.cohort}__{body.scenario}__{body.model}",
            "advertencia": "Estudio exploratorio: los resultados oficiales salen de la CV anidada."}


@router.post("/optuna/studies/{name}/export")
def optuna_export(name: str):
    from diabetes_ml.optuna_ui import export_best
    return {"yaml": export_best(name)}


# ---------------- Fase 0: LLM ----------------
class AskIn(BaseModel):
    pregunta: str = Field(..., min_length=3)


@router.post("/llm/ask")
def llm_ask(body: AskIn):
    from backend.app.llm.deepseek_client import guarded_answer
    return guarded_answer(body.pregunta)
