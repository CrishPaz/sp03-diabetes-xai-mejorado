"""Fase 6: inferencia con probabilidad calibrada, conformal, OOD, severidad en dos capas y deriva (PSI/KS)."""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from scipy import stats

from diabetes_ml.config import ARTIFACTS, DISCLAIMER, load
from diabetes_ml.training import prediction_set

LABELS = {"edad": "edad", "imc": "IMC", "glucosa": "glucosa", "hba1c": "HbA1c", "presion_sistolica": "presión sistólica",
          "presion_diastolica": "presión diastólica", "colesterol_total": "colesterol total", "colesterol_hdl": "colesterol HDL",
          "trigliceridos": "triglicéridos", "antecedente_familiar_diabetes": "antecedente familiar de diabetes",
          "antecedente_hipertension": "antecedente de hipertensión", "tratamiento_antihipertensivo": "tratamiento antihipertensivo",
          "embarazos": "embarazos", "pliegue_cutaneo": "pliegue cutáneo", "insulina": "insulina", "pedigri_diabetes": "pedigrí de diabetes"}
UNITS = {"edad": "años", "imc": "kg/m²", "glucosa": "mg/dL", "hba1c": "%", "presion_sistolica": "mmHg", "presion_diastolica": "mmHg",
         "colesterol_total": "mg/dL", "colesterol_hdl": "mg/dL", "trigliceridos": "mg/dL", "insulina": "µU/mL", "pliegue_cutaneo": "mm"}
MODIFIABLE = ("imc", "presion_sistolica", "presion_diastolica", "colesterol_total", "colesterol_hdl", "trigliceridos")


def available() -> List[str]:
    return sorted(p.name for p in ARTIFACTS.glob("*__*") if (p / "bundle.joblib").exists())


@lru_cache(maxsize=8)
def bundle(key: str) -> Dict[str, Any]:
    return joblib.load(ARTIFACTS / key / "bundle.joblib")


def severity_model(p: float) -> str:
    cfg = load("models")["severidad_modelo"]
    return cfg["etiquetas"][int(np.searchsorted(cfg["cortes"], p, side="right"))]


def severity_ada(features: Dict[str, float], cohort: str = "nhanes_2017_2018") -> Optional[Dict[str, Any]]:
    """Capa 2 (no proviene del modelo). Pima mide glucosa a 2 h de OGTT (ADA: <140 / 140-199 / >=200 mg/dL);
    NHANES BIOPRO mide glucosa sérica no necesariamente en ayunas, por lo que se rotula como orientativa."""
    out = {}
    a1c, glu = features.get("hba1c"), features.get("glucosa")
    if a1c is not None:
        out["hba1c"] = "normal" if a1c < 5.7 else "rango de prediabetes" if a1c < 6.5 else "rango compatible con diabetes"
    if glu is not None:
        if cohort == "pima_indians":
            out["glucosa_ogtt_2h"] = "normal" if glu < 140 else "rango de prediabetes" if glu < 200 else "rango compatible con diabetes"
        else:
            out["glucosa_serica_orientativa"] = "normal" if glu < 100 else "rango de prediabetes" if glu < 126 else "rango compatible con diabetes"
    if not out:
        return None
    out["categoria"] = max(out.values(), key=["normal", "rango de prediabetes", "rango compatible con diabetes"].index)
    out["nota"] = "Requiere confirmación clínica con repetición de la prueba (criterios ADA)."
    return out


def mmol_to_mgdl(glucose_mmol: float) -> float:
    return round(glucose_mmol / 0.05551, 1)


def predict(key: str, features: Dict[str, float], model: Optional[str] = None, narrative: bool = True) -> Dict[str, Any]:
    b = bundle(key)
    feats = b["predictores"]
    X = pd.DataFrame([{f: features.get(f, np.nan) for f in feats}])
    missing = [f for f in feats if pd.isna(X.iloc[0][f])]
    model = model or b["modelo_recomendado"]
    if model not in b["final"]:
        model = next(iter(b["final"]))
    probs = {m: float(b["calibradores"][m](est.predict_proba(X)[:, 1])[0]) for m, est in b["final"].items()}
    p = probs[model]
    ref = b["referencia"]
    imp = b["final"][next(iter(b["final"]))].named_steps["imp"]
    z = imp.transform(X.values.astype(float))[0]
    d = float(np.sqrt((z - ref["media"]) @ np.array(ref["icov"]) @ (z - ref["media"])))
    q = b["conformal_q"][model]
    pset = prediction_set(p, q)
    layer1 = severity_model(p)
    layer2 = severity_ada(features, b["cohorte"])
    high_model = layer1 in ("Alto", "Muy alto")
    high_ada = layer2 is not None and layer2["categoria"] != "normal"
    expl = []
    if "xgb" in b["final"]:
        from diabetes_ml.explain import shap_values
        sv = shap_values("xgb", b["final"]["xgb"], X, b["dev_X"].iloc[:100])[0]
        for j in np.argsort(-np.abs(sv))[:5]:
            f = feats[j]
            val = features.get(f)
            txt = f"{LABELS.get(f, f)}" + (f" de {val:g} {UNITS.get(f, '')}".rstrip() if val is not None else " (imputado)")
            expl.append({"variable": f, "shap": round(float(sv[j]), 4),
                         "frase": f"Un valor de {txt} {'aumenta' if sv[j] > 0 else 'reduce'} el riesgo estimado."})
    out = {"cohorte_escenario": key, "modelo": model, "probabilidad": round(p, 4),
           "probabilidades_por_modelo": {m: round(v, 4) for m, v in probs.items()},
           "dispersion_modelos": round(float(np.std(list(probs.values()))), 4),
           "desacuerdo_modelos": bool(max(probs.values()) - min(probs.values()) > 0.25),
           "conjunto_conformal": {"clases": pset, "alpha": load("cv")["conformal"]["alpha"],
                                  "lectura": "ambigua (ambas clases)" if len(pset) == 2 else ("riesgo" if pset == [1] else "sin riesgo") if pset else "vacío"},
           "severidad_modelo": layer1, "categoria_ada": layer2,
           "discordancia_capas": bool(layer2 is not None and high_model != high_ada),
           "fuera_de_distribucion": {"distancia_mahalanobis": round(d, 3), "umbral": round(ref["umbral_ood"], 3), "alerta": d > ref["umbral_ood"]},
           "variables_imputadas": missing, "confiabilidad_reducida": bool(missing),
           "top5_shap": expl, "aviso": DISCLAIMER}
    if narrative:
        from backend.app.llm.deepseek_client import guarded_narrative
        out["narrativa"] = guarded_narrative(out)
    return out


def what_if(key: str, features: Dict[str, float], variable: str, values: List[float]) -> Dict[str, Any]:
    if variable not in MODIFIABLE:
        raise ValueError("Solo variables modificables: " + ", ".join(MODIFIABLE))
    rows = [{"valor": v, "probabilidad": predict(key, {**features, variable: v}, narrative=False)["probabilidad"]} for v in values]
    return {"variable": variable, "curva": rows, "nota": "Sensibilidad del modelo; no es una recomendación clínica."}


def drift(key: str, batch: pd.DataFrame) -> Dict[str, Any]:
    b = bundle(key)
    ref = b["referencia"]
    sample = pd.DataFrame(ref["muestra"], columns=b["predictores"])
    rows = []
    for f in b["predictores"]:
        if f not in batch:
            continue
        x = batch[f].dropna().astype(float)
        edges = np.array(ref["deciles"][f])
        if len(edges) < 3 or x.empty:
            continue
        e = np.histogram(np.clip(sample[f], edges[0], edges[-1]), edges)[0] / len(sample) + 1e-4
        a = np.histogram(np.clip(x, edges[0], edges[-1]), edges)[0] / len(x) + 1e-4
        psi = float(np.sum((a - e) * np.log(a / e)))
        ks = stats.ks_2samp(x, sample[f])
        rows.append({"variable": f, "psi": round(psi, 4), "ks_p": float(f"{ks.pvalue:.3g}"), "alerta": psi > 0.2})
    return {"variables": rows, "alerta_global": any(r["alerta"] for r in rows),
            "regla": "PSI > 0.2 indica deriva relevante frente al entrenamiento."}
