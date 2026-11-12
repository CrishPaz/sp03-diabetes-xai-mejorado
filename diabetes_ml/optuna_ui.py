"""Fase 7: lectura del storage de Optuna y datos para visualizaciones (sin plotly: JSON para React)."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import optuna
import yaml

from diabetes_ml.config import CONFIG_DIR, interp
from diabetes_ml.training import storage_url


def list_studies() -> List[Dict[str, Any]]:
    st = storage_url()
    if not st:
        return []
    out = []
    for s in optuna.get_all_study_summaries(storage=st):
        best = None
        try:
            best = s.best_trial.value if s.best_trial else None
        except Exception:
            pass
        out.append({"nombre": s.study_name, "n_trials": s.n_trials, "mejor_valor": best,
                    "exploratorio": s.study_name.startswith("explor__"), "inicio": str(s.datetime_start)})
    return sorted(out, key=lambda r: r["nombre"])


def study_detail(name: str) -> Dict[str, Any]:
    study = optuna.load_study(study_name=name, storage=storage_url())
    trials = study.get_trials(deepcopy=False)
    multi = len(study.directions) > 1
    rows = [{"numero": t.number, "estado": t.state.name, "valor": (t.values if multi else t.value), "params": t.params,
             "duracion_s": t.duration.total_seconds() if t.duration else None, "intermedios": t.intermediate_values} for t in trials]
    done = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    counts = {s: sum(t.state.name == s for t in trials) for s in ("COMPLETE", "PRUNED", "FAIL", "RUNNING")}
    out = {"nombre": name, "multiobjetivo": multi, "trials": rows, "conteos": counts}
    if not done:
        return out
    if multi:
        out["pareto"] = [{"numero": t.number, "valores": t.values} for t in study.best_trials]
        return out
    vals = [t.value for t in done]
    out["historial"] = {"numero": [t.number for t in done], "valor": vals, "mejor_acumulado": np.maximum.accumulate(vals).tolist()}
    out["edf"] = {"valor": sorted(vals), "fraccion": (np.arange(1, len(vals) + 1) / len(vals)).tolist()}
    imp = {}
    if len(done) >= 3 and done[0].params:
        try:
            imp = {k: round(float(v), 4) for k, v in optuna.importance.get_param_importances(study).items()}
        except Exception:
            imp = {}
    out["importancia"] = imp
    out["coordenadas_paralelas"] = [{"valor": t.value, **t.params} for t in done]
    if len(imp) >= 2:
        a, b = list(imp)[:2]
        out["contorno"] = {"x": a, "y": b, "puntos": [{"x": t.params.get(a), "y": t.params.get(b), "valor": t.value} for t in done]}
    best = study.best_trial
    top = next(iter(imp), None)
    out["mejor"] = {"valor": best.value, "params": best.params}
    out["interpretabilidad"] = interp(
        "Historial e importancia de hiperparámetros del estudio.",
        "La curva de mejor acumulado debe estabilizarse; la importancia (fANOVA) indica qué hiperparámetro explica más la variación del objetivo.",
        (f"{top} concentra el {100 * imp[top]:.0f} % de la importancia; " if top else "") + f"mejor ROC-AUC interno = {best.value:.4f}.",
        "Los estudios exploratorios no sustituyen la CV anidada; el AUC interno es optimista.")
    return out


def export_best(name: str) -> str:
    study = optuna.load_study(study_name=name, storage=storage_url())
    d = CONFIG_DIR / "optuna_best"
    d.mkdir(exist_ok=True)
    path = d / f"{name}.yaml"
    path.write_text(yaml.safe_dump({"estudio": name, "valor": study.best_value, "params": study.best_params}, allow_unicode=True), encoding="utf-8")
    return str(path.relative_to(CONFIG_DIR.parent))
