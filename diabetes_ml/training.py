"""Fase 2: CV anidada 5x3 con Optuna, modelo final, calibración y umbral en Validación, conformal.
La Prueba se lee una sola vez, al final, desde `run()`."""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List, Optional

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold

from diabetes_ml import data as D
from diabetes_ml import evaluation as E
from diabetes_ml import models as M
from diabetes_ml.config import ARTIFACTS, ROOT, DISCLAIMER, interp, load, seed, set_seeds
from diabetes_ml.splitting import index_hash, partition_table, split_70_15_15

optuna.logging.set_verbosity(optuna.logging.WARNING)
Emit = Callable[[str, Dict[str, Any]], None]
TUNE_SPY: List[np.ndarray] = []   # tests: registra las filas que ve Optuna


def _emit_default(event: str, payload: Dict[str, Any]) -> None:
    print("EVENT " + json.dumps({"event": event, **payload}, default=float), flush=True)


def storage_url() -> Optional[str]:
    url = os.getenv("OPTUNA_STORAGE", load("search_spaces")["optuna"]["storage"])
    if url.startswith("sqlite:///") and not url.startswith("sqlite:////"):
        path = ROOT / url[len("sqlite:///"):]
        path.parent.mkdir(parents=True, exist_ok=True)
        url = "sqlite:///" + path.as_posix()
    return None if url == "memory" else url


def make_study(name: str, sampler_name: str = "tpe", pruner_name: str = "median", directions=None) -> optuna.Study:
    st = storage_url()
    if st:
        try:
            optuna.delete_study(study_name=name, storage=st)
        except KeyError:
            pass
    samplers = {"tpe": lambda: optuna.samplers.TPESampler(seed=seed(), multivariate=True),
                "random": lambda: optuna.samplers.RandomSampler(seed=seed()),
                "cmaes": lambda: optuna.samplers.CmaEsSampler(seed=seed()),
                "nsga2": lambda: optuna.samplers.NSGAIISampler(seed=seed())}
    pruners = {"median": lambda: optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
               "hyperband": lambda: optuna.pruners.HyperbandPruner(), "none": lambda: optuna.pruners.NopPruner()}
    kw = {"directions": directions} if directions else {"direction": "maximize"}
    return optuna.create_study(study_name=name, storage=st, sampler=samplers[sampler_name](), pruner=pruners[pruner_name](), **kw)


def tune(name: str, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, study_name: str, n_trials: int,
         emit: Emit = _emit_default, space: Optional[Dict] = None, sampler: str = "tpe", pruner: str = "median",
         multiobjective: bool = False) -> Dict[str, Any]:
    """Búsqueda Optuna usando solo las filas recibidas (3 pliegues internos agrupados)."""
    TUNE_SPY.append(np.asarray(X.index))
    if n_trials <= 0:
        return {"params": M.default_params(name), "best_inner": None, "n_trials": 0}
    space = space or load("search_spaces")[name]
    inner = list(StratifiedGroupKFold(n_splits=load("cv")["cv_anidada"]["internos"], shuffle=True,
                                      random_state=seed()).split(X, y, groups))

    def objective(trial):
        params = M.suggest(trial, space)
        aucs, briers = [], []
        for step, (tr, va) in enumerate(inner):
            pipe = M.fit(name, params, X.iloc[tr], y[tr], groups[tr])
            p = pipe.predict_proba(X.iloc[va])[:, 1]
            aucs.append(roc_auc_score(y[va], p))
            briers.append(float(np.mean((p - y[va]) ** 2)))
            if not multiobjective:
                trial.report(float(np.mean(aucs)), step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        emit("trial_completed", {"estudio": study_name, "trial": trial.number, "valor": float(np.mean(aucs)), "params": params})
        return (float(np.mean(aucs)), float(np.mean(briers))) if multiobjective else float(np.mean(aucs))

    study = make_study(study_name, sampler, pruner, ["maximize", "minimize"] if multiobjective else None)
    study.optimize(objective, n_trials=n_trials)
    if multiobjective:
        best = max(study.best_trials, key=lambda t: t.values[0])
        return {"params": {**M.default_params(name), **best.params} if name == "tabnet" else best.params,
                "best_inner": best.values[0], "n_trials": n_trials}
    params = study.best_params
    return {"params": {**M.default_params(name), **params} if name == "tabnet" else params,
            "best_inner": float(study.best_value), "n_trials": n_trials}


def _trials(name: str, override: Optional[int]) -> int:
    return override if override is not None else int(load("search_spaces")["optuna"]["n_trials"][name])


def nested_cv(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, models: List[str], prefix: str,
              emit: Emit = _emit_default, n_trials_override: Optional[int] = None) -> Dict[str, Any]:
    cfg = load("cv")["cv_anidada"]
    outer = list(StratifiedGroupKFold(n_splits=cfg["externos"], shuffle=True, random_state=seed()).split(X, y, groups))
    base = [m for m in models if m in M.BASE_MODELS]
    oof = {m: np.zeros(len(y)) for m in base}
    res = {m: {"auc_externo": [], "best_inner": [], "params": [], "segundos": []} for m in base}
    xgb_gain = []
    for k, (tr, va) in enumerate(outer):
        emit("fold_started", {"pliegue": k + 1, "de": len(outer)})
        for m in base:
            t0 = time.time()
            tuned = tune(m, X.iloc[tr], y[tr], groups[tr], f"{prefix}__{m}__outer{k}", _trials(m, n_trials_override), emit)
            pipe = M.fit(m, tuned["params"], X.iloc[tr], y[tr], groups[tr])
            p = pipe.predict_proba(X.iloc[va])[:, 1]   # outer_val evaluado UNA sola vez
            oof[m][va] = p
            auc = float(roc_auc_score(y[va], p))
            res[m]["auc_externo"].append(auc)
            res[m]["best_inner"].append(tuned["best_inner"])
            res[m]["params"].append(tuned["params"])
            res[m]["segundos"].append(round(time.time() - t0, 1))
            if m == "xgb":
                xgb_gain.append(dict(zip(X.columns, pipe.named_steps["clf"].feature_importances_.round(5).tolist())))
            if m == "tabnet":
                emit("epoch", {"pliegue": k + 1, "historial": pipe.named_steps["clf"].history_})
            emit("fold_completed", {"pliegue": k + 1, "modelo": m, "auc": auc})
    if "stack" in models and len(base) >= 2:
        S = np.column_stack([oof[m] for m in base])
        res["stack"] = {"auc_externo": [], "best_inner": [], "params": [], "segundos": []}
        for k, (tr, va) in enumerate(outer):   # meta-LR entrenado sin el pliegue evaluado
            meta = LogisticRegression(C=1.0, max_iter=2000).fit(S[tr], y[tr])
            res["stack"]["auc_externo"].append(float(roc_auc_score(y[va], meta.predict_proba(S[va])[:, 1])))
    summary = {}
    for m, r in res.items():
        a = np.array(r["auc_externo"])
        from scipy import stats
        half = stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))
        inner = [v for v in r["best_inner"] if v is not None]
        summary[m] = {"auc_externo": [round(v, 4) for v in a], "media": round(float(a.mean()), 4), "de": round(float(a.std(ddof=1)), 4),
                      "ic95_t": [round(float(a.mean() - half), 4), round(float(a.mean() + half), 4)],
                      "brecha_optimismo": round(float(np.mean(inner) - a.mean()), 4) if inner else None,
                      "params_por_pliegue": r["params"], "segundos": r["segundos"]}
    n_tr = int(np.mean([len(tr) for tr, _ in outer]))
    n_te = int(np.mean([len(va) for _, va in outer]))
    return {"resumen": summary, "oof": oof, "importancia_xgb_por_pliegue": xgb_gain,
            "pruebas": E.nested_tests({m: summary[m]["auc_externo"] for m in summary}, n_tr, n_te),
            "interpretabilidad": interp(
                "Desempeño del procedimiento (CV anidada 5 externos x 3 internos).",
                "Cada AUC externo se obtiene en un pliegue que Optuna nunca vio; la brecha de optimismo es mejor AUC interno menos AUC externo.",
                "; ".join(f"{m}: {s['media']} ± {s['de']}" for m, s in summary.items()),
                "Estima el procedimiento, no un modelo concreto; IC t con n = 5 es amplio.")}


class Calibrator:
    def __init__(self, method: str):
        self.method = method

    def fit(self, p, y):
        if self.method == "isotonica":
            self.m = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(p, y)
        else:
            self.m = LogisticRegression(C=1e6, max_iter=2000).fit(E._logit(p).reshape(-1, 1), y)
        return self

    def __call__(self, p):
        p = np.asarray(p, dtype=float)
        return self.m.predict(p) if self.method == "isotonica" else self.m.predict_proba(E._logit(p).reshape(-1, 1))[:, 1]


def choose_threshold(y, p) -> float:
    cfg = load("cv")["umbral"]
    fpr, tpr, thr = roc_curve(y, p)
    if cfg["metodo"] == "sensibilidad":
        ok = np.where(tpr >= cfg["sensibilidad_objetivo"])[0]
        return float(thr[ok[0]]) if len(ok) else 0.5
    return float(np.clip(thr[np.argmax(tpr - fpr)], 1e-4, 1 - 1e-4))


def conformal_q(y, p, alpha: float) -> float:
    scores = 1 - np.where(y == 1, p, 1 - p)
    n = len(scores)
    return float(np.quantile(scores, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher"))


def prediction_set(p: float, q: float) -> List[int]:
    return [c for c, pc in ((0, 1 - p), (1, p)) if 1 - pc <= q]


class StackModel:
    def __init__(self, bases: Dict[str, Any], meta: LogisticRegression):
        self.bases, self.meta = bases, meta

    def predict_proba(self, X):
        S = np.column_stack([self.bases[m].predict_proba(X)[:, 1] for m in self.bases])
        return self.meta.predict_proba(S)


def _mlflow_log(name: str, params: Dict, metrics: Dict, nested: Dict) -> Optional[str]:
    try:
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        import mlflow
        mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
        mlflow.set_experiment("sp03_v3")
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            for m, s in nested.items():
                for k, auc in enumerate(s["auc_externo"]):
                    with mlflow.start_run(run_name=f"{m}_outer{k}", nested=True):
                        mlflow.log_metric("auc_externo", auc)
            return run.info.run_id
    except Exception as exc:  # MLflow es trazabilidad, no bloquea el entrenamiento
        return f"no registrado: {exc}"


def run(cohort: str, scenario: str, models: List[str], emit: Emit = _emit_default,
        n_trials_override: Optional[int] = None) -> Dict[str, Any]:
    set_seeds()
    t0 = time.time()
    df, feats, audit = D.cohort_frame(cohort, scenario)
    parts = split_70_15_15(df)
    groups_check = D.validate_groups(df, feats, parts)
    emit("particion", {"tabla": partition_table(df, parts), "validacion_grupos": groups_check["estado"]})
    dev, val, test = (df.loc[parts[k]].reset_index(drop=True) for k in ("desarrollo", "validacion", "prueba"))
    Xd, yd, gd = dev[feats], dev["outcome"].values, dev["patient_id"].values
    prefix = f"{cohort}__{scenario}"
    nested = nested_cv(Xd, yd, gd, models, prefix, emit, n_trials_override)

    final, tuned_final, cal, thr, q = {}, {}, {}, {}, {}
    alpha = load("cv")["conformal"]["alpha"]
    yv = val["outcome"].values
    method = "isotonica" if yv.sum() >= load("cv")["calibracion"]["isotonica_min_eventos"] else "sigmoide"
    for m in [m for m in models if m in M.BASE_MODELS]:
        emit("final_fit", {"modelo": m})
        tuned_final[m] = tune(m, Xd, yd, gd, f"{prefix}__{m}__final", _trials(m, n_trials_override), emit)
        final[m] = M.fit(m, tuned_final[m]["params"], Xd, yd, gd)
    if "stack" in models and len(final) >= 2:
        S = np.column_stack([nested["oof"][m] for m in final])
        final["stack"] = StackModel(dict(final), LogisticRegression(C=1.0, max_iter=2000).fit(S, yd))
    for m, est in final.items():
        pv = est.predict_proba(val[feats])[:, 1]
        cal[m] = Calibrator(method).fit(pv, yv)
        pvc = cal[m](pv)
        thr[m] = choose_threshold(yv, pvc)
        q[m] = conformal_q(yv, pvc, alpha)

    # ---- Prueba: lectura única ----
    yt = test["outcome"].values
    probs = {m: cal[m](est.predict_proba(test[feats])[:, 1]) for m, est in final.items()}
    groups = {"sexo": test["sexo"].astype(str).values, "edad": D.age_band(test["edad"]).values}
    evaluation = E.evaluate_test(yt, probs, thr, groups)
    coverage = {m: round(float(np.mean([int(yt[i]) in prediction_set(float(probs[m][i]), q[m]) for i in range(len(yt))])), 4) for m in probs}
    best = max(nested["resumen"], key=lambda k: nested["resumen"][k]["media"])

    imp_dev = final[next(iter(final))].named_steps["imp"] if hasattr(final[next(iter(final))], "named_steps") else None
    Zd = pd.DataFrame(imp_dev.transform(Xd) if imp_dev is not None else Xd, columns=feats)
    mu, cov = Zd.mean().values, np.cov(Zd.values, rowvar=False) + 1e-6 * np.eye(len(feats))
    icov = np.linalg.pinv(cov)
    dist = np.sqrt(np.einsum("ij,jk,ik->i", Zd.values - mu, icov, Zd.values - mu))
    reference = {"media": mu.tolist(), "icov": icov.tolist(), "umbral_ood": float(np.percentile(dist, load("models")["ood"]["percentil"])),
                 "medianas": Zd.median().to_dict(), "deciles": {f: np.unique(np.quantile(Zd[f], np.linspace(0, 1, 11))).tolist() for f in feats},
                 "muestra": Zd.sample(min(2000, len(Zd)), random_state=seed()).values.tolist()}

    out_dir = ARTIFACTS / prefix
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "cohorte": cohort, "escenario": scenario, "auditoria": audit, "predictores": feats,
        "particion": {"tabla": partition_table(df, parts), "hash_prueba": index_hash(parts["prueba"]),
                      "validacion_grupos": groups_check,
                      "interpretabilidad": interp("Partición 70/15/15 agrupada y estratificada.",
                                                  "Desarrollo: CV anidada y modelo final; Validación: calibración, umbral y conformal; Prueba: evaluación única.",
                                                  f"Validación de grupos: {groups_check['estado']}.",
                                                  "La agrupación puede desviar las proporciones en ±1 %.")},
        "cv_anidada": {k: v for k, v in nested.items() if k != "oof"},
        "modelo_final": {m: {"params": t["params"], "best_inner": t["best_inner"]} for m, t in tuned_final.items()},
        "calibracion": {"metodo": method, "umbrales": thr, "conformal": {"alpha": alpha, "q": q, "cobertura_prueba": coverage}},
        "evaluacion_prueba": evaluation, "modelo_recomendado": best,
        "supervivencia": load("models")["supervivencia"], "semilla": seed(), "segundos": round(time.time() - t0, 1),
        "aviso": DISCLAIMER,
    }
    results["mlflow_run"] = _mlflow_log(prefix, {"cohorte": cohort, "escenario": scenario, "modelos": ",".join(models), "semilla": seed()},
                                        {f"auc_test_{m}": v["metricas"]["roc_auc"] for m, v in evaluation["por_modelo"].items()},
                                        nested["resumen"])
    bundle = {"final": final, "calibradores": cal, "umbrales": thr, "conformal_q": q, "predictores": feats,
              "referencia": reference, "modelo_recomendado": best, "cohorte": cohort, "escenario": scenario,
              "test": {"X": test[feats], "y": yt, "probs": probs}, "dev_X": Xd.sample(min(500, len(Xd)), random_state=seed())}
    joblib.dump(bundle, out_dir / "bundle.joblib")
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    emit("completed", {"cohorte": cohort, "escenario": scenario, "segundos": results["segundos"]})
    return results
