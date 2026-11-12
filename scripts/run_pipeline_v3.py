"""CLI del pipeline v3 (Fases 1-7). Ejemplos:
  python -m scripts.run_pipeline_v3 --cohort nhanes_2017_2018 --scenario B --models lr,rf,xgb,tabnet,stack
  python -m scripts.run_pipeline_v3 --all
  python -m scripts.run_pipeline_v3 --explore --cohort pima_indians --scenario A --model xgb --n-trials 30 --sampler tpe
  python -m scripts.run_pipeline_v3 --hash-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diabetes_ml import jobs  # noqa: E402
from diabetes_ml.config import ARTIFACTS, ROOT, load, sha256_path  # noqa: E402


def emit(event, payload):
    print("EVENT " + json.dumps({"event": event, "t": round(time.time(), 1), **payload}, default=float, ensure_ascii=False), flush=True)


def _resources(stop):
    try:
        import psutil
    except ImportError:
        return
    while not stop.is_set():
        emit("resource", {"cpu": psutil.cpu_percent(interval=None), "ram_pct": psutil.virtual_memory().percent})
        stop.wait(15)


def train_one(cohort, scenario, models, n_trials):
    from diabetes_ml import explain, reports, training
    import joblib
    res = training.run(cohort, scenario, models, emit, n_trials)
    key = f"{cohort}__{scenario}"
    emit("explain_started", {"clave": key})
    exp = explain.explain_bundle(joblib.load(ARTIFACTS / key / "bundle.joblib"), res)
    (ARTIFACTS / key / "explain.json").write_text(json.dumps(exp, ensure_ascii=False, default=float), encoding="utf-8")
    emit("reports_started", {"clave": key})
    rep = reports.generate(key)
    emit("reports_completed", {"archivos": rep["archivos"]})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort"); ap.add_argument("--scenario", default="A")
    ap.add_argument("--models", default=",".join(load("models")["catalogo"]))
    ap.add_argument("--n-trials", type=int, default=None)
    ap.add_argument("--all", action="store_true"); ap.add_argument("--hash-only", action="store_true")
    ap.add_argument("--explore", action="store_true"); ap.add_argument("--model", default="xgb")
    ap.add_argument("--sampler", default="tpe"); ap.add_argument("--pruner", default="median")
    ap.add_argument("--multiobjective", action="store_true"); ap.add_argument("--space", default=None)
    ap.add_argument("--job-id", default=None)
    a = ap.parse_args()
    stop = threading.Event()
    threading.Thread(target=_resources, args=(stop,), daemon=True).start()
    try:
        if a.hash_only:
            for cid, meta in load("datasets").items():
                print(cid, sha256_path(ROOT / meta["ruta"]))
            return
        if a.explore:
            from diabetes_ml import data as D, training
            from diabetes_ml.splitting import split_70_15_15
            df, feats, _ = D.cohort_frame(a.cohort, a.scenario)
            dev = df.loc[split_70_15_15(df)["desarrollo"]].reset_index(drop=True)
            name = f"explor__{a.cohort}__{a.scenario}__{a.model}"
            space = json.loads(a.space) if a.space else None
            r = training.tune(a.model, dev[feats], dev["outcome"].values, dev["patient_id"].values, name, a.n_trials or 20,
                              emit, space, a.sampler, a.pruner, a.multiobjective)
            emit("completed", {"estudio": name, "mejor": r})
        else:
            runs = ([(c, s) for c in ("nhanes_2017_2018", "pima_indians") for s in ("A", "B")] if a.all else [(a.cohort, a.scenario)])
            for c, s in runs:
                train_one(c, s, a.models.split(","), a.n_trials)
        if a.job_id:
            jobs.write_status(a.job_id, estado="completed")
    except Exception as exc:
        traceback.print_exc()
        if a.job_id:
            jobs.write_status(a.job_id, estado="failed", error=str(exc))
        sys.exit(1)
    finally:
        stop.set()


if __name__ == "__main__":
    main()
