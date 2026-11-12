"""Ejecutor de trabajos largos en procesos separados; estado persistido en JSON y eventos en un log."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from diabetes_ml.config import ROOT

JOBS = ROOT / "logs" / "jobs"


def _path(job_id: str, ext: str):
    return JOBS / f"{job_id}.{ext}"


def write_status(job_id: str, **fields) -> Dict[str, Any]:
    JOBS.mkdir(parents=True, exist_ok=True)
    p = _path(job_id, "json")
    cur = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    cur.update(fields, actualizado=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    p.write_text(json.dumps(cur, ensure_ascii=False), encoding="utf-8")
    return cur


def start(kind: str, args: List[str]) -> Dict[str, Any]:
    """kind: train | optuna. Un trabajo de entrenamiento por cohorte a la vez."""
    cohort = args[args.index("--cohort") + 1] if "--cohort" in args else None
    for j in list_jobs():
        if j.get("estado") in ("queued", "running") and j.get("cohorte") == cohort and j.get("tipo") == kind:
            raise RuntimeError(f"Ya hay un trabajo activo para {cohort}: {j['id']}")
    job_id = uuid.uuid4().hex[:10]
    write_status(job_id, id=job_id, tipo=kind, cohorte=cohort, args=args, estado="queued")
    log = open(_path(job_id, "log"), "w", encoding="utf-8")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "MLFLOW_DISABLE_AGENT_HINT": "1"}
    proc = subprocess.Popen([sys.executable, "-u", "-m", "scripts.run_pipeline_v3", "--job-id", job_id, *args],
                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
    return write_status(job_id, pid=proc.pid, estado="running")


def get(job_id: str) -> Optional[Dict[str, Any]]:
    p = _path(job_id, "json")
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def list_jobs() -> List[Dict[str, Any]]:
    JOBS.mkdir(parents=True, exist_ok=True)
    return sorted((json.loads(p.read_text(encoding="utf-8")) for p in JOBS.glob("*.json")), key=lambda j: j.get("actualizado", ""), reverse=True)


def cancel(job_id: str) -> Dict[str, Any]:
    j = get(job_id)
    if j and j.get("estado") == "running" and j.get("pid"):
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(j["pid"]), "/T", "/F"], capture_output=True)
            else:
                os.kill(j["pid"], 9)
        except Exception:
            pass
    return write_status(job_id, estado="cancelled")


def events(job_id: str, offset: int = 0) -> Dict[str, Any]:
    p = _path(job_id, "log")
    if not p.exists():
        return {"eventos": [], "offset": offset, "logs": []}
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    ev, logs = [], []
    for ln in lines[offset:]:
        if ln.startswith("EVENT "):
            try:
                ev.append(json.loads(ln[6:]))
            except json.JSONDecodeError:
                pass
        elif ln.strip():
            logs.append(ln[:400])
    return {"eventos": ev, "logs": logs[-50:], "offset": len(lines)}
