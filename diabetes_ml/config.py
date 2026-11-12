from __future__ import annotations

import hashlib
import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"
ARTIFACTS = ROOT / "models" / "v3"
DISCLAIMER = ("Prototipo académico de investigación (no es un dispositivo médico). "
              "Las salidas son riesgos estimados y requieren confirmación clínica.")


@lru_cache(maxsize=None)
def load(name: str) -> Dict[str, Any]:
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed() -> int:
    return int(os.getenv("RANDOM_SEED", load("cv")["random_seed"]))


def set_seeds() -> int:
    s = seed()
    random.seed(s)
    np.random.seed(s)
    try:
        import torch
        torch.manual_seed(s)
    except ImportError:
        pass
    return s


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_path(path: Path) -> str:
    """Hash de un archivo o del conjunto ordenado de archivos de un directorio."""
    if path.is_file():
        return sha256_file(path)
    h = hashlib.sha256()
    for p in sorted(path.glob("*")):
        if p.is_file():
            h.update(p.name.encode())
            h.update(sha256_file(p).encode())
    return h.hexdigest()


def interp(que: str, como: str, interpretacion: str, limitaciones: str) -> Dict[str, str]:
    """Bloque de interpretabilidad obligatorio por tabla/figura (regla 6)."""
    return {"que_muestra": que, "como_leerla": como, "interpretacion": interpretacion, "limitaciones": limitaciones}
