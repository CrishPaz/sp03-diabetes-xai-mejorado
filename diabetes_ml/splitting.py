"""Fase 2: partición 70/15/15 agrupada y estratificada."""
from __future__ import annotations

import hashlib
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from diabetes_ml.config import load, seed


def split_70_15_15(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    cfg = load("cv")["particion"]
    sgkf = StratifiedGroupKFold(n_splits=cfg["n_splits"], shuffle=True, random_state=seed())
    folds = [te for _, te in sgkf.split(df, df["outcome"], df["patient_id"])]
    nt, nv = cfg["folds_prueba"], cfg["folds_validacion"]
    return {"prueba": np.sort(np.concatenate(folds[:nt])),
            "validacion": np.sort(np.concatenate(folds[nt:nt + nv])),
            "desarrollo": np.sort(np.concatenate(folds[nt + nv:]))}


def partition_table(df: pd.DataFrame, parts: Dict[str, np.ndarray]) -> list:
    rows = []
    for cls in (0, 1):
        row = {"clase": "Con DM2" if cls else "Sin DM2"}
        for k in ("desarrollo", "validacion", "prueba"):
            row[k] = int((df.loc[parts[k], "outcome"] == cls).sum())
        row["total"] = int((df["outcome"] == cls).sum())
        rows.append(row)
    return rows


def index_hash(idx: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()[:16]
