"""Fase 1: carga, limpieza, diagrama de exclusiones, EDA y validación de grupos."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from diabetes_ml.config import ROOT, interp, load, sha256_path

NHANES_FILES = {
    "DEMO_J": ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "INDFMPIR"],
    "BMX_J": ["SEQN", "BMXBMI"],
    "BPX_J": ["SEQN", "BPXSY1", "BPXDI1"],
    "BIOPRO_J": ["SEQN", "LBXSGL", "LBXSTR"],
    "GHB_J": ["SEQN", "LBXGH"],
    "TCHOL_J": ["SEQN", "LBXTC"],
    "HDL_J": ["SEQN", "LBDHDD"],
    "DIQ_J": ["SEQN", "DIQ010"],
    "MCQ_J": ["SEQN", "MCQ300C"],
    "BPQ_J": ["SEQN", "BPQ020", "BPQ040A"],
}
RENAME = {"SEQN": "patient_id", "RIDAGEYR": "edad", "BMXBMI": "imc", "LBXSGL": "glucosa", "LBXGH": "hba1c",
          "BPXSY1": "presion_sistolica", "BPXDI1": "presion_diastolica", "LBXTC": "colesterol_total",
          "LBDHDD": "colesterol_hdl", "LBXSTR": "trigliceridos"}
SUBGROUP_ALERT = "Representatividad (>= 30 eventos por subgrupo)"
BALANCE_ALERT = "Balance entre particiones (SMD < 0.1)"


def _flow(steps: List[Tuple[str, int]]) -> List[Dict[str, Any]]:
    out, prev = [], None
    for motivo, n in steps:
        out.append({"paso": motivo, "n": int(n), "excluidos": None if prev is None else int(prev - n)})
        prev = n
    return out


def load_nhanes() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    d = ROOT / "data/raw/nhanes"
    df = None
    for name, cols in NHANES_FILES.items():
        part = pd.read_sas(d / f"{name}.xpt", format="xport")[cols]
        df = part if df is None else df.merge(part, on="SEQN", how="left")
    n0 = len(df)
    df = df[df["RIDAGEYR"] >= 20]
    n1 = len(df)
    df = df[df["DIQ010"].isin([1.0, 2.0, 3.0])]
    n2 = len(df)
    df = df.dropna(subset=["LBXGH", "LBXSGL", "BMXBMI"]).copy()
    n3 = len(df)
    df.loc[df["BPXDI1"] < 30, "BPXDI1"] = np.nan
    df.loc[df["BPXSY1"] < 60, "BPXSY1"] = np.nan
    df["antecedente_familiar_diabetes"] = (df["MCQ300C"] == 1).astype(float)
    df["antecedente_hipertension"] = (df["BPQ020"] == 1).astype(float)
    df["tratamiento_antihipertensivo"] = ((df["BPQ020"] == 1) & (df["BPQ040A"] == 1)).astype(float)
    df["outcome"] = (df["DIQ010"] == 1).astype(int)
    df["sexo"] = df["RIAGENDR"].map({1.0: "Masculino", 2.0: "Femenino"})
    df = df.rename(columns=RENAME)
    df["patient_id"] = df["patient_id"].astype(int).astype(str)
    flow = _flow([("Participantes DEMO_J", n0), ("Adultos >= 20 años", n1),
                  ("Respuesta válida DIQ010 (1/2/3)", n2), ("HbA1c, glucosa e IMC presentes", n3)])
    return df.reset_index(drop=True), {"flujo_exclusiones": flow}


def load_pima() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    cols = ["embarazos", "glucosa", "presion_diastolica", "pliegue_cutaneo", "insulina", "imc",
            "pedigri_diabetes", "edad", "outcome"]
    df = pd.read_csv(ROOT / load("datasets")["pima_indians"]["ruta"], header=None, names=cols)
    zeros = {}
    for c in load("features")["pima_indians"]["ceros_a_nan"]:
        zeros[c] = int((df[c] == 0).sum())
        df.loc[df[c] == 0, c] = np.nan
    df["patient_id"] = [f"P{i:04d}" for i in range(len(df))]
    df["sexo"] = "Femenino"
    return df, {"flujo_exclusiones": _flow([("Registros NIDDK", len(df))]), "ceros_convertidos_a_nan": zeros}


def _dedup(path_key: str, label: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = pd.read_csv(ROOT / load("datasets")[path_key]["ruta"])
    n0 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    return df, {"flujo_exclusiones": _flow([(label, n0), ("Sin duplicados exactos", len(df))])}


def load_sylhet():
    return _dedup("early_stage_sylhet", "Registros originales")


def load_brfss():
    return _dedup("brfss_2015", "Registros 50/50")


def load_hospitals() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = pd.read_csv(ROOT / load("datasets")["diabetes_130_hospitals"]["ruta"])
    n0 = len(df)
    df = df.drop_duplicates("patient_nbr", keep="first")
    n1 = len(df)
    df = df[~df["discharge_disposition_id"].isin([11, 13, 14, 19, 20, 21])].reset_index(drop=True)
    return df, {"flujo_exclusiones": _flow([("Encuentros", n0), ("Primer encuentro por paciente", n1),
                                            ("Sin fallecimiento/hospicio al alta", len(df))])}


LOADERS = {"nhanes_2017_2018": load_nhanes, "pima_indians": load_pima, "early_stage_sylhet": load_sylhet,
           "brfss_2015": load_brfss, "diabetes_130_hospitals": load_hospitals}
MODELED = ("nhanes_2017_2018", "pima_indians")


def cohort_frame(cohort: str, scenario: str = "A") -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    if cohort not in MODELED:
        raise ValueError(f"Cohorte sin pipeline de modelado: {cohort}")
    df, audit = LOADERS[cohort]()
    fcfg = load("features")[cohort]
    feats = fcfg["escenarios"][scenario]["predictores"]
    audit.update({"cohorte": cohort, "escenario": scenario, "n_oficial": int(len(df)),
                  "eventos": int(df["outcome"].sum()), "prevalencia": round(float(df["outcome"].mean()), 4),
                  "etiqueta": fcfg["etiqueta"], "descripcion_escenario": fcfg["escenarios"][scenario]["descripcion"],
                  "predictores": feats, "sha256_datos": sha256_path(ROOT / load("datasets")[cohort]["ruta"])})
    return df, feats, audit


def registry() -> List[Dict[str, Any]]:
    out = []
    for cid, meta in load("datasets").items():
        df, audit = LOADERS[cid]()
        out.append({"id": cid, **meta, "n_oficial": int(len(df)), "flujo_exclusiones": audit["flujo_exclusiones"]})
    return out


def _smd(a: pd.Series, b: pd.Series) -> float:
    s = np.sqrt((a.var() + b.var()) / 2)
    return float((a.mean() - b.mean()) / s) if s > 0 else 0.0


def table_one(df: pd.DataFrame, feats: List[str]) -> Dict[str, Any]:
    rows = []
    pos, neg = df[df.outcome == 1], df[df.outcome == 0]
    for f in feats:
        a, b = pos[f].dropna(), neg[f].dropna()
        if set(df[f].dropna().unique()) <= {0.0, 1.0}:
            tab = pd.crosstab(df[f], df.outcome)
            p = stats.chi2_contingency(tab)[1] if tab.shape == (2, 2) else np.nan
            dp, dn = f"{int(a.sum())} ({100*a.mean():.1f} %)", f"{int(b.sum())} ({100*b.mean():.1f} %)"
        elif len(a) > 20 and stats.normaltest(a).pvalue > 0.05 and stats.normaltest(b).pvalue > 0.05:
            dp, dn = f"{a.mean():.1f} ± {a.std():.1f}", f"{b.mean():.1f} ± {b.std():.1f}"
            p = stats.ttest_ind(a, b, equal_var=False).pvalue
        else:
            dp = f"{a.median():.1f} [{a.quantile(.25):.1f}-{a.quantile(.75):.1f}]"
            dn = f"{b.median():.1f} [{b.quantile(.25):.1f}-{b.quantile(.75):.1f}]"
            p = stats.mannwhitneyu(a, b).pvalue
        rows.append({"variable": f, "con_dm2": dp, "sin_dm2": dn, "faltantes_pct": round(100 * df[f].isna().mean(), 1),
                     "p_descriptivo": None if np.isnan(p) else float(f"{p:.3g}"), "smd": round(_smd(a, b), 3)})
    top = max(rows, key=lambda r: abs(r["smd"]))
    return {"filas": rows, "n_con_dm2": int(len(pos)), "n_sin_dm2": int(len(neg)),
            "interpretabilidad": interp(
                "Características basales por clase (Tabla 1).",
                "Media ± DE si ambas clases son normales (D'Agostino-Pearson); si no, mediana [RIC]. |SMD| > 0.2 indica diferencia relevante.",
                f"La mayor diferencia estandarizada es {top['variable']} (SMD = {top['smd']}).",
                "Los valores p son descriptivos, no inferenciales, y no ajustan por confusores.")}


def eda(cohort: str, scenario: str = "A") -> Dict[str, Any]:
    df, feats, audit = cohort_frame(cohort, scenario)
    hist = {}
    for f in feats:
        x = df[f].dropna()
        if x.nunique() > 2:
            edges = np.histogram_bin_edges(x, bins=20)
            hist[f] = {"edges": edges.round(3).tolist(),
                       "con_dm2": np.histogram(df.loc[df.outcome == 1, f].dropna(), edges)[0].tolist(),
                       "sin_dm2": np.histogram(df.loc[df.outcome == 0, f].dropna(), edges)[0].tolist()}
    corr = df[feats].corr(method="spearman").round(3)
    return {"auditoria": audit, "tabla1": table_one(df, feats), "histogramas": hist,
            "spearman": {"variables": feats, "matriz": corr.fillna(0).values.tolist()},
            "faltantes": {f: round(100 * df[f].isna().mean(), 2) for f in feats},
            "balance_clases": {"0": int((df.outcome == 0).sum()), "1": int((df.outcome == 1).sum())}}


def age_band(age: pd.Series) -> pd.Series:
    return pd.cut(age, [0, 39, 59, 200], labels=["18-39", "40-59", ">=60"]).astype(str)


def validate_groups(df: pd.DataFrame, feats: List[str], parts: Dict[str, np.ndarray] | None = None) -> Dict[str, Any]:
    checks = []
    dup = int(df["patient_id"].duplicated().sum())
    checks.append({"control": "Identificador único por paciente", "ok": dup == 0, "detalle": f"{dup} duplicados"})
    if parts:
        ids = {k: set(df.loc[v, "patient_id"]) for k, v in parts.items()}
        keys = list(ids)
        inter = sum(len(ids[a] & ids[b]) for i, a in enumerate(keys) for b in keys[i + 1:])
        checks.append({"control": "Intersección vacía entre particiones", "ok": inter == 0, "detalle": f"{inter} pacientes compartidos"})
        dev = df.loc[parts["desarrollo"]]
        rest = df.loc[np.concatenate([parts["validacion"], parts["prueba"]])]
        smds = {f: round(abs(_smd(dev[f].dropna(), rest[f].dropna())), 3) for f in feats}
        checks.append({"control": BALANCE_ALERT, "ok": max(smds.values()) < 0.1, "detalle": smds})
    if "edad" in df:
        g = df.assign(banda=age_band(df["edad"])).groupby(["banda", "sexo"])["outcome"].agg(["size", "sum"])
        low = [f"{a}/{b}: {int(r['sum'])}" for (a, b), r in g.iterrows() if r["sum"] < 30]
        checks.append({"control": SUBGROUP_ALERT, "ok": not low, "detalle": low or "todos los subgrupos >= 30 eventos"})
    critical = [c for c in checks if c["control"] not in (SUBGROUP_ALERT, BALANCE_ALERT)]
    alerts = [c["control"] for c in checks if c["control"] in (SUBGROUP_ALERT, BALANCE_ALERT) and not c["ok"]]
    return {"estado": "PASS" if all(c["ok"] for c in critical) else "FAIL", "alertas": alerts, "controles": checks}
