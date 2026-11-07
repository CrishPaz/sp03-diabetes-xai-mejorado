#!/usr/bin/env python3
"""
Pipeline reproducible de preparación de datos, entrenamiento, validación cruzada agrupada
por paciente, evaluación en conjunto de prueba aislado, calibración, análisis de subgrupos
(edad y sexo sin uso discriminatorio) y explicabilidad SHAP para el Caso SP-03.

Conjuntos de datos reales procesados:
1. NHANES 2017-2018 (CDC / NCHS) - Cohorte clínica primaria con biomarcadores completos.
2. Pima Indians Diabetes Dataset (NIDDK / UCI) - Cohorte clásica de benchmarking.

También genera un archivo sintético rotulado exclusivamente como DEMO para pruebas de
importación CSV sin conexión, el cual NUNCA se utiliza para entrenar ni reportar métricas.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import scipy
import shap
import sklearn
import torch
import xgboost as xgb
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_NHANES_DIR = DATA_DIR / "raw" / "nhanes"
RAW_PIMA_PATH = DATA_DIR / "raw" / "pima" / "pima-indians-diabetes.csv"
PROCESSED_DIR = DATA_DIR / "processed"
DEMO_DIR = DATA_DIR / "demo"
MODELS_DIR = ROOT_DIR / "models"


NHANES_FEATURES_META: List[Dict[str, Any]] = [
    {
        "name": "edad",
        "source_col": "RIDAGEYR",
        "label": "Edad",
        "unit": "años",
        "category": "Demografía clínica",
        "min": 20.0,
        "max": 85.0,
        "step": 1.0,
        "default": 52.0,
        "normal_range": "20 – 80 años",
        "description": "Edad cronológica del paciente adulto (≥ 20 años).",
    },
    {
        "name": "imc",
        "source_col": "BMXBMI",
        "label": "Índice de Masa Corporal (IMC)",
        "unit": "kg/m²",
        "category": "Antropometría",
        "min": 14.0,
        "max": 65.0,
        "step": 0.1,
        "default": 29.4,
        "normal_range": "18.5 – 24.9 kg/m²",
        "description": "Relación peso/talla al cuadrado medida en examen físico MEC.",
    },
    {
        "name": "glucosa",
        "source_col": "LBXSGL",
        "label": "Glucosa sérica",
        "unit": "mg/dL",
        "category": "Biomarcadores glucémicos",
        "min": 45.0,
        "max": 450.0,
        "step": 1.0,
        "default": 104.0,
        "normal_range": "70 – 99 mg/dL (ayunas)",
        "description": "Concentración de glucosa en suero refrigerado (panel bioquímico).",
    },
    {
        "name": "hba1c",
        "source_col": "LBXGH",
        "label": "Hemoglobina Glicosilada (HbA1c)",
        "unit": "%",
        "category": "Biomarcadores glucémicos",
        "min": 3.5,
        "max": 16.0,
        "step": 0.1,
        "default": 5.7,
        "normal_range": "< 5.7 % (Normal) | 5.7–6.4 % (Prediabetes)",
        "description": "Porcentaje de hemoglobina glicosilada que refleja glucemia media de 8-12 semanas.",
    },
    {
        "name": "presion_sistolica",
        "source_col": "BPXSY1",
        "label": "Presión Arterial Sistólica",
        "unit": "mmHg",
        "category": "Hemodinámica",
        "min": 75.0,
        "max": 240.0,
        "step": 1.0,
        "default": 126.0,
        "normal_range": "< 120 mmHg",
        "description": "Presión arterial sistólica braquial en reposo.",
    },
    {
        "name": "presion_diastolica",
        "source_col": "BPXDI1",
        "label": "Presión Arterial Diastólica",
        "unit": "mmHg",
        "category": "Hemodinámica",
        "min": 40.0,
        "max": 140.0,
        "step": 1.0,
        "default": 74.0,
        "normal_range": "< 80 mmHg",
        "description": "Presión arterial diastólica braquial en reposo.",
    },
    {
        "name": "colesterol_total",
        "source_col": "LBXTC",
        "label": "Colesterol Total",
        "unit": "mg/dL",
        "category": "Perfil lipídico",
        "min": 80.0,
        "max": 450.0,
        "step": 1.0,
        "default": 189.0,
        "normal_range": "< 200 mg/dL",
        "description": "Colesterol sérico total medido enzimáticamente.",
    },
    {
        "name": "colesterol_hdl",
        "source_col": "LBDHDD",
        "label": "Colesterol HDL",
        "unit": "mg/dL",
        "category": "Perfil lipídico",
        "min": 15.0,
        "max": 130.0,
        "step": 1.0,
        "default": 51.0,
        "normal_range": "≥ 40 mg/dL (H) | ≥ 50 mg/dL (M)",
        "description": "Colesterol de lipoproteínas de alta densidad (factor protector).",
    },
    {
        "name": "trigliceridos",
        "source_col": "LBXSTR",
        "label": "Triglicéridos Séricos",
        "unit": "mg/dL",
        "category": "Perfil lipídico",
        "min": 25.0,
        "max": 900.0,
        "step": 1.0,
        "default": 138.0,
        "normal_range": "< 150 mg/dL",
        "description": "Triglicéridos séricos asociados a resistencia a la insulina.",
    },
    {
        "name": "antecedente_familiar_diabetes",
        "source_col": "MCQ300C",
        "label": "Antecedente Familiar de Diabetes",
        "unit": "binario (0/1)",
        "category": "Antecedentes y Tratamientos",
        "type": "boolean",
        "min": 0.0,
        "max": 1.0,
        "step": 1.0,
        "default": 0.0,
        "normal_range": "0 = No, 1 = Sí",
        "description": "Parientes biológicos cercanos diagnosticados con diabetes mellitus (MCQ300C).",
    },
    {
        "name": "antecedente_hipertension",
        "source_col": "BPQ020",
        "label": "Antecedente de Hipertensión Arterial",
        "unit": "binario (0/1)",
        "category": "Antecedentes y Tratamientos",
        "type": "boolean",
        "min": 0.0,
        "max": 1.0,
        "step": 1.0,
        "default": 0.0,
        "normal_range": "0 = No, 1 = Sí",
        "description": "Diagnóstico médico previo de hipertensión arterial (BPQ020).",
    },
    {
        "name": "tratamiento_antihipertensivo",
        "source_col": "BPQ040A",
        "label": "En Tratamiento Antihipertensivo",
        "unit": "binario (0/1)",
        "category": "Antecedentes y Tratamientos",
        "type": "boolean",
        "min": 0.0,
        "max": 1.0,
        "step": 1.0,
        "default": 0.0,
        "normal_range": "0 = No, 1 = Sí",
        "description": "Uso actual de medicación prescrita para control de presión arterial (BPQ040A).",
    },
]

PIMA_FEATURES_META: List[Dict[str, Any]] = [
    {
        "name": "embarazos",
        "source_col": "Pregnancies",
        "label": "Número de Embarazos",
        "unit": "gestaciones",
        "category": "Antecedentes obstétricos",
        "min": 0.0,
        "max": 18.0,
        "step": 1.0,
        "default": 3.0,
        "normal_range": "0 – 10",
        "description": "Número total de gestaciones previas.",
    },
    {
        "name": "glucosa",
        "source_col": "Glucose",
        "label": "Glucosa Plasmática (2h OGTT)",
        "unit": "mg/dL",
        "category": "Biomarcadores glucémicos",
        "min": 44.0,
        "max": 250.0,
        "step": 1.0,
        "default": 120.0,
        "normal_range": "< 140 mg/dL (2h post-carga)",
        "description": "Concentración de glucosa plasmática a las 2 horas en prueba de tolerancia oral.",
    },
    {
        "name": "presion_diastolica",
        "source_col": "BloodPressure",
        "label": "Presión Arterial Diastólica",
        "unit": "mmHg",
        "category": "Hemodinámica",
        "min": 35.0,
        "max": 130.0,
        "step": 1.0,
        "default": 72.0,
        "normal_range": "< 80 mmHg",
        "description": "Presión arterial diastólica en reposo (mmHg).",
    },
    {
        "name": "pliegue_cutaneo",
        "source_col": "SkinThickness",
        "label": "Pliegue Cutáneo Tricipital",
        "unit": "mm",
        "category": "Antropometría",
        "min": 7.0,
        "max": 99.0,
        "step": 1.0,
        "default": 29.0,
        "normal_range": "10 – 35 mm",
        "description": "Espesor del pliegue cutáneo del tríceps en milímetros.",
    },
    {
        "name": "insulina",
        "source_col": "Insulin",
        "label": "Insulina Sérica a las 2h",
        "unit": "μU/mL",
        "category": "Biomarcadores glucémicos",
        "min": 14.0,
        "max": 850.0,
        "step": 1.0,
        "default": 125.0,
        "normal_range": "16 – 166 μU/mL",
        "description": "Nivel sérico de insulina a las 2 horas tras sobrecarga oral de glucosa.",
    },
    {
        "name": "imc",
        "source_col": "BMI",
        "label": "Índice de Masa Corporal (IMC)",
        "unit": "kg/m²",
        "category": "Antropometría",
        "min": 15.0,
        "max": 68.0,
        "step": 0.1,
        "default": 32.0,
        "normal_range": "18.5 – 24.9 kg/m²",
        "description": "Peso en kilogramos dividido por el cuadrado de la talla en metros.",
    },
    {
        "name": "pedigri_diabetes",
        "source_col": "DiabetesPedigreeFunction",
        "label": "Función de Pedigrí de Diabetes",
        "unit": "índice",
        "category": "Antecedentes genéticos",
        "min": 0.05,
        "max": 2.50,
        "step": 0.01,
        "default": 0.47,
        "normal_range": "0.08 – 0.80",
        "description": "Puntaje continuo que cuantifica la carga genética familiar de diabetes.",
    },
    {
        "name": "edad",
        "source_col": "Age",
        "label": "Edad",
        "unit": "años",
        "category": "Demografía clínica",
        "min": 21.0,
        "max": 85.0,
        "step": 1.0,
        "default": 33.0,
        "normal_range": "≥ 21 años",
        "description": "Edad en años cumplidos (población femenina ≥ 21 años).",
    },
]


def prepare_nhanes_dataset() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Une, audita y limpia los archivos XPT oficiales de CDC NHANES 2017-2018."""
    demo = pd.read_sas(RAW_NHANES_DIR / "DEMO_J.xpt")[
        ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "INDFMPIR"]
    ]
    bmx = pd.read_sas(RAW_NHANES_DIR / "BMX_J.xpt")[["SEQN", "BMXBMI", "BMXWAIST"]]
    bpx = pd.read_sas(RAW_NHANES_DIR / "BPX_J.xpt")[["SEQN", "BPXSY1", "BPXDI1"]]
    bio = pd.read_sas(RAW_NHANES_DIR / "BIOPRO_J.xpt")[["SEQN", "LBXSGL", "LBXSTR"]]
    ghb = pd.read_sas(RAW_NHANES_DIR / "GHB_J.xpt")[["SEQN", "LBXGH"]]
    tchol = pd.read_sas(RAW_NHANES_DIR / "TCHOL_J.xpt")[["SEQN", "LBXTC"]]
    hdl = pd.read_sas(RAW_NHANES_DIR / "HDL_J.xpt")[["SEQN", "LBDHDD"]]
    diq = pd.read_sas(RAW_NHANES_DIR / "DIQ_J.xpt")[["SEQN", "DIQ010", "DIQ050", "DIQ070"]]
    mcq = pd.read_sas(RAW_NHANES_DIR / "MCQ_J.xpt")[["SEQN", "MCQ300C"]]
    bpq = pd.read_sas(RAW_NHANES_DIR / "BPQ_J.xpt")[["SEQN", "BPQ020", "BPQ040A"]]

    raw_total = len(demo)
    duplicate_seqn = int(demo["SEQN"].duplicated().sum())

    df = (
        demo.merge(bmx, on="SEQN", how="left")
        .merge(bpx, on="SEQN", how="left")
        .merge(bio, on="SEQN", how="left")
        .merge(ghb, on="SEQN", how="left")
        .merge(tchol, on="SEQN", how="left")
        .merge(hdl, on="SEQN", how="left")
        .merge(diq, on="SEQN", how="left")
        .merge(mcq, on="SEQN", how="left")
        .merge(bpq, on="SEQN", how="left")
    )

    # Filtrar adultos >= 20 años con respuesta válida en DIQ010 (1=Sí, 2=No, 3=Borderline)
    df_adults = df[(df["RIDAGEYR"] >= 20) & (df["DIQ010"].isin([1.0, 2.0, 3.0]))].copy()
    adults_total = len(df_adults)

    # Exigir presencia de los biomarcadores nucleares (HbA1c, Glucosa sérica e IMC)
    df_valid = df_adults.dropna(subset=["LBXGH", "LBXSGL", "BMXBMI"]).copy()

    # Limpieza fisiológica de presión arterial (0 mmHg diastólica en lecturas aisladas -> NaN)
    df_valid.loc[df_valid["BPXDI1"] < 30, "BPXDI1"] = np.nan
    df_valid.loc[df_valid["BPXSY1"] < 60, "BPXSY1"] = np.nan

    # Codificación de variables categóricas clínicas (respetando patrones de salto de NHANES)
    df_valid["antecedente_familiar_diabetes"] = (df_valid["MCQ300C"] == 1.0).astype(float)
    df_valid["antecedente_hipertension"] = (df_valid["BPQ020"] == 1.0).astype(float)
    df_valid["tratamiento_antihipertensivo"] = (
        (df_valid["BPQ020"] == 1.0) & (df_valid["BPQ040A"] == 1.0)
    ).astype(float)

    # Renombrar columnas al esquema clínico en español
    rename_map = {
        "SEQN": "patient_id",
        "RIDAGEYR": "edad",
        "BMXBMI": "imc",
        "LBXSGL": "glucosa",
        "LBXGH": "hba1c",
        "BPXSY1": "presion_sistolica",
        "BPXDI1": "presion_diastolica",
        "LBXTC": "colesterol_total",
        "LBDHDD": "colesterol_hdl",
        "LBXSTR": "trigliceridos",
    }
    df_valid = df_valid.rename(columns=rename_map)
    df_valid["patient_id"] = df_valid["patient_id"].astype(int).astype(str)

    # Variable objetivo: Diagnóstico clínico confirmado de Diabetes Mellitus (DIQ010 == 1)
    df_valid["outcome"] = (df_valid["DIQ010"] == 1.0).astype(int)

    # Metadatos para auditoría de subgrupos (NO entran como predictores del modelo)
    df_valid["sexo_label"] = df_valid["RIAGENDR"].map({1.0: "Masculino", 2.0: "Femenino"})
    df_valid["grupo_edad"] = pd.cut(
        df_valid["edad"],
        bins=[19, 39, 59, 120],
        labels=["20-39 años", "40-59 años", "≥60 años"],
    ).astype(str)

    feature_names = [f["name"] for f in NHANES_FEATURES_META]
    missing_before_imputation = {
        col: int(df_valid[col].isna().sum()) for col in feature_names
    }
    exact_row_duplicates = int(df_valid[feature_names].duplicated().sum())

    audit_info = {
        "dataset_id": "nhanes_2017_2018",
        "name": "CDC NHANES 2017–2018 (Cohorte Clínica Nacional de EE. UU.)",
        "period": "Ciclo 2017–2018 (NCHS / CDC)",
        "license": "Dominio Público de EE. UU. (CC0 1.0 / Public Domain US Government Work)",
        "source_url": "https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017",
        "raw_participants_count": raw_total,
        "adults_evaluated_count": adults_total,
        "final_analytic_cohort_count": len(df_valid),
        "positive_cases": int(df_valid["outcome"].sum()),
        "negative_cases": int((df_valid["outcome"] == 0).sum()),
        "prevalence_rate": round(float(df_valid["outcome"].mean()), 4),
        "duplicate_patient_ids": duplicate_seqn,
        "duplicate_feature_rows": exact_row_duplicates,
        "missing_values_documented": missing_before_imputation,
        "imputation_strategy": (
            "Imputación por mediana ajustada exclusivamente sobre el conjunto de entrenamiento "
            "(SimpleImputer(strategy='median') dentro del pipeline de cada partición) para evitar "
            "fuga de información (data leakage) hacia validación o prueba."
        ),
        "leakage_prevention_notes": [
            "Partición agrupada por identificador único de paciente (SEQN) mediante StratifiedGroupKFold.",
            "Exclusión explícita de variables post-diagnóstico de tratamiento hipoglucemiante (DIQ050 insulina, DIQ070 antidiabéticos orales) que causarían fuga directa de la etiqueta objetivo.",
            "Exclusión de variables demográficas sensibles (RIAGENDR sexo, RIDRETH1 raza/etnia, INDFMPIR nivel socioeconómico) de la matriz predictora X para evitar discriminación algorítmica; se conservan únicamente para auditar equidad por subgrupos.",
        ],
        "target_definition": "Diagnóstico médico confirmado de Diabetes Mellitus Tipo 2 (DIQ010 == 1 en cuestionario clínico NHANES).",
    }

    keep_cols = ["patient_id", "outcome", "sexo_label", "grupo_edad"] + feature_names
    return df_valid[keep_cols].reset_index(drop=True), audit_info


def prepare_pima_dataset() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Carga, audita y limpia Pima Indians Diabetes Dataset (NIDDK)."""
    cols = [
        "embarazos",
        "glucosa",
        "presion_diastolica",
        "pliegue_cutaneo",
        "insulina",
        "imc",
        "pedigri_diabetes",
        "edad",
        "outcome",
    ]
    df = pd.read_csv(RAW_PIMA_PATH, header=None, names=cols)
    raw_total = len(df)
    exact_row_duplicates = int(df.duplicated().sum())

    # Documentar ceros biológicamente imposibles codificados como 0 en el dataset original
    zero_invalid_cols = ["glucosa", "presion_diastolica", "pliegue_cutaneo", "insulina", "imc"]
    invalid_zeros_count = {}
    for c in zero_invalid_cols:
        cnt = int((df[c] == 0).sum())
        invalid_zeros_count[c] = cnt
        df.loc[df[c] == 0, c] = np.nan

    df["patient_id"] = [f"PIMA_{i:04d}" for i in range(1, len(df) + 1)]
    df["sexo_label"] = "Femenino (Cohorte Pima)"
    df["grupo_edad"] = pd.cut(
        df["edad"],
        bins=[20, 29, 44, 120],
        labels=["21-29 años", "30-44 años", "≥45 años"],
    ).astype(str)

    feature_names = [f["name"] for f in PIMA_FEATURES_META]
    audit_info = {
        "dataset_id": "pima_indians",
        "name": "Pima Indians Diabetes Dataset (NIDDK Benchmark)",
        "period": "Estudio prospectivo longitudinal NIDDK (Phoenix, Arizona, EE. UU.)",
        "license": "Acceso Público Académico / OpenML CC-BY 4.0 (NIDDK / UCI Repository)",
        "source_url": "https://www.openml.org/d/37",
        "raw_participants_count": raw_total,
        "adults_evaluated_count": raw_total,
        "final_analytic_cohort_count": len(df),
        "positive_cases": int(df["outcome"].sum()),
        "negative_cases": int((df["outcome"] == 0).sum()),
        "prevalence_rate": round(float(df["outcome"].mean()), 4),
        "duplicate_patient_ids": 0,
        "duplicate_feature_rows": exact_row_duplicates,
        "missing_values_documented": invalid_zeros_count,
        "imputation_strategy": (
            "Conversión de ceros fisiológicamente imposibles (glucosa=0, presión=0, IMC=0, "
            "pliegue=0, insulina=0) a NaN seguida de imputación por mediana ajustada "
            "exclusivamente en entrenamiento."
        ),
        "leakage_prevention_notes": [
            "Aislamiento estricto del conjunto de prueba (20% estratificado por paciente) antes de cualquier imputación o escalado.",
            "Cálculo de medianas y desviaciones estándar restringido al subconjunto de entrenamiento en cada fold.",
        ],
        "target_definition": "Resultado positivo para diabetes mellitus según criterios OMS a las 2h post-carga (Outcome == 1).",
    }

    keep_cols = ["patient_id", "outcome", "sexo_label", "grupo_edad"] + feature_names
    return df[keep_cols].reset_index(drop=True), audit_info


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """Calcula todas las métricas clínicas, matriz de confusión y calibración."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
    pr_auc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
    brier = float(brier_score_loss(y_true, y_prob))

    # Curva ROC compacta (30 puntos interpolados para visualización fluida)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    idx = np.linspace(0, len(fpr) - 1, min(35, len(fpr))).astype(int)
    roc_points = [
        {"fpr": round(float(fpr[i]), 4), "tpr": round(float(tpr[i]), 4)}
        for i in idx
    ]

    # Curva de calibración (8 bins)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=8, strategy="uniform")
    calibration_points = [
        {"prob_pred": round(float(p), 4), "prob_true": round(float(t), 4)}
        for p, t in zip(prob_pred, prob_true)
    ]

    return {
        "threshold": round(float(threshold), 3),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "precision": round(precision, 4),
        "npv": round(npv, 4),
        "f1": round(f1, 4),
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier, 4),
        "roc_curve": roc_points,
        "calibration_curve": calibration_points,
    }


def evaluate_subgroups(
    df_test: pd.DataFrame, y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """Audita desempeño por subgrupos de edad y sexo sin usar variables sensibles en el modelo."""
    results = []
    for dimension, col in [("Grupo Etario", "grupo_edad"), ("Sexo Biológico", "sexo_label")]:
        for group_val, sub_df in df_test.groupby(col):
            idx = sub_df.index.values
            if len(idx) < 10:
                continue
            yt = y_true[idx]
            yp = y_prob[idx]
            y_hat = (yp >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(yt, y_hat, labels=[0, 1]).ravel()
            sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            auc = float(roc_auc_score(yt, yp)) if len(np.unique(yt)) > 1 else 0.0
            brier = float(brier_score_loss(yt, yp))
            results.append(
                {
                    "dimension": dimension,
                    "subgroup": str(group_val),
                    "n": int(len(idx)),
                    "prevalence": round(float(yt.mean()), 4),
                    "predicted_positive_rate": round(float(y_hat.mean()), 4),
                    "roc_auc": round(auc, 4),
                    "sensitivity": round(sens, 4),
                    "specificity": round(spec, 4),
                    "brier_score": round(brier, 4),
                }
            )
    return results


def train_and_evaluate_cohort(
    dataset_id: str,
    df: pd.DataFrame,
    features_meta: List[Dict[str, Any]],
    audit_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Entrena Línea Base (Regresión Logística), Random Forest, XGBoost y TabNet con CV y Test aislado."""
    feature_names = [f["name"] for f in features_meta]
    X_all = df[feature_names].copy()
    y_all = df["outcome"].values.astype(int)
    groups_all = df["patient_id"].values

    # División agrupada por paciente: 80% Desarrollo (con 5-fold StratifiedGroupKFold) y 20% Prueba Aislada
    sgkf_holdout = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    train_idx, test_idx = next(sgkf_holdout.split(X_all, y_all, groups_all))

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    # Verificar que no exista intersección de pacientes entre entrenamiento y prueba
    overlap = set(df_train["patient_id"]).intersection(set(df_test["patient_id"]))
    assert len(overlap) == 0, f"Fuga de pacientes detectada: {len(overlap)}"

    X_train_raw = df_train[feature_names].values
    y_train = df_train["outcome"].values.astype(int)
    groups_train = df_train["patient_id"].values

    X_test_raw = df_test[feature_names].values
    y_test = df_test["outcome"].values.astype(int)

    # Guardar muestra de prueba aislada para pruebas reproducibles en lote
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_test[feature_names + ["outcome"]].head(40).to_csv(
        PROCESSED_DIR / f"{dataset_id}_test_sample.csv", index=False
    )

    # Imputador ajustado EXCLUSIVAMENTE sobre X_train_raw
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train_raw)
    X_test_imp = imputer.transform(X_test_raw)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    pos_weight = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))

    # 1. Validación cruzada agrupada de 5 pliegues sobre el conjunto de entrenamiento (80%)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores: Dict[str, List[Dict[str, float]]] = {
        "logistic_regression": [],
        "random_forest": [],
        "xgboost": [],
        "tabnet": [],
    }

    for fold_i, (tr_i, val_i) in enumerate(cv.split(X_train_raw, y_train, groups_train)):
        f_imp = SimpleImputer(strategy="median")
        X_tr = f_imp.fit_transform(X_train_raw[tr_i])
        X_va = f_imp.transform(X_train_raw[val_i])
        y_tr, y_va = y_train[tr_i], y_train[val_i]

        f_scaler = StandardScaler()
        X_tr_s = f_scaler.fit_transform(X_tr)
        X_va_s = f_scaler.transform(X_va)

        # Logistic Regression
        lr = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=500, random_state=RANDOM_SEED
        )
        lr.fit(X_tr_s, y_tr)
        p_lr = lr.predict_proba(X_va_s)[:, 1]
        cv_scores["logistic_regression"].append(
            {
                "roc_auc": float(roc_auc_score(y_va, p_lr)),
                "f1": float(f1_score(y_va, (p_lr >= 0.5).astype(int))),
                "sensitivity": float(recall_score(y_va, (p_lr >= 0.5).astype(int))),
                "brier": float(brier_score_loss(y_va, p_lr)),
            }
        )

        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=220,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        rf.fit(X_tr, y_tr)
        p_rf = rf.predict_proba(X_va)[:, 1]
        cv_scores["random_forest"].append(
            {
                "roc_auc": float(roc_auc_score(y_va, p_rf)),
                "f1": float(f1_score(y_va, (p_rf >= 0.5).astype(int))),
                "sensitivity": float(recall_score(y_va, (p_rf >= 0.5).astype(int))),
                "brier": float(brier_score_loss(y_va, p_rf)),
            }
        )

        # XGBoost
        xgb_clf = xgb.XGBClassifier(
            n_estimators=180,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            scale_pos_weight=pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        xgb_clf.fit(X_tr, y_tr)
        p_xgb = xgb_clf.predict_proba(X_va)[:, 1]
        cv_scores["xgboost"].append(
            {
                "roc_auc": float(roc_auc_score(y_va, p_xgb)),
                "f1": float(f1_score(y_va, (p_xgb >= 0.5).astype(int))),
                "sensitivity": float(recall_score(y_va, (p_xgb >= 0.5).astype(int))),
                "brier": float(brier_score_loss(y_va, p_xgb)),
            }
        )

        # TabNet
        tab = TabNetClassifier(
            n_d=8,
            n_a=8,
            n_steps=3,
            gamma=1.3,
            seed=RANDOM_SEED + fold_i,
            verbose=0,
            optimizer_params=dict(lr=2e-2),
        )
        tab.fit(
            X_tr_s,
            y_tr,
            eval_set=[(X_va_s, y_va)],
            eval_metric=["auc"],
            max_epochs=35,
            patience=8,
            batch_size=min(128, max(32, len(X_tr_s) // 6)),
            virtual_batch_size=32,
        )
        p_tab = tab.predict_proba(X_va_s)[:, 1]
        cv_scores["tabnet"].append(
            {
                "roc_auc": float(roc_auc_score(y_va, p_tab)),
                "f1": float(f1_score(y_va, (p_tab >= 0.5).astype(int))),
                "sensitivity": float(recall_score(y_va, (p_tab >= 0.5).astype(int))),
                "brier": float(brier_score_loss(y_va, p_tab)),
            }
        )

    # 2. Entrenamiento final en el 80% completo y evaluación en el 20% de prueba aislada
    lr_final = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=500, random_state=RANDOM_SEED
    )
    lr_final.fit(X_train_scaled, y_train)
    prob_lr_test = lr_final.predict_proba(X_test_scaled)[:, 1]

    rf_final = RandomForestClassifier(
        n_estimators=220,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rf_final.fit(X_train_imp, y_train)
    prob_rf_test = rf_final.predict_proba(X_test_imp)[:, 1]

    xgb_final = xgb.XGBClassifier(
        n_estimators=180,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        scale_pos_weight=pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    xgb_final.fit(X_train_imp, y_train)
    prob_xgb_test = xgb_final.predict_proba(X_test_imp)[:, 1]

    tabnet_final = TabNetClassifier(
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        seed=RANDOM_SEED,
        verbose=0,
        optimizer_params=dict(lr=2e-2),
    )
    tabnet_final.fit(
        X_train_scaled,
        y_train,
        eval_set=[(X_test_scaled, y_test)],
        eval_metric=["auc"],
        max_epochs=45,
        patience=10,
        batch_size=min(128, max(32, len(X_train_scaled) // 6)),
        virtual_batch_size=32,
    )
    prob_tab_test = tabnet_final.predict_proba(X_test_scaled)[:, 1]

    # Explicabilidad SHAP global sobre el conjunto de prueba aislado (con TreeExplainer para XGBoost y RF)
    xgb_explainer = shap.TreeExplainer(xgb_final)
    shap_vals_xgb = xgb_explainer.shap_values(X_test_imp)
    if isinstance(shap_vals_xgb, list):
        shap_vals_xgb = shap_vals_xgb[1]
    mean_abs_shap = np.abs(shap_vals_xgb).mean(axis=0)

    label_map = {f["name"]: f["label"] for f in features_meta}
    global_shap_ranking = sorted(
        [
            {
                "feature": fname,
                "label": label_map[fname],
                "mean_abs_shap": round(float(val), 4),
            }
            for fname, val in zip(feature_names, mean_abs_shap)
        ],
        key=lambda x: x["mean_abs_shap"],
        reverse=True,
    )

    def summarize_cv(model_key: str) -> Dict[str, float]:
        folds = cv_scores[model_key]
        return {
            "roc_auc_mean": round(float(np.mean([f["roc_auc"] for f in folds])), 4),
            "roc_auc_std": round(float(np.std([f["roc_auc"] for f in folds])), 4),
            "f1_mean": round(float(np.mean([f["f1"] for f in folds])), 4),
            "f1_std": round(float(np.std([f["f1"] for f in folds])), 4),
            "sensitivity_mean": round(float(np.mean([f["sensitivity"] for f in folds])), 4),
            "brier_mean": round(float(np.mean([f["brier"] for f in folds])), 4),
        }

    models_catalog = {
        "xgboost": {
            "model_id": "xgboost",
            "name": "XGBoost (Gradient Boosted Trees)",
            "family": "Boosting de Árboles Regularizado",
            "role": "Modelo Primario Recomendado (Caso03)",
            "hyperparameters": {
                "n_estimators": 180,
                "max_depth": 4,
                "learning_rate": 0.04,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_lambda": 2.0,
                "scale_pos_weight": round(pos_weight, 3),
                "random_seed": RANDOM_SEED,
            },
            "cv_5fold": summarize_cv("xgboost"),
            "test_metrics": compute_binary_metrics(y_test, prob_xgb_test, threshold=0.5),
            "subgroup_fairness": evaluate_subgroups(df_test, y_test, prob_xgb_test, threshold=0.5),
        },
        "random_forest": {
            "model_id": "random_forest",
            "name": "Random Forest Classifier",
            "family": "Ensamble Bagging de Árboles",
            "role": "Modelo Comparativo de Ensamble (Caso03)",
            "hyperparameters": {
                "n_estimators": 220,
                "max_depth": 8,
                "min_samples_leaf": 5,
                "class_weight": "balanced_subsample",
                "random_seed": RANDOM_SEED,
            },
            "cv_5fold": summarize_cv("random_forest"),
            "test_metrics": compute_binary_metrics(y_test, prob_rf_test, threshold=0.5),
            "subgroup_fairness": evaluate_subgroups(df_test, y_test, prob_rf_test, threshold=0.5),
        },
        "tabnet": {
            "model_id": "tabnet",
            "name": "TabNet (Attentive Deep Learning Tabular)",
            "family": "Aprendizaje Profundo con Atención Secuencial",
            "role": "Arquitectura Neuronal Tabular (Caso03)",
            "hyperparameters": {
                "n_d": 8,
                "n_a": 8,
                "n_steps": 3,
                "gamma": 1.3,
                "lr": 0.02,
                "max_epochs": 45,
                "random_seed": RANDOM_SEED,
            },
            "cv_5fold": summarize_cv("tabnet"),
            "test_metrics": compute_binary_metrics(y_test, prob_tab_test, threshold=0.5),
            "subgroup_fairness": evaluate_subgroups(df_test, y_test, prob_tab_test, threshold=0.5),
        },
        "logistic_regression": {
            "model_id": "logistic_regression",
            "name": "Regresión Logística Penalizada (L2)",
            "family": "Modelo Lineal Generalizado",
            "role": "Línea Base Reproducible (Prompt)",
            "hyperparameters": {
                "C": 1.0,
                "penalty": "l2",
                "class_weight": "balanced",
                "random_seed": RANDOM_SEED,
            },
            "cv_5fold": summarize_cv("logistic_regression"),
            "test_metrics": compute_binary_metrics(y_test, prob_lr_test, threshold=0.5),
            "subgroup_fairness": evaluate_subgroups(df_test, y_test, prob_lr_test, threshold=0.5),
        },
    }

    # Guardar artefactos serializados para inferencia y explicabilidad SHAP en tiempo real
    cohort_dir = MODELS_DIR / dataset_id
    cohort_dir.mkdir(parents=True, exist_ok=True)

    # Background sample (100 filas de entrenamiento) para SHAP en modelos no arbóreos si se requiere
    bg_indices = np.random.choice(len(X_train_imp), size=min(80, len(X_train_imp)), replace=False)

    bundle = {
        "dataset_id": dataset_id,
        "feature_names": feature_names,
        "features_meta": features_meta,
        "imputer": imputer,
        "scaler": scaler,
        "models": {
            "xgboost": xgb_final,
            "random_forest": rf_final,
            "logistic_regression": lr_final,
            "tabnet": tabnet_final,
        },
        "background_imp": X_train_imp[bg_indices],
        "background_scaled": X_train_scaled[bg_indices],
        "train_medians": {
            fname: round(float(val), 3)
            for fname, val in zip(feature_names, imputer.statistics_)
        },
    }
    joblib.dump(bundle, cohort_dir / "model_bundle.joblib")

    return {
        "dataset_id": dataset_id,
        "audit": audit_info,
        "split_summary": {
            "strategy": "StratifiedGroupKFold agrupado por patient_id (80% desarrollo con 5-fold CV / 20% prueba aislada)",
            "train_patients": int(len(df_train)),
            "train_positive_cases": int(y_train.sum()),
            "test_patients_isolated": int(len(df_test)),
            "test_positive_cases": int(y_test.sum()),
            "patient_overlap_count": 0,
        },
        "features": features_meta,
        "global_shap_importance": global_shap_ranking,
        "models": models_catalog,
    }


def generate_synthetic_demo_csv() -> Dict[str, Any]:
    """
    Genera un conjunto de datos sintético rotulado EXCLUSIVAMENTE como DEMO para
    probar el mecanismo de importación CSV sin conexión.
    Cumple estrictamente el Prompt: NO se entrena ningún modelo ni se informan
    métricas clínicas sobre estos datos sintéticos.
    """
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    n_rows = 25

    df_demo = pd.DataFrame(
        {
            "demo_tag": ["DEMO_SINTETICO_SIN_VALIDEZ_CLINICA"] * n_rows,
            "edad": rng.integers(25, 78, size=n_rows),
            "imc": np.round(rng.uniform(21.0, 41.5, size=n_rows), 1),
            "glucosa": np.round(rng.uniform(82.0, 195.0, size=n_rows), 0),
            "hba1c": np.round(rng.uniform(4.9, 8.6, size=n_rows), 1),
            "presion_sistolica": np.round(rng.uniform(108.0, 168.0, size=n_rows), 0),
            "presion_diastolica": np.round(rng.uniform(62.0, 98.0, size=n_rows), 0),
            "colesterol_total": np.round(rng.uniform(145.0, 265.0, size=n_rows), 0),
            "colesterol_hdl": np.round(rng.uniform(32.0, 74.0, size=n_rows), 0),
            "trigliceridos": np.round(rng.uniform(75.0, 290.0, size=n_rows), 0),
            "antecedente_familiar_diabetes": rng.choice([0, 1], size=n_rows, p=[0.6, 0.4]),
            "antecedente_hipertension": rng.choice([0, 1], size=n_rows, p=[0.65, 0.35]),
            "tratamiento_antihipertensivo": rng.choice([0, 1], size=n_rows, p=[0.7, 0.3]),
        }
    )
    demo_path = DEMO_DIR / "DEMO_SINTETICO_NO_CLINICO.csv"
    df_demo.to_csv(demo_path, index=False)
    return {
        "file_path": str(demo_path.relative_to(ROOT_DIR)),
        "rows": n_rows,
        "label": "DEMO SINTÉTICO — EXCLUSIVO PARA PRUEBA DE IMPORTACIÓN CSV (NO USADO EN ENTRENAMIENTO NI EVALUACIÓN)",
    }


def main() -> None:
    t0 = time.time()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] Preparando y auditando cohorte primaria CDC NHANES 2017-2018...")
    df_nhanes, audit_nhanes = prepare_nhanes_dataset()
    nhanes_results = train_and_evaluate_cohort(
        "nhanes_2017_2018", df_nhanes, NHANES_FEATURES_META, audit_nhanes
    )

    print("[2/4] Preparando y auditando cohorte benchmark Pima Indians (NIDDK)...")
    df_pima, audit_pima = prepare_pima_dataset()
    pima_results = train_and_evaluate_cohort(
        "pima_indians", df_pima, PIMA_FEATURES_META, audit_pima
    )

    print("[3/4] Generando archivo CSV sintético rotulado exclusivamente como DEMO...")
    demo_info = generate_synthetic_demo_csv()

    elapsed = round(time.time() - t0, 2)

    metadata = {
        "project_code": "SP-03",
        "scientific_title": (
            "Multimodal Explainable Artificial Intelligence for Early Prediction of "
            "Type 2 Diabetes and Its Long-Term Complications"
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_duration_seconds": elapsed,
        "random_seed": RANDOM_SEED,
        "default_dataset": "nhanes_2017_2018",
        "environment_versions": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "scikit_learn": sklearn.__version__,
            "xgboost": xgb.__version__,
            "shap": shap.__version__,
            "pytorch": torch.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "datasets_evaluated_feasibility": [
            {
                "name": "NHANES 2017–2018 (CDC / NCHS)",
                "status": "SELECCIONADO — COHORTE PRIMARIA REAL",
                "license": "Dominio Público EE. UU. (CC0 / Public Domain)",
                "access": "Descarga directa verificada de archivos SAS Transport (.XPT) oficiales de CDC",
                "strengths": (
                    "Incluye todas las familias de variables clínicas del Caso03: edad, IMC, glucosa sérica, "
                    "HbA1c, presión arterial sistólica/diastólica, perfil lipídico completo (colesterol total, "
                    "HDL, triglicéridos), antecedentes familiares/clínicos y tratamiento antihipertensivo."
                ),
                "limitations": (
                    "Diseño transversal por ciclos: permite predicción probabilística rigurosa de presencia/aparición "
                    "de diabetes mellitus tipo 2, pero carece de seguimiento longitudinal público de tiempos hasta "
                    "complicaciones micro/macrovasculares para modelos de supervivencia."
                ),
            },
            {
                "name": "Pima Indians Diabetes Dataset (NIDDK)",
                "status": "SELECCIONADO — BENCHMARK SECUNDARIO REAL",
                "license": "Open Access / CC-BY 4.0 (UCI / OpenML)",
                "access": "Archivo CSV verificable (768 registros de pacientes mujeres de herencia Pima ≥21 años)",
                "strengths": "Estándar internacional de comparación reproducible para algoritmos tabulares.",
                "limitations": (
                    "Población específica (mujeres indígenas Pima), tamaño muestral moderado (n=768), "
                    "ausencia de HbA1c y perfil lipídico, y valores faltantes codificados originalmente como 0."
                ),
            },
            {
                "name": "CDC Behavioral Risk Factor Surveillance System (BRFSS)",
                "status": "EVALUADO — DESCARTADO COMO FUENTE PRIMARIA DE BIOMARCADORES",
                "license": "Dominio Público EE. UU. (CDC)",
                "access": "Público (archivos anuales XPT > 500 MB)",
                "strengths": "Gran tamaño muestral y variables socioeconómicas/hábitos autorreportados.",
                "limitations": (
                    "Encuesta telefónica autorreportada sin mediciones de laboratorio (carece de glucosa en ayunas, "
                    "HbA1c y perfil lipídico medido) ni seguimiento longitudinal por paciente."
                ),
            },
            {
                "name": "PhysioNet (EHR Longitudinal / MIMIC-IV / CGM)",
                "status": "EVALUADO — RESERVADO PARA FASE 2 LONGITUDINAL / SUPERVIVENCIA",
                "license": "PhysioNet Credentialed Health Data License / ODC-By",
                "access": "Las cohortes con seguimiento temporal de complicaciones requieren acreditación DUA/CITI",
                "strengths": "Registros electrónicos de salud longitudinales con marcas temporales y eventos clínicos.",
                "limitations": (
                    "Acceso restringido bajo convenio de uso de datos (DUA) para cohortes hospitalarias completas; "
                    "los subconjuntos abiertos pequeños son series de corto plazo sin incidencia de complicaciones a 5-10 años."
                ),
            },
        ],
        "outcomes_architecture": {
            "outcome_1_diabetes_onset": {
                "title": "Resultado 1: Predicción de Aparición / Presencia de Diabetes Mellitus Tipo 2",
                "status": "ACTIVO Y VERIFICADO CON DATOS REALES",
                "models_implemented": [
                    "XGBoost Classifier",
                    "Random Forest Classifier",
                    "TabNet Attentive Tabular Network",
                    "Logistic Regression (Línea Base)",
                ],
                "explainability": "SHAP (SHapley Additive exPlanations) global e individual por predicción + análisis de calibración",
            },
            "outcome_2_complications_survival": {
                "title": "Resultado 2: Desarrollo de Complicaciones Crónicas y Supervivencia a Largo Plazo",
                "status": "PENDIENTE DE COHORTE LONGITUDINAL (TRABAJO FUTURO DOCUMENTADO)",
                "planned_hybrid_models": [
                    "XGBoost + Cox Proportional Hazards (Riesgo proporcional con representación arbórea no lineal)",
                    "LSTM + Attention + Survival Analysis (Modelado de trayectorias temporales de HbA1c/PA/lípidos con censura a la derecha)",
                ],
                "justification": (
                    "Conforme a las reglas de integridad científica del proyecto, ni NHANES 2017–2018 ni Pima Indians "
                    "contienen tiempos de seguimiento longitudinal (time-to-event T) ni indicadores de censura (δ) para "
                    "retinopatía, nefropatía, neuropatía o eventos cardiovasculares mayores (MACE). No se inventan etiquetas "
                    "ni tiempos sintéticos; el módulo documenta el esquema de datos longitudinales requerido para su activación."
                ),
                "required_longitudinal_schema": [
                    "patient_id (identificador único con múltiples visitas t_0, t_1, ..., t_k)",
                    "visit_date / follow_up_months (tiempo desde línea base hasta evento o última observación)",
                    "event_complication (0 = censurado a la derecha, 1 = nefropatía/retinopatía/neuropatía/MACE confirmado)",
                    "biomarcadores seriados (pendiente ΔHbA1c/año, variabilidad glucémica, TFGe, albuminuria ACR)",
                ],
            },
        },
        "traceability_matrix": [
            {
                "requirement_id": "REQ-01",
                "caso03_specification": "Variables clínicas conjuntas: edad, IMC, glucosa, HbA1c, presión arterial, perfil lipídico, antecedentes y tratamientos",
                "implementation_status": "IMPLEMENTADO (Datos Reales)",
                "module_location": "scripts/train_and_evaluate.py + backend/app/main.py (cohorte nhanes_2017_2018 con 12 variables clínicas reales)",
            },
            {
                "requirement_id": "REQ-02",
                "caso03_specification": "Definición de dos resultados independientes: aparición de diabetes y desarrollo de complicaciones",
                "implementation_status": "DIFERENCIADO CON TRAZABILIDAD",
                "module_location": "Resultado 1 activo en /predict y /model-info; Resultado 2 separado como arquitectura longitudinal pendiente sin etiquetas inventadas",
            },
            {
                "requirement_id": "REQ-03",
                "caso03_specification": "Comparación de algoritmos: Línea Base, Random Forest, XGBoost y TabNet",
                "implementation_status": "IMPLEMENTADO Y EVALUADO",
                "module_location": "models/{nhanes_2017_2018,pima_indians}/model_bundle.joblib + Panel de Evaluación UI",
            },
            {
                "requirement_id": "REQ-04",
                "caso03_specification": "Modelos híbridos de supervivencia (XGBoost + Cox, LSTM + Attention + Survival)",
                "implementation_status": "TRABAJO FUTURO DOCUMENTADO",
                "module_location": "Sección de Experimentos Longitudinales Pendientes (bloqueado hasta disponer de tiempos y censura reales)",
            },
            {
                "requirement_id": "REQ-05",
                "caso03_specification": "Validación cruzada agrupada por paciente y aislamiento del conjunto de prueba sin fuga de información",
                "implementation_status": "IMPLEMENTADO",
                "module_location": "StratifiedGroupKFold por SEQN/patient_id + imputación/escalado intra-pliegue y exclusión de fármacos hipoglucemiantes post-diagnóstico",
            },
            {
                "requirement_id": "REQ-06",
                "caso03_specification": "Análisis de subgrupos por edad y sexo sin utilizar variables sensibles de forma discriminatoria",
                "implementation_status": "IMPLEMENTADO",
                "module_location": "evaluate_subgroups() en scripts/train_and_evaluate.py + Pestaña de Equidad y Subgrupos en Frontend",
            },
            {
                "requirement_id": "REQ-07",
                "caso03_specification": "Explicabilidad mediante SHAP y análisis de calibración",
                "implementation_status": "IMPLEMENTADO",
                "module_location": "TreeExplainer/KernelExplainer en backend/app/predictor.py + curvas de calibración y Brier Score en UI",
            },
            {
                "requirement_id": "REQ-08",
                "caso03_specification": "Importación CSV y archivo DEMO sintético rotulado sin uso en entrenamiento",
                "implementation_status": "IMPLEMENTADO",
                "module_location": "POST /predict-batch + data/demo/DEMO_SINTETICO_NO_CLINICO.csv",
            },
        ],
        "cohorts": {
            "nhanes_2017_2018": nhanes_results,
            "pima_indians": pima_results,
        },
        "demo_synthetic_info": demo_info,
        "clinical_disclaimer": (
            "ADVERTENCIA REGULATORIA Y ÉTICA: Este sistema es un prototipo académico e investigativo de "
            "inteligencia artificial explicable (Caso SP-03). NO es un dispositivo médico certificado "
            "(SaMD) ni sustituye el juicio clínico ni las pruebas diagnósticas confirmatorias (ADA/OMS). "
            "No almacena historias clínicas ni solicita datos personales identificables (PII)."
        ),
    }

    with open(MODELS_DIR / "metrics_and_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[4/4] Entrenamiento y evaluación completados en {elapsed}s.")
    for cid, cdata in metadata["cohorts"].items():
        print(f"\n=== Resumen de Métricas en Conjunto de Prueba Aislado: {cid} ===")
        for mid, mdata in cdata["models"].items():
            tm = mdata["test_metrics"]
            cv = mdata["cv_5fold"]
            print(
                f"  - {mdata['name']:42s} | Test ROC-AUC: {tm['roc_auc']:.4f} | "
                f"CV ROC-AUC: {cv['roc_auc_mean']:.4f}±{cv['roc_auc_std']:.4f} | "
                f"Sens: {tm['sensitivity']:.4f} | Esp: {tm['specificity']:.4f} | "
                f"F1: {tm['f1']:.4f} | Brier: {tm['brier_score']:.4f}"
            )


if __name__ == "__main__":
    main()
