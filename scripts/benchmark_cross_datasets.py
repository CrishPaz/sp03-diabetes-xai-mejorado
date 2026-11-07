#!/usr/bin/env python3
"""
Benchmark Comparativo de Modelos de Machine Learning sobre Cohortes Reales:
1. CDC NHANES 2017–2018 (Biomarcadores Clínicos de Laboratorio HPLC, N=4,829)
2. NIDDK Pima Indians (Benchmark Histórico, N=768)
3. CDC BRFSS 2015 (Salud Poblacional y Estilo de Vida, N=70,692)
4. Diabetes 130-US Hospitals (Complicaciones y Readmisión Prematura, N=71,515 pacientes únicos)
5. Early Stage Diabetes Risk (Sintomatología Prodrómica, N=520)

Algoritmos comparados por cohorte:
- Random Forest Classifier
- XGBoost Classifier
- Logistic Regression (Baseline Lineal Regularizado)
- Stacking Ensemble (Híbrido Calibrado)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
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
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_cohort(
    name: str,
    description: str,
    category: str,
    target_name: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"Evaluando Cohorte: {name} ({category})")
    print(f"Muestra N = {X.shape[0]}, Features = {X.shape[1]}, Positivos = {int(y.sum())} ({y.mean():.2%})")
    print(f"=======================================================")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scale_pos = float((len(y) - y.sum()) / max(y.sum(), 1))

    # Definir modelos
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=6, class_weight="balanced", random_state=random_state, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05, scale_pos_weight=scale_pos,
            random_state=random_state, n_jobs=-1, eval_metric="logloss"
        ),
        "Regresion Logistica": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=random_state
        ),
        "Stacking Hibrido": StackingClassifier(
            estimators=[
                ("rf", RandomForestClassifier(n_estimators=60, max_depth=5, class_weight="balanced", random_state=random_state, n_jobs=-1)),
                ("xgb", XGBClassifier(n_estimators=60, max_depth=3, learning_rate=0.05, scale_pos_weight=scale_pos, random_state=random_state, n_jobs=-1, eval_metric="logloss")),
            ],
            final_estimator=LogisticRegression(),
            cv=3,
            n_jobs=-1,
        ),
    }

    results = {}

    for model_name, model in models.items():
        t0 = time.time()
        fold_aucs: List[float] = []
        fold_praucs: List[float] = []
        fold_accs: List[float] = []
        fold_baccs: List[float] = []
        fold_f1s: List[float] = []
        fold_recalls: List[float] = []
        fold_specificities: List[float] = []
        fold_briers: List[float] = []

        for train_idx, test_idx in skf.split(X, y):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            # Imputación y estandarización dentro de cada pliegue
            imputer = SimpleImputer(strategy="median")
            scaler = StandardScaler()

            X_tr_proc = scaler.fit_transform(imputer.fit_transform(X_tr))
            X_te_proc = scaler.transform(imputer.transform(X_te))

            # Ajustar modelo
            model.fit(X_tr_proc, y_tr)
            probs = model.predict_proba(X_te_proc)[:, 1]
            preds = (probs >= 0.5).astype(int)

            # Métricas
            auc = roc_auc_score(y_te, probs)
            prauc = average_precision_score(y_te, probs)
            acc = accuracy_score(y_te, preds)
            bacc = balanced_accuracy_score(y_te, preds)
            f1 = f1_score(y_te, preds, zero_division=0)
            recall = recall_score(y_te, preds, zero_division=0)
            
            tn, fp, fn, tp = confusion_matrix(y_te, preds, labels=[0, 1]).ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            brier = brier_score_loss(y_te, probs)

            fold_aucs.append(float(auc))
            fold_praucs.append(float(prauc))
            fold_accs.append(float(acc))
            fold_baccs.append(float(bacc))
            fold_f1s.append(float(f1))
            fold_recalls.append(float(recall))
            fold_specificities.append(float(spec))
            fold_briers.append(float(brier))

        t1 = time.time()
        res_summary = {
            "roc_auc_mean": round(float(np.mean(fold_aucs)), 4),
            "roc_auc_std": round(float(np.std(fold_aucs)), 4),
            "pr_auc_mean": round(float(np.mean(fold_praucs)), 4),
            "pr_auc_std": round(float(np.std(fold_praucs)), 4),
            "accuracy_mean": round(float(np.mean(fold_accs)), 4),
            "balanced_accuracy_mean": round(float(np.mean(fold_baccs)), 4),
            "f1_mean": round(float(np.mean(fold_f1s)), 4),
            "sensitivity_mean": round(float(np.mean(fold_recalls)), 4),
            "specificity_mean": round(float(np.mean(fold_specificities)), 4),
            "brier_score_mean": round(float(np.mean(fold_briers)), 4),
            "fit_time_seconds": round(t1 - t0, 2),
        }
        results[model_name] = res_summary
        print(f"  -> {model_name:20s} | ROC-AUC: {res_summary['roc_auc_mean']:.4f} ± {res_summary['roc_auc_std']:.4f} | PR-AUC: {res_summary['pr_auc_mean']:.4f} | F1: {res_summary['f1_mean']:.4f} | Tiempo: {res_summary['fit_time_seconds']}s")

    return {
        "cohort_name": name,
        "category": category,
        "description": description,
        "target_name": target_name,
        "sample_size": int(X.shape[0]),
        "num_features": int(X.shape[1]),
        "positive_cases": int(y.sum()),
        "prevalence": round(float(y.mean()), 4),
        "feature_names": feature_names,
        "models": results,
    }


def main():
    benchmarks = {}

    # 1. CDC NHANES 2017-2018 (Laboratorio HPLC)
    nhanes_path = PROCESSED_DIR / "nhanes_2017_2018_dataset_completo.csv"
    if nhanes_path.exists():
        df_n = pd.read_csv(nhanes_path)
        feat_n = [
            "edad", "imc", "glucosa", "hba1c", "presion_sistolica", "presion_diastolica",
            "colesterol_total", "colesterol_hdl", "trigliceridos",
            "antecedente_familiar_diabetes", "antecedente_hipertension", "tratamiento_antihipertensivo"
        ]
        X_n = df_n[feat_n].values
        y_n = df_n["outcome"].values
        benchmarks["CDC_NHANES_2017_2018"] = evaluate_cohort(
            name="CDC NHANES 2017–2018",
            description="Examen clínico con biomarcadores de laboratorio HPLC (HbA1c, glucosa basal, perfil lipídico).",
            category="Biomarcadores de Laboratorio",
            target_name="Diagnóstico DM2 confirmado",
            X=X_n,
            y=y_n,
            feature_names=feat_n,
        )

    # 2. NIDDK Pima Indians
    pima_path = PROCESSED_DIR / "pima_indians_dataset_completo.csv"
    if pima_path.exists():
        df_p = pd.read_csv(pima_path)
        feat_p = [
            "embarazos", "glucosa", "presion_diastolica", "pliegue_cutaneo",
            "insulina", "imc", "pedigri_diabetes", "edad"
        ]
        X_p = df_p[feat_p].values
        y_p = df_p["outcome"].values
        benchmarks["NIDDK_Pima_Indians"] = evaluate_cohort(
            name="NIDDK Pima Indians",
            description="Benchmark clásico biomédico de mujeres indígenas Pima evaluadas con OGTT.",
            category="Benchmark Histórico",
            target_name="Diagnóstico DM2 a 5 años",
            X=X_p,
            y=y_p,
            feature_names=feat_p,
        )

    # 3. CDC BRFSS 2015 (50/50 split, N=70,692)
    brfss_path = PROCESSED_DIR / "cdc_brfss_5050_clean.csv"
    if brfss_path.exists():
        df_b = pd.read_csv(brfss_path)
        feat_b = [c for c in df_b.columns if c != "target"]
        X_b = df_b[feat_b].values
        y_b = df_b["target"].values
        benchmarks["CDC_BRFSS_2015"] = evaluate_cohort(
            name="CDC BRFSS 2015",
            description="Tamizaje de salud poblacional y estilo de vida (Big Data de 70,692 casos balanceados).",
            category="Salud Poblacional y Estilo de Vida",
            target_name="Diabetes / Prediabetes Diagnosticada",
            X=X_b,
            y=y_b,
            feature_names=feat_b,
        )

    # 4. Diabetes 130-US Hospitals (UCI 296, N=71,515 pacientes únicos)
    hosp_path = PROCESSED_DIR / "diabetes_130_hospitals_clean.csv"
    if hosp_path.exists():
        df_h = pd.read_csv(hosp_path)
        feat_h = [
            "gender_male", "age_numeric", "time_in_hospital", "num_lab_procedures",
            "num_procedures", "num_medications", "number_outpatient", "number_emergency",
            "number_inpatient", "number_diagnoses", "insulin_treatment",
            "diabetes_med_prescribed", "medication_changed", "a1c_tested", "a1c_elevated"
        ]
        X_h = df_h[feat_h].values
        y_h = df_h["target"].values
        benchmarks["Diabetes_130_Hospitals"] = evaluate_cohort(
            name="Diabetes 130-US Hospitals (UCI 296)",
            description="Registros hospitalarios de pacientes únicos con descompensación diabética y farmacoterapia.",
            category="Complicaciones y Reingreso Hospitalario",
            target_name="Reingreso prematuro (< 30 días)",
            X=X_h,
            y=y_h,
            feature_names=feat_h,
        )

    # 5. Early Stage Diabetes Risk (UCI 529 - Sylhet)
    early_path = PROCESSED_DIR / "early_stage_diabetes_sylhet_clean.csv"
    if early_path.exists():
        df_e = pd.read_csv(early_path)
        feat_e = [c for c in df_e.columns if c != "target"]
        X_e = df_e[feat_e].values
        y_e = df_e["target"].values
        benchmarks["Early_Stage_Diabetes_Sylhet"] = evaluate_cohort(
            name="Early Stage Diabetes (UCI 529 - Sylhet)",
            description="Detección temprana en atención primaria basada en sintomatología prodrómica (poliuria, polidipsia).",
            category="Sintomatología Prodrómica",
            target_name="Diabetes Confirmada",
            X=X_e,
            y=y_e,
            feature_names=feat_e,
        )

    # Guardar resultados en JSON
    json_path = MODELS_DIR / "benchmark_cross_dataset_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(benchmarks, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Benchmark multiplataforma guardado en: {json_path}")

    # Generar tabla markdown comparativa
    md_path = MODELS_DIR / "BENCHMARK_COMPARATIVO_DATASETS.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Benchmark Científico Comparativo de Modelos sobre Cohortes Reales\n\n")
        f.write("Resultados obtenidos mediante **5-Fold Stratified Cross-Validation** con preprocesamiento anti-fuga dentro de cada pliegue.\n\n")
        f.write("| Cohorte / Dataset | Muestra ($N$) | Tipo de Variables | Algoritmo | ROC-AUC | PR-AUC | F1-Score | Sensibilidad | Especificidad | Brier Score |\n")
        f.write("| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")

        for ckey, cdata in benchmarks.items():
            first = True
            for mname, mdata in cdata["models"].items():
                c_col = f"**{cdata['cohort_name']}**<br>*(Prev: {cdata['prevalence']*100:.1f}%)*" if first else ""
                n_col = f"{cdata['sample_size']}" if first else ""
                t_col = f"{cdata['category']}" if first else ""
                first = False
                f.write(
                    f"| {c_col} | {n_col} | {t_col} | **{mname}** | "
                    f"{mdata['roc_auc_mean']:.4f} ± {mdata['roc_auc_std']:.3f} | "
                    f"{mdata['pr_auc_mean']:.4f} | "
                    f"{mdata['f1_mean']:.4f} | "
                    f"{mdata['sensitivity_mean']:.4f} | "
                    f"{mdata['specificity_mean']:.4f} | "
                    f"{mdata['brier_score_mean']:.4f} |\n"
                )
    print(f"✅ Reporte markdown guardado en: {md_path}")


if __name__ == "__main__":
    main()
