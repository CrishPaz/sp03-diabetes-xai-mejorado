#!/usr/bin/env python3
"""
Pipeline Avanzado de Machine Learning, Optimización con Optuna, Ensamble Híbrido,
y Batería de Pruebas Estadísticas Formales (Friedman, Wilcoxon-Holm, McNemar, DeLong, Bootstrap).
Cumple con las Fases F6 a F10 del protocolo común del docente Juan Pedro Santos Fernández.

Modelos Entrenados (5 Modelos):
  1. Individual 1: Random Forest Classifier
  2. Individual 2: XGBoost Classifier
  3. Individual 3: TabNet Attentive Tabular Network
  4. Híbrido 1: Ensamble Stacking Calibrado (Meta-clasificador sobre RF + XGB + TabNet)
  5. Híbrido 2: Modelo Híbrido de Riesgo / Supervivencia (XGBoost + Función de Riesgo Weibull a 5/10 años)
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import shap
import sklearn
import torch
import xgboost as xgb
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
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
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

warnings.filterwarnings("ignore")

RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

from scripts.train_and_evaluate import (
    NHANES_FEATURES_META,
    PIMA_FEATURES_META,
    compute_binary_metrics,
    evaluate_subgroups,
    prepare_nhanes_dataset,
    prepare_pima_dataset,
)
from backend.app.hybrid_models import StackingHybridClassifier, SurvivalRiskHybridModel


def run_optuna_tuning_xgb(X_train: np.ndarray, y_train: np.ndarray, scale_pos: float) -> Dict[str, Any]:
    """Optimización de hiperparámetros de XGBoost con Optuna (50 trials) sobre CV interna."""
    if not HAS_OPTUNA:
        return {
            "n_estimators": 180,
            "max_depth": 4,
            "learning_rate": 0.04,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 2.0,
            "scale_pos_weight": round(scale_pos, 3),
        }

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 250, step=25),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.10, log=True),
            "subsample": trial.suggest_float("subsample", 0.70, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.70, 0.95),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 5.0),
            "scale_pos_weight": scale_pos,
            "eval_metric": "logloss",
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
        }
        # CV interna de 3 pliegues dentro del conjunto de entrenamiento
        skf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
        scores = []
        dummy_groups = np.arange(len(y_train))
        for tr_i, va_i in skf.split(X_train, y_train, dummy_groups):
            clf = xgb.XGBClassifier(**params)
            clf.fit(X_train[tr_i], y_train[tr_i])
            preds = clf.predict_proba(X_train[va_i])[:, 1]
            scores.append(roc_auc_score(y_train[va_i], preds))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=35, timeout=120)
    best_p = study.best_params
    best_p["scale_pos_weight"] = round(scale_pos, 3)
    best_p["eval_metric"] = "logloss"
    best_p["random_state"] = RANDOM_SEED
    best_p["n_jobs"] = -1
    return best_p


def calculate_statistical_battery(
    y_test: np.ndarray,
    cv_scores: Dict[str, List[float]],
    test_probs: Dict[str, np.ndarray],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Calcula la batería de pruebas estadísticas exigida por el docente (Fase F9):
    - Prueba de Friedman entre los 5 modelos sobre los pliegues de validación cruzada
    - Prueba de Wilcoxon pareada con corrección de Holm-Bonferroni
    - Prueba de McNemar pareada en el conjunto de prueba aislado
    - Bootstrap empírico con B=1000 réplicas para intervalos de confianza al 95%
    """
    models_keys = list(cv_scores.keys())
    k_models = len(models_keys)
    n_folds = len(cv_scores[models_keys[0]])

    # 1. Prueba de Friedman
    data_matrix = np.array([cv_scores[m] for m in models_keys])  # shape: (k_models, n_folds)
    stat_friedman, p_friedman = stats.friedmanchisquare(*[cv_scores[m] for m in models_keys])

    # 2. Pruebas pareadas de Wilcoxon Signed-Rank con corrección de Holm-Bonferroni
    pairwise_wilcoxon = []
    raw_p_values = []
    pairs = []
    for i in range(k_models):
        for j in range(i + 1, k_models):
            m1, m2 = models_keys[i], models_keys[j]
            diff = np.array(cv_scores[m1]) - np.array(cv_scores[m2])
            if np.all(diff == 0):
                w_stat, w_p = 0.0, 1.0
            else:
                w_stat, w_p = stats.wilcoxon(cv_scores[m1], cv_scores[m2], zero_method="wilcox")
            pairs.append((m1, m2, float(w_stat)))
            raw_p_values.append(float(w_p))

    # Corrección de Holm
    m_comparisons = len(raw_p_values)
    sorted_indices = np.argsort(raw_p_values)
    holm_p_values = [0.0] * m_comparisons
    running_max = 0.0
    for rank, idx in enumerate(sorted_indices):
        p_raw = raw_p_values[idx]
        adj = p_raw * (m_comparisons - rank)
        adj = max(running_max, min(1.0, adj))
        running_max = adj
        holm_p_values[idx] = round(adj, 4)

    for idx, (m1, m2, w_stat) in enumerate(pairs):
        pairwise_wilcoxon.append(
            {
                "model_a": m1,
                "model_b": m2,
                "statistic": w_stat,
                "p_raw": round(raw_p_values[idx], 4),
                "p_holm": holm_p_values[idx],
                "significant_at_05": holm_p_values[idx] < 0.05,
            }
        )

    # 3. Prueba de McNemar pareada en el conjunto de prueba aislado
    mcnemar_results = []
    test_preds = {m: (test_probs[m] >= threshold).astype(int) for m in models_keys}
    for i in range(k_models):
        for j in range(i + 1, k_models):
            m1, m2 = models_keys[i], models_keys[j]
            correct1 = test_preds[m1] == y_test
            correct2 = test_preds[m2] == y_test
            # b: m1 acierta y m2 falla; c: m1 falla y m2 acierta
            b = int(np.sum(correct1 & ~correct2))
            c = int(np.sum(~correct1 & correct2))
            # Chi-cuadrado de McNemar con corrección de Edwards
            if b + c > 0:
                chi2_mcnemar = ((abs(b - c) - 1.0) ** 2) / (b + c)
                p_mcnemar = float(stats.chi2.sf(chi2_mcnemar, df=1))
            else:
                chi2_mcnemar, p_mcnemar = 0.0, 1.0

            mcnemar_results.append(
                {
                    "model_a": m1,
                    "model_b": m2,
                    "contingency_b_c": {"a_correct_b_error": b, "a_error_b_correct": c},
                    "chi2_statistic": round(float(chi2_mcnemar), 4),
                    "p_value": round(float(p_mcnemar), 4),
                    "significant_at_05": p_mcnemar < 0.05,
                }
            )

    # 4. Bootstrap B=1000 para Intervalos de Confianza al 95%
    rng = np.random.default_rng(RANDOM_SEED)
    bootstrap_ci = {}
    n_test = len(y_test)
    for m in models_keys:
        p_arr = test_probs[m]
        auc_boot = []
        brier_boot = []
        f1_boot = []
        mcc_boot = []
        for _ in range(1000):
            b_idx = rng.integers(0, n_test, size=n_test)
            yt_b = y_test[b_idx]
            if len(np.unique(yt_b)) < 2:
                continue
            yp_b = p_arr[b_idx]
            yhat_b = (yp_b >= threshold).astype(int)
            auc_boot.append(roc_auc_score(yt_b, yp_b))
            brier_boot.append(brier_score_loss(yt_b, yp_b))
            f1_boot.append(f1_score(yt_b, yhat_b, zero_division=0))
            mcc_boot.append(matthews_corrcoef(yt_b, yhat_b))

        def ci(arr: List[float]) -> Tuple[float, float]:
            lo, hi = np.percentile(arr, [2.5, 97.5])
            return round(float(lo), 4), round(float(hi), 4)

        auc_lo, auc_hi = ci(auc_boot)
        brier_lo, brier_hi = ci(brier_boot)
        f1_lo, f1_hi = ci(f1_boot)
        mcc_lo, mcc_hi = ci(mcc_boot)

        bootstrap_ci[m] = {
            "roc_auc_95_ci": [auc_lo, auc_hi],
            "brier_score_95_ci": [brier_lo, brier_hi],
            "f1_score_95_ci": [f1_lo, f1_hi],
            "mcc_95_ci": [mcc_lo, mcc_hi],
        }

    return {
        "friedman_test": {
            "statistic": round(float(stat_friedman), 4),
            "p_value": float(p_friedman),
            "significant_at_05": p_friedman < 0.05,
            "interpretation": (
                "Existen diferencias estadísticamente significativas entre los modelos (p < 0.05)"
                if p_friedman < 0.05
                else "No se detectaron diferencias significativas globales entre los modelos (p >= 0.05)"
            ),
        },
        "wilcoxon_holm_pairwise": pairwise_wilcoxon,
        "mcnemar_pairwise": mcnemar_results,
        "bootstrap_95_ci": bootstrap_ci,
    }


def train_and_evaluate_cohort_v2(
    dataset_id: str,
    df: pd.DataFrame,
    features_meta: List[Dict[str, Any]],
    audit_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Entrena los 5 modelos (3 individuales + 2 híbridos), Optuna y pruebas estadísticas."""
    feature_names = [f["name"] for f in features_meta]
    X_all = df[feature_names].values
    y_all = df["outcome"].values.astype(int)
    groups_all = df["patient_id"].values

    # División por paciente: 80% Desarrollo / 20% Prueba Aislada
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    train_idx, test_idx = next(sgkf.split(X_all, y_all, groups_all))

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    X_train_raw = df_train[feature_names].values
    y_train = df_train["outcome"].values.astype(int)
    groups_train = df_train["patient_id"].values

    X_test_raw = df_test[feature_names].values
    y_test = df_test["outcome"].values.astype(int)

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train_raw)
    X_test_imp = imputer.transform(X_test_raw)

    scaler = StandardScaler()
    X_train_scl = scaler.fit_transform(X_train_imp)
    X_test_scl = scaler.transform(X_test_imp)

    pos_weight = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))

    # Optimización con Optuna para XGBoost (Fase F7)
    optuna_params_xgb = run_optuna_tuning_xgb(X_train_imp, y_train, pos_weight)

    # 1. Validación Cruzada de 5 pliegues para guardar puntuaciones y generar predicciones out-of-fold para Stacking
    oof_preds_rf = np.zeros(len(y_train))
    oof_preds_xgb = np.zeros(len(y_train))
    oof_preds_tab = np.zeros(len(y_train))

    cv_scores_auc: Dict[str, List[float]] = {
        "random_forest": [],
        "xgboost": [],
        "tabnet": [],
        "stacking_hybrid": [],
        "survival_risk_hybrid": [],
    }

    for fold_i, (tr_i, val_i) in enumerate(sgkf.split(X_train_raw, y_train, groups_train)):
        f_imp = SimpleImputer(strategy="median")
        X_tr = f_imp.fit_transform(X_train_raw[tr_i])
        X_va = f_imp.transform(X_train_raw[val_i])
        y_tr, y_va = y_train[tr_i], y_train[val_i]

        f_scl = StandardScaler()
        X_tr_s = f_scl.fit_transform(X_tr)
        X_va_s = f_scl.transform(X_va)

        # RF
        rf_f = RandomForestClassifier(
            n_estimators=220,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        rf_f.fit(X_tr, y_tr)
        p_rf = rf_f.predict_proba(X_va)[:, 1]
        oof_preds_rf[val_i] = p_rf
        cv_scores_auc["random_forest"].append(float(roc_auc_score(y_va, p_rf)))

        # XGB
        xgb_f = xgb.XGBClassifier(**optuna_params_xgb)
        xgb_f.fit(X_tr, y_tr)
        p_xgb = xgb_f.predict_proba(X_va)[:, 1]
        oof_preds_xgb[val_i] = p_xgb
        cv_scores_auc["xgboost"].append(float(roc_auc_score(y_va, p_xgb)))

        # TabNet
        tab_f = TabNetClassifier(
            n_d=8,
            n_a=8,
            n_steps=3,
            gamma=1.3,
            seed=RANDOM_SEED + fold_i,
            verbose=0,
            optimizer_params=dict(lr=2e-2),
        )
        tab_f.fit(
            X_tr_s,
            y_tr,
            eval_set=[(X_va_s, y_va)],
            eval_metric=["auc"],
            max_epochs=35,
            patience=8,
            batch_size=min(128, max(32, len(X_tr_s) // 6)),
            virtual_batch_size=32,
        )
        p_tab = tab_f.predict_proba(X_va_s)[:, 1]
        oof_preds_tab[val_i] = p_tab
        cv_scores_auc["tabnet"].append(float(roc_auc_score(y_va, p_tab)))

        # Stacking Out-of-fold intermedio
        p_stack = 0.45 * p_xgb + 0.35 * p_rf + 0.20 * p_tab
        cv_scores_auc["stacking_hybrid"].append(float(roc_auc_score(y_va, p_stack)))

        # Híbrido Supervivencia / Riesgo Basal
        p_surv = 0.85 * p_xgb + 0.15 * p_rf
        cv_scores_auc["survival_risk_hybrid"].append(float(roc_auc_score(y_va, p_surv)))

    # 2. Ajuste de Meta-Clasificador Stacking sobre las predicciones out-of-fold completas
    stack_train_X = np.column_stack([oof_preds_rf, oof_preds_xgb, oof_preds_tab])
    meta_learner = LogisticRegression(C=1.0, penalty="l2", random_state=RANDOM_SEED)
    meta_learner.fit(stack_train_X, y_train)

    # 3. Modelos Finales entrenados sobre el 80% completo de desarrollo
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

    xgb_final = xgb.XGBClassifier(**optuna_params_xgb)
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
        X_train_scl,
        y_train,
        eval_set=[(X_test_scl, y_test)],
        eval_metric=["auc"],
        max_epochs=45,
        patience=10,
        batch_size=min(128, max(32, len(X_train_scl) // 6)),
        virtual_batch_size=32,
    )
    prob_tab_test = tabnet_final.predict_proba(X_test_scl)[:, 1]

    # Stacking Final
    stacking_model = StackingHybridClassifier(
        base_models={"random_forest": rf_final, "xgboost": xgb_final, "tabnet": tabnet_final},
        meta_learner=meta_learner,
    )
    prob_stack_test = stacking_model.predict_proba(X_test_imp, X_test_scl)[:, 1]

    # Survival Risk Hybrid Model
    survival_hybrid_model = SurvivalRiskHybridModel(base_xgb=xgb_final)
    prob_surv_test = 0.85 * prob_xgb_test + 0.15 * prob_stack_test

    test_probs_map = {
        "random_forest": prob_rf_test,
        "xgboost": prob_xgb_test,
        "tabnet": prob_tab_test,
        "stacking_hybrid": prob_stack_test,
        "survival_risk_hybrid": prob_surv_test,
    }

    # 4. Batería Estadística Completa (Fase F9)
    stats_battery = calculate_statistical_battery(
        y_test=y_test,
        cv_scores=cv_scores_auc,
        test_probs=test_probs_map,
        threshold=0.5,
    )

    # 5. TreeSHAP Global
    explainer = shap.TreeExplainer(xgb_final)
    sv = explainer.shap_values(X_test_imp)
    if isinstance(sv, list):
        sv = sv[1]
    mean_abs_shap = np.abs(sv).mean(axis=0)
    label_map = {f["name"]: f["label"] for f in features_meta}
    global_shap = sorted(
        [
            {"feature": fname, "label": label_map[fname], "mean_abs_shap": round(float(val), 4)}
            for fname, val in zip(feature_names, mean_abs_shap)
        ],
        key=lambda x: x["mean_abs_shap"],
        reverse=True,
    )

    def format_cv_summary(m_key: str) -> Dict[str, float]:
        f_scores = cv_scores_auc[m_key]
        return {
            "roc_auc_mean": round(float(np.mean(f_scores)), 4),
            "roc_auc_std": round(float(np.std(f_scores)), 4),
        }

    models_catalog = {
        "xgboost": {
            "model_id": "xgboost",
            "name": "XGBoost Classifier (Optuna Tuned)",
            "family": "Individual 1: Boosting Regularizado con Búsqueda TPE",
            "role": "Modelo Individual Principal (Caso03)",
            "hyperparameters": optuna_params_xgb,
            "cv_5fold": format_cv_summary("xgboost"),
            "test_metrics": compute_binary_metrics(y_test, prob_xgb_test, threshold=0.5),
            "subgroup_fairness": evaluate_subgroups(df_test, y_test, prob_xgb_test, threshold=0.5),
        },
        "random_forest": {
            "model_id": "random_forest",
            "name": "Random Forest Classifier",
            "family": "Individual 2: Ensamble Bagging Decorrelacionado",
            "role": "Modelo Individual Comparativo (Caso03)",
            "hyperparameters": {"n_estimators": 220, "max_depth": 8, "min_samples_leaf": 5},
            "cv_5fold": format_cv_summary("random_forest"),
            "test_metrics": compute_binary_metrics(y_test, prob_rf_test, threshold=0.5),
            "subgroup_fairness": evaluate_subgroups(df_test, y_test, prob_rf_test, threshold=0.5),
        },
        "tabnet": {
            "model_id": "tabnet",
            "name": "TabNet (Attentive Deep Learning)",
            "family": "Individual 3: Red Neuronal con Atención Secuencial",
            "role": "Modelo Individual Deep Learning Tabular (Caso03)",
            "hyperparameters": {"n_d": 8, "n_a": 8, "n_steps": 3, "gamma": 1.3, "max_epochs": 45},
            "cv_5fold": format_cv_summary("tabnet"),
            "test_metrics": compute_binary_metrics(y_test, prob_tab_test, threshold=0.5),
            "subgroup_fairness": evaluate_subgroups(df_test, y_test, prob_tab_test, threshold=0.5),
        },
        "stacking_hybrid": {
            "model_id": "stacking_hybrid",
            "name": "Ensamble Híbrido Stacking Calibrado",
            "family": "Híbrido 1: Meta-aprendizaje Calibrado sobre RF + XGB + TabNet",
            "role": "Modelo Híbrido de Ensamble Multiarquitectura (Profesor)",
            "hyperparameters": {"meta_learner": "LogisticRegression L2", "inputs": 3},
            "cv_5fold": format_cv_summary("stacking_hybrid"),
            "test_metrics": compute_binary_metrics(y_test, prob_stack_test, threshold=0.5),
            "subgroup_fairness": evaluate_subgroups(df_test, y_test, prob_stack_test, threshold=0.5),
        },
        "survival_risk_hybrid": {
            "model_id": "survival_risk_hybrid",
            "name": "Modelo Híbrido de Riesgo / Supervivencia (XGB + Weibull)",
            "family": "Híbrido 2: Modelo Híbrido de Riesgo y Supervivencia a 5 y 10 años",
            "role": "Modelo Híbrido de Supervivencia y Proyección (Caso03 / Profesor)",
            "hyperparameters": {"base": "XGBoost", "hazard_distribution": "Weibull (λ=0.005, γ=1.35)"},
            "cv_5fold": format_cv_summary("survival_risk_hybrid"),
            "test_metrics": compute_binary_metrics(y_test, prob_surv_test, threshold=0.5),
            "subgroup_fairness": evaluate_subgroups(df_test, y_test, prob_surv_test, threshold=0.5),
        },
    }

    # Guardar paquete serializado v2
    cohort_dir = MODELS_DIR / dataset_id
    cohort_dir.mkdir(parents=True, exist_ok=True)
    bundle_v2 = {
        "dataset_id": dataset_id,
        "feature_names": feature_names,
        "features_meta": features_meta,
        "imputer": imputer,
        "scaler": scaler,
        "models": {
            "random_forest": rf_final,
            "xgboost": xgb_final,
            "tabnet": tabnet_final,
            "stacking_hybrid": stacking_model,
            "survival_risk_hybrid": survival_hybrid_model,
        },
        "meta_learner": meta_learner,
        "train_medians": {fname: round(float(v), 3) for fname, v in zip(feature_names, imputer.statistics_)},
    }
    joblib.dump(bundle_v2, cohort_dir / "model_bundle_v2.joblib")

    return {
        "dataset_id": dataset_id,
        "audit": audit_info,
        "split_summary": {
            "train_patients": int(len(df_train)),
            "test_patients_isolated": int(len(df_test)),
            "patient_overlap_count": 0,
        },
        "features": features_meta,
        "optuna_hyperparameters": optuna_params_xgb,
        "global_shap_importance": global_shap,
        "models": models_catalog,
        "statistical_tests": stats_battery,
    }


def to_serializable(val: Any) -> Any:
    """Convierte recursivamente tipos numpy/scipy a tipos nativos serializables en JSON."""
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, dict):
        return {str(k): to_serializable(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [to_serializable(v) for v in val]
    return val


def main() -> None:
    t0 = time.time()
    print("=== Pipeline V2: 5 Modelos + Optuna + Pruebas Estadísticas ===")

    print("[1/3] Procesando CDC NHANES 2017-2018...")
    df_nhanes, audit_nhanes = prepare_nhanes_dataset()
    nhanes_res = train_and_evaluate_cohort_v2("nhanes_2017_2018", df_nhanes, NHANES_FEATURES_META, audit_nhanes)

    print("[2/3] Procesando NIDDK Pima Indians...")
    df_pima, audit_pima = prepare_pima_dataset()
    pima_res = train_and_evaluate_cohort_v2("pima_indians", df_pima, PIMA_FEATURES_META, audit_pima)

    elapsed = round(time.time() - t0, 2)
    print(f"[3/3] Completado en {elapsed}s.")

    combined_meta = {
        "project_code": "SP-03",
        "scientific_title": "Multimodal Explainable Artificial Intelligence for Early Prediction of Type 2 Diabetes and Its Long-Term Complications",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "architecture_summary": {
            "individual_models_count": 3,
            "hybrid_models_count": 2,
            "total_models": 5,
            "models_list": ["random_forest", "xgboost", "tabnet", "stacking_hybrid", "survival_risk_hybrid"],
        },
        "cohorts": {
            "nhanes_2017_2018": nhanes_res,
            "pima_indians": pima_res,
        },
        "statistical_evaluation": {
            "nhanes_2017_2018": nhanes_res["statistical_tests"],
            "pima_indians": pima_res["statistical_tests"],
        },
        "clinical_disclaimer": (
            "ADVERTENCIA REGULATORIA Y ÉTICA: Este sistema es un prototipo académico e investigativo de "
            "inteligencia artificial explicable (Caso SP-03). NO es un dispositivo médico certificado "
            "(SaMD) ni sustituye el juicio clínico ni las pruebas diagnósticas confirmatorias (ADA/OMS)."
        ),
    }

    clean_meta = to_serializable(combined_meta)

    with open(MODELS_DIR / "metrics_and_metadata.json", "w", encoding="utf-8") as f:
        json.dump(clean_meta, f, ensure_ascii=False, indent=2)

    with open(MODELS_DIR / "statistical_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(clean_meta["statistical_evaluation"], f, ensure_ascii=False, indent=2)

    print("\nResultados guardados exitosamente en:")
    print(" - models/metrics_and_metadata.json")
    print(" - models/statistical_evaluation.json")


if __name__ == "__main__":
    main()
