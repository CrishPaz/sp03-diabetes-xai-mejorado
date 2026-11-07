#!/usr/bin/env python3
"""
Genera las 6 figuras científicas de calidad Q1 (300 DPI) y calcula estadísticas
inferenciales adicionales (intervalos de confianza al 95% por bootstrap, p-valores
pareados y estadísticos descriptivos por grupo clínico) a partir de los datos y
modelos reales entrenados en el proyecto SP-03.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from scripts.train_and_evaluate import (
    NHANES_FEATURES_META,
    PIMA_FEATURES_META,
    RANDOM_SEED,
    prepare_nhanes_dataset,
    prepare_pima_dataset,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
FIG_DIR = ROOT_DIR / "figuras"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "figure.dpi": 300,
    }
)


def compute_inferential_stats(
    df_nhanes: pd.DataFrame, df_pima: pd.DataFrame, meta: Dict[str, Any]
) -> Dict[str, Any]:
    """Calcula estadísticos descriptivos, IC 95% bootstrap y p-valores reales."""
    rng = np.random.default_rng(RANDOM_SEED)
    out: Dict[str, Any] = {"nhanes_descriptive": {}, "bootstrap_ci": {}}

    # Descriptivos NHANES por estado de diabetes (outcome=1 vs outcome=0)
    fnames = [f["name"] for f in NHANES_FEATURES_META]
    pos = df_nhanes[df_nhanes["outcome"] == 1]
    neg = df_nhanes[df_nhanes["outcome"] == 0]
    for col in fnames:
        v_pos = pos[col].dropna().values
        v_neg = neg[col].dropna().values
        stat_u, p_val = stats.mannwhitneyu(v_pos, v_neg, alternative="two-sided")
        out["nhanes_descriptive"][col] = {
            "dm2_mean": round(float(np.mean(v_pos)), 2),
            "dm2_sd": round(float(np.std(v_pos)), 2),
            "non_dm2_mean": round(float(np.mean(v_neg)), 2),
            "non_dm2_sd": round(float(np.std(v_neg)), 2),
            "p_value": float(p_val),
        }

    # Bootstrap IC 95% sobre el conjunto de prueba aislado de NHANES y Pima
    for cid, df_coh, fmeta in [
        ("nhanes_2017_2018", df_nhanes, NHANES_FEATURES_META),
        ("pima_indians", df_pima, PIMA_FEATURES_META),
    ]:
        cols = [f["name"] for f in fmeta]
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        _, test_idx = next(sgkf.split(df_coh[cols], df_coh["outcome"].values, df_coh["patient_id"].values))
        df_test = df_coh.iloc[test_idx].reset_index(drop=True)
        y_test = df_test["outcome"].values.astype(int)

        bundle = joblib.load(MODELS_DIR / cid / "model_bundle.joblib")
        X_imp = bundle["imputer"].transform(df_test[cols].values)
        X_scl = bundle["scaler"].transform(X_imp)

        probs_map = {}
        for mid in ["xgboost", "random_forest", "tabnet", "logistic_regression"]:
            mdl = bundle["models"][mid]
            probs_map[mid] = (
                mdl.predict_proba(X_scl)[:, 1]
                if mid in ("logistic_regression", "tabnet")
                else mdl.predict_proba(X_imp)[:, 1]
            )

        out["bootstrap_ci"][cid] = {}
        n_test = len(y_test)
        for mid, y_prob in probs_map.items():
            aucs = []
            for _ in range(600):
                b_idx = rng.integers(0, n_test, size=n_test)
                if len(np.unique(y_test[b_idx])) < 2:
                    continue
                aucs.append(roc_auc_score(y_test[b_idx], y_prob[b_idx]))
            lo, hi = np.percentile(aucs, [2.5, 97.5])
            out["bootstrap_ci"][cid][mid] = {
                "auc_ci_lower": round(float(lo), 4),
                "auc_ci_upper": round(float(hi), 4),
            }

        # P-valor pareado por bootstrap entre XGBoost y los demás modelos
        p_xgb = probs_map["xgboost"]
        for other_id in ["random_forest", "tabnet", "logistic_regression"]:
            p_oth = probs_map[other_id]
            diffs = []
            for _ in range(600):
                b_idx = rng.integers(0, n_test, size=n_test)
                if len(np.unique(y_test[b_idx])) < 2:
                    continue
                diffs.append(
                    roc_auc_score(y_test[b_idx], p_xgb[b_idx])
                    - roc_auc_score(y_test[b_idx], p_oth[b_idx])
                )
            diffs_arr = np.array(diffs)
            p_two = 2.0 * min(float(np.mean(diffs_arr <= 0)), float(np.mean(diffs_arr >= 0)))
            out["bootstrap_ci"][cid][f"pval_xgb_vs_{other_id}"] = round(min(1.0, p_two), 4)

    return out


def make_figure_1() -> Path:
    """Figura 1: Diagrama STROBE/TRIPOD+AI de selección de cohortes y arquitectura de doble resultado."""
    fig, ax = plt.subplots(figsize=(11, 6.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def draw_box(x, y, w, h, title, body, fc="#f8fafc", ec="#0f766e", lw=1.6, title_color="#0f172a"):
        rect = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.6,rounding_size=1.8",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h - 3.2,
            title,
            ha="center",
            va="top",
            fontsize=9.3,
            fontweight="bold",
            color=title_color,
        )
        ax.text(
            x + w / 2,
            y + h / 2 - 1.5,
            body,
            ha="center",
            va="center",
            fontsize=8.1,
            color="#334155",
        )

    def arrow(x1, y1, x2, y2):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="#334155", lw=1.4),
        )

    # Columna izquierda: NHANES
    draw_box(
        3,
        79,
        43,
        16,
        "Cohorte Primaria: CDC NHANES 2017–2018 (Archivos SAS .XPT)",
        "N = 9,254 participantes totales evaluados en examen MEC\nAdultos ≥ 20 años con cuestionario DIQ010 válido: n = 5,565\nExclusión por ausencia de panel bioquímico/HbA1c/IMC: n = 736",
        fc="#f0fdfa",
        ec="#0f766e",
    )
    arrow(24.5, 79, 24.5, 73)
    draw_box(
        3,
        55,
        43,
        17.5,
        "Cohorte Analítica Primaria Verificada (n = 4,829 pacientes únicos)",
        "Casos DM2 confirmados (DIQ010=1): 772 (15.99%) | Sin DM2: 4,057 (84.01%)\n12 predictores clínicos (glucosa, HbA1c, IMC, PA, lípidos, antecedentes)\nExclusión de insulina/hipoglucemiantes (fuga) y de sexo/raza (equidad)",
        fc="#ffffff",
        ec="#0f766e",
    )
    arrow(24.5, 55, 24.5, 49)
    draw_box(
        3,
        33,
        43,
        15.5,
        "Partición Agrupada por Paciente (StratifiedGroupKFold por SEQN)",
        "Desarrollo (80%): n = 3,863 (618 DM2) → 5-Fold CV + Imputación intra-fold\nPrueba Aislada (20% Holdout): n = 966 (154 DM2) → Intersección Train∩Test = 0",
        fc="#f8fafc",
        ec="#1d4ed8",
    )

    # Columna derecha: Pima Benchmark
    draw_box(
        54,
        79,
        43,
        16,
        "Cohorte Benchmark: NIDDK Pima Indians Diabetes Dataset",
        "N = 768 registros de pacientes mujeres de herencia Pima (≥ 21 años)\nCasos positivos (OMS 2h OGTT): 268 (34.90%) | Negativos: 500 (65.10%)\n8 variables: embarazos, glucosa 2h, PA diastólica, pliegue, insulina, IMC, pedigrí, edad",
        fc="#eff6ff",
        ec="#1d4ed8",
    )
    arrow(75.5, 79, 75.5, 73)
    draw_box(
        54,
        55,
        43,
        17.5,
        "Auditoría Fisiológica y Preprocesamiento sin Fuga (n = 768)",
        "Detección de ceros biológicamente imposibles codificados como 0:\nInsulina (374), Pliegue (227), PA diastólica (35), IMC (11), Glucosa (5)\nRecodificación a NaN e imputación por mediana restringida a entrenamiento",
        fc="#ffffff",
        ec="#1d4ed8",
    )
    arrow(75.5, 55, 75.5, 49)
    draw_box(
        54,
        33,
        43,
        15.5,
        "Partición Estratificada Aislada (Semilla Fija = 42)",
        "Desarrollo (80%): n = 614 (214 DM2) → 5-Fold CV Agrupado\nPrueba Aislada (20% Holdout): n = 154 (54 DM2) → Intersección Train∩Test = 0",
        fc="#f8fafc",
        ec="#1d4ed8",
    )

    # Resultado 1 vs Resultado 2 en la base
    arrow(24.5, 33, 28, 25)
    arrow(75.5, 33, 42, 25)
    draw_box(
        3,
        4,
        54,
        20.5,
        "RESULTADO 1 (IMPLEMENTADO Y VERIFICADO EN DATOS REALES):\nPredicción Temprana de Aparición / Presencia de DM2 + Explicabilidad XAI",
        "• Algoritmos entrenados: XGBoost, Random Forest, TabNet y Regresión Logística L2\n• Explicabilidad global y local SHAP (TreeSHAP / LinearSHAP) + Curvas de Calibración (Brier)\n• Auditoría de equidad por Edad y Sexo | API FastAPI (/predict, /model-info) + UI React",
        fc="#ecfdf5",
        ec="#15803d",
        lw=2.0,
        title_color="#14532d",
    )

    draw_box(
        60,
        4,
        37,
        20.5,
        "RESULTADO 2 (ARQUITECTURA PROPUESTA / SIMULADA):\nComplicaciones Crónicas y Supervivencia",
        "• Modelos híbridos: XGBoost + Cox PH y\n  LSTM + Attention + Survival Analysis\n• Requiere tiempo a evento (T) y censura (δ)\n  ausentes en cohortes transversales NHANES/Pima",
        fc="#fffbeb",
        ec="#d97706",
        lw=1.8,
        title_color="#92400e",
    )

    fig.suptitle(
        "Figura 1. Flujo de selección de cohortes (STROBE/TRIPOD+AI) y arquitectura de doble resultado del sistema SP-03",
        fontsize=11.5,
        fontweight="bold",
        y=0.98,
    )
    out_path = FIG_DIR / "Figura_1_Arquitectura_Flujo_Cohortes.png"
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def make_figure_2(meta: Dict[str, Any]) -> Path:
    """Figura 2: Curvas ROC comparativas reales en NHANES (n=966) y Pima (n=154)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    colors = {
        "xgboost": "#0f766e",
        "random_forest": "#1d4ed8",
        "tabnet": "#7c3aed",
        "logistic_regression": "#ea580c",
    }

    for ax, (cid, title_panel) in zip(
        axes,
        [
            ("nhanes_2017_2018", "A) Cohorte Primaria: CDC NHANES 2017–2018 (Prueba Aislada, n = 966)"),
            ("pima_indians", "B) Cohorte Benchmark: NIDDK Pima Indians (Prueba Aislada, n = 154)"),
        ],
    ):
        models_dict = meta["cohorts"][cid]["models"]
        ax.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", lw=1.2, label="Azar (AUC = 0.5000)")
        for mid in ["xgboost", "random_forest", "tabnet", "logistic_regression"]:
            m = models_dict[mid]
            pts = m["test_metrics"]["roc_curve"]
            fpr = [p["fpr"] for p in pts]
            tpr = [p["tpr"] for p in pts]
            auc = m["test_metrics"]["roc_auc"]
            cv_auc = m["cv_5fold"]["roc_auc_mean"]
            ax.plot(
                fpr,
                tpr,
                lw=2.2,
                color=colors[mid],
                label=f"{m['name']} (Test AUC={auc:.4f} | CV={cv_auc:.4f})",
            )
        ax.set_title(title_panel, fontsize=10.2, fontweight="bold")
        ax.set_xlabel("Tasa de Falsos Positivos (1 − Especificidad)")
        ax.set_ylabel("Tasa de Verdaderos Positivos (Sensibilidad)")
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.02)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower right", frameon=True, fontsize=7.8)

    out_path = FIG_DIR / "Figura_2_Curvas_ROC_Comparativas.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def make_figure_3(meta: Dict[str, Any]) -> Path:
    """Figura 3: Curvas de calibración y comparación de desempeño en NHANES 2017-2018."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    colors = {
        "xgboost": "#0f766e",
        "random_forest": "#1d4ed8",
        "tabnet": "#7c3aed",
        "logistic_regression": "#ea580c",
    }
    models_nhanes = meta["cohorts"]["nhanes_2017_2018"]["models"]

    # Panel A: Curvas de calibración
    ax1 = axes[0]
    ax1.plot([0, 1], [0, 1], linestyle="--", color="#64748b", lw=1.3, label="Calibración Ideal Perfecta")
    for mid in ["xgboost", "random_forest", "tabnet", "logistic_regression"]:
        m = models_nhanes[mid]
        cal = m["test_metrics"]["calibration_curve"]
        pred_p = [c["prob_pred"] for c in cal]
        true_p = [c["prob_true"] for c in cal]
        brier = m["test_metrics"]["brier_score"]
        ax1.plot(
            pred_p,
            true_p,
            marker="o",
            markersize=4.5,
            lw=2.0,
            color=colors[mid],
            label=f"{m['name']} (Brier = {brier:.4f})",
        )
    ax1.set_title("A) Curvas de Calibración en Prueba Aislada (NHANES, n = 966)", fontsize=10.2)
    ax1.set_xlabel("Probabilidad Media Predicha por el Modelo")
    ax1.set_ylabel("Proporción Empírica Observada de DM2")
    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", fontsize=7.8)

    # Panel B: Comparación de métricas operativas en prueba aislada
    ax2 = axes[1]
    metric_keys = ["roc_auc", "sensitivity", "specificity", "precision", "f1"]
    metric_labels = ["ROC-AUC", "Sensibilidad", "Especificidad", "Precisión (VPP)", "F1-Score"]
    x = np.arange(len(metric_keys))
    width = 0.19
    for idx, mid in enumerate(["xgboost", "random_forest", "tabnet", "logistic_regression"]):
        m = models_nhanes[mid]
        vals = [m["test_metrics"][k] for k in metric_keys]
        ax2.bar(
            x + (idx - 1.5) * width,
            vals,
            width=width,
            color=colors[mid],
            label=m["name"],
            alpha=0.9,
        )
    ax2.set_title("B) Perfil Multidimensional de Métricas (NHANES Test, n = 966)", fontsize=10.2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metric_labels)
    ax2.set_ylim(0, 1.08)
    ax2.set_ylabel("Valor de la Métrica (Proporción 0–1)")
    ax2.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", fontsize=7.8)

    out_path = FIG_DIR / "Figura_3_Calibracion_y_Metricas.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def make_figure_4(meta: Dict[str, Any]) -> Path:
    """Figura 4: Importancia global SHAP en NHANES y Pima + explicación local SHAP individual."""
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.9))

    # Panel A: Global SHAP NHANES
    shap_nhanes = list(reversed(meta["cohorts"]["nhanes_2017_2018"]["global_shap_importance"]))
    labels_n = [item["label"] for item in shap_nhanes]
    vals_n = [item["mean_abs_shap"] for item in shap_nhanes]

    ax1 = axes[0]
    bars1 = ax1.barh(labels_n, vals_n, color="#0f766e", edgecolor="#115e59", height=0.68)
    for bar, v in zip(bars1, vals_n):
        ax1.text(v + 0.02, bar.get_y() + bar.get_height() / 2, f"{v:.4f}", va="center", fontsize=8)
    ax1.set_xlim(0, max(vals_n) * 1.18)
    ax1.set_title("A) Importancia Global TreeSHAP — CDC NHANES (n = 966)", fontsize=10.2)
    ax1.set_xlabel("Impacto Marginal Medio Absoluto |SHAP| (unidades log-odds)")
    ax1.grid(True, axis="x", linestyle=":", alpha=0.6)

    # Panel B: Explicación individual SHAP para un paciente de alto riesgo real
    from backend.app.predictor import DiabetesPredictorService

    svc = DiabetesPredictorService()
    sample_pred = svc.predict_single(
        dataset_id="nhanes_2017_2018",
        model_id="xgboost",
        raw_features={
            "edad": 58.0,
            "imc": 33.4,
            "glucosa": 146.0,
            "hba1c": 6.6,
            "presion_sistolica": 142.0,
            "presion_diastolica": 86.0,
            "colesterol_total": 218.0,
            "colesterol_hdl": 41.0,
            "trigliceridos": 210.0,
            "antecedente_familiar_diabetes": 1,
            "antecedente_hipertension": 1,
            "tratamiento_antihipertensivo": 1,
        },
    )
    contribs = list(reversed(sample_pred.shap_contributions[:8]))
    c_labels = [f"{c.label} ({c.value:g} {c.unit})" for c in contribs]
    c_vals = [c.shap_value for c in contribs]
    c_colors = ["#dc2626" if v > 0 else "#0f766e" for v in c_vals]

    ax2 = axes[1]
    bars2 = ax2.barh(c_labels, c_vals, color=c_colors, height=0.65)
    ax2.axvline(0, color="#334155", lw=1.2)
    for bar, v in zip(bars2, c_vals):
        offset = 0.04 if v >= 0 else -0.04
        ha = "left" if v >= 0 else "right"
        ax2.text(
            v + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{v:+.3f}",
            va="center",
            ha=ha,
            fontsize=8,
            fontweight="bold",
        )
    ax2.set_xlim(min(c_vals) - 0.45, max(c_vals) + 0.55)
    ax2.set_title(
        f"B) Explicación Local SHAP Caso Individual (Prob. = {sample_pred.percentage:.1f}%)",
        fontsize=10.2,
    )
    ax2.set_xlabel("Contribución Individual SHAP al Puntaje de Riesgo (log-odds)")
    ax2.grid(True, axis="x", linestyle=":", alpha=0.6)

    out_path = FIG_DIR / "Figura_4_Importancia_Global_y_Local_SHAP.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def make_figure_5(meta: Dict[str, Any]) -> Path:
    """Figura 5: Auditoría de equidad por subgrupos de Edad y Sexo en NHANES (n=966)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    models_nhanes = meta["cohorts"]["nhanes_2017_2018"]["models"]

    subgroups_order = ["20-39 años", "40-59 años", "≥60 años", "Femenino", "Masculino"]
    x = np.arange(len(subgroups_order))
    width = 0.19
    colors = {
        "xgboost": "#0f766e",
        "random_forest": "#1d4ed8",
        "tabnet": "#7c3aed",
        "logistic_regression": "#ea580c",
    }

    # Panel A: ROC-AUC por subgrupo demográfico
    ax1 = axes[0]
    for idx, mid in enumerate(["xgboost", "random_forest", "tabnet", "logistic_regression"]):
        sg_map = {s["subgroup"]: s for s in models_nhanes[mid]["subgroup_fairness"]}
        vals = [sg_map[k]["roc_auc"] for k in subgroups_order]
        ax1.bar(
            x + (idx - 1.5) * width,
            vals,
            width=width,
            color=colors[mid],
            label=models_nhanes[mid]["name"],
        )
    ax1.set_title("A) Discriminación (ROC-AUC) por Subgrupo de Edad y Sexo", fontsize=10.2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(subgroups_order, rotation=12)
    ax1.set_ylim(0.70, 1.02)
    ax1.set_ylabel("ROC-AUC en Conjunto de Prueba Aislado")
    ax1.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", fontsize=7.6)

    # Panel B: Calibración (Brier Score) y Prevalencia Observada vs Predicha (XGBoost)
    ax2 = axes[1]
    sg_xgb = {s["subgroup"]: s for s in models_nhanes["xgboost"]["subgroup_fairness"]}
    prev_real = [sg_xgb[k]["prevalence"] * 100 for k in subgroups_order]
    prev_pred = [sg_xgb[k]["predicted_positive_rate"] * 100 for k in subgroups_order]
    brier_vals = [sg_xgb[k]["brier_score"] * 100 for k in subgroups_order]

    w2 = 0.26
    ax2.bar(x - w2, prev_real, width=w2, color="#334155", label="Prevalencia Real Observada (%)")
    ax2.bar(x, prev_pred, width=w2, color="#0f766e", label="Tasa Positivos Predichos XGBoost (%)")
    ax2.bar(x + w2, brier_vals, width=w2, color="#f59e0b", label="Brier Score × 100")
    ax2.set_title("B) Paridad Epidemiológica y Error de Calibración por Subgrupo", fontsize=10.2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(subgroups_order, rotation=12)
    ax2.set_ylabel("Porcentaje (%) / Escala ×100")
    ax2.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", fontsize=7.8)

    out_path = FIG_DIR / "Figura_5_Equidad_Subgrupos_Edad_Sexo.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def make_figure_6() -> Path:
    """
    Figura 6: Escenario SIMULADO / HIPOTÉTICO de supervivencia libre de complicaciones
    crónicas a 10 años (120 meses) para Resultado 2 (XGBoost+Cox y LSTM+Attention+Survival).
    Rotulado explícitamente como SIMULADO / HIPOTÉTICO en título, ejes y leyenda.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    months = np.linspace(0, 120, 61)

    # Curvas de supervivencia de Weibull plausibles por cuartil de riesgo [SIMULADAS]
    s_q1 = np.exp(-((months / 360.0) ** 1.35))
    s_q2 = np.exp(-((months / 235.0) ** 1.40))
    s_q3 = np.exp(-((months / 155.0) ** 1.48))
    s_q4 = np.exp(-((months / 98.0) ** 1.55))

    ax1 = axes[0]
    ax1.plot(months, s_q1 * 100, lw=2.2, color="#15803d", label="Cuartil 1: Riesgo Bajo [SIMULADO]")
    ax1.plot(months, s_q2 * 100, lw=2.2, color="#0284c7", label="Cuartil 2: Riesgo Moderado [SIMULADO]")
    ax1.plot(months, s_q3 * 100, lw=2.2, color="#d97706", label="Cuartil 3: Riesgo Alto [SIMULADO]")
    ax1.plot(months, s_q4 * 100, lw=2.2, color="#dc2626", label="Cuartil 4: Riesgo Muy Alto [SIMULADO]")
    ax1.set_title(
        "A) Supervivencia Libre de Complicaciones Micro/Macrovasculares\n[DATOS SIMULADOS / HIPOTÉTICOS — Resultado 2 Propuesto]",
        fontsize=9.8,
    )
    ax1.set_xlabel("Seguimiento Longitudinal desde el Diagnóstico (Meses)")
    ax1.set_ylabel("Supervivencia Libre de Eventos S(t) [%] (Simulada)")
    ax1.set_xlim(0, 120)
    ax1.set_ylim(30, 102)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower left", fontsize=7.8)

    # Panel B: Comparación simulada de C-index de Harrell por complicación
    ax2 = axes[1]
    complications = ["Nefropatía\nDiabética", "Retinopatía\nDiabética", "Neuropatía\nPeriférica", "Eventos MACE\nCardiovasculares"]
    x = np.arange(len(complications))
    w = 0.25
    c_cox = [0.742, 0.728, 0.715, 0.751]
    c_xgb_cox = [0.794, 0.781, 0.768, 0.806]
    c_lstm_surv = [0.818, 0.809, 0.791, 0.829]

    ax2.bar(x - w, c_cox, width=w, color="#64748b", label="Cox PH Clásico [SIMULADO]")
    ax2.bar(x, c_xgb_cox, width=w, color="#0f766e", label="XGBoost + Cox PH [SIMULADO]")
    ax2.bar(x + w, c_lstm_surv, width=w, color="#7c3aed", label="LSTM + Attention + Survival [SIMULADO]")
    ax2.set_title(
        "B) Índice de Concordancia (C-index) por Tipo de Complicación\n[VALORES HIPOTÉTICOS / SIMULADOS — Experimento Futuro]",
        fontsize=9.8,
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(complications)
    ax2.set_ylim(0.55, 0.90)
    ax2.set_ylabel("C-index de Harrell [Escenario Simulado]")
    ax2.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", fontsize=7.8)

    out_path = FIG_DIR / "Figura_6_Supervivencia_Complicaciones_Simulada.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def main() -> None:
    with open(MODELS_DIR / "metrics_and_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    df_nhanes, _ = prepare_nhanes_dataset()
    df_pima, _ = prepare_pima_dataset()

    stats_extra = compute_inferential_stats(df_nhanes, df_pima, meta)
    with open(MODELS_DIR / "article_inferential_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats_extra, f, ensure_ascii=False, indent=2)

    f1 = make_figure_1()
    f2 = make_figure_2(meta)
    f3 = make_figure_3(meta)
    f4 = make_figure_4(meta)
    f5 = make_figure_5(meta)
    f6 = make_figure_6()

    print("Figuras generadas exitosamente:")
    for p in [f1, f2, f3, f4, f5, f6]:
        print(" -", p.relative_to(ROOT_DIR))


if __name__ == "__main__":
    main()
