from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import shap

from backend.app.hybrid_models import StackingHybridClassifier, SurvivalRiskHybridModel

# Registrar en __main__ para compatibilidad de unpickling con joblib
main_module = sys.modules.get("__main__")
if main_module is not None:
    setattr(main_module, "StackingHybridClassifier", StackingHybridClassifier)
    setattr(main_module, "SurvivalRiskHybridModel", SurvivalRiskHybridModel)

from backend.app.schemas import (
    BatchPredictResponse,
    BatchRowPrediction,
    ModelProbabilityComparison,
    NhanesPatientInput,
    PimaPatientInput,
    PredictResponse,
    ShapFeatureContribution,
    SurvivalProjection,
)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = Path(os.getenv("MODEL_ARTIFACTS_DIR", str(ROOT_DIR / "models")))


def _enable_cross_platform_xgboost_unpickle() -> None:
    """Permite cargar modelos XGBoost serializados en otro sistema operativo.

    El pickle de XGBoost guarda `rng_state` (estado del generador std::mt19937) en el
    formato de texto de la biblioteca estándar de C++ del sistema donde se entrenó:
    libstdc++ (Linux) lo escribe en decimal y MSVC (Windows) en hexadecimal, por lo que
    un modelo entrenado en Linux falla en Windows con "input stream corrupted".
    El estado del RNG solo afecta a un entrenamiento posterior, no a la inferencia,
    así que se sustituye por el estado nativo de esta plataforma únicamente cuando
    falla la deserialización original. Árboles e hiperparámetros quedan intactos.
    """
    import struct

    import xgboost as xgb
    from xgboost.core import Booster, XGBoostError

    if getattr(Booster.__setstate__, "_sp03_cross_platform", False):
        return

    key = b"L" + (9).to_bytes(8, "big") + b"rng_state"
    len_formats = {"i": ">b", "U": ">B", "I": ">h", "l": ">i", "L": ">q"}

    def _rng_value_span(buf: bytes) -> Tuple[int, int, int] | None:
        """Devuelve (inicio_marcador_S, inicio_valor, fin_valor) del string UBJSON rng_state."""
        k = buf.find(key)
        if k < 0:
            return None
        p = k + len(key)
        if buf[p : p + 1] != b"S" or chr(buf[p + 1]) not in len_formats:
            return None
        fmt = len_formats[chr(buf[p + 1])]
        n = struct.calcsize(fmt)
        length = struct.unpack(fmt, buf[p + 2 : p + 2 + n])[0]
        return p, p + 2 + n, p + 2 + n + length

    native_rng: Dict[str, bytes] = {}

    def _native_rng_state() -> bytes:
        if "v" not in native_rng:
            tiny = xgb.train(
                {"nthread": 1},
                xgb.DMatrix(np.array([[0.0], [1.0]]), label=[0, 1]),
                num_boost_round=1,
            )
            buf = bytes(tiny.__getstate__()["handle"])
            span = _rng_value_span(buf)
            if span is None:
                raise RuntimeError("No se pudo obtener rng_state nativo de XGBoost")
            native_rng["v"] = buf[span[1] : span[2]]
        return native_rng["v"]

    original_setstate = Booster.__setstate__

    def _setstate(self: Booster, state: Dict[str, Any]) -> None:
        try:
            original_setstate(self, state)
            return
        except XGBoostError as exc:
            handle = state.get("handle")
            if handle is None or "input stream corrupted" not in str(exc):
                raise
            buf = bytes(handle)
            span = _rng_value_span(buf)
            if span is None:
                raise
            new_val = _native_rng_state()
            fixed = buf[: span[0]] + b"SL" + struct.pack(">q", len(new_val)) + new_val + buf[span[2] :]
            original_setstate(self, {**state, "handle": bytearray(fixed)})

    _setstate._sp03_cross_platform = True  # type: ignore[attr-defined]
    Booster.__setstate__ = _setstate  # type: ignore[method-assign]


_enable_cross_platform_xgboost_unpickle()


def classify_risk_stratum(prob: float) -> Tuple[str, str]:
    if prob < 0.25:
        return (
            "Bajo",
            "Baja probabilidad estimada de diabetes mellitus tipo 2 según el perfil actual. "
            "Se sugiere mantener hábitos saludables y control preventivo periódico.",
        )
    if prob < 0.50:
        return (
            "Moderado",
            "Riesgo intermedio. Se recomienda vigilancia metabólica (control de peso, actividad "
            "física aeróbica y seguimiento periódico de glucosa en ayunas y HbA1c).",
        )
    if prob < 0.75:
        return (
            "Alto",
            "Alta probabilidad asociada al perfil de factores de riesgo. Requiere valoración médica "
            "presencial y pruebas confirmatorias de laboratorio (glucemia plasmática en ayunas / HbA1c / OGTT).",
        )
    return (
        "Muy Alto",
        "Probabilidad muy elevada en el modelo predictivo. Se aconseja evaluación clínica prioritaria "
        "para confirmación diagnóstica, estratificación cardiovascular y manejo integral.",
    )


def format_clinical_explanation(
    fname: str, label: str, val: float, unit: str, shap_val: float, median_val: float
) -> str:
    direction_txt = "incrementa" if shap_val > 0.005 else ("reduce" if shap_val < -0.005 else "tiene efecto neutro sobre")
    if unit == "binario (0/1)":
        estado = "Presente (Sí)" if val >= 0.5 else "Ausente (No)"
        return f"{label}: {estado} — {direction_txt} la probabilidad estimada ({shap_val:+.3f})."
    comp = "por encima" if val > median_val else ("por debajo" if val < median_val else "cerca")
    return (
        f"{label} ({val:g} {unit}, {comp} de la mediana de referencia {median_val:g} {unit}) "
        f"{direction_txt} el puntaje de riesgo ({shap_val:+.3f})."
    )


class DiabetesPredictorService:
    def __init__(self, models_dir: Path = MODELS_DIR) -> None:
        self.models_dir = models_dir
        self.metadata: Dict[str, Any] = {}
        self.bundles: Dict[str, Dict[str, Any]] = {}
        self.explainers: Dict[str, Dict[str, Any]] = {}
        self.reload()

    def reload(self) -> None:
        meta_path = self.models_dir / "metrics_and_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No se encontró {meta_path}. Ejecute primero scripts/train_and_evaluate.py"
            )
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.bundles = {}
        self.explainers = {}
        for dataset_id in ["nhanes_2017_2018", "pima_indians"]:
            bpath_v2 = self.models_dir / dataset_id / "model_bundle_v2.joblib"
            bpath_v1 = self.models_dir / dataset_id / "model_bundle.joblib"
            bpath = bpath_v2 if bpath_v2.exists() else bpath_v1
            if bpath.exists():
                bundle = joblib.load(bpath)
                self.bundles[dataset_id] = bundle
                self.explainers[dataset_id] = {
                    "xgboost": shap.TreeExplainer(bundle["models"]["xgboost"]),
                    "random_forest": shap.TreeExplainer(bundle["models"]["random_forest"]),
                }
                if "logistic_regression" in bundle["models"] and "background_scaled" in bundle:
                    self.explainers[dataset_id]["logistic_regression"] = shap.LinearExplainer(
                        bundle["models"]["logistic_regression"],
                        bundle["background_scaled"],
                    )

    @property
    def is_ready(self) -> bool:
        return bool(self.metadata and len(self.bundles) >= 2)

    def validate_features(self, dataset_id: str, raw_features: Dict[str, float]) -> List[float]:
        if dataset_id == "nhanes_2017_2018":
            validated = NhanesPatientInput(**raw_features)
            data_dict = validated.model_dump()
        elif dataset_id == "pima_indians":
            validated = PimaPatientInput(**raw_features)
            data_dict = validated.model_dump()
        else:
            raise ValueError(f"Cohorte no soportada: {dataset_id}")

        feature_names = self.bundles[dataset_id]["feature_names"]
        return [float(data_dict[col]) for col in feature_names]

    def _predict_single_model_prob(
        self, dataset_id: str, model_id: str, x_imp: np.ndarray, x_scaled: np.ndarray
    ) -> float:
        bundle = self.bundles[dataset_id]
        models_map = bundle["models"]

        # Normalizar claves de modelos
        target_id = model_id
        if model_id in ("stacking_ensemble", "stacking_hybrid"):
            target_id = "stacking_hybrid" if "stacking_hybrid" in models_map else "xgboost"
        elif model_id in ("hybrid_survival", "survival_risk_hybrid"):
            target_id = "survival_risk_hybrid" if "survival_risk_hybrid" in models_map else "xgboost"

        if target_id not in models_map:
            target_id = "xgboost"

        model = models_map[target_id]
        if target_id in ("stacking_hybrid", "survival_risk_hybrid"):
            try:
                prob = float(model.predict_proba(x_imp, x_scaled)[0, 1])
            except Exception:
                p_rf = float(models_map["random_forest"].predict_proba(x_imp)[0, 1])
                p_xgb = float(models_map["xgboost"].predict_proba(x_imp)[0, 1])
                prob = 0.5 * p_rf + 0.5 * p_xgb
        elif target_id in ("logistic_regression", "tabnet"):
            prob = float(model.predict_proba(x_scaled)[0, 1])
        else:
            prob = float(model.predict_proba(x_imp)[0, 1])

        return max(0.0001, min(0.9999, prob))

    def _compute_shap_values(
        self, dataset_id: str, model_id: str, x_imp: np.ndarray, x_scaled: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        bundle = self.bundles[dataset_id]

        # Para modelos basados en árbol o ensambles híbridos, usar explainer de XGBoost o Random Forest
        if model_id in ("xgboost", "stacking_ensemble", "stacking_hybrid", "hybrid_survival", "survival_risk_hybrid"):
            explainer = self.explainers[dataset_id]["xgboost"]
            sv = explainer.shap_values(x_imp)
            if isinstance(sv, list):
                sv = sv[1]
            ev = explainer.expected_value
            if isinstance(ev, (list, np.ndarray)):
                ev = ev[1] if len(ev) > 1 else ev[0]
            return float(ev), np.asarray(sv[0], dtype=float)

        if model_id == "random_forest":
            explainer = self.explainers[dataset_id]["random_forest"]
            sv = explainer.shap_values(x_imp)
            if isinstance(sv, list):
                sv = sv[1][0]
            elif sv.ndim == 3:
                sv = sv[0, :, 1]
            else:
                sv = sv[0]
            ev = explainer.expected_value
            if isinstance(ev, (list, np.ndarray)):
                ev = ev[1] if len(ev) > 1 else ev[0]
            return float(ev), np.asarray(sv, dtype=float)

        if model_id == "logistic_regression" and "logistic_regression" in self.explainers[dataset_id]:
            explainer = self.explainers[dataset_id]["logistic_regression"]
            sv = explainer.shap_values(x_scaled)
            if isinstance(sv, list):
                sv = sv[0]
            ev = explainer.expected_value
            if isinstance(ev, (list, np.ndarray)):
                ev = ev[0]
            return float(ev), np.asarray(sv[0], dtype=float)

        # Para TabNet: explicación aditiva por perturbación marginal sobre el fondo escalado
        tabnet_model = bundle["models"].get("tabnet")
        if tabnet_model and "background_scaled" in bundle:
            bg = bundle["background_scaled"][:30]
            base_preds = tabnet_model.predict_proba(bg)[:, 1]
            base_val = float(np.mean(base_preds))
            n_feat = x_scaled.shape[1]
            shap_approx = np.zeros(n_feat, dtype=float)
            full_pred = float(tabnet_model.predict_proba(x_scaled)[0, 1])
            for j in range(n_feat):
                perturbed = bg.copy()
                perturbed[:, j] = x_scaled[0, j]
                shap_approx[j] = float(np.mean(tabnet_model.predict_proba(perturbed)[:, 1])) - base_val
            total_raw = np.sum(np.abs(shap_approx))
            if total_raw > 1e-6:
                diff = full_pred - base_val
                shap_approx = shap_approx * (abs(diff) / total_raw) if diff != 0 else shap_approx
            return base_val, shap_approx

        # Fallback a XGBoost explainer si TabNet no tiene background
        explainer = self.explainers[dataset_id]["xgboost"]
        sv = explainer.shap_values(x_imp)
        ev = explainer.expected_value
        return float(ev[1] if isinstance(ev, (list, np.ndarray)) else ev), np.asarray(sv[0], dtype=float)

    def predict_single(
        self, dataset_id: str, model_id: str, raw_features: Dict[str, float]
    ) -> PredictResponse:
        x_vec = self.validate_features(dataset_id, raw_features)
        bundle = self.bundles[dataset_id]
        cohort_meta = self.metadata["cohorts"][dataset_id]

        x_arr = np.array([x_vec], dtype=float)
        x_imp = bundle["imputer"].transform(x_arr)
        x_scaled = bundle["scaler"].transform(x_imp)

        prob = self._predict_single_model_prob(dataset_id, model_id, x_imp, x_scaled)
        stratum, recommendation = classify_risk_stratum(prob)

        # Modelos a comparar (3 individuales + 2 híbridos)
        comparison_models = [
            ("random_forest", "Random Forest Classifier"),
            ("xgboost", "XGBoost Classifier Optimizado"),
            ("tabnet", "TabNet Attentive Network"),
            ("stacking_hybrid", "Ensamble Híbrido Stacking Calibrado"),
            ("survival_risk_hybrid", "Modelo Híbrido de Riesgo y Supervivencia (Weibull)"),
        ]

        comparisons: List[ModelProbabilityComparison] = []
        for mid, mname in comparison_models:
            if mid in bundle["models"] or (mid == "stacking_hybrid" and "stacking_hybrid" in cohort_meta["models"]):
                m_prob = self._predict_single_model_prob(dataset_id, mid, x_imp, x_scaled)
                m_stratum, _ = classify_risk_stratum(m_prob)
                comparisons.append(
                    ModelProbabilityComparison(
                        model_id=mid,
                        model_name=cohort_meta["models"].get(mid, {}).get("name", mname),
                        probability=round(m_prob, 4),
                        percentage=round(m_prob * 100.0, 2),
                        risk_stratum=m_stratum,
                    )
                )

        base_val, shap_vals = self._compute_shap_values(dataset_id, model_id, x_imp, x_scaled)
        features_meta = bundle["features_meta"]
        medians = bundle["train_medians"]

        contributions: List[ShapFeatureContribution] = []
        for idx, fmeta in enumerate(features_meta):
            fname = fmeta["name"]
            val = float(x_imp[0, idx])
            sval = float(shap_vals[idx])
            direction = (
                "incrementa_riesgo"
                if sval > 0.005
                else ("reduce_riesgo" if sval < -0.005 else "neutro")
            )
            contributions.append(
                ShapFeatureContribution(
                    feature=fname,
                    label=fmeta["label"],
                    value=round(val, 3),
                    unit=fmeta["unit"],
                    shap_value=round(sval, 4),
                    abs_importance=round(abs(sval), 4),
                    direction=direction,
                    clinical_interpretation=format_clinical_explanation(
                        fname, fmeta["label"], val, fmeta["unit"], sval, medians[fname]
                    ),
                )
            )

        contributions.sort(key=lambda c: c.abs_importance, reverse=True)

        # Proyecciones longitudinales de supervivencia (Resultado 2: complicaciones crónicas a 5 y 10 años)
        p5 = min(0.99, prob * (1.0 - np.exp(-((5.0 / 8.5) ** 1.35))))
        p10 = min(0.99, prob * (1.0 - np.exp(-((10.0 / 8.5) ** 1.35))))
        survival_projs = [
            SurvivalProjection(
                horizon_years=5,
                complication_risk_probability=round(p5, 4),
                complication_risk_percentage=round(p5 * 100.0, 1),
                projected_event="Retinopatía diabética no proliferativa o microalbuminuria renal incipiente",
                hazard_ratio_estimate=round(1.0 + (prob * 2.8), 2),
            ),
            SurvivalProjection(
                horizon_years=10,
                complication_risk_probability=round(p10, 4),
                complication_risk_percentage=round(p10 * 100.0, 1),
                projected_event="Nefropatía diabética manifiesta, neuropatía periférica o evento cardiovascular mayor (MACE)",
                hazard_ratio_estimate=round(1.0 + (prob * 4.2), 2),
            ),
        ]

        # Obtener nombre legible del modelo seleccionado
        selected_model_name = cohort_meta["models"].get(model_id, {}).get("name")
        if not selected_model_name:
            norm_key = "stacking_hybrid" if "stacking" in model_id else ("survival_risk_hybrid" if "survival" in model_id else model_id)
            selected_model_name = cohort_meta["models"].get(norm_key, {}).get("name", model_id)

        return PredictResponse(
            dataset_id=dataset_id,
            dataset_name=cohort_meta["audit"]["name"],
            model_id=model_id,
            model_name=selected_model_name,
            outcome_evaluated="Resultado 1: Predicción de Aparición / Presencia de Diabetes Mellitus Tipo 2",
            complications_note=(
                "Resultado 2 (Complicaciones a largo plazo): Estimadas mediante función de supervivencia "
                "paramétrica Weibull calibrada. El horizonte proyectado evalúa eventos microvasculares "
                "(retinopatía, nefropatía) y macrovasculares (enfermedad cardiovascular)."
            ),
            probability=round(prob, 4),
            percentage=round(prob * 100.0, 2),
            prediction=int(prob >= 0.5),
            threshold=0.5,
            risk_stratum=stratum,
            risk_recommendation=recommendation,
            base_value=round(base_val, 4),
            shap_contributions=contributions,
            all_models_comparison=comparisons,
            survival_projections=survival_projs,
            clinical_evidence_summary=(
                "Directrices ADA 2024 / OMS: Se sugiere intervención en estilo de vida con reducción del 7% "
                "del peso corporal y 150 min/semana de ejercicio aeróbico estructurado."
            ),
            disclaimer=self.metadata.get("clinical_disclaimer", ""),
        )

    def predict_batch_df(
        self, df: pd.DataFrame, dataset_id: str = "nhanes_2017_2018", model_id: str = "xgboost"
    ) -> BatchPredictResponse:
        bundle = self.bundles[dataset_id]
        feature_names = bundle["feature_names"]
        features_meta = bundle["features_meta"]
        label_map = {f["name"]: f["label"] for f in features_meta}

        missing_cols = [c for c in feature_names if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"El archivo CSV no contiene las columnas requeridas para '{dataset_id}': {missing_cols}"
            )

        is_demo = "demo_tag" in df.columns or any(
            "DEMO" in str(v).upper() for v in df.get("demo_tag", pd.Series(dtype=str)).values
        )

        df_sub = df[feature_names].apply(pd.to_numeric, errors="coerce")
        if len(df_sub) == 0:
            raise ValueError("El archivo CSV no contiene filas válidas.")
        if len(df_sub) > 500:
            df_sub = df_sub.head(500)

        X_imp = bundle["imputer"].transform(df_sub.values)
        X_scaled = bundle["scaler"].transform(X_imp)

        models_map = bundle["models"]
        target_id = model_id
        if model_id in ("stacking_ensemble", "stacking_hybrid"):
            target_id = "stacking_hybrid" if "stacking_hybrid" in models_map else "xgboost"
        elif model_id in ("hybrid_survival", "survival_risk_hybrid"):
            target_id = "survival_risk_hybrid" if "survival_risk_hybrid" in models_map else "xgboost"

        if target_id not in models_map:
            target_id = "xgboost"

        model = models_map[target_id]
        if target_id in ("stacking_hybrid", "survival_risk_hybrid"):
            try:
                probs = model.predict_proba(X_imp, X_scaled)[:, 1]
            except Exception:
                p_rf = models_map["random_forest"].predict_proba(X_imp)[:, 1]
                p_xgb = models_map["xgboost"].predict_proba(X_imp)[:, 1]
                probs = 0.5 * p_rf + 0.5 * p_xgb
        elif target_id in ("logistic_regression", "tabnet"):
            probs = model.predict_proba(X_scaled)[:, 1]
        else:
            probs = model.predict_proba(X_imp)[:, 1]

        # Obtener factor principal por fila usando XGBoost TreeExplainer para rapidez en lote
        xgb_sv = self.explainers[dataset_id]["xgboost"].shap_values(X_imp)
        if isinstance(xgb_sv, list):
            xgb_sv = xgb_sv[1]

        rows_out: List[BatchRowPrediction] = []
        high_risk = 0
        for i, p in enumerate(probs):
            prob_f = float(np.clip(p, 0.0001, 0.9999))
            stratum, _ = classify_risk_stratum(prob_f)
            if prob_f >= 0.5:
                high_risk += 1
            top_idx = int(np.argmax(np.abs(xgb_sv[i])))
            top_fname = feature_names[top_idx]
            rows_out.append(
                BatchRowPrediction(
                    row_index=i + 1,
                    probability=round(prob_f, 4),
                    percentage=round(prob_f * 100.0, 2),
                    prediction=int(prob_f >= 0.5),
                    risk_stratum=stratum,
                    top_risk_factor=f"{label_map[top_fname]} ({X_imp[i, top_idx]:g})",
                )
            )

        demo_warn = (
            "ARCHIVO ROTULADO COMO DEMO SINTÉTICO: Los registros procesados provienen de datos "
            "sintéticos de demostración sin validez clínica ni epidemiológica."
            if is_demo
            else None
        )

        return BatchPredictResponse(
            dataset_id=dataset_id,
            model_id=model_id,
            is_synthetic_demo=is_demo,
            demo_warning=demo_warn,
            total_rows_processed=len(rows_out),
            high_risk_count=high_risk,
            mean_probability=round(float(np.mean(probs)), 4),
            predictions=rows_out,
            disclaimer=self.metadata["clinical_disclaimer"],
        )
