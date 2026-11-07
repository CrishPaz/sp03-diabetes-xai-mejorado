"""
Módulo de modelos híbridos y de supervivencia:
- StackingHybridClassifier: Meta-clasificador calibrado sobre predicciones out-of-fold de RF, XGBoost y TabNet.
- SurvivalRiskHybridModel: Combinación de riesgo f(x) con función Weibull para supervivencia libre de eventos.
"""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
from sklearn.linear_model import LogisticRegression


class StackingHybridClassifier:
    """Ensamble Híbrido Stacking que calibra las predicciones out-of-fold de RF, XGBoost y TabNet."""

    def __init__(self, base_models: Dict[str, Any], meta_learner: LogisticRegression) -> None:
        self.base_models = base_models
        self.meta_learner = meta_learner

    def predict_proba(self, X_imp: np.ndarray, X_scaled: np.ndarray) -> np.ndarray:
        p_rf = self.base_models["random_forest"].predict_proba(X_imp)[:, 1]
        p_xgb = self.base_models["xgboost"].predict_proba(X_imp)[:, 1]
        p_tab = self.base_models["tabnet"].predict_proba(X_scaled)[:, 1]
        stack_X = np.column_stack([p_rf, p_xgb, p_tab])
        return self.meta_learner.predict_proba(stack_X)


class SurvivalRiskHybridModel:
    """
    Modelo Híbrido de Riesgo / Supervivencia (Resultado 2 / Caso03):
    Combina la probabilidad de riesgo cardiometabólico basal f(x) con una función
    paramétrica de riesgo acumulado h(t) = λγ(λt)^(γ-1) para proyectar supervivencia libre
    de complicaciones crónicas a 60 y 120 meses.
    """

    def __init__(self, base_xgb: Any, baseline_lambda: float = 0.005, shape_gamma: float = 1.35) -> None:
        self.base_xgb = base_xgb
        self.baseline_lambda = baseline_lambda
        self.shape_gamma = shape_gamma

    def predict_proba(self, X_imp: np.ndarray, X_scaled: np.ndarray = None) -> np.ndarray:
        return self.base_xgb.predict_proba(X_imp)

    def predict_survival_curve(self, X_imp: np.ndarray, time_points_months: List[float]) -> np.ndarray:
        p_base = self.base_xgb.predict_proba(X_imp)[:, 1]
        surv_curves = np.zeros((len(X_imp), len(time_points_months)))
        for i, p in enumerate(p_base):
            rr = np.exp((p - 0.15) * 3.2)
            for j, t in enumerate(time_points_months):
                cum_hazard = (self.baseline_lambda * t) ** self.shape_gamma * rr
                surv_curves[i, j] = np.exp(-cum_hazard)
        return surv_curves
