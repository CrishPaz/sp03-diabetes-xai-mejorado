"""Fase 4: SHAP, máscaras de atención TabNet, equivalente tabular de Grad-CAM++ (Integrated Gradients,
Saliency, Gradiente x Entrada sobre la red PyTorch de TabNet), concordancia, estabilidad y plausibilidad."""
from __future__ import annotations

from itertools import combinations
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from diabetes_ml.config import interp, load, seed

GRADCAM_NOTE = ("Grad-CAM++ pondera mapas de activación de capas convolucionales; un modelo tabular no los tiene. "
                "Se usa su equivalente por gradientes (Integrated Gradients, Saliency, Gradiente x Entrada) sobre TabNet.")
CAUSAL_NOTE = "Las explicaciones describen asociaciones aprendidas por el modelo, no relaciones causales ni recomendaciones terapéuticas."


def _transform(pipe, X):
    Z = X.values.astype(float)
    for name, step in pipe.steps[:-1]:
        Z = step.transform(Z)
    return Z


def shap_values(name: str, pipe, X: pd.DataFrame, background: pd.DataFrame) -> np.ndarray:
    import shap
    Z, B = _transform(pipe, X), _transform(pipe, background)
    clf = pipe.named_steps["clf"]
    if name in ("xgb", "rf"):
        sv = shap.TreeExplainer(clf).shap_values(Z, check_additivity=True)
    else:
        sv = shap.LinearExplainer(clf, B).shap_values(Z)
    sv = sv[1] if isinstance(sv, list) else sv
    return sv[..., 1] if sv.ndim == 3 else sv


def tabnet_masks(pipe, X: pd.DataFrame) -> Dict[str, Any]:
    Z = _transform(pipe, X).astype(np.float32)
    agg, masks = pipe.named_steps["clf"].model_.explain(Z)
    return {"agregada": agg, "por_paso": {int(k): v for k, v in masks.items()},
            "importancia_global": pipe.named_steps["clf"].model_.feature_importances_}


def gradient_attributions(pipe, X: pd.DataFrame, baseline: np.ndarray, steps: int = 50) -> Dict[str, Any]:
    import torch
    net = pipe.named_steps["clf"].model_.network
    net.eval()
    Z = torch.tensor(_transform(pipe, X), dtype=torch.float32)
    b = torch.tensor(baseline, dtype=torch.float32).unsqueeze(0)

    def f(inp):
        return torch.softmax(net(inp)[0], dim=1)[:, 1]

    Zs = Z.clone().requires_grad_(True)
    sal = torch.autograd.grad(f(Zs).sum(), Zs)[0]
    total = torch.zeros_like(Z)
    for a in np.linspace(0, 1, steps):
        xi = (b + a * (Z - b)).requires_grad_(True)
        total += torch.autograd.grad(f(xi).sum(), xi)[0]
    ig = (Z - b) * total / steps
    with torch.no_grad():
        delta = (ig.sum(1) - (f(Z) - f(b.expand_as(Z)))).abs().mean().item()
    return {"ig": ig.detach().numpy(), "saliency": sal.abs().numpy(), "grad_x_input": (sal * Z).detach().numpy(),
            "delta_convergencia": round(delta, 5)}


def permutation_importance(predict, X: pd.DataFrame, y, repeats: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed())
    base = roc_auc_score(y, predict(X))
    out = []
    for c in X.columns:
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[c] = rng.permutation(Xp[c].values)
            drops.append(base - roc_auc_score(y, predict(Xp)))
        out.append(np.mean(drops))
    return np.array(out)


def explain_bundle(bundle: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
    feats, final, cal = bundle["predictores"], bundle["final"], bundle["calibradores"]
    Xt, yt = bundle["test"]["X"], bundle["test"]["y"]
    bg = bundle["dev_X"].iloc[:100]
    rankings, out = {}, {"nota_gradcam": GRADCAM_NOTE, "advertencia": CAUSAL_NOTE, "global": {}}
    for m in ("xgb", "rf", "lr"):
        if m in final:
            sv = shap_values(m, final[m], Xt, bg)
            imp = np.abs(sv).mean(0)
            rankings[f"shap_{m}"] = imp
            out["global"][f"shap_{m}"] = dict(zip(feats, imp.round(5).tolist()))
            if m == "xgb":
                Zt = _transform(final[m], Xt)
                exp = load("features")["direccion_esperada"]
                plaus = []
                for j, fname in enumerate(feats):
                    rho = stats.spearmanr(Zt[:, j], sv[:, j]).statistic if np.std(Zt[:, j]) > 0 else 0.0
                    sign = "+" if rho > 0.05 else "-" if rho < -0.05 else "0"
                    plaus.append({"variable": fname, "rango_importancia": int(np.argsort(-imp).tolist().index(j)) + 1,
                                  "direccion_modelo": sign, "direccion_ada": exp.get(fname, "?"),
                                  "coherente": "sí" if sign == exp.get(fname) else "revisar"})
                out["plausibilidad"] = sorted(plaus, key=lambda r: r["rango_importancia"])
                order = np.argsort(-imp)[:3]
                out["dependencia_xgb"] = {feats[j]: {"x": Zt[:, j].round(3).tolist()[:400], "shap": sv[:, j].round(4).tolist()[:400]} for j in order}
    for m in final:
        rankings[f"permutacion_{m}"] = permutation_importance(lambda X, m=m: cal[m](final[m].predict_proba(X)[:, 1]), Xt, yt)
        out["global"][f"permutacion_{m}"] = dict(zip(feats, rankings[f"permutacion_{m}"].round(5).tolist()))
    if "tabnet" in final:
        rng = np.random.default_rng(seed())
        pos, neg = np.where(yt == 1)[0], np.where(yt == 0)[0]
        k = min(200, len(yt))
        idx = np.concatenate([rng.choice(pos, min(len(pos), k // 2), replace=False), rng.choice(neg, min(len(neg), k - k // 2), replace=False)])
        Xs, ys = Xt.iloc[idx], yt[idx]
        masks = tabnet_masks(final["tabnet"], Xs)
        p = final["tabnet"].predict_proba(Xs)[:, 1]
        order = np.argsort(-p)
        base = np.array([bundle["referencia"]["medianas"][f] for f in feats])
        base = final["tabnet"].named_steps["sc"].transform(base.reshape(1, -1))[0]
        grads = gradient_attributions(final["tabnet"], Xs, base)
        rankings["atencion_tabnet"] = masks["importancia_global"]
        rankings["ig_tabnet"] = np.abs(grads["ig"]).mean(0)
        out["global"]["atencion_tabnet"] = dict(zip(feats, masks["importancia_global"].round(5).tolist()))
        out["global"]["ig_tabnet"] = dict(zip(feats, rankings["ig_tabnet"].round(5).tolist()))
        sparsity = {s: float((m_ > 1e-6).mean()) for s, m_ in masks["por_paso"].items()}
        out["tabnet"] = {
            "mapa_calor": {"pacientes_por_prob": p[order].round(4).tolist(), "matriz": masks["agregada"][order].round(4).tolist()},
            "mascara_media_por_clase": {"con_dm2": masks["agregada"][ys == 1].mean(0).round(4).tolist(), "sin_dm2": masks["agregada"][ys == 0].mean(0).round(4).tolist()},
            "paciente_ejemplo": {"pasos": {s: m_[order[0]].round(4).tolist() for s, m_ in masks["por_paso"].items()}, "agregada": masks["agregada"][order[0]].round(4).tolist(), "prob": float(p[order[0]])},
            "dispersion_por_paso": sparsity, "delta_convergencia_ig": grads["delta_convergencia"],
            "ig_local_ejemplo": grads["ig"][order[0]].round(5).tolist(),
            "saliency_global": dict(zip(feats, grads["saliency"].mean(0).round(5).tolist())),
            "grad_x_input_global": dict(zip(feats, np.abs(grads["grad_x_input"]).mean(0).round(5).tolist()))}
    names = list(rankings)
    agree = []
    for a, b in combinations(names, 2):
        top_a, top_b = set(np.argsort(-rankings[a])[:5]), set(np.argsort(-rankings[b])[:5])
        agree.append({"metodo_a": a, "metodo_b": b, "spearman": round(float(stats.spearmanr(rankings[a], rankings[b]).statistic), 3),
                      "jaccard_top5": round(len(top_a & top_b) / len(top_a | top_b), 3)})
    out["concordancia"] = agree
    gains = results["cv_anidada"].get("importancia_xgb_por_pliegue") or []
    if gains:
        G = pd.DataFrame(gains)
        out["estabilidad_pliegues"] = [{"variable": c, "media": round(float(G[c].mean()), 4), "rango": [round(float(G[c].min()), 4), round(float(G[c].max()), 4)],
                                        "cv": round(float(G[c].std() / G[c].mean()), 3) if G[c].mean() > 0 else None} for c in G.columns]
    med = np.median([r["spearman"] for r in agree]) if agree else float("nan")
    incoh = [r["variable"] for r in out.get("plausibilidad", []) if r["coherente"] == "revisar"]
    out["interpretabilidad"] = interp(
        "Importancia global por método y concordancia entre métodos.",
        "Spearman alto y Jaccard top-5 alto indican que los métodos coinciden en qué variables importan.",
        f"Spearman mediano entre métodos = {med:.2f}. Variables a revisar por dirección: {', '.join(incoh) or 'ninguna'}.",
        CAUSAL_NOTE)
    return out
