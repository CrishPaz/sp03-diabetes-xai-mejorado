"""Fase 3: métricas con IC bootstrap, pruebas con potencia real (DeLong, Nadeau-Bengio, McNemar,
bootstrap pareado, Friedman + Nemenyi), calibración, curva de decisión y equidad."""
from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List

import numpy as np
from scipy import stats
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss, matthews_corrcoef,
                             precision_recall_curve, roc_auc_score, roc_curve)

from diabetes_ml.config import interp, load, seed


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibration_slope_intercept(y, p) -> Dict[str, float]:
    import statsmodels.api as sm
    lp = _logit(p)
    try:
        slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1]
        inter = sm.GLM(y, np.ones_like(lp), family=sm.families.Binomial(), offset=lp).fit().params[0]
        return {"pendiente": float(slope), "intercepto": float(inter)}
    except Exception:
        return {"pendiente": float("nan"), "intercepto": float("nan")}


def ece(y, p, bins: int = 10) -> float:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def point_metrics(y, p, thr: float) -> Dict[str, float]:
    yhat = (p >= thr).astype(int)
    tp, tn = int(((yhat == 1) & (y == 1)).sum()), int(((yhat == 0) & (y == 0)).sum())
    fp, fn = int(((yhat == 1) & (y == 0)).sum()), int(((yhat == 0) & (y == 1)).sum())
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    ppv = tp / max(1, tp + fp)
    npv = tn / max(1, tn + fn)
    out = {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
           "exactitud": (tp + tn) / len(y), "exactitud_balanceada": (sens + spec) / 2,
           "sensibilidad": sens, "especificidad": spec, "vpp": ppv, "vpn": npv,
           "f1": 2 * ppv * sens / max(1e-12, ppv + sens), "mcc": matthews_corrcoef(y, yhat),
           "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)), "brier": brier_score_loss(y, p), "ece": ece(y, p)}
    out.update(calibration_slope_intercept(y, p))
    out = {k: round(float(v), 4) for k, v in out.items()}
    out["matriz_confusion"] = {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
    return out


def bootstrap_ci(y, p, thr, B: int, keys=("roc_auc", "pr_auc", "sensibilidad", "especificidad", "f1", "brier")) -> Dict[str, List[float]]:
    rng = np.random.default_rng(seed())
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    acc = {k: [] for k in keys}
    for _ in range(B):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])  # estratificado; 1 fila = 1 paciente
        yb, pb = y[idx], p[idx]
        yh = pb >= thr
        tp = (yh & (yb == 1)).sum()
        fn = (~yh & (yb == 1)).sum()
        tn = (~yh & (yb == 0)).sum()
        fp = (yh & (yb == 0)).sum()
        sens, ppv = tp / max(1, tp + fn), tp / max(1, tp + fp)
        vals = {"roc_auc": roc_auc_score(yb, pb), "pr_auc": average_precision_score(yb, pb), "sensibilidad": sens,
                "especificidad": tn / max(1, tn + fp), "f1": 2 * ppv * sens / max(1e-12, ppv + sens),
                "brier": float(np.mean((pb - yb) ** 2))}
        for k in keys:
            acc[k].append(vals[k])
    return {k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] for k, v in acc.items()}


# ---------------- DeLong (Sun & Xu, 2014) ----------------
def _midrank(x):
    order = np.argsort(x)
    xs = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def delong(y, p1, p2) -> Dict[str, float]:
    pos, neg = p1[y == 1], p1[y == 0]
    preds = np.vstack([p1, p2])
    m, n = int((y == 1).sum()), int((y == 0).sum())
    X, Y = preds[:, y == 1], preds[:, y == 0]
    tx = np.array([_midrank(r) for r in X])
    ty = np.array([_midrank(r) for r in Y])
    tz = np.array([_midrank(r) for r in np.hstack([X, Y])])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    return {"auc_1": float(aucs[0]), "auc_2": float(aucs[1]), "z": float(z), "p": float(2 * stats.norm.sf(abs(z)))}


def mcnemar(y, yhat1, yhat2) -> Dict[str, Any]:
    c1, c2 = yhat1 == y, yhat2 == y
    b, c = int((c1 & ~c2).sum()), int((~c1 & c2).sum())
    if b + c == 0:
        return {"b": b, "c": c, "estadistico": 0.0, "p": 1.0, "metodo": "sin discordantes"}
    if b + c < 25:
        return {"b": b, "c": c, "estadistico": float(min(b, c)), "p": float(stats.binomtest(min(b, c), b + c).pvalue), "metodo": "exacto binomial"}
    chi = (abs(b - c) - 1) ** 2 / (b + c)
    return {"b": b, "c": c, "estadistico": float(chi), "p": float(stats.chi2.sf(chi, 1)), "metodo": "chi2 con corrección de Edwards"}


def paired_bootstrap_auc(y, p1, p2, B: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed())
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    d = []
    for _ in range(B):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        d.append(roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p2[idx]))
    d = np.array(d)
    return {"diferencia": float(roc_auc_score(y, p1) - roc_auc_score(y, p2)),
            "ic95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}


def holm(pvals: List[float]) -> List[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return [float(a) for a in adj]


def nadeau_bengio(a: List[float], b: List[float], n_train: int, n_test: int) -> Dict[str, float]:
    d = np.array(a) - np.array(b)
    k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return {"t": 0.0, "p": 1.0, "media_diferencia": float(d.mean())}
    t = d.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    return {"t": float(t), "p": float(2 * stats.t.sf(abs(t), k - 1)), "media_diferencia": float(d.mean())}


def nested_tests(outer_auc: Dict[str, List[float]], n_train: int, n_test: int) -> Dict[str, Any]:
    import pandas as pd
    import scikit_posthocs as sp
    names = list(outer_auc)
    fr = stats.friedmanchisquare(*[outer_auc[m] for m in names]) if len(names) >= 3 else None
    nem = sp.posthoc_nemenyi_friedman(pd.DataFrame(outer_auc).values)
    nem.index = nem.columns = names
    ranks = pd.DataFrame(outer_auc).rank(axis=1, ascending=False).mean().to_dict()
    pairs = []
    for a, b in combinations(names, 2):
        r = nadeau_bengio(outer_auc[a], outer_auc[b], n_train, n_test)
        r.update({"modelo_a": a, "modelo_b": b, "p_nemenyi": float(nem.loc[a, b])})
        pairs.append(r)
    adj = holm([r["p"] for r in pairs])
    for r, pa in zip(pairs, adj):
        r["p_holm"] = pa
    k = len(next(iter(outer_auc.values())))
    cd = float(stats.studentized_range.ppf(0.95, len(names), np.inf) / np.sqrt(2) * np.sqrt(len(names) * (len(names) + 1) / (6 * k)))
    return {"friedman": None if fr is None else {"chi2": float(fr.statistic), "p": float(fr.pvalue)},
            "rangos_medios": {k2: round(v, 3) for k2, v in ranks.items()}, "diferencia_critica_nemenyi": round(cd, 3),
            "pares_nadeau_bengio": pairs,
            "nota_wilcoxon": f"Con {k} pliegues, Wilcoxon bilateral tiene un p mínimo exacto de {2 / 2 ** k:.4f}; se reporta solo como descriptivo.",
            "interpretabilidad": interp(
                "Comparación de modelos sobre las estimaciones externas de la CV anidada.",
                "Friedman contrasta si algún modelo difiere; Nemenyi y Nadeau-Bengio (t corregida por solapamiento de pliegues) comparan pares, con ajuste de Holm.",
                ("Friedman p = %.3f." % fr.pvalue if fr else "") + (" No se demostraron diferencias tras Holm." if all(r["p_holm"] >= 0.05 for r in pairs) else " Hay pares con diferencia significativa tras Holm."),
                f"Solo {k} observaciones externas por modelo: potencia baja.")}


def test_tests(y, probs: Dict[str, np.ndarray], thresholds: Dict[str, float], B: int) -> Dict[str, Any]:
    rows = []
    for a, b in combinations(probs, 2):
        dl = delong(y, probs[a], probs[b])
        mc = mcnemar(y, (probs[a] >= thresholds[a]).astype(int), (probs[b] >= thresholds[b]).astype(int))
        bs = paired_bootstrap_auc(y, probs[a], probs[b], B=min(B, 1000))
        rows.append({"modelo_a": a, "modelo_b": b, "delong_z": round(dl["z"], 3), "delong_p": dl["p"],
                     "mcnemar_p": mc["p"], "mcnemar_metodo": mc["metodo"], "dif_auc": round(bs["diferencia"], 4),
                     "dif_auc_ic95": [round(x, 4) for x in bs["ic95"]]})
    for key in ("delong_p", "mcnemar_p"):
        for r, pa in zip(rows, holm([r[key] for r in rows])):
            r[key + "_holm"] = pa
    sig = [f"{r['modelo_a']} vs {r['modelo_b']}" for r in rows if r["delong_p_holm"] < 0.05]
    return {"pares": rows, "interpretabilidad": interp(
        "Comparaciones pareadas en la prueba aislada.",
        "DeLong compara AUC sobre los mismos pacientes; McNemar compara aciertos al umbral fijado en Validación; p ajustados por Holm.",
        ("Diferencias de AUC significativas tras Holm: " + ", ".join(sig)) if sig else "No se demostraron diferencias de AUC tras Holm; no se afirma un «mejor modelo».",
        "Una sola partición de prueba; la significancia depende de su tamaño.")}


def curves(y, p) -> Dict[str, Any]:
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    sel = lambda a: np.linspace(0, len(a) - 1, min(len(a), 120)).astype(int)
    i, j = sel(fpr), sel(prec)
    q = np.quantile(p, np.linspace(0, 1, 11))
    idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, 9)
    cal = [{"pred": float(p[idx == b].mean()), "obs": float(y[idx == b].mean()), "n": int((idx == b).sum())} for b in range(10) if (idx == b).any()]
    return {"roc": {"fpr": fpr[i].round(4).tolist(), "tpr": tpr[i].round(4).tolist()},
            "pr": {"recall": rec[j].round(4).tolist(), "precision": prec[j].round(4).tolist(), "prevalencia": float(y.mean())},
            "calibracion": cal, "histograma_prob": np.histogram(p, bins=20, range=(0, 1))[0].tolist()}


def decision_curve(y, p, thresholds=None) -> List[Dict[str, float]]:
    thresholds = np.round(np.arange(0.05, 0.51, 0.01), 2) if thresholds is None else thresholds
    n, prev = len(y), y.mean()
    out = []
    for t in thresholds:
        yh = p >= t
        tp, fp = (yh & (y == 1)).sum(), (yh & (y == 0)).sum()
        w = t / (1 - t)
        out.append({"umbral": float(t), "modelo": float(tp / n - fp / n * w), "tratar_todos": float(prev - (1 - prev) * w), "ninguno": 0.0})
    return out


def fairness(y, p, thr, groups: Dict[str, np.ndarray], B: int = 500) -> Dict[str, Any]:
    out = {}
    for gname, g in groups.items():
        rows = []
        for val in sorted(set(g)):
            m = g == val
            if m.sum() < 10 or len(set(y[m])) < 2:
                continue
            pm = point_metrics(y[m], p[m], thr)
            ci = bootstrap_ci(y[m], p[m], thr, B, keys=("roc_auc", "sensibilidad", "especificidad"))
            rows.append({"grupo": str(val), "n": int(m.sum()), "eventos": int(y[m].sum()), "roc_auc": pm["roc_auc"],
                         "sensibilidad": pm["sensibilidad"], "especificidad": pm["especificidad"], "ic95": ci,
                         "prob_media": round(float(p[m].mean()), 4), "prevalencia_obs": round(float(y[m].mean()), 4)})
        sens = [r["sensibilidad"] for r in rows]
        gap = round(max(sens) - min(sens), 4) if sens else 0.0
        out[gname] = {"filas": rows, "brecha_sensibilidad": gap, "alerta": gap > 0.10}
    return out


def evaluate_test(y, probs: Dict[str, np.ndarray], thresholds: Dict[str, float], groups: Dict[str, np.ndarray]) -> Dict[str, Any]:
    B = load("cv")["bootstrap"]["B"]
    per_model = {}
    for m, p in probs.items():
        per_model[m] = {"metricas": point_metrics(y, p, thresholds[m]), "ic95": bootstrap_ci(y, p, thresholds[m], B),
                        "umbral": thresholds[m], "curvas": curves(y, p), "curva_decision": decision_curve(y, p),
                        "equidad": fairness(y, p, thresholds[m], groups)}
    best = max(per_model, key=lambda k: per_model[k]["metricas"]["roc_auc"])
    return {"por_modelo": per_model, "pruebas": test_tests(y, probs, thresholds, B),
            "interpretabilidad": interp(
                "Métricas en la prueba aislada (15 %), leída una sola vez.",
                f"IC 95 % por bootstrap estratificado (B = {B}); el umbral proviene de Validación.",
                f"El AUC más alto es de {best} ({per_model[best]['metricas']['roc_auc']}); ver la tabla de pruebas antes de declarar superioridad.",
                "Prevalencia de la muestra NHANES no ponderada; no extrapolar VPP/VPN a otras poblaciones.")}
