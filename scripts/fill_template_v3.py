"""Calcula, desde el sistema, los datos que faltan en la plantilla del artículo (campos [ ] resaltados)
y ejecuta los análisis de sensibilidad de la sección 2.11. Salida: docs/DATOS_PLANTILLA_V3.md y .json."""
import json
import os
import platform
import sys
from pathlib import Path

os.environ["OPTUNA_STORAGE"] = "memory"
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.metrics import brier_score_loss, roc_auc_score  # noqa: E402

from diabetes_ml import data as D, evaluation as E, models as M, training as T  # noqa: E402
from diabetes_ml.config import ARTIFACTS, load, set_seeds  # noqa: E402
from diabetes_ml.inference import bundle  # noqa: E402
from diabetes_ml.splitting import split_70_15_15  # noqa: E402

NAMES = {"lr": "Reg. logística L2", "rf": "Random Forest", "xgb": "XGBoost", "tabnet": "TabNet", "stack": "Stacking calibrado"}
NH = "nhanes_2017_2018"
CODES = {"edad": ("DEMO_J", "RIDAGEYR", "años"), "imc": ("BMX_J", "BMXBMI", "kg/m²"), "glucosa": ("BIOPRO_J", "LBXSGL", "mg/dL"),
         "hba1c": ("GHB_J", "LBXGH", "%"), "presion_sistolica": ("BPX_J", "BPXSY1", "mmHg"), "presion_diastolica": ("BPX_J", "BPXDI1", "mmHg"),
         "colesterol_total": ("TCHOL_J", "LBXTC", "mg/dL"), "colesterol_hdl": ("HDL_J", "LBDHDD", "mg/dL"), "trigliceridos": ("BIOPRO_J", "LBXSTR", "mg/dL"),
         "antecedente_familiar_diabetes": ("MCQ_J", "MCQ300C = 1", "0/1"), "antecedente_hipertension": ("BPQ_J", "BPQ020 = 1", "0/1"),
         "tratamiento_antihipertensivo": ("BPQ_J", "BPQ020 = 1 y BPQ040A = 1", "0/1")}
PIMA_UNITS = {"embarazos": "n", "glucosa": "mg/dL (2 h PTOG)", "presion_diastolica": "mmHg", "pliegue_cutaneo": "mm", "insulina": "µU/mL (2 h)",
              "imc": "kg/m²", "pedigri_diabetes": "índice", "edad": "años"}
OUT, MD = {}, []


def res(key):
    return json.loads((ARTIFACTS / key / "results.json").read_text(encoding="utf-8"))


def exp(key):
    return json.loads((ARTIFACTS / key / "explain.json").read_text(encoding="utf-8"))


def table(head, rows):
    return ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows] + [""]


def f3(x):
    return f"{x:.3f}"


def section(title, where):
    MD.extend([f"## {title}", "", f"**Dónde va en la plantilla:** {where}", ""])


def main():
    global MD
    set_seeds()
    df, featsA, _ = D.cohort_frame(NH, "A")
    featsB = load("features")[NH]["escenarios"]["B"]["predictores"]
    parts = split_70_15_15(df)
    test = df.loc[parts["prueba"]].reset_index(drop=True)
    MD.extend(["# Datos para completar la plantilla del artículo (SP-03 · pipeline v3)", "",
               "Generado por `scripts/fill_template_v3.py` desde los artefactos reales (`models/v3/`) y los datos crudos. "
               "Cada sección indica el lugar exacto de la plantilla y el valor que reemplaza al campo resaltado. "
               "Las cifras no deben editarse a mano.", ""])

    # 1. Exclusiones
    flow = D.load_nhanes()[1]["flujo_exclusiones"]
    exc = {"edad_menor_20": flow[1]["excluidos"], "DIQ010_invalido": flow[2]["excluidos"], "sin_hba1c_glucosa_imc": flow[3]["excluidos"]}
    OUT["exclusiones"] = {"inicial": flow[0]["n"], **exc, "total_excluidos": flow[0]["n"] - flow[-1]["n"], "final": flow[-1]["n"]}
    section("1. Exclusiones de NHANES", "§2.1 (texto) y Figura 1 (caja «Excluidos»)")
    MD += table(["Paso", "n restante", "Excluidos"], [[s["paso"], s["n"], s["excluidos"] if s["excluidos"] is not None else "—"] for s in flow])
    MD += [f"Texto para la Figura 1: «Excluidos (n = {OUT['exclusiones']['total_excluidos']:,}): edad < 20 años: n = {exc['edad_menor_20']:,}; "
           f"DIQ010 inválido (7/9/faltante): n = {exc['DIQ010_invalido']:,}; sin HbA1c, glucosa o IMC: n = {exc['sin_hba1c_glucosa_imc']:,}».", ""]

    # 2. Particiones y tamaño muestral
    rows, psz = [], {}
    for k in ("desarrollo", "validacion", "prueba"):
        sub = df.loc[parts[k]]
        psz[k] = {"n": len(sub), "eventos": int(sub.outcome.sum())}
        rows.append([k, len(sub), int(sub.outcome.sum()), f"{100 * sub.outcome.mean():.1f} %"])
    pima = D.cohort_frame("pima_indians", "A")[0]
    pp = split_70_15_15(pima)
    OUT["particiones"] = {"nhanes": psz, "pima": {k: {"n": len(v), "eventos": int(pima.loc[v, "outcome"].sum())} for k, v in pp.items()}}
    epp = psz["desarrollo"]["eventos"] / len(featsB)
    OUT["eventos_por_parametro_B"] = round(epp, 1)
    section("2. Tamaño de las particiones y eventos", "§2.4 (texto «alrededor de 540 eventos… 117 eventos») y Figura 1 (cajas de partición)")
    MD += table(["Partición NHANES", "n", "Eventos", "Prevalencia"], rows)
    MD += [f"Eventos por parámetro candidato en el escenario B: {psz['desarrollo']['eventos']} / {len(featsB)} = {epp:.1f}; escenario A: "
           f"{psz['desarrollo']['eventos']} / {len(featsA)} = {psz['desarrollo']['eventos'] / len(featsA):.1f}. "
           f"Pima: desarrollo n = {len(pp['desarrollo'])} ({int(pima.loc[pp['desarrollo'], 'outcome'].sum())} eventos), validación n = {len(pp['validacion'])}, "
           f"prueba n = {len(pp['prueba'])} ({int(pima.loc[pp['prueba'], 'outcome'].sum())} eventos).", ""]

    # 3. Tabla 1
    t1 = D.table_one(df, featsA)
    pos, neg = df[df.outcome == 1], df[df.outcome == 0]
    fem = lambda s: f"{int((s.sexo == 'Femenino').sum())} ({100 * (s.sexo == 'Femenino').mean():.1f} %)"
    p_sex = stats.chi2_contingency(pd.crosstab(df.sexo, df.outcome))[1]
    miss = lambda s: f"{100 * s[featsA].isna().any(axis=1).mean():.1f} %"
    OUT["tabla1"] = t1["filas"]
    section("3. Tabla 1 (NHANES por estado de DM2)", "§3.1, Tabla 1 — reemplaza todas las celdas [ ]")
    MD += [f"Sin DM2 n = {len(neg):,}; con DM2 n = {len(pos):,}. Media ± DE si ambas clases son normales (D'Agostino-Pearson); si no, mediana [RIC].", ""]
    rows = [[r["variable"], r["sin_dm2"], r["con_dm2"], r["smd"], f"{r['faltantes_pct']} %", r["p_descriptivo"]] for r in t1["filas"]]
    rows.insert(1, ["sexo femenino, n (%)", fem(neg), fem(pos), "—", "0.0 %", float(f"{p_sex:.3g}")])
    rows.append(["participantes con ≥ 1 dato imputado", miss(neg), miss(pos), "—", "—", "—"])
    MD += table(["Variable", "Sin DM2", "Con DM2", "SMD", "% faltantes (total)", "p descriptivo"], rows)

    # 4. Tabla S0
    s0 = []
    for f in featsA:
        x = df[f]
        s0.append([f, CODES[f][0], CODES[f][1], CODES[f][2], f"{x.min():g}–{x.max():g}", f"{100 * x.isna().mean():.1f} %", "sí" if f in featsB else "solo A"])
    pdf = D.cohort_frame("pima_indians", "A")[0]
    s0p = [[f, PIMA_UNITS[f], f"{pdf[f].min():g}–{pdf[f].max():g}", f"{100 * pdf[f].isna().mean():.1f} %"] for f in load("features")["pima_indians"]["escenarios"]["A"]["predictores"]]
    OUT["tabla_s0"] = s0
    section("4. Tabla S0 (diccionario de variables)", "Material suplementario, Tabla S0; también resuelve el pendiente de §2.2 (códigos NHANES) y §2.3 (faltantes)")
    MD += table(["Variable (código)", "Archivo NHANES", "Variable NHANES", "Unidad", "Rango observado", "% faltantes", "Escenario B"], s0)
    MD += ["Pima Indians (ceros fisiológicamente imposibles convertidos a NaN antes de imputar):", ""]
    MD += table(["Variable", "Unidad", "Rango observado", "% faltantes"], s0p)
    tg = df["trigliceridos"].isna().mean() * 100
    MD += [f"**Fuente de triglicéridos (§2.3):** LBXSTR del panel bioquímico estándar (BIOPRO_J), no LBXTR de la submuestra en ayunas. "
           f"Faltantes: {tg:.1f} %, por lo que la imputación por mediana afecta a una fracción pequeña. La glucosa (LBXSGL) también proviene de BIOPRO_J "
           "y no requiere ayuno.", "",
           "**Verificación de predictores (§2.2):** escenario B = " + ", ".join(featsB) + f" ({len(featsB)}); escenario A añade hba1c y glucosa ({len(featsA)}). Coincide con configs/features.yaml.", ""]

    # 5. Δ AUC A→B con IC pareado
    bA, bB = bundle(f"{NH}__A"), bundle(f"{NH}__B")
    y = bA["test"]["y"]
    rng = np.random.default_rng(42)
    pi, ni = np.where(y == 1)[0], np.where(y == 0)[0]
    boots = [np.concatenate([rng.choice(pi, len(pi)), rng.choice(ni, len(ni))]) for _ in range(2000)]
    drow, OUT["delta_auc"] = [], {}
    for m in bA["test"]["probs"]:
        pa, pb = bA["test"]["probs"][m], bB["test"]["probs"][m]
        d = np.array([roc_auc_score(y[i], pb[i]) - roc_auc_score(y[i], pa[i]) for i in boots])
        dl = E.delong(y, pb, pa)
        point = roc_auc_score(y, pb) - roc_auc_score(y, pa)
        OUT["delta_auc"][m] = {"delta": round(point, 3), "ic95": [round(np.percentile(d, 2.5), 3), round(np.percentile(d, 97.5), 3)], "p_delong": dl["p"]}
        drow.append([NAMES[m], f"{point:.3f}", f"[{np.percentile(d, 2.5):.3f}, {np.percentile(d, 97.5):.3f}]", f"{dl['p']:.2e}"])
    section("5. IC 95 % de cada Δ ROC-AUC (A → B)", "§3.3, nota de la Tabla 3 («Añadir el IC 95 % de cada Δ»)")
    MD += ["Bootstrap pareado estratificado (B = 2000) sobre la misma prueba aislada de 726 pacientes, que es idéntica en ambos escenarios; p de DeLong pareado.", ""]
    MD += table(["Modelo", "Δ ROC-AUC (B − A)", "IC 95 %", "p DeLong"], drow)

    # 6. Pruebas estadísticas
    section("6. Pares Nadeau–Bengio, Nemenyi y McNemar", "§3.4, Tabla 4 (columna «Nadeau–Bengio + Holm») y nota; Tabla S2 para Pima")
    OUT["pruebas"] = {}
    for key in (f"{NH}__A", f"{NH}__B", "pima_indians__A", "pima_indians__B"):
        r = res(key)
        pr = r["cv_anidada"]["pruebas"]
        nb = pr["pares_nadeau_bengio"]
        sig = [f"{p['modelo_a']} vs {p['modelo_b']}" for p in nb if p["p_holm"] < 0.05]
        nem = [f"{p['modelo_a']} vs {p['modelo_b']}" for p in nb if p["p_nemenyi"] < 0.05]
        mc = [f"{p['modelo_a']} vs {p['modelo_b']}" for p in r["evaluacion_prueba"]["pruebas"]["pares"] if p["mcnemar_p_holm"] < 0.05]
        OUT["pruebas"][key] = {"nadeau_bengio_holm_sig": sig, "nemenyi_sig": nem, "mcnemar_holm_sig": mc, "rangos": pr["rangos_medios"], "cd": pr["diferencia_critica_nemenyi"]}
        MD += [f"### {key}", "", f"- Nadeau–Bengio + Holm significativos: {', '.join(sig) or 'ninguno'}.",
               f"- Nemenyi significativos (p < 0.05): {', '.join(nem) or 'ninguno'}. Rangos medios: " + ", ".join(f"{k} {v}" for k, v in pr["rangos_medios"].items()) + f"; diferencia crítica = {pr['diferencia_critica_nemenyi']}.",
               f"- McNemar + Holm al umbral de Youden: {', '.join(mc) or 'sin diferencias'}.", ""]
        MD += table(["Par", "Dif. media AUC", "t", "p", "p Holm", "p Nemenyi"],
                    [[f"{p['modelo_a']} vs {p['modelo_b']}", f"{p['media_diferencia']:.4f}", f"{p['t']:.2f}", f"{p['p']:.4f}", f"{p['p_holm']:.4f}", f"{p['p_nemenyi']:.4f}"] for p in nb])

    # 7. Conformal ambiguo y decil superior
    section("7. Conjuntos conformales ambiguos y decil superior", "§3.5 («Reportar la proporción de pacientes con conjunto ambiguo {0, 1}») y verificación de «probabilidad media del decil superior ≤ 0.63»")
    rows, OUT["conformal"] = [], {}
    for key in (f"{NH}__B", f"{NH}__A"):
        b = bundle(key)
        for m, p in b["test"]["probs"].items():
            q = b["conformal_q"][m]
            sets = [(1 - p_ <= q) + (p_ <= q) * 1 for p_ in p]   # tamaño del conjunto: incluye 1 si 1-p<=q, incluye 0 si p<=q
            amb = float(np.mean([((1 - p_) <= q) and (p_ <= q) for p_ in p]))
            empty = float(np.mean([((1 - p_) > q) and (p_ > q) for p_ in p]))
            dec = float(np.mean(np.sort(p)[-int(np.ceil(len(p) / 10)):]))
            OUT["conformal"][f"{key}__{m}"] = {"ambiguo": round(amb, 3), "vacio": round(empty, 3), "decil_superior": round(dec, 3)}
            rows.append([key.split("__")[1], NAMES[m], f"{100 * amb:.1f} %", f"{100 * empty:.1f} %", f"{dec:.3f}"])
    MD += table(["Escenario", "Modelo", "Conjunto ambiguo {0,1}", "Conjunto vacío", "Prob. media del decil superior"], rows)

    # 8. Tabla 5
    eB = exp(f"{NH}__B")
    g = eB["global"]
    rank = lambda d: {k: i + 1 for i, (k, _) in enumerate(sorted(d.items(), key=lambda kv: -kv[1]))}
    rs, rp, rig = rank(g["shap_xgb"]), rank(g["permutacion_xgb"]), rank(g.get("ig_tabnet", {}))
    pl = {r["variable"]: r for r in eB["plausibilidad"]}
    jac = [c["jaccard_top5"] for c in eB["concordancia"]]
    rho = [c["spearman"] for c in eB["concordancia"]]
    rows5 = [[f, rs[f], rp[f], rig.get(f, "—"), {"+": "↑ riesgo", "-": "↓ riesgo", "0": "nula"}[pl[f]["direccion_modelo"]], pl[f]["direccion_ada"], pl[f]["coherente"]]
             for f in sorted(rs, key=rs.get)]
    OUT["tabla5"] = rows5
    section("8. Tabla 5 (concordancia y plausibilidad, NHANES · B)", "§3.6, Tabla 5 — reemplaza las columnas [ ] y el Jaccard top-5 de la nota")
    MD += table(["Variable", "Rango SHAP (XGBoost)", "Rango permutación (XGBoost)", "Rango IG (TabNet)", "Dirección observada", "Esperada (ADA, config)", "¿Coherente?"], rows5)
    MD += [f"Spearman mediano entre métodos: {np.median(rho):.2f} (rango {min(rho):.2f}–{max(rho):.2f}); Jaccard top-5 mediano: {np.median(jac):.2f} (rango {min(jac):.2f}–{max(jac):.2f}). "
           "Dirección observada: signo de la correlación de Spearman entre el valor de la variable y su SHAP en la prueba.", ""]
    OUT["concordancia"] = {k: {"rho_mediano": round(float(np.median([c["spearman"] for c in exp(k)["concordancia"]])), 2),
                               "jaccard_mediano": round(float(np.median([c["jaccard_top5"] for c in exp(k)["concordancia"]])), 2)}
                           for k in (f"{NH}__A", f"{NH}__B", "pima_indians__A", "pima_indians__B")}
    MD += ["Concordancia por corrida (verificación de §3.6): " + "; ".join(f"{k}: ρ = {v['rho_mediano']}, Jaccard = {v['jaccard_mediano']}" for k, v in OUT["concordancia"].items()) + ".", ""]

    # 9. Equidad
    section("9. Equidad por subgrupo (Tabla 6 y Tabla S2)", "§3.7, Tabla 6 (columna «Eventos por subgrupo») y nota «Añadir sensibilidad, especificidad y AUC con IC 95 %»; §2.10 (cortes de Pima)")
    MD += ["Los grupos de edad se rotulan 18–39 en el código, pero NHANES excluye a menores de 20, así que el grupo real es 20–39. En Pima los cortes son 21–39, 40–59 y ≥ 60 (misma función).", ""]
    OUT["equidad"] = {}
    for key in (f"{NH}__A", f"{NH}__B", "pima_indians__A", "pima_indians__B"):
        r = res(key)
        m = r["modelo_recomendado"]
        eq = r["evaluacion_prueba"]["por_modelo"][m]["equidad"]
        OUT["equidad"][key] = {"modelo": m, **eq}
        rows = []
        for gname, e in eq.items():
            for f in e["filas"]:
                ci = f["ic95"]
                rows.append([gname, f["grupo"].replace("18-39", "20-39" if key.startswith("nhanes") else "21-39"), f["n"], f["eventos"],
                             f"{f['sensibilidad']:.3f} [{ci['sensibilidad'][0]:.3f}–{ci['sensibilidad'][1]:.3f}]",
                             f"{f['especificidad']:.3f} [{ci['especificidad'][0]:.3f}–{ci['especificidad'][1]:.3f}]",
                             f"{f['roc_auc']:.3f} [{ci['roc_auc'][0]:.3f}–{ci['roc_auc'][1]:.3f}]"])
        MD += [f"### {key} · modelo recomendado: {NAMES[m]}", ""]
        MD += table(["Atributo", "Grupo", "n", "Eventos", "Sensibilidad [IC 95 %]", "Especificidad [IC 95 %]", "AUC [IC 95 %]"], rows)

    # 10. Verificación de afirmaciones del texto
    section("10. Verificación de afirmaciones de la plantilla", "Revisar §3.2, §3.5 y el Resumen; corregir si alguna no coincide")
    rB = res(f"{NH}__B")["evaluacion_prueba"]["por_modelo"]
    dc = {m: {round(r["umbral"], 2): r["modelo"] for r in v["curva_decision"]} for m, v in rB.items()}
    best_at = {t: max(dc, key=lambda m: dc[m][t]) for t in dc["lr"]}
    lr_band = [t for t in best_at if best_at[t] == "lr"]
    pos_nb = all(min(dc[m].values()) > 0 for m in dc)
    beat_all = all(all(dc[m][t] >= next(r["tratar_todos"] for r in rB[m]["curva_decision"] if round(r["umbral"], 2) == t) for t in dc[m]) for m in dc)
    gap = res(f"{NH}__B")["cv_anidada"]["resumen"]
    gaps = [gap[m]["brecha_optimismo"] for m in ("lr", "rf", "xgb")]
    prauc = [v["metricas"]["pr_auc"] for v in rB.values()]
    OUT["verificacion"] = {"lr_mejor_nb_umbrales": lr_band, "nb_positivo_todos": pos_nb, "supera_tratar_todos": beat_all, "brecha_B": gaps}
    MD += [f"- Beneficio neto positivo de los cinco modelos en 5–50 %: {'sí' if pos_nb else 'no'}; todos ≥ «tratar a todos» en todo el rango: {'sí' if beat_all else 'no'}.",
           f"- Umbrales donde la RL tiene el mayor beneficio neto: {', '.join(f'{t:.2f}' for t in sorted(lr_band)) or 'ninguno'} (la plantilla dice «entre el 10 % y el 25 %»).",
           f"- Brecha de optimismo B (RL, RF, XGBoost): {', '.join(f'{x:.4f}' for x in gaps)}; rango {min(gaps):.4f} a {max(gaps):.4f}.",
           f"- PR-AUC B: {min(prauc):.3f}–{max(prauc):.3f}, es decir {min(prauc) / 0.16:.1f} a {max(prauc) / 0.16:.1f} veces la prevalencia de 0.160.",
           f"- Δ ROC-AUC A→B: {min(v['delta'] for v in OUT['delta_auc'].values()):.3f} a {max(v['delta'] for v in OUT['delta_auc'].values()):.3f}.", ""]

    # 11. Hardware
    import psutil, torch
    OUT["hardware"] = {"cpu": "11th Gen Intel Core i7-1165G7 @ 2.80 GHz (4 núcleos, 8 hilos)", "ram_gb": round(psutil.virtual_memory().total / 2 ** 30, 1),
                       "gpu": "Ninguna usada (Intel Iris Xe integrada; PyTorch CPU, CUDA no disponible)", "so": platform.platform(), "python": platform.python_version()}
    section("11. Hardware", "§2.12 («Indicar el hardware»)")
    MD += [f"Texto: «Los análisis se ejecutaron en CPU ({OUT['hardware']['cpu']}, {OUT['hardware']['ram_gb']} GB de RAM, {OUT['hardware']['so'].split('-')[0]} 11) "
           "sin GPU; PyTorch se usó en su versión para CPU.»", ""]

    # 12. Pesos muestrales (sensibilidad b)
    demo = pd.read_sas(ROOT / "data/raw/nhanes/DEMO_J.xpt", format="xport")[["SEQN", "WTMEC2YR"]]
    demo["patient_id"] = demo.SEQN.astype(int).astype(str)
    w = test.merge(demo, on="patient_id", how="left")["WTMEC2YR"].values
    rows, OUT["sens_b"] = [], {}
    for key in (f"{NH}__B", f"{NH}__A"):
        b = bundle(key)
        for m, p in b["test"]["probs"].items():
            wa, ua = roc_auc_score(y, p, sample_weight=w), roc_auc_score(y, p)
            OUT["sens_b"][f"{key}__{m}"] = {"auc_ponderado": round(wa, 3), "auc_sin_ponderar": round(ua, 3), "brier_ponderado": round(brier_score_loss(y, p, sample_weight=w), 3)}
            rows.append([key.split("__")[1], NAMES[m], f"{ua:.3f}", f"{wa:.3f}", f"{brier_score_loss(y, p, sample_weight=w):.3f}"])
    prev_w = float(np.average(y, weights=w))
    section("12. Análisis de sensibilidad (b): pesos muestrales WTMEC2YR", "§2.11 (b); resultados como nueva Tabla S4 y una frase en §3.2 o §4.7")
    MD += [f"Métricas de la prueba aislada reponderadas con WTMEC2YR (sin reentrenar). Prevalencia ponderada en la prueba: {prev_w:.3f} (sin ponderar 0.161).", ""]
    MD += table(["Escenario", "Modelo", "ROC-AUC sin ponderar", "ROC-AUC ponderado", "Brier ponderado"], rows)

    # 13. Sensibilidad (a) y (d) reentrenando LR y XGBoost con el mismo protocolo del modelo final
    def quick(dfx, feats, tag, models=("lr", "xgb"), trials=None):
        set_seeds()
        prt = split_70_15_15(dfx)
        dv, vl, ts = (dfx.loc[prt[k]].reset_index(drop=True) for k in ("desarrollo", "validacion", "prueba"))
        out = {}
        for m in models:
            n_tr = trials if trials is not None else T._trials(m, None)
            tu = T.tune(m, dv[feats], dv.outcome.values, dv.patient_id.values, f"sens__{tag}__{m}", n_tr, emit=lambda *a: None,
                        space=load("search_spaces")["tabnet"] if m == "tabnet" else None)
            est = M.fit(m, tu["params"], dv[feats], dv.outcome.values, dv.patient_id.values)
            cal = T.Calibrator("sigmoide").fit(est.predict_proba(vl[feats])[:, 1], vl.outcome.values)
            thr = T.choose_threshold(vl.outcome.values, cal(est.predict_proba(vl[feats])[:, 1]))
            pt = cal(est.predict_proba(ts[feats])[:, 1])
            pm = E.point_metrics(ts.outcome.values, pt, thr)
            ci = E.bootstrap_ci(ts.outcome.values, pt, thr, 2000, keys=("roc_auc",))
            out[m] = {"auc": pm["roc_auc"], "ic95": ci["roc_auc"], "sens": pm["sensibilidad"], "espec": pm["especificidad"], "brier": pm["brier"],
                      "pendiente": pm["pendiente"], "n_test": len(ts), "eventos_test": int(ts.outcome.sum()), "eventos_total": int(dfx.outcome.sum())}
            print("sens", tag, m, out[m]["auc"], flush=True)
        return out

    sens = {}
    sens["referencia_B"] = quick(df, featsB, "refB")
    a1 = df.copy()
    a1["outcome"] = ((a1.outcome == 1) | (a1.hba1c >= 6.5)).astype(int)
    sens["a1_hba1c"] = quick(a1, featsB, "a1")
    a2 = df.copy()
    a2["outcome"] = ((a2.outcome == 1) | (a2.hba1c >= 6.5) | (a2.glucosa >= 126)).astype(int)
    sens["a2_hba1c_o_glucosa"] = quick(a2, featsB, "a2")
    sens["d_sin_col_trat"] = quick(df, [f for f in featsB if f not in ("colesterol_total", "tratamiento_antihipertensivo")], "d")
    sens["c_tabnet_optuna"] = quick(df, featsB, "c", models=("tabnet",), trials=10)
    OUT["sensibilidad"] = sens
    section("13. Análisis de sensibilidad (a), (c) y (d)", "§2.11 (quitar el resaltado «ejecutar o eliminar»), nueva Tabla S5 y una frase en §3.2/§4.4/§4.7")
    MD += ["Protocolo: igual al del modelo final del pipeline (Optuna en desarrollo con 3 pliegues internos, calibración sigmoide y umbral de Youden en Validación, "
           "prueba aislada leída una vez, IC bootstrap B = 2000). La fila de referencia reproduce el escenario B principal. En (a) cambia la etiqueta, "
           "así que la partición estratificada se recalcula.", ""]
    desc = {"referencia_B": "Referencia: escenario B (DIQ010 = 1)",
            "a1_hba1c": "(a1) Caso = DIQ010 = 1 o HbA1c ≥ 6.5 % sin diagnóstico",
            "a2_hba1c_o_glucosa": "(a2) Caso = DIQ010 = 1 o HbA1c ≥ 6.5 % o glucosa ≥ 126 mg/dL (glucosa sin ayuno garantizado)",
            "d_sin_col_trat": "(d) Escenario B sin colesterol total ni tratamiento antihipertensivo",
            "c_tabnet_optuna": "(c) TabNet con búsqueda Optuna (10 ensayos)"}
    rows = []
    for k, v in sens.items():
        for m, r in v.items():
            rows.append([desc[k], NAMES[m], r["eventos_total"], f"{r['auc']:.3f} [{r['ic95'][0]:.3f}–{r['ic95'][1]:.3f}]", f"{r['sens']:.3f}", f"{r['espec']:.3f}", f"{r['brier']:.3f}", f"{r['pendiente']:.2f}"])
    MD += table(["Análisis", "Modelo", "Eventos (total)", "ROC-AUC prueba [IC 95 %]", "Sensibilidad", "Especificidad", "Brier", "Pendiente"], rows)

    (ROOT / "docs/DATOS_PLANTILLA_V3.md").write_text("\n".join(MD), encoding="utf-8")
    (ROOT / "docs/datos_plantilla_v3.json").write_text(json.dumps(OUT, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("OK docs/DATOS_PLANTILLA_V3.md")


if __name__ == "__main__":
    main()
