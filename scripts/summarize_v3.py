"""Genera docs/RESULTADOS_V3.md a partir de models/v3/*/results.json (solo cifras reales)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
NAMES = {"lr": "Reg. logística L2", "rf": "Random Forest", "xgb": "XGBoost", "tabnet": "TabNet", "stack": "Stacking calibrado"}


def ci(v, k):
    lo, hi = v["ic95"][k]
    return f"{v['metricas'][k]:.3f} [{lo:.3f}–{hi:.3f}]"


def main():
    out = ["# Resultados del pipeline v3", "",
           "Generado automáticamente por `scripts/summarize_v3.py` desde `models/v3/*/results.json`. "
           "Partición 70/15/15 agrupada; CV anidada 5 externos × 3 internos con Optuna (TPE, semilla 42); "
           "calibración, umbral (Youden) y conformal en Validación; Prueba leída una sola vez. IC 95 % por bootstrap estratificado (B = 2000).", ""]
    for p in sorted((ROOT / "models/v3").glob("*__*/results.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        a = r["auditoria"]
        out += [f"## {r['cohorte']} · escenario {r['escenario']}", "",
                f"{a['descripcion_escenario']}. n = {a['n_oficial']}, eventos = {a['eventos']} (prevalencia {a['prevalencia']:.3f}). "
                f"Validación de grupos: {r['particion']['validacion_grupos']['estado']}. Tiempo: {r['segundos']:.0f} s.", "",
                "| Modelo | AUC CV anidada (media ± DE) | Brecha optimismo | ROC-AUC prueba [IC95] | PR-AUC prueba | Sensibilidad | Especificidad | Brier | Pendiente calib. | Cobertura conformal |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        pm, nested = r["evaluacion_prueba"]["por_modelo"], r["cv_anidada"]["resumen"]
        for m, v in pm.items():
            s = nested.get(m, {})
            out.append(f"| {NAMES.get(m, m)} | {s.get('media', float('nan')):.3f} ± {s.get('de', float('nan')):.3f} | {s.get('brecha_optimismo') if s.get('brecha_optimismo') is not None else '—'} | "
                       f"{ci(v, 'roc_auc')} | {v['metricas']['pr_auc']:.3f} | {ci(v, 'sensibilidad')} | {ci(v, 'especificidad')} | {v['metricas']['brier']:.3f} | "
                       f"{v['metricas']['pendiente']:.2f} | {r['calibracion']['conformal']['cobertura_prueba'][m]:.3f} |")
        fr = r["cv_anidada"]["pruebas"]["friedman"]
        sig = [f"{x['modelo_a']} vs {x['modelo_b']}" for x in r["evaluacion_prueba"]["pruebas"]["pares"] if x["delong_p_holm"] < 0.05]
        out += ["", f"- Friedman (CV anidada): χ² = {fr['chi2']:.3f}, p = {fr['p']:.3f}.",
                f"- DeLong con Holm en prueba: {'diferencias significativas en ' + ', '.join(sig) if sig else 'no se demostraron diferencias de AUC entre modelos'}.",
                f"- Modelo recomendado (mayor AUC en CV anidada): {NAMES.get(r['modelo_recomendado'])}.",
                f"- Equidad ({NAMES.get(r['modelo_recomendado'])}): " + "; ".join(
                    f"{g}: brecha de sensibilidad {e['brecha_sensibilidad']:.3f}{' (alerta)' if e['alerta'] else ''}"
                    for g, e in pm[r["modelo_recomendado"]]["equidad"].items()), ""]
    (ROOT / "docs/RESULTADOS_V3.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
