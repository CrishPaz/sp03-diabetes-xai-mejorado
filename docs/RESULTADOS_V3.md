# Resultados del pipeline v3

Generado automáticamente por `scripts/summarize_v3.py` desde `models/v3/*/results.json`. Partición 70/15/15 agrupada; CV anidada 5 externos × 3 internos con Optuna (TPE, semilla 42); calibración, umbral (Youden) y conformal en Validación; Prueba leída una sola vez. IC 95 % por bootstrap estratificado (B = 2000).

## nhanes_2017_2018 · escenario A

Deteccion con biomarcadores glucemicos (circular: HbA1c y glucosa son criterios diagnosticos). n = 4829, eventos = 772 (prevalencia 0.160). Validación de grupos: PASS. Tiempo: 671 s.

| Modelo | AUC CV anidada (media ± DE) | Brecha optimismo | ROC-AUC prueba [IC95] | PR-AUC prueba | Sensibilidad | Especificidad | Brier | Pendiente calib. | Cobertura conformal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Reg. logística L2 | 0.941 ± 0.006 | -0.002 | 0.945 [0.924–0.963] | 0.774 | 0.829 [0.761–0.897] | 0.916 [0.893–0.938] | 0.063 | 0.85 | 0.891 |
| Random Forest | 0.946 ± 0.007 | -0.0019 | 0.952 [0.933–0.968] | 0.827 | 0.863 [0.803–0.923] | 0.878 [0.851–0.905] | 0.058 | 1.20 | 0.912 |
| XGBoost | 0.948 ± 0.004 | -0.0036 | 0.947 [0.926–0.964] | 0.818 | 0.838 [0.769–0.897] | 0.900 [0.875–0.923] | 0.060 | 1.09 | 0.906 |
| TabNet | 0.930 ± 0.019 | — | 0.935 [0.910–0.956] | 0.761 | 0.846 [0.778–0.906] | 0.892 [0.865–0.916] | 0.065 | 1.08 | 0.888 |
| Stacking calibrado | 0.948 ± 0.006 | — | 0.952 [0.932–0.968] | 0.820 | 0.846 [0.778–0.906] | 0.906 [0.882–0.928] | 0.058 | 1.07 | 0.897 |

- Friedman (CV anidada): χ² = 15.840, p = 0.003.
- DeLong con Holm en prueba: no se demostraron diferencias de AUC entre modelos.
- Modelo recomendado (mayor AUC en CV anidada): XGBoost.
- Equidad (XGBoost): sexo: brecha de sensibilidad 0.101 (alerta); edad: brecha de sensibilidad 0.362 (alerta)

## nhanes_2017_2018 · escenario B

Prediccion temprana sin HbA1c ni glucosa. n = 4829, eventos = 772 (prevalencia 0.160). Validación de grupos: PASS. Tiempo: 763 s.

| Modelo | AUC CV anidada (media ± DE) | Brecha optimismo | ROC-AUC prueba [IC95] | PR-AUC prueba | Sensibilidad | Especificidad | Brier | Pendiente calib. | Cobertura conformal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Reg. logística L2 | 0.834 ± 0.007 | -0.001 | 0.856 [0.818–0.889] | 0.502 | 0.872 [0.812–0.932] | 0.759 [0.723–0.793] | 0.101 | 1.02 | 0.908 |
| Random Forest | 0.837 ± 0.004 | -0.0014 | 0.852 [0.810–0.890] | 0.571 | 0.863 [0.795–0.923] | 0.675 [0.635–0.713] | 0.097 | 1.03 | 0.902 |
| XGBoost | 0.844 ± 0.004 | -0.0031 | 0.856 [0.815–0.892] | 0.555 | 0.880 [0.821–0.940] | 0.658 [0.619–0.696] | 0.097 | 1.02 | 0.909 |
| TabNet | 0.819 ± 0.008 | — | 0.807 [0.759–0.851] | 0.437 | 0.761 [0.684–0.838] | 0.767 [0.731–0.800] | 0.109 | 0.84 | 0.894 |
| Stacking calibrado | 0.845 ± 0.004 | — | 0.855 [0.814–0.892] | 0.537 | 0.880 [0.821–0.940] | 0.649 [0.609–0.686] | 0.097 | 1.03 | 0.898 |

- Friedman (CV anidada): χ² = 18.720, p = 0.001.
- DeLong con Holm en prueba: diferencias significativas en lr vs tabnet, rf vs tabnet, xgb vs tabnet, tabnet vs stack.
- Modelo recomendado (mayor AUC en CV anidada): Stacking calibrado.
- Equidad (Stacking calibrado): sexo: brecha de sensibilidad 0.070; edad: brecha de sensibilidad 0.954 (alerta)

## pima_indians · escenario A

Con glucosa a 2 h de OGTT. n = 768, eventos = 268 (prevalencia 0.349). Validación de grupos: PASS. Tiempo: 684 s.

| Modelo | AUC CV anidada (media ± DE) | Brecha optimismo | ROC-AUC prueba [IC95] | PR-AUC prueba | Sensibilidad | Especificidad | Brier | Pendiente calib. | Cobertura conformal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Reg. logística L2 | 0.823 ± 0.041 | -0.0064 | 0.882 [0.815–0.938] | 0.805 | 0.833 [0.714–0.929] | 0.760 [0.666–0.853] | 0.136 | 1.30 | 0.940 |
| Random Forest | 0.828 ± 0.063 | -0.0112 | 0.863 [0.788–0.924] | 0.812 | 0.833 [0.714–0.929] | 0.720 [0.613–0.813] | 0.142 | 1.07 | 0.923 |
| XGBoost | 0.820 ± 0.068 | -0.003 | 0.851 [0.773–0.917] | 0.808 | 0.857 [0.738–0.952] | 0.653 [0.547–0.760] | 0.147 | 1.10 | 0.932 |
| TabNet | 0.776 ± 0.067 | — | 0.821 [0.735–0.896] | 0.701 | 0.762 [0.619–0.881] | 0.747 [0.640–0.840] | 0.170 | 0.82 | 0.906 |
| Stacking calibrado | 0.831 ± 0.061 | — | 0.873 [0.800–0.931] | 0.826 | 0.833 [0.714–0.929] | 0.787 [0.693–0.880] | 0.138 | 1.07 | 0.923 |

- Friedman (CV anidada): χ² = 12.646, p = 0.013.
- DeLong con Holm en prueba: no se demostraron diferencias de AUC entre modelos.
- Modelo recomendado (mayor AUC en CV anidada): Stacking calibrado.
- Equidad (Stacking calibrado): sexo: brecha de sensibilidad 0.000; edad: brecha de sensibilidad 0.035

## pima_indians · escenario B

Sin glucosa ni insulina. n = 768, eventos = 268 (prevalencia 0.349). Validación de grupos: PASS. Tiempo: 487 s.

| Modelo | AUC CV anidada (media ± DE) | Brecha optimismo | ROC-AUC prueba [IC95] | PR-AUC prueba | Sensibilidad | Especificidad | Brier | Pendiente calib. | Cobertura conformal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Reg. logística L2 | 0.733 ± 0.035 | -0.0019 | 0.765 [0.669–0.844] | 0.613 | 0.786 [0.667–0.905] | 0.560 [0.440–0.667] | 0.189 | 0.94 | 0.906 |
| Random Forest | 0.745 ± 0.048 | -0.009 | 0.767 [0.675–0.846] | 0.618 | 0.667 [0.524–0.809] | 0.653 [0.547–0.747] | 0.183 | 1.03 | 0.949 |
| XGBoost | 0.740 ± 0.045 | 0.0016 | 0.762 [0.669–0.839] | 0.597 | 0.786 [0.667–0.905] | 0.627 [0.520–0.733] | 0.186 | 1.17 | 0.966 |
| TabNet | 0.675 ± 0.066 | — | 0.767 [0.668–0.849] | 0.591 | 0.857 [0.738–0.952] | 0.587 [0.467–0.693] | 0.191 | 0.93 | 0.966 |
| Stacking calibrado | 0.743 ± 0.047 | — | 0.769 [0.676–0.847] | 0.616 | 0.809 [0.691–0.905] | 0.613 [0.507–0.720] | 0.183 | 1.07 | 0.949 |

- Friedman (CV anidada): χ² = 8.480, p = 0.075.
- DeLong con Holm en prueba: no se demostraron diferencias de AUC entre modelos.
- Modelo recomendado (mayor AUC en CV anidada): Random Forest.
- Equidad (Random Forest): sexo: brecha de sensibilidad 0.000; edad: brecha de sensibilidad 0.520 (alerta)
