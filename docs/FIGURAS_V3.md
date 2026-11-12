# Figuras del pipeline v3

Índice para actualizar el artículo. Cada figura tiene versión PNG/TIFF a 600 DPI y PDF vectorial (misma ruta, cambia la extensión). Cifras de origen: `docs/RESULTADOS_V3.md`.

| N.º | Cohorte · escenario | Figura | Archivo PNG | Pie sugerido |
| --- | --- | --- | --- | --- |
| 1 | NHANES · escenario A (con HbA1c y glucosa) | Curvas ROC | `reports/v3/nhanes_2017_2018__A/fig_roc.png` | Curvas ROC en la prueba aislada para los cinco modelos; entre paréntesis, el ROC-AUC. La diagonal es el azar. |
| 2 | NHANES · escenario A (con HbA1c y glucosa) | Curvas precisión-sensibilidad | `reports/v3/nhanes_2017_2018__A/fig_pr.png` | Curvas precisión-sensibilidad; la línea punteada es la prevalencia (0.160). |
| 3 | NHANES · escenario A (con HbA1c y glucosa) | Calibración | `reports/v3/nhanes_2017_2018__A/fig_calibracion.png` | Diagrama de confiabilidad (10 bins por cuantiles) tras calibración sigmoide en Validación; la pendiente ideal es 1. |
| 4 | NHANES · escenario A (con HbA1c y glucosa) | Curva de decisión | `reports/v3/nhanes_2017_2018__A/fig_curva_decision.png` | Curva de decisión: beneficio neto frente a «tratar a todos» y «ninguno» para umbrales de 5–50 %. |
| 5 | NHANES · escenario A (con HbA1c y glucosa) | Importancia SHAP global (XGBoost) | `reports/v3/nhanes_2017_2018__A/fig_shap_global.png` | Importancia global por media del valor SHAP absoluto en la prueba aislada. La variable con mayor peso es hba1c. Describe asociaciones aprendidas por el modelo, no causalidad. |
| 6 | NHANES · escenario A (con HbA1c y glucosa) | Mapa de atención de TabNet | `reports/v3/nhanes_2017_2018__A/fig_tabnet_atencion.png` | Máscara de atención agregada de TabNet en 200 pacientes de prueba estratificados, ordenados por probabilidad. Los colores claros indican las variables que el modelo selecciona en cada paciente. |
| 7 | NHANES · escenario B (sin HbA1c ni glucosa) | Curvas ROC | `reports/v3/nhanes_2017_2018__B/fig_roc.png` | Curvas ROC en la prueba aislada para los cinco modelos; entre paréntesis, el ROC-AUC. La diagonal es el azar. |
| 8 | NHANES · escenario B (sin HbA1c ni glucosa) | Curvas precisión-sensibilidad | `reports/v3/nhanes_2017_2018__B/fig_pr.png` | Curvas precisión-sensibilidad; la línea punteada es la prevalencia (0.160). |
| 9 | NHANES · escenario B (sin HbA1c ni glucosa) | Calibración | `reports/v3/nhanes_2017_2018__B/fig_calibracion.png` | Diagrama de confiabilidad (10 bins por cuantiles) tras calibración sigmoide en Validación; la pendiente ideal es 1. |
| 10 | NHANES · escenario B (sin HbA1c ni glucosa) | Curva de decisión | `reports/v3/nhanes_2017_2018__B/fig_curva_decision.png` | Curva de decisión: beneficio neto frente a «tratar a todos» y «ninguno» para umbrales de 5–50 %. |
| 11 | NHANES · escenario B (sin HbA1c ni glucosa) | Importancia SHAP global (XGBoost) | `reports/v3/nhanes_2017_2018__B/fig_shap_global.png` | Importancia global por media del valor SHAP absoluto en la prueba aislada. La variable con mayor peso es edad. Describe asociaciones aprendidas por el modelo, no causalidad. |
| 12 | NHANES · escenario B (sin HbA1c ni glucosa) | Mapa de atención de TabNet | `reports/v3/nhanes_2017_2018__B/fig_tabnet_atencion.png` | Máscara de atención agregada de TabNet en 200 pacientes de prueba estratificados, ordenados por probabilidad. Los colores claros indican las variables que el modelo selecciona en cada paciente. |
| 13 | Pima · escenario A (con glucosa OGTT) | Curvas ROC | `reports/v3/pima_indians__A/fig_roc.png` | Curvas ROC en la prueba aislada para los cinco modelos; entre paréntesis, el ROC-AUC. La diagonal es el azar. |
| 14 | Pima · escenario A (con glucosa OGTT) | Curvas precisión-sensibilidad | `reports/v3/pima_indians__A/fig_pr.png` | Curvas precisión-sensibilidad; la línea punteada es la prevalencia (0.349). |
| 15 | Pima · escenario A (con glucosa OGTT) | Calibración | `reports/v3/pima_indians__A/fig_calibracion.png` | Diagrama de confiabilidad (10 bins por cuantiles) tras calibración sigmoide en Validación; la pendiente ideal es 1. |
| 16 | Pima · escenario A (con glucosa OGTT) | Curva de decisión | `reports/v3/pima_indians__A/fig_curva_decision.png` | Curva de decisión: beneficio neto frente a «tratar a todos» y «ninguno» para umbrales de 5–50 %. |
| 17 | Pima · escenario A (con glucosa OGTT) | Importancia SHAP global (XGBoost) | `reports/v3/pima_indians__A/fig_shap_global.png` | Importancia global por media del valor SHAP absoluto en la prueba aislada. La variable con mayor peso es glucosa. Describe asociaciones aprendidas por el modelo, no causalidad. |
| 18 | Pima · escenario A (con glucosa OGTT) | Mapa de atención de TabNet | `reports/v3/pima_indians__A/fig_tabnet_atencion.png` | Máscara de atención agregada de TabNet en 200 pacientes de prueba estratificados, ordenados por probabilidad. Los colores claros indican las variables que el modelo selecciona en cada paciente. |
| 19 | Pima · escenario B (sin glucosa ni insulina) | Curvas ROC | `reports/v3/pima_indians__B/fig_roc.png` | Curvas ROC en la prueba aislada para los cinco modelos; entre paréntesis, el ROC-AUC. La diagonal es el azar. |
| 20 | Pima · escenario B (sin glucosa ni insulina) | Curvas precisión-sensibilidad | `reports/v3/pima_indians__B/fig_pr.png` | Curvas precisión-sensibilidad; la línea punteada es la prevalencia (0.349). |
| 21 | Pima · escenario B (sin glucosa ni insulina) | Calibración | `reports/v3/pima_indians__B/fig_calibracion.png` | Diagrama de confiabilidad (10 bins por cuantiles) tras calibración sigmoide en Validación; la pendiente ideal es 1. |
| 22 | Pima · escenario B (sin glucosa ni insulina) | Curva de decisión | `reports/v3/pima_indians__B/fig_curva_decision.png` | Curva de decisión: beneficio neto frente a «tratar a todos» y «ninguno» para umbrales de 5–50 %. |
| 23 | Pima · escenario B (sin glucosa ni insulina) | Importancia SHAP global (XGBoost) | `reports/v3/pima_indians__B/fig_shap_global.png` | Importancia global por media del valor SHAP absoluto en la prueba aislada. La variable con mayor peso es edad. Describe asociaciones aprendidas por el modelo, no causalidad. |
| 24 | Pima · escenario B (sin glucosa ni insulina) | Mapa de atención de TabNet | `reports/v3/pima_indians__B/fig_tabnet_atencion.png` | Máscara de atención agregada de TabNet en 200 pacientes de prueba estratificados, ordenados por probabilidad. Los colores claros indican las variables que el modelo selecciona en cada paciente. |

## Correspondencia sugerida con la plantilla del artículo

- Figuras 2–5 del artículo: usar ROC, PR, calibración y curva de decisión de **NHANES · B** (resultado principal) y las de NHANES · A como comparación.
- Figura 6: SHAP global de NHANES · B. Figura 7: mapa de atención TabNet de NHANES · B.
- Pima: material suplementario.

Regenerar: `python -m scripts.run_pipeline_v3 --all` y luego `python -m scripts.figures_v3_doc`.