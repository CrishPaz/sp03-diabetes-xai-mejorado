# Datos para completar la plantilla del artículo (SP-03 · pipeline v3)

Generado por `scripts/fill_template_v3.py` desde los artefactos reales (`models/v3/`) y los datos crudos. Cada sección indica el lugar exacto de la plantilla y el valor que reemplaza al campo resaltado. Las cifras no deben editarse a mano.

## 0. Correcciones al texto de la plantilla (aplicar primero)

Al recalcular desde el sistema, estas afirmaciones de la plantilla no coinciden con los datos:

| Dónde | Dice la plantilla | Dato real | Texto corregido sugerido |
| --- | --- | --- | --- |
| Resumen, Abstract | «Todos los modelos aportaron beneficio neto positivo con umbrales del 5 % al 50 %» | TabNet cae a −0.001 en los umbrales 0.49–0.50; RL, RF, XGBoost y Stacking son positivos en todo el rango, y los cinco superan a «tratar a todos» | «RL, Random Forest, XGBoost y Stacking aportaron beneficio neto positivo entre el 5 % y el 50 %, y los cinco superaron a la estrategia de tratar a todos» |
| §3 (frase inicial), Puntos clave, §5 | «En ningún escenario un modelo complejo superó de forma significativa a la regresión logística» | En el escenario A, el Stacking superó a la RL en la CV anidada (Nadeau–Bengio + Holm, p = 0.027; diferencia media 0.008); en la prueba aislada (DeLong) no hubo diferencias | «En el escenario B ningún modelo superó a la RL; en el A, solo el Stacking la superó en la CV anidada (diferencia 0.008), sin confirmarse en la prueba aislada» |
| §3.5 | «La RL tuvo un beneficio neto ligeramente mayor entre el 10 % y el 25 %» | La RL tiene el mayor beneficio neto entre el 7 % y el 24 % | «…entre el 7 % y el 24 %…» |
| §2.4, Figura 1 | Desarrollo n ≈ 3,380; Validación n ≈ 724; Prueba n ≈ 725 | Desarrollo 3,377 (538 eventos); Validación 726 (117); Prueba 726 (117) | Usar los n exactos de la sección 2 |
| §3.4, Tabla 4 | Solo DeLong y Friedman | McNemar + Holm en B sí detecta diferencias de clasificación al umbral (RL frente a RF, XGBoost y Stacking; TabNet frente a XGBoost, RF y Stacking) | Añadir: «McNemar detectó diferencias en la clasificación al umbral de Youden, que reflejan distintos equilibrios entre sensibilidad y especificidad, no diferencias de discriminación» |
| §3.7, §4.6 | «probablemente refleja muy pocos eventos en el grupo de 20 a 39 años» | Confirmado: 6 eventos en 199 pacientes de 20–39 años; sensibilidad 0.000 y AUC 0.498 [0.315–0.698] con el Stacking en B | Reemplazar «probablemente» por el dato |
| §2.10, Tabla S2 | Cortes de Pima por definir | 21–39, 40–59 y ≥ 60; el grupo ≥ 60 de la prueba tiene 4 pacientes (1 evento) y se omite de la auditoría (mínimo 10) | Declarar los cortes y la omisión |
| §3.5 | Solo cobertura conformal | En B, 8.0–13.5 % de conjuntos ambiguos {0, 1}; en A, 0 % ambiguos pero 2.2–5.0 % de conjuntos vacíos | Añadir la frase con los dos datos |
| §4.3 | «su inferioridad podría estar sobrestimada» (TabNet con hiperparámetros fijos) | TabNet con Optuna (10 ensayos) obtuvo 0.819 [0.778–0.857] frente a 0.807: mejora poco y sigue siendo el más bajo | «Con búsqueda de hiperparámetros, TabNet mejoró solo a 0.819, por lo que su inferioridad no se explica por la configuración fija» |
| §2.11 | «Propuesto…; ejecutar o eliminar» | Los cuatro análisis (a)–(d) están ejecutados (secciones 12 y 13) | Quitar el resaltado, pasar §2.11 a tiempo pasado y añadir las Tablas S4 y S5 |

Hallazgos nuevos útiles para la Discusión:

- **(b) Pesos muestrales:** el AUC ponderado con WTMEC2YR es mayor (B: 0.837–0.879; A: 0.948–0.959), con una prevalencia ponderada de 0.123. Las conclusiones no cambian.
- **(a) Diabetes no diagnosticada:** al contar como casos a 176 personas con HbA1c ≥ 6.5 % sin diagnóstico (948 casos en total), el AUC en B es de 0.837 (RL) y 0.850 (XGBoost). El ruido de etiqueta tiene un efecto pequeño.
- **(d) Causalidad inversa:** al quitar el colesterol total y el tratamiento antihipertensivo, el AUC baja de 0.856 a 0.845 (RL) y a 0.838 (XGBoost). El modelo no depende de esas dos variables.
- **Tabla 5:** el colesterol total tiene dirección inversa (↓ riesgo), coherente con la hipótesis de tratamiento con estatinas tras el diagnóstico (§4.4). La presión diastólica también sale inversa y hay que revisarla.

## 1. Exclusiones de NHANES

**Dónde va en la plantilla:** §2.1 (texto) y Figura 1 (caja «Excluidos»)

| Paso | n restante | Excluidos |
| --- | --- | --- |
| Participantes DEMO_J | 9254 | — |
| Adultos >= 20 años | 5569 | 3685 |
| Respuesta válida DIQ010 (1/2/3) | 5565 | 4 |
| HbA1c, glucosa e IMC presentes | 4829 | 736 |

Texto para la Figura 1: «Excluidos (n = 4,425): edad < 20 años: n = 3,685; DIQ010 inválido (7/9/faltante): n = 4; sin HbA1c, glucosa o IMC: n = 736».

## 2. Tamaño de las particiones y eventos

**Dónde va en la plantilla:** §2.4 (texto «alrededor de 540 eventos… 117 eventos») y Figura 1 (cajas de partición)

| Partición NHANES | n | Eventos | Prevalencia |
| --- | --- | --- | --- |
| desarrollo | 3377 | 538 | 15.9 % |
| validacion | 726 | 117 | 16.1 % |
| prueba | 726 | 117 | 16.1 % |

Eventos por parámetro candidato en el escenario B: 538 / 10 = 53.8; escenario A: 538 / 12 = 44.8. Pima: desarrollo n = 537 (187 eventos), validación n = 114, prueba n = 117 (42 eventos).

## 3. Tabla 1 (NHANES por estado de DM2)

**Dónde va en la plantilla:** §3.1, Tabla 1 — reemplaza todas las celdas [ ]

Sin DM2 n = 4,057; con DM2 n = 772. Media ± DE si ambas clases son normales (D'Agostino-Pearson); si no, mediana [RIC].

| Variable | Sin DM2 | Con DM2 | SMD | % faltantes (total) | p descriptivo |
| --- | --- | --- | --- | --- | --- |
| edad | 49.0 [34.0-63.0] | 64.0 [55.0-73.0] | 0.945 | 0.0 % | 9.49e-96 |
| sexo femenino, n (%) | 2167 (53.4 %) | 341 (44.2 %) | — | 0.0 % | 2.98e-06 |
| imc | 28.3 [24.5-33.1] | 31.1 [26.8-36.2] | 0.396 | 0.0 % | 7.26e-26 |
| glucosa | 92.0 [86.0-99.0] | 125.0 [103.0-168.0] | 1.033 | 0.0 % | 6.29e-189 |
| hba1c | 5.5 [5.2-5.8] | 7.0 [6.3-8.1] | 1.478 | 0.0 % | 1.11e-301 |
| presion_sistolica | 122.0 [112.0-136.0] | 130.0 [120.0-144.0] | 0.391 | 10.8 % | 7.14e-24 |
| presion_diastolica | 72.0 [66.0-80.0] | 72.0 [64.0-78.0] | -0.162 | 11.6 % | 0.000646 |
| colesterol_total | 187.0 [163.0-215.0] | 169.0 [144.0-198.5] | -0.373 | 0.1 % | 3.22e-26 |
| colesterol_hdl | 52.0 [43.0-63.0] | 46.0 [39.0-54.0] | -0.419 | 0.1 % | 8.36e-29 |
| trigliceridos | 113.0 [80.0-165.0] | 146.0 [108.0-205.2] | 0.306 | 0.0 % | 4.15e-32 |
| antecedente_familiar_diabetes | 1761 (43.4 %) | 559 (72.4 %) | 0.614 | 0.0 % | 3.34e-49 |
| antecedente_hipertension | 1306 (32.2 %) | 532 (68.9 %) | 0.789 | 0.0 % | 2.51e-82 |
| tratamiento_antihipertensivo | 1158 (28.5 %) | 517 (67.0 %) | 0.833 | 0.0 % | 1.46e-93 |
| participantes con ≥ 1 dato imputado | 11.3 % | 13.6 % | — | — | — |

## 4. Tabla S0 (diccionario de variables)

**Dónde va en la plantilla:** Material suplementario, Tabla S0; también resuelve el pendiente de §2.2 (códigos NHANES) y §2.3 (faltantes)

| Variable (código) | Archivo NHANES | Variable NHANES | Unidad | Rango observado | % faltantes | Escenario B |
| --- | --- | --- | --- | --- | --- | --- |
| edad | DEMO_J | RIDAGEYR | años | 20–80 | 0.0 % | sí |
| imc | BMX_J | BMXBMI | kg/m² | 14.2–86.2 | 0.0 % | sí |
| glucosa | BIOPRO_J | LBXSGL | mg/dL | 47–626 | 0.0 % | solo A |
| hba1c | GHB_J | LBXGH | % | 4.1–16.2 | 0.0 % | solo A |
| presion_sistolica | BPX_J | BPXSY1 | mmHg | 78–224 | 10.8 % | sí |
| presion_diastolica | BPX_J | BPXDI1 | mmHg | 30–124 | 11.6 % | sí |
| colesterol_total | TCHOL_J | LBXTC | mg/dL | 76–446 | 0.1 % | sí |
| colesterol_hdl | HDL_J | LBDHDD | mg/dL | 10–189 | 0.1 % | sí |
| trigliceridos | BIOPRO_J | LBXSTR | mg/dL | 25–2923 | 0.0 % | sí |
| antecedente_familiar_diabetes | MCQ_J | MCQ300C = 1 | 0/1 | 0–1 | 0.0 % | sí |
| antecedente_hipertension | BPQ_J | BPQ020 = 1 | 0/1 | 0–1 | 0.0 % | sí |
| tratamiento_antihipertensivo | BPQ_J | BPQ020 = 1 y BPQ040A = 1 | 0/1 | 0–1 | 0.0 % | sí |

Pima Indians (ceros fisiológicamente imposibles convertidos a NaN antes de imputar):

| Variable | Unidad | Rango observado | % faltantes |
| --- | --- | --- | --- |
| embarazos | n | 0–17 | 0.0 % |
| glucosa | mg/dL (2 h PTOG) | 44–199 | 0.7 % |
| presion_diastolica | mmHg | 24–122 | 4.6 % |
| pliegue_cutaneo | mm | 7–99 | 29.6 % |
| insulina | µU/mL (2 h) | 14–846 | 48.7 % |
| imc | kg/m² | 18.2–67.1 | 1.4 % |
| pedigri_diabetes | índice | 0.078–2.42 | 0.0 % |
| edad | años | 21–81 | 0.0 % |

**Fuente de triglicéridos (§2.3):** LBXSTR del panel bioquímico estándar (BIOPRO_J), no LBXTR de la submuestra en ayunas. Faltantes: 0.0 %, por lo que la imputación por mediana afecta a una fracción pequeña. La glucosa (LBXSGL) también proviene de BIOPRO_J y no requiere ayuno.

**Verificación de predictores (§2.2):** escenario B = edad, imc, presion_sistolica, presion_diastolica, colesterol_total, colesterol_hdl, trigliceridos, antecedente_familiar_diabetes, antecedente_hipertension, tratamiento_antihipertensivo (10); escenario A añade hba1c y glucosa (12). Coincide con configs/features.yaml.

## 5. IC 95 % de cada Δ ROC-AUC (A → B)

**Dónde va en la plantilla:** §3.3, nota de la Tabla 3 («Añadir el IC 95 % de cada Δ»)

Bootstrap pareado estratificado (B = 2000) sobre la misma prueba aislada de 726 pacientes, que es idéntica en ambos escenarios; p de DeLong pareado.

| Modelo | Δ ROC-AUC (B − A) | IC 95 % | p DeLong |
| --- | --- | --- | --- |
| Reg. logística L2 | -0.089 | [-0.120, -0.061] | 1.59e-08 |
| Random Forest | -0.100 | [-0.139, -0.065] | 1.50e-07 |
| XGBoost | -0.091 | [-0.127, -0.056] | 6.44e-07 |
| TabNet | -0.128 | [-0.171, -0.087] | 1.73e-09 |
| Stacking calibrado | -0.097 | [-0.132, -0.063] | 8.21e-08 |

## 6. Pares Nadeau–Bengio, Nemenyi y McNemar

**Dónde va en la plantilla:** §3.4, Tabla 4 (columna «Nadeau–Bengio + Holm») y nota; Tabla S2 para Pima

### nhanes_2017_2018__A

- Nadeau–Bengio + Holm significativos: lr vs stack.
- Nemenyi significativos (p < 0.05): xgb vs tabnet, tabnet vs stack. Rangos medios: lr 4.2, rf 2.4, xgb 1.8, tabnet 4.8, stack 1.8; diferencia crítica = 2.728.
- McNemar + Holm al umbral de Youden: sin diferencias.

| Par | Dif. media AUC | t | p | p Holm | p Nemenyi |
| --- | --- | --- | --- | --- | --- |
| lr vs rf | -0.0057 | -1.84 | 0.1394 | 1.0000 | 0.3735 |
| lr vs xgb | -0.0077 | -4.03 | 0.0158 | 0.1420 | 0.1152 |
| lr vs tabnet | 0.0107 | 1.04 | 0.3578 | 1.0000 | 0.9751 |
| lr vs stack | -0.0078 | -6.62 | 0.0027 | 0.0269 | 0.1152 |
| rf vs xgb | -0.0021 | -0.92 | 0.4076 | 1.0000 | 0.9751 |
| rf vs tabnet | 0.0164 | 1.23 | 0.2860 | 1.0000 | 0.1152 |
| rf vs stack | -0.0021 | -0.97 | 0.3878 | 1.0000 | 0.9751 |
| xgb vs tabnet | 0.0184 | 1.55 | 0.1954 | 1.0000 | 0.0227 |
| xgb vs stack | -0.0001 | -0.07 | 0.9493 | 1.0000 | 1.0000 |
| tabnet vs stack | -0.0185 | -1.64 | 0.1755 | 1.0000 | 0.0227 |

### nhanes_2017_2018__B

- Nadeau–Bengio + Holm significativos: ninguno.
- Nemenyi significativos (p < 0.05): xgb vs tabnet, tabnet vs stack. Rangos medios: lr 3.8, rf 3.2, xgb 1.8, tabnet 5.0, stack 1.2; diferencia crítica = 2.728.
- McNemar + Holm al umbral de Youden: lr vs rf, lr vs xgb, lr vs stack, rf vs tabnet, xgb vs tabnet, tabnet vs stack.

| Par | Dif. media AUC | t | p | p Holm | p Nemenyi |
| --- | --- | --- | --- | --- | --- |
| lr vs rf | -0.0028 | -0.69 | 0.5306 | 0.5306 | 0.9751 |
| lr vs xgb | -0.0095 | -3.22 | 0.0324 | 0.1619 | 0.2659 |
| lr vs tabnet | 0.0156 | 2.65 | 0.0571 | 0.2282 | 0.7514 |
| lr vs stack | -0.0108 | -3.96 | 0.0166 | 0.1329 | 0.0703 |
| rf vs xgb | -0.0067 | -3.51 | 0.0246 | 0.1476 | 0.6277 |
| rf vs tabnet | 0.0184 | 2.53 | 0.0643 | 0.2282 | 0.3735 |
| rf vs stack | -0.0081 | -5.37 | 0.0058 | 0.0580 | 0.2659 |
| xgb vs tabnet | 0.0251 | 3.87 | 0.0181 | 0.1329 | 0.0120 |
| xgb vs stack | -0.0014 | -1.47 | 0.2165 | 0.4331 | 0.9751 |
| tabnet vs stack | -0.0265 | -4.22 | 0.0134 | 0.1210 | 0.0014 |

### pima_indians__A

- Nadeau–Bengio + Holm significativos: ninguno.
- Nemenyi significativos (p < 0.05): tabnet vs stack. Rangos medios: lr 3.3, rf 2.2, xgb 3.2, tabnet 4.8, stack 1.5; diferencia crítica = 2.728.
- McNemar + Holm al umbral de Youden: sin diferencias.

| Par | Dif. media AUC | t | p | p Holm | p Nemenyi |
| --- | --- | --- | --- | --- | --- |
| lr vs rf | -0.0055 | -0.38 | 0.7261 | 1.0000 | 0.8066 |
| lr vs xgb | 0.0029 | 0.16 | 0.8800 | 1.0000 | 1.0000 |
| lr vs tabnet | 0.0473 | 1.75 | 0.1548 | 0.9290 | 0.5624 |
| lr vs stack | -0.0084 | -0.63 | 0.5609 | 1.0000 | 0.3735 |
| rf vs xgb | 0.0084 | 1.91 | 0.1292 | 0.9044 | 0.8555 |
| rf vs tabnet | 0.0528 | 2.08 | 0.1061 | 0.8548 | 0.0703 |
| rf vs stack | -0.0029 | -1.48 | 0.2132 | 0.9290 | 0.9565 |
| xgb vs tabnet | 0.0444 | 1.64 | 0.1770 | 0.9290 | 0.4973 |
| xgb vs stack | -0.0113 | -2.18 | 0.0950 | 0.8548 | 0.4339 |
| tabnet vs stack | -0.0557 | -2.30 | 0.0830 | 0.8297 | 0.0086 |

### pima_indians__B

- Nadeau–Bengio + Holm significativos: ninguno.
- Nemenyi significativos (p < 0.05): ninguno. Rangos medios: lr 2.8, rf 2.2, xgb 2.6, tabnet 4.8, stack 2.6; diferencia crítica = 2.728.
- McNemar + Holm al umbral de Youden: sin diferencias.

| Par | Dif. media AUC | t | p | p Holm | p Nemenyi |
| --- | --- | --- | --- | --- | --- |
| lr vs rf | -0.0114 | -0.87 | 0.4338 | 1.0000 | 0.9751 |
| lr vs xgb | -0.0063 | -0.36 | 0.7400 | 1.0000 | 0.9996 |
| lr vs tabnet | 0.0577 | 2.13 | 0.1000 | 0.6999 | 0.2659 |
| lr vs stack | -0.0100 | -0.75 | 0.4956 | 1.0000 | 0.9996 |
| rf vs xgb | 0.0051 | 0.50 | 0.6455 | 1.0000 | 0.9946 |
| rf vs tabnet | 0.0691 | 3.09 | 0.0366 | 0.3293 | 0.0703 |
| rf vs stack | 0.0014 | 0.45 | 0.6734 | 1.0000 | 0.9946 |
| xgb vs tabnet | 0.0640 | 2.43 | 0.0717 | 0.5738 | 0.1796 |
| xgb vs stack | -0.0038 | -0.44 | 0.6804 | 1.0000 | 1.0000 |
| tabnet vs stack | -0.0677 | -3.24 | 0.0316 | 0.3159 | 0.1796 |

## 7. Conjuntos conformales ambiguos y decil superior

**Dónde va en la plantilla:** §3.5 («Reportar la proporción de pacientes con conjunto ambiguo {0, 1}») y verificación de «probabilidad media del decil superior ≤ 0.63»

| Escenario | Modelo | Conjunto ambiguo {0,1} | Conjunto vacío | Prob. media del decil superior |
| --- | --- | --- | --- | --- |
| B | Reg. logística L2 | 13.5 % | 0.0 % | 0.626 |
| B | Random Forest | 8.0 % | 0.0 % | 0.615 |
| B | XGBoost | 10.7 % | 0.0 % | 0.611 |
| B | TabNet | 10.7 % | 0.0 % | 0.554 |
| B | Stacking calibrado | 9.1 % | 0.0 % | 0.622 |
| A | Reg. logística L2 | 0.0 % | 4.3 % | 0.884 |
| A | Random Forest | 0.0 % | 2.5 % | 0.824 |
| A | XGBoost | 0.0 % | 2.2 % | 0.835 |
| A | TabNet | 0.0 % | 4.4 % | 0.799 |
| A | Stacking calibrado | 0.0 % | 5.0 % | 0.833 |

## 8. Tabla 5 (concordancia y plausibilidad, NHANES · B)

**Dónde va en la plantilla:** §3.6, Tabla 5 — reemplaza las columnas [ ] y el Jaccard top-5 de la nota

| Variable | Rango SHAP (XGBoost) | Rango permutación (XGBoost) | Rango IG (TabNet) | Dirección observada | Esperada (ADA, config) | ¿Coherente? |
| --- | --- | --- | --- | --- | --- | --- |
| edad | 1 | 1 | 4 | ↑ riesgo | + | sí |
| antecedente_familiar_diabetes | 2 | 2 | 1 | ↑ riesgo | + | sí |
| colesterol_total | 3 | 3 | 3 | ↓ riesgo | + | revisar |
| trigliceridos | 4 | 4 | 5 | ↑ riesgo | + | sí |
| tratamiento_antihipertensivo | 5 | 6 | 2 | ↑ riesgo | + | sí |
| imc | 6 | 7 | 6 | ↑ riesgo | + | sí |
| colesterol_hdl | 7 | 5 | 7 | ↓ riesgo | - | sí |
| antecedente_hipertension | 8 | 10 | 9 | ↑ riesgo | + | sí |
| presion_diastolica | 9 | 8 | 8 | ↓ riesgo | + | revisar |
| presion_sistolica | 10 | 9 | 10 | ↑ riesgo | + | sí |

Spearman mediano entre métodos: 0.85 (rango 0.66–1.00); Jaccard top-5 mediano: 0.67 (rango 0.43–1.00). Dirección observada: signo de la correlación de Spearman entre el valor de la variable y su SHAP en la prueba.

Concordancia por corrida (verificación de §3.6): nhanes_2017_2018__A: ρ = 0.79, Jaccard = 0.67; nhanes_2017_2018__B: ρ = 0.85, Jaccard = 0.67; pima_indians__A: ρ = 0.62, Jaccard = 0.67; pima_indians__B: ρ = 0.71, Jaccard = 0.67.

## 9. Equidad por subgrupo (Tabla 6 y Tabla S2)

**Dónde va en la plantilla:** §3.7, Tabla 6 (columna «Eventos por subgrupo») y nota «Añadir sensibilidad, especificidad y AUC con IC 95 %»; §2.10 (cortes de Pima)

Los grupos de edad se rotulan 18–39 en el código, pero NHANES excluye a menores de 20, así que el grupo real es 20–39. En Pima los cortes son 21–39, 40–59 y ≥ 60 (misma función).

### nhanes_2017_2018__A · modelo recomendado: XGBoost

| Atributo | Grupo | n | Eventos | Sensibilidad [IC 95 %] | Especificidad [IC 95 %] | AUC [IC 95 %] |
| --- | --- | --- | --- | --- | --- | --- |
| sexo | Femenino | 368 | 50 | 0.780 [0.669–0.900] | 0.918 [0.888–0.947] | 0.935 [0.897–0.965] |
| sexo | Masculino | 358 | 67 | 0.881 [0.806–0.955] | 0.880 [0.842–0.914] | 0.956 [0.935–0.976] |
| edad | 20-39 | 199 | 6 | 0.500 [0.167–0.833] | 1.000 [1.000–1.000] | 0.942 [0.838–1.000] |
| edad | 40-59 | 215 | 24 | 0.833 [0.667–0.958] | 0.895 [0.848–0.937] | 0.939 [0.885–0.974] |
| edad | >=60 | 312 | 87 | 0.862 [0.793–0.926] | 0.818 [0.764–0.867] | 0.930 [0.899–0.955] |

### nhanes_2017_2018__B · modelo recomendado: Stacking calibrado

| Atributo | Grupo | n | Eventos | Sensibilidad [IC 95 %] | Especificidad [IC 95 %] | AUC [IC 95 %] |
| --- | --- | --- | --- | --- | --- | --- |
| sexo | Femenino | 368 | 50 | 0.840 [0.740–0.940] | 0.670 [0.623–0.720] | 0.835 [0.762–0.897] |
| sexo | Masculino | 358 | 67 | 0.910 [0.836–0.970] | 0.625 [0.574–0.673] | 0.865 [0.817–0.907] |
| edad | 20-39 | 199 | 6 | 0.000 [0.000–0.000] | 0.979 [0.961–0.995] | 0.498 [0.315–0.698] |
| edad | 40-59 | 215 | 24 | 0.833 [0.708–0.958] | 0.670 [0.607–0.736] | 0.878 [0.806–0.938] |
| edad | >=60 | 312 | 87 | 0.954 [0.908–0.989] | 0.347 [0.289–0.404] | 0.797 [0.739–0.846] |

### pima_indians__A · modelo recomendado: Stacking calibrado

| Atributo | Grupo | n | Eventos | Sensibilidad [IC 95 %] | Especificidad [IC 95 %] | AUC [IC 95 %] |
| --- | --- | --- | --- | --- | --- | --- |
| sexo | Femenino | 117 | 42 | 0.833 [0.714–0.929] | 0.787 [0.693–0.880] | 0.873 [0.805–0.927] |
| edad | 21-39 | 88 | 25 | 0.840 [0.680–0.960] | 0.841 [0.746–0.921] | 0.882 [0.796–0.951] |
| edad | 40-59 | 25 | 16 | 0.875 [0.688–1.000] | 0.444 [0.111–0.778] | 0.833 [0.646–0.965] |

### pima_indians__B · modelo recomendado: Random Forest

| Atributo | Grupo | n | Eventos | Sensibilidad [IC 95 %] | Especificidad [IC 95 %] | AUC [IC 95 %] |
| --- | --- | --- | --- | --- | --- | --- |
| sexo | Femenino | 117 | 42 | 0.667 [0.524–0.809] | 0.653 [0.547–0.747] | 0.767 [0.677–0.851] |
| edad | 21-39 | 88 | 25 | 0.480 [0.320–0.680] | 0.730 [0.619–0.825] | 0.761 [0.667–0.858] |
| edad | 40-59 | 25 | 16 | 1.000 [1.000–1.000] | 0.111 [0.000–0.333] | 0.722 [0.510–0.896] |

## 10. Verificación de afirmaciones de la plantilla

**Dónde va en la plantilla:** Revisar §3.2, §3.5 y el Resumen; corregir si alguna no coincide

- Beneficio neto positivo de los cinco modelos en 5–50 %: no (TabNet −0.001 en 0.49–0.50; los otros cuatro siempre positivos); todos ≥ «tratar a todos» en todo el rango: sí.
- Umbrales donde la RL tiene el mayor beneficio neto: 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24 (la plantilla dice «entre el 10 % y el 25 %»).
- Brecha de optimismo B (RL, RF, XGBoost): -0.0010, -0.0014, -0.0031; rango -0.0031 a -0.0010.
- PR-AUC B: 0.437–0.571, es decir 2.7 a 3.6 veces la prevalencia de 0.160.
- Δ ROC-AUC A→B: -0.128 a -0.089.

## 11. Hardware

**Dónde va en la plantilla:** §2.12 («Indicar el hardware»)

Texto: «Los análisis se ejecutaron en CPU (11th Gen Intel Core i7-1165G7 @ 2.80 GHz (4 núcleos, 8 hilos), 19.8 GB de RAM, Windows 11) sin GPU; PyTorch se usó en su versión para CPU.»

## 12. Análisis de sensibilidad (b): pesos muestrales WTMEC2YR

**Dónde va en la plantilla:** §2.11 (b); resultados como nueva Tabla S4 y una frase en §3.2 o §4.7

Métricas de la prueba aislada reponderadas con WTMEC2YR (sin reentrenar). Prevalencia ponderada en la prueba: 0.123 (sin ponderar 0.161).

| Escenario | Modelo | ROC-AUC sin ponderar | ROC-AUC ponderado | Brier ponderado |
| --- | --- | --- | --- | --- |
| B | Reg. logística L2 | 0.856 | 0.879 | 0.080 |
| B | Random Forest | 0.852 | 0.878 | 0.075 |
| B | XGBoost | 0.856 | 0.872 | 0.078 |
| B | TabNet | 0.807 | 0.837 | 0.088 |
| B | Stacking calibrado | 0.855 | 0.874 | 0.078 |
| A | Reg. logística L2 | 0.945 | 0.955 | 0.051 |
| A | Random Forest | 0.952 | 0.959 | 0.048 |
| A | XGBoost | 0.947 | 0.953 | 0.051 |
| A | TabNet | 0.935 | 0.948 | 0.054 |
| A | Stacking calibrado | 0.952 | 0.959 | 0.049 |

## 13. Análisis de sensibilidad (a), (c) y (d)

**Dónde va en la plantilla:** §2.11 (quitar el resaltado «ejecutar o eliminar»), nueva Tabla S5 y una frase en §3.2/§4.4/§4.7

Protocolo: igual al del modelo final del pipeline (Optuna en desarrollo con 3 pliegues internos, calibración sigmoide y umbral de Youden en Validación, prueba aislada leída una vez, IC bootstrap B = 2000). La fila de referencia reproduce el escenario B principal. En (a) cambia la etiqueta, así que la partición estratificada se recalcula.

| Análisis | Modelo | Eventos (total) | ROC-AUC prueba [IC 95 %] | Sensibilidad | Especificidad | Brier | Pendiente |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Referencia: escenario B (DIQ010 = 1) | Reg. logística L2 | 772 | 0.856 [0.818–0.889] | 0.872 | 0.759 | 0.101 | 1.02 |
| Referencia: escenario B (DIQ010 = 1) | XGBoost | 772 | 0.856 [0.815–0.892] | 0.880 | 0.658 | 0.097 | 1.02 |
| (a1) Caso = DIQ010 = 1 o HbA1c ≥ 6.5 % sin diagnóstico | Reg. logística L2 | 948 | 0.837 [0.804–0.869] | 0.826 | 0.706 | 0.121 | 1.08 |
| (a1) Caso = DIQ010 = 1 o HbA1c ≥ 6.5 % sin diagnóstico | XGBoost | 948 | 0.850 [0.818–0.881] | 0.750 | 0.795 | 0.117 | 1.12 |
| (a2) Caso = DIQ010 = 1 o HbA1c ≥ 6.5 % o glucosa ≥ 126 mg/dL (glucosa sin ayuno garantizado) | Reg. logística L2 | 1004 | 0.844 [0.807–0.876] | 0.821 | 0.736 | 0.118 | 1.24 |
| (a2) Caso = DIQ010 = 1 o HbA1c ≥ 6.5 % o glucosa ≥ 126 mg/dL (glucosa sin ayuno garantizado) | XGBoost | 1004 | 0.861 [0.826–0.891] | 0.887 | 0.668 | 0.114 | 1.33 |
| (d) Escenario B sin colesterol total ni tratamiento antihipertensivo | Reg. logística L2 | 772 | 0.845 [0.806–0.880] | 0.812 | 0.775 | 0.104 | 1.01 |
| (d) Escenario B sin colesterol total ni tratamiento antihipertensivo | XGBoost | 772 | 0.838 [0.796–0.876] | 0.889 | 0.675 | 0.103 | 1.03 |
| (c) TabNet con búsqueda Optuna (10 ensayos) | TabNet | 772 | 0.819 [0.778–0.857] | 0.821 | 0.642 | 0.108 | 0.94 |
