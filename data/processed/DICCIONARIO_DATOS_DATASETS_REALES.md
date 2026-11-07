# Diccionario de Datos y Especificación Científica de Datasets Reales
## Proyecto SP-03: Predicción Temprana de Diabetes Mellitus Tipo 2 y Riesgo de Complicaciones con IA Explicable

---

## 1. Resumen de Cohortes y Datasets Clínicos Reales Disponibles

El proyecto cuenta con **cinco conjuntos de datos reales y auditados** de acuerdo con las directrices PRISMA 2020 y el protocolo común de investigación recomendado por la cátedra:

| Cohorte / Dataset | Archivos Generados | Muestra Real ($n$) | Variables | Naturaleza de los Datos | Objetivo Clínico / Científico |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **1. CDC NHANES 2017–2018** *(Cohorte Primaria)* | `nhanes_2017_2018_dataset_completo.csv`<br>`nhanes_2017_2018_dataset_completo.xlsx` | **4,829** adultos (≥20 años) | 16 cols (12 predictoras) | Biomarcadores de laboratorio HPLC (HbA1c, glucosa, lípidos) y examen físico | Predicción clínica ambulatoria de alta precisión diagnóstica basada en laboratorio (Estándar Q1). |
| **2. NIDDK Pima Indians** *(Benchmark Histórico)* | `pima_indians_dataset_completo.csv`<br>`pima_indians_dataset_completo.xlsx` | **768** pacientes | 12 cols (8 predictoras) | Examen médico, OGTT, grosor de pliegue y antecedentes genéticos | Comparabilidad con la literatura clásica biomédica de las últimas décadas. |
| **3. CDC BRFSS 2015** *(Salud Poblacional)* | `cdc_brfss_5050_clean.csv`<br>`cdc_brfss_5050_sample_5k.xlsx` | **70,692** encuestados (50/50) | 22 cols (21 predictoras) | Factores de riesgo conductuales, comorbilidades y estilo de vida (Big Data) | Tamizaje epidemiológico masivo a nivel poblacional sin requerir punción venosa previa. |
| **4. Diabetes 130-US Hospitals** *(UCI 296)* | `diabetes_130_hospitals_clean.csv`<br>`diabetes_130_hospitals_sample_5k.xlsx` | **71,515** pacientes únicos (sin fuga) | 18 cols (16 predictoras) | Registros hospitalarios, farmacoterapia (insulina), admisiones y diagnósticos | Predicción de complicaciones agudas, descompensación y reingreso prematuro (<30 días). |
| **5. Early Stage Diabetes** *(UCI 529 - Sylhet)* | `early_stage_diabetes_sylhet_clean.csv`<br>`early_stage_diabetes_sylhet_clean.xlsx` | **520** pacientes | 17 cols (16 predictoras) | Signos y síntomas tempranos prodrómicos (poliuria, polidipsia, astenia) | Detección precoz en atención primaria basada exclusivamente en sintomatología inicial. |

---

## 2. Diccionario de Variables — CDC NHANES 2017–2018 (Cohorte Primaria)

Archivo: `data/processed/nhanes_2017_2018_dataset_completo.csv` (y `.xlsx`)  
Número de registros: **4,829 filas**  
Número de variables: **16 columnas**

| # | Columna | Código CDC | Tipo | Unidad | Rango / Valores | Norma Clínica (ADA 2024 / OMS) | Descripción Médica |
| :-: | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| 1 | `patient_id` | `SEQN` | Texto | ID | 93703 – 102956 | Identificador único | Código de secuencia anonimizado asignado por el NCHS/CDC. |
| 2 | `outcome` | `DIQ010` | Binario | 0 / 1 | 0: No / 1: Sí | Criterio Diagnóstico DM2 | Variable objetivo: Diagnóstico médico confirmado de diabetes mellitus tipo 2. |
| 3 | `sexo_label` | `RIAGENDR` | Categórico | Texto | Masculino / Femenino | Demografía clínica | Variable sensible reservada para auditoría de equidad (fairness). |
| 4 | `grupo_edad` | `RIDAGEYR` | Categórico | Rango | 20-39 / 40-59 / ≥60 | Estratificación | Grupo etario utilizado para evaluar paridad estadística por subgrupo. |
| 5 | `edad` | `RIDAGEYR` | Numérico | años | 20.0 – 80.0 | Adultos (≥20 años) | Edad en años cumplidos en el examen médico MEC. |
| 6 | `imc` | `BMXBMI` | Numérico | kg/m² | 14.1 – 67.3 | 18.5–24.9 (Normal)<br>25.0–29.9 (Sobrepeso)<br>≥30.0 (Obesidad) | Índice de Masa Corporal (peso en kg / talla en m²). |
| 7 | `glucosa` | `LBXSGL` | Numérico | mg/dL | 49.0 – 505.0 | < 100 mg/dL (Normal)<br>100–125 (Prediabetes)<br>≥ 126 mg/dL (Diabetes) | Glucemia sérica en ayunas de 8 horas (método enzimático hexocinasa). |
| 8 | `hba1c` | `LBXGH` | Numérico | % | 4.0 – 16.6 | < 5.7 % (Normal)<br>5.7–6.4 % (Prediabetes)<br>≥ 6.5 % (Diabetes) | Hemoglobina glicosilada medida por HPLC certificada por NGSP. |
| 9 | `presion_sistolica` | `BPXSY1` | Numérico | mmHg | 72.0 – 236.0 | < 120 mmHg (Normal)<br>120–129 (Elevada)<br>≥ 130 mmHg (Hipertensión) | Presión arterial sistólica en reposo con esfigmomanómetro. |
| 10 | `presion_diastolica` | `BPXDI1` | Numérico | mmHg | 32.0 – 122.0 | < 80 mmHg (Normal)<br>≥ 80 mmHg (Hipertensión) | Presión arterial diastólica (fase V de Korotkoff). |
| 11 | `colesterol_total` | `LBXTC` | Numérico | mg/dL | 78.0 – 450.0 | < 200 mg/dL (Deseable)<br>200–239 (Límite alto)<br>≥ 240 mg/dL (Alto) | Colesterol total sérico enzimático. |
| 12 | `colesterol_hdl` | `LBDHDD` | Numérico | mg/dL | 18.0 – 148.0 | > 40 mg/dL (Hombres)<br>> 50 mg/dL (Mujeres) | Lipoproteínas de alta densidad (colesterol "protector"). |
| 13 | `trigliceridos` | `LBXSTR` | Numérico | mg/dL | 26.0 – 1480.0 | < 150 mg/dL (Normal)<br>150–199 (Límite)<br>≥ 200 mg/dL (Elevado) | Triglicéridos séricos en ayunas. Marcador de resistencia insulínica. |
| 14 | `antecedente_familiar_diabetes` | `MCQ300C` | Binario | 0 / 1 | 0: No / 1: Sí | Riesgo genético | Antecedente de diabetes en familiares biológicos de primer grado. |
| 15 | `antecedente_hipertension` | `BPQ020` | Binario | 0 / 1 | 0: No / 1: Sí | Comorbilidad | Diagnóstico previo de hipertensión arterial emitido por médico. |
| 16 | `tratamiento_antihipertensivo` | `BPQ040A` | Binario | 0 / 1 | 0: No / 1: Sí | Farmacoterapia | Uso actual de fármacos antihipertensivos prescritos. |

---

## 3. Diccionario de Variables — CDC BRFSS 2015 (Salud Poblacional y Estilo de Vida)

Archivo: `data/processed/cdc_brfss_5050_clean.csv`  
Número de registros: **70,692 filas** (balanceado: 35,346 diabetes / 35,346 controles sanos)  
Número de variables: **22 columnas**

| # | Variable | Tipo | Rango | Descripción y Significado Clínico-Epidemiológico |
| :-: | :--- | :---: | :---: | :--- |
| 1 | `target` | Binario | 0 / 1 | Variable objetivo: 0 = No diabetes, 1 = Diabetes o Prediabetes diagnosticada. |
| 2 | `HighBP` | Binario | 0 / 1 | Presión arterial alta diagnosticada por profesional de la salud. |
| 3 | `HighChol` | Binario | 0 / 1 | Colesterol sanguíneo elevado reportado por laboratorio. |
| 4 | `CholCheck` | Binario | 0 / 1 | Chequeo de colesterol realizado en los últimos 5 años. |
| 5 | `BMI` | Numérico | 12 – 98 | Índice de Masa Corporal autodeclarado (kg/m²). |
| 6 | `Smoker` | Binario | 0 / 1 | Ha fumado al menos 100 cigarrillos a lo largo de toda su vida. |
| 7 | `Stroke` | Binario | 0 / 1 | Diagnóstico previo de Accidente Cerebrovascular (ACV). |
| 8 | `HeartDiseaseorAttack` | Binario | 0 / 1 | Enfermedad coronaria (infarto agudo de miocardio o angina). |
| 9 | `PhysActivity` | Binario | 0 / 1 | Realiza actividad física o ejercicio regular fuera de las labores rutinarias. |
| 10 | `Fruits` | Binario | 0 / 1 | Consume fruta 1 o más veces al día. |
| 11 | `Veggies` | Binario | 0 / 1 | Consume verduras 1 o más veces al día. |
| 12 | `HvyAlcoholConsump` | Binario | 0 / 1 | Consumo excesivo de alcohol (>14 tragos/semana hombres, >7 mujeres). |
| 13 | `AnyHealthcare` | Binario | 0 / 1 | Cuenta con cualquier cobertura médica o seguro de salud. |
| 14 | `NoDocbcCost` | Binario | 0 / 1 | No pudo visitar al médico en los últimos 12 meses por limitaciones de costo. |
| 15 | `GenHlth` | Ordinal | 1 – 5 | Autopercepción del estado de salud general (1 = Excelente a 5 = Pobre). |
| 16 | `MentHlth` | Entero | 0 – 30 | Días del último mes con afectación de salud mental (estrés, depresión). |
| 17 | `PhysHlth` | Entero | 0 – 30 | Días del último mes con enfermedad física o lesiones. |
| 18 | `DiffWalk` | Binario | 0 / 1 | Dificultad severa para caminar o subir escaleras (movilidad reducida). |
| 19 | `Sex` | Binario | 0 / 1 | Sexo biológico (0 = Femenino, 1 = Masculino). |
| 20 | `Age` | Ordinal | 1 – 13 | Categoría de edad en intervalos de 5 años (1 = 18-24 hasta 13 = ≥80 años). |
| 21 | `Education` | Ordinal | 1 – 6 | Nivel educativo máximo completado (1 = Primaria incompleta a 6 = Universidad). |
| 22 | `Income` | Ordinal | 1 – 8 | Escala de ingresos anuales del hogar (1 = <$10k a 8 = ≥$75k). |

---

## 4. Diccionario de Variables — Diabetes 130-US Hospitals (UCI 296: Complicaciones y Reingreso)

Archivo: `data/processed/diabetes_130_hospitals_clean.csv`  
Número de registros: **71,515 pacientes únicos** (filtrado con regla anti-fuga: primer encuentro hospitalario por paciente)  
Número de variables: **18 columnas**

| # | Variable | Tipo | Rango / Valores | Significado Clínico y Pronóstico |
| :-: | :--- | :---: | :---: | :--- |
| 1 | `patient_nbr` | Texto | ID único | Identificador del paciente (para agrupamiento y validación cruzada). |
| 2 | `target` | Binario | 0 / 1 | **Variable objetivo:** Reingreso hospitalario prematuro en < 30 días (1 = Sí, 0 = No o >30 días). Marcador de complicación aguda o descompensación. |
| 3 | `gender_male` | Binario | 0 / 1 | Sexo del paciente (1 = Masculino, 0 = Femenino). |
| 4 | `age_numeric` | Numérico | 5 – 95 | Punto medio del rango de edad en años (e.g. `[60-70)` = 65 años). |
| 5 | `time_in_hospital` | Entero | 1 – 14 | Días de estancia en el centro hospitalario. |
| 6 | `num_lab_procedures` | Entero | 1 – 132 | Número de exámenes de laboratorio realizados durante la estancia. |
| 7 | `num_procedures` | Entero | 0 – 6 | Número de procedimientos médicos no de laboratorio realizados. |
| 8 | `num_medications` | Entero | 1 – 81 | Número de medicamentos prescritos al paciente durante el encuentro. |
| 9 | `number_outpatient` | Entero | 0 – 42 | Visitas ambulatorias en el año previo a la admisión. |
| 10 | `number_emergency` | Entero | 0 – 76 | Visitas a emergencias médicas en el año previo a la admisión. |
| 11 | `number_inpatient` | Entero | 0 – 21 | Hospitalizaciones previas en el año anterior. |
| 12 | `number_diagnoses` | Entero | 1 – 16 | Número total de diagnósticos clínicos asentados en el historial. |
| 13 | `insulin_treatment` | Binario | 0 / 1 | Paciente con prescripción de insulina (Up, Down, Steady vs No). |
| 14 | `diabetes_med_prescribed` | Binario | 0 / 1 | Cualquier fármaco antidiabético prescrito al alta médica. |
| 15 | `medication_changed` | Binario | 0 / 1 | Se modificó la dosis o el principio activo antidiabético durante la estancia. |
| 16 | `a1c_tested` | Binario | 0 / 1 | Se cuantificó la hemoglobina glicosilada durante la estancia hospitalaria. |
| 17 | `a1c_elevated` | Binario | 0 / 1 | HbA1c descompensada (>7% o >8%). |
| 18 | `readmitted` | Texto | `<30`, `>30`, `NO` | Registro original de reingreso para análisis multinomial. |

---

## 5. Diccionario de Variables — Early Stage Diabetes Risk (UCI 529 - Sylhet)

Archivo: `data/processed/early_stage_diabetes_sylhet_clean.csv` (y `.xlsx`)  
Número de registros: **520 filas** (320 positivas / 200 negativas)  
Número de variables: **17 columnas**

| # | Variable | Tipo | Valores | Significado Médico y Prodromos |
| :-: | :--- | :---: | :---: | :--- |
| 1 | `target` | Binario | 0 / 1 | Variable objetivo: 1 = Diabetes confirmada por clínica, 0 = No diabetes. |
| 2 | `age` | Numérico | 16 – 90 | Edad del paciente en años. |
| 3 | `gender` | Binario | 1 / 0 | Sexo biológico (1 = Masculino, 0 = Femenino). |
| 4 | `polyuria` | Binario | 0 / 1 | Exceso de volumen urinario (>3 L/día). Síntoma cardinal de hiperglucemia. |
| 5 | `polydipsia` | Binario | 0 / 1 | Sed excesiva y constante derivada de la diuresis osmótica. |
| 6 | `sudden_weight_loss` | Binario | 0 / 1 | Pérdida de peso involuntaria y repentina por catabolismo lipídico y proteico. |
| 7 | `weakness` | Binario | 0 / 1 | Fatiga generalizada, astenia o falta de energía persistente. |
| 8 | `polyphagia` | Binario | 0 / 1 | Hambre voraz secundaria a la falta de captación celular de glucosa. |
| 9 | `genital_thrush` | Binario | 0 / 1 | Candidiasis genital recurrente favorecida por glucosuria. |
| 10 | `visual_blurring` | Binario | 0 / 1 | Visión borrosa por cambios osmóticos en el cristalino. |
| 11 | `itching` | Binario | 0 / 1 | Prurito cutáneo generalizado o localizado. |
| 12 | `irritability` | Binario | 0 / 1 | Labilidad emocional o irritabilidad por fluctuaciones glucémicas. |
| 13 | `delayed_healing` | Binario | 0 / 1 | Cicatrización retardada de heridas o úlceras dérmicas. |
| 14 | `partial_paresis` | Binario | 0 / 1 | Paresia o debilidad neuromuscular periférica (neuropatía diabética). |
| 15 | `muscle_stiffness` | Binario | 0 / 1 | Rigidez musculoesquelética o calambres nocturnos. |
| 16 | `alopecia` | Binario | 0 / 1 | Caída inusual de cabello relacionada con desregulación metabólica. |
| 17 | `obesity` | Binario | 0 / 1 | Obesidad clínica observable o diagnosticada. |

---

## 6. Diccionario de Variables — NIDDK Pima Indians (Cohorte Benchmark)

Archivo: `data/processed/pima_indians_dataset_completo.csv` (y `.xlsx`)  
Número de registros: **768 filas**  
Número de variables: **12 columnas**

| # | Columna | Tipo | Unidad | Valores | Descripción Médica |
| :-: | :--- | :---: | :---: | :---: | :--- |
| 1 | `patient_id` | Texto | ID | PIMA_001 – PIMA_768 | Identificador de fila para agrupamiento sin fuga. |
| 2 | `outcome` | Binario | 0 / 1 | 0: Negativo / 1: Positivo | Diagnóstico de diabetes a los 5 años según criterios OMS. |
| 3 | `sexo_label` | Categórico | Texto | Femenino | Cohorte 100% femenina de herencia indígena Pima. |
| 4 | `grupo_edad` | Categórico | Rango | 21-39 / 40-59 / ≥60 | Grupo etario para auditoría de subgrupos. |
| 5 | `embarazos` | Numérico | gestaciones | 0 – 17 | Número de gestaciones previas. |
| 6 | `glucosa` | Numérico | mg/dL | 44 – 199 | Glucemia plasmática a las 2 horas de sobrecarga oral (OGTT). |
| 7 | `presion_diastolica` | Numérico | mmHg | 30 – 122 | Presión arterial diastólica (mmHg). |
| 8 | `pliegue_cutaneo` | Numérico | mm | 7 – 99 | Grosor del pliegue cutáneo del tríceps (mm). |
| 9 | `insulina` | Numérico | μU/mL | 14 – 846 | Insulina sérica a las 2 horas de sobrecarga de glucosa. |
| 10 | `imc` | Numérico | kg/m² | 18.2 – 67.1 | Índice de Masa Corporal. |
| 11 | `pedigri_diabetes` | Numérico | índice | 0.078 – 2.42 | Función de antecedentes genéticos familiares. |
| 12 | `edad` | Numérico | años | 21 – 81 | Edad en años cumplidos (población ≥ 21 años). |

---

## 7. Garantías Científicas y Prevención de Fuga de Información (*Zero Data Leakage*)

1. **Desacoplamiento por Paciente Único:** En la cohorte hospitalaria UCI 296, se filtraron múltiples admisiones para dejar un solo encuentro por paciente (`patient_nbr`), eliminando la contaminación cruzada entre pliegues de entrenamiento y validación.
2. **Validación Cruzada Estratificada:** En todas las cohortes se utiliza partición estratificada por la variable objetivo para conservar la prevalencia epidemiológica real en cada pliegue.
3. **Imputación y Escalamiento Ciegos al Test:** Los transformadores (`StandardScaler`, `SimpleImputer`) se ajustan únicamente en el conjunto de entrenamiento de cada iteración y se transforman sobre la prueba sin filtrar estadísticas globales.
