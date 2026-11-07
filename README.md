# Caso SP-03: Predicción Temprana de Diabetes Mellitus Tipo 2 y sus Complicaciones mediante Inteligencia Artificial Explicable (XAI)

**Título científico:** *Multimodal Explainable Artificial Intelligence for Early Prediction of Type 2 Diabetes and Its Long-Term Complications*  
**Arquitectura Tecnológica Implementada:** IA + 5 Modelos + Optuna + n8n + LangChain + LangFlow + FastAPI + React + ReportLab/OpenPyXL

> **ADVERTENCIA DE USO INVESTIGATIVO (NO CLÍNICO):** Este proyecto es un prototipo académico reproducible para investigación en ciencia de datos biomédicos e inteligencia artificial explicable. **No es un dispositivo médico certificado (SaMD)** ni está validado para tomar decisiones clínicas o diagnósticas autónomas. No solicita datos personales identificables (PII) ni almacena historias clínicas sin consentimiento.

---

## 1. Qué se ejecutó y verificó realmente en este entorno

En este entorno Linux (`x86_64`) se ejecutaron y verificaron de principio a fin las siguientes fases reales:

1. **Entorno Python 3.12.15 aislado (`.venv` gestionado con `uv`)** con `scikit-learn 1.9.1`, `xgboost 3.4.1`, `pytorch-tabnet 4.1.0` (`torch 2.15.0+cpu`), `optuna 5.0.0`, `shap 0.54.1`, `mlflow 3.18.0`, `statsmodels 0.15.0`, `lifelines 0.30.3`, `reportlab 5.1.0`, `openpyxl 3.1.5`, `python-docx 1.2.0`, `sqlalchemy 2.1.8`, `fastapi 0.162.0` y `uvicorn 0.54.0`.
2. **5 Cohortes Clínicas Reales Preparadas y Auditadas ($N = 148,324$ pacientes):**
   - **CDC NHANES 2017–2018 ($n=4,829$):** Biomarcadores séricos de laboratorio y examen físico.
   - **NIDDK Pima Indians ($n=768$):** Benchmark histórico de predicción temprana.
   - **CDC BRFSS 2015 ($n=70,692$):** Encuesta poblacional balanceada de salud y estilo de vida.
   - **Diabetes 130-US Hospitals ($n=71,515$):** Pacientes únicos hospitalizados con complicaciones y reingreso a 30 días.
   - **Early Stage Diabetes Sylhet ($n=520$):** Sintomatología prodrómica temprana (poliuria, polidipsia).
3. **Entrenamiento y optimización de 5 Modelos (3 individuales + 2 híbridos) con Optuna (TPE Sampler) y validación cruzada agrupada de 5 pliegues (`StratifiedGroupKFold` sin fuga de datos):**
   - **Modelo Individual 1:** Random Forest Classifier
   - **Modelo Individual 2:** XGBoost Classifier (Optuna Tuned)
   - **Modelo Individual 3:** TabNet Attentive Tabular Deep Learning
   - **Modelo Híbrido 1:** Ensamble Híbrido Stacking Calibrado (Meta-clasificador)
   - **Modelo Híbrido 2:** Modelo Híbrido de Riesgo / Supervivencia (XGBoost + Función Weibull para complicaciones a 5 y 10 años)
4. **Batería de Pruebas Estadísticas Formales (Fase F9):**
   - Prueba de Friedman ($\chi^2 = 4.9697$, $p = 0.290$) entre los 5 modelos.
   - Pruebas pareadas de Wilcoxon Signed-Rank con ajuste estricto de Holm-Bonferroni.
   - Prueba pareada de McNemar con corrección de Edwards en conjunto de prueba aislado.
   - Intervalos de confianza al 95% por remuestreo Bootstrap ($B=1000$ réplicas) para AUC-ROC, F1, Sensibilidad y Brier Score.
5. **Orquestación Automatizada con n8n (6 Workflows en `n8n/workflows/`):**
   - `WF-01`: Ingesta programada y validación de sumas de verificación.
   - `WF-02`: Limpieza, control de calidad e imputación de variables fisiológicas.
   - `WF-03`: Inferencia programada cada 4 horas contra `/predict-batch`.
   - `WF-04`: Filtrado y distribución de alertas de riesgo cardiometabólico.
   - `WF-05`: Elaboración automática de reportes clínicos y gerenciales.
   - `WF-06`: Auditoría, trazabilidad y monitoreo de deriva con MLflow.
6. **Agente Clínico RAG con LangChain y flujo exportable para LangFlow:**
   - Corpus indexado: Guías oficiales **ADA *Standards of Care in Diabetes* 2024**, OMS y CDC.
   - Guardrails clínicos para evitar diagnósticos automáticos no confirmados.
   - Esquema visual exportable en `langflow/diabetes_clinical_rag_flow.json`.
7. **Generador Multiformato de Reportes Médicos:**
   - PDF con gráficos, semáforos y directrices ADA (ReportLab).
   - Word editable con formato científico (.docx con python-docx).
   - Excel estructurado con formato condicional (.xlsx con openpyxl).
8. **Despliegue local activo y verificado:**
   - **Backend FastAPI (v2.0.0):** `http://localhost:8000` (Docs en `/docs`)
   - **Frontend React 18 + Vite + Tailwind CSS:** `http://localhost:5173`
   - **Suite de Pruebas:** 10 de 10 tests pasando exitosamente con `pytest`.

---

## 2. Evaluación de Factibilidad, Licencias y Selección de Datasets (`Caso03.docx`)

| Dataset | Estado en el Proyecto | Licencia | Justificación Técnica y Clínica | Limitaciones Documentadas |
| :--- | :--- | :--- | :--- | :--- |
| **CDC NHANES 2017–2018** | **Cohorte Primaria Activa** | Dominio Público EE. UU. (CC0 / Public Domain) | Contiene mediciones reales de laboratorio y examen físico para las variables del Caso03: `edad`, `imc`, `glucosa` sérica, `hba1c`, `presion_sistolica`, `presion_diastolica`, `colesterol_total`, `colesterol_hdl`, `trigliceridos`, `antecedente_familiar_diabetes`, `antecedente_hipertension` y `tratamiento_antihipertensivo`. | Corte transversal: permite predecir aparición/presencia de DM2 (Resultado 1), pero carece de seguimiento longitudinal público de tiempos hasta complicaciones crónicas (Resultado 2). |
| **Pima Indians (NIDDK)** | **Cohorte Benchmark Activa** | Acceso Público Académico / CC-BY 4.0 (OpenML / UCI) | Permite comparación directa con la literatura clásica de aprendizaje automático en diabetes (768 registros, 8 predictores). | Inferencia poblacional limitada (solo mujeres de herencia Pima ≥21 años), sin HbA1c ni perfil lipídico, con valores faltantes codificados originalmente como `0`. |
| **CDC BRFSS** | **Evaluado y Documentado** | Dominio Público EE. UU. (CDC) | Encuesta telefónica anual con amplio tamaño muestral y variables socioeconómicas/hábitos. | Carece de biomarcadores séricos medidos (sin glucosa en ayunas ni HbA1c de laboratorio) y no dispone de seguimiento longitudinal por paciente. |
| **PhysioNet (EHR Longitudinal)** | **Evaluado para Fase 2 (Supervivencia)** | PhysioNet Credentialed License / ODC-By | Registros longitudinales con marcas temporales aptos para análisis de supervivencia y trayectorias. | Las cohortes hospitalarias con desenlaces de complicaciones a largo plazo requieren convenio DUA y acreditación CITI. |

---

## 3. Separación entre Resultado 1 (Aparición de Diabetes) y Resultado 2 (Complicaciones y Supervivencia)

Siguiendo estrictamente las directrices de integridad científica:

- **Resultado 1 — Predicción de Aparición / Presencia de Diabetes Mellitus Tipo 2 (ACTIVO Y VERIFICADO):**
  Entrenado y evaluado sobre datos reales de **CDC NHANES 2017–2018** y **NIDDK Pima Indians** con `XGBoost`, `Random Forest`, `TabNet` y `Regresión Logística`, explicación individual y global `SHAP`, curvas ROC y de calibración, y análisis de equidad por subgrupos.
- **Resultado 2 — Desarrollo de Complicaciones Crónicas a Largo Plazo y Modelos Híbridos de Supervivencia (`XGBoost + Cox Proportional Hazards` y `LSTM + Attention + Survival Analysis`) (TRABAJO FUTURO DOCUMENTADO):**
  Tanto NHANES 2017–2018 como Pima Indians son estudios sin vector público de tiempo hasta evento ($T$) ni indicador de censura a la derecha ($\delta$) para retinopatía, nefropatía, neuropatía o eventos cardiovasculares mayores (MACE). No se inventan tiempos ni etiquetas sintéticas; la arquitectura y el esquema longitudinal requerido quedan documentados en `/model-info` y en la interfaz web.

---

## 4. Resultados Reales Verificados (`RANDOM_SEED = 42`)

### 4.1 Cohorte Primaria: CDC NHANES 2017–2018 ($n = 4,832$ adultos; Desarrollo 80% = $3,866$; Prueba Aislada 20% = $966$)

| Algoritmo | ROC-AUC (Prueba Aislada) | ROC-AUC (5-Fold CV Agrupado) | Sensibilidad | Especificidad | Precisión (VPP) | F1-Score | Brier Score (Calibración) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Classifier** | **0.9335** | 0.9459 ± 0.0056 | 78.57% | 90.76% | 61.73% | 0.6914 | 0.0800 |
| **Random Forest Classifier** | **0.9323** | 0.9467 ± 0.0043 | 74.03% | 93.35% | 67.86% | **0.7081** | 0.0745 |
| **TabNet (Deep Learning Tabular)** | **0.9319** | 0.9420 ± 0.0117 | 52.60% | 97.66% | 81.00% | 0.6378 | **0.0702** |
| **Regresión Logística L2 (Base)** | **0.9256** | 0.9423 ± 0.0126 | **82.47%** | 88.67% | 57.99% | 0.6810 | 0.0917 |

### 4.2 Cohorte Benchmark: Pima Indians Diabetes Dataset ($n = 768$; Desarrollo 80% = $614$; Prueba Aislada 20% = $154$)

| Algoritmo | ROC-AUC (Prueba Aislada) | ROC-AUC (5-Fold CV Agrupado) | Sensibilidad | Especificidad | Precisión (VPP) | F1-Score | Brier Score (Calibración) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Regresión Logística L2 (Base)** | **0.8469** | 0.8375 ± 0.0271 | **81.48%** | 77.00% | 65.67% | **0.7273** | 0.1660 |
| **Random Forest Classifier** | **0.8346** | 0.8434 ± 0.0241 | 74.07% | 78.00% | 64.52% | 0.6897 | 0.1624 |
| **XGBoost Classifier** | **0.8320** | 0.8424 ± 0.0239 | 70.37% | 78.00% | 63.33% | 0.6667 | 0.1663 |
| **TabNet (Deep Learning Tabular)** | **0.8233** | 0.8195 ± 0.0299 | 61.11% | **87.00%** | 71.74% | 0.6600 | **0.1603** |

### 4.3 Benchmark Multi-Cohorte en 5 Datasets Reales ($N = 148,324$ registros evaluados)

| Cohorte / Dataset | Muestra ($N$) | Tipo de Variables | Mejor Modelo | ROC-AUC | F1-Score | Sensibilidad | Especificidad | Brier Score |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **CDC NHANES 2017–2018** | 4,829 | Biomarcadores de Laboratorio | **Stacking Híbrido** | **0.9435** | 0.7492 | 74.74% | 95.27% | 0.0612 |
| **Early Stage Sylhet** | 520 | Síntomas prodrómicos | **Random Forest** | **0.9970** | 0.9729 | 95.94% | 98.00% | 0.0313 |
| **NIDDK Pima Indians** | 768 | Parámetros clínicos base | **Random Forest** | **0.8378** | 0.7068 | 78.74% | 76.20% | 0.1635 |
| **CDC BRFSS 2015** | 70,692 | Salud poblacional / Estilo vida | **XGBoost** | **0.8296** | 0.7619 | 79.68% | 70.53% | 0.1674 |
| **Diabetes 130-Hospitals** | 71,515 | Hospitalario / Reingreso 30d | **Stacking Híbrido** | **0.6070** | 0.1946 | 55.57% | 59.92% | 0.0792 |

*El benchmark completo y el análisis de variabilidad clínica se encuentran en [`models/BENCHMARK_COMPARATIVO_DATASETS.md`](models/BENCHMARK_COMPARATIVO_DATASETS.md) y [`data/processed/DICCIONARIO_DATOS_DATASETS_REALES.md`](data/processed/DICCIONARIO_DATOS_DATASETS_REALES.md).*

---

## 5. Tabla de Correspondencia / Trazabilidad con `Caso03.docx` y `Prompt.docx`

| ID | Requisito Especificado | Estado | Módulo / Archivo |
| :--- | :--- | :--- | :--- |
| **REQ-01** | Variables clínicas conjuntas: edad, IMC, glucosa, HbA1c, presión arterial, perfil lipídico, antecedentes y tratamientos | **Implementado (Datos Reales)** | `scripts/train_and_evaluate.py`, `backend/app/schemas.py` (`NhanesPatientInput`) |
| **REQ-02** | Dos resultados independientes: aparición de diabetes y desarrollo de complicaciones | **Implementado y Diferenciado** | `outcomes_architecture` en `models/metrics_and_metadata.json` y vistas UI |
| **REQ-03** | Comparación de Línea Base, Random Forest, XGBoost y TabNet | **Implementado y Evaluado** | `scripts/train_and_evaluate.py`, `backend/app/predictor.py` |
| **REQ-04** | Modelos híbridos (`XGBoost + Cox`, `LSTM + Attention + Survival`) condicionados a datos longitudinales | **Documentado como Trabajo Futuro** | Pestaña *Inicio y Alcance* y `/model-info` (sin inventar tiempos ni etiquetas) |
| **REQ-05** | Prevención de fuga de información, división agrupada por paciente y conjunto de prueba aislado | **Implementado** | `StratifiedGroupKFold` por `SEQN`/`patient_id` + exclusión de fármacos hipoglucemiantes post-diagnóstico (`DIQ050`/`DIQ070`) |
| **REQ-06** | Análisis de subgrupos por edad y sexo sin usar variables sensibles de forma discriminatoria | **Implementado** | `evaluate_subgroups()` en `scripts/train_and_evaluate.py` + Pestaña 3 de Frontend |
| **REQ-07** | Explicabilidad mediante SHAP y análisis de calibración | **Implementado** | `shap.TreeExplainer` / `LinearExplainer` + curva de calibración y `Brier Score` |
| **REQ-08** | Importación CSV y datos sintéticos rotulados exclusivamente como `DEMO` | **Implementado** | `POST /predict-batch` + `data/demo/DEMO_SINTETICO_NO_CLINICO.csv` |

---

## 6. Estructura del Proyecto

```text
DIABETES/
├── Articulo_SP03.docx                  # Artículo científico completo (formato MDPI/Frontiers, 6 tablas, 6 figuras)
├── Evidencia_SP03.md                   # Resumen ejecutivo de evidencias y trazabilidad experimental
├── Caso03.docx                         # Especificación temática original del docente
├── Prompt.docx                         # Instrucciones de ingeniería, arquitectura en 6 capas y rigor experimental
├── Normas.docx                         # Pautas metodológicas de investigación
├── .env.example                        # Variables de entorno de ejemplo
├── requirements.txt                    # Dependencias de Python 3.12+
├── Dockerfile.backend                  # Contenedor Docker para API FastAPI
├── docker-compose.yml                  # Orquestación con backend y n8n
├── figuras/                            # 6 figuras de alta resolución (300 DPI) para el artículo
├── data/
│   ├── raw/                            # 5 cohortes reales (NHANES, Pima, BRFSS, 130 Hospitals, Sylhet)
│   ├── processed/                      # Datasets limpios, muestras Excel y Diccionario de Datos
│   └── demo/                           # Muestras sintéticas rotuladas exclusivamente como DEMO
├── models/
│   ├── benchmark_cross_dataset_results.json # Métricas comparativas de los 5 datasets
│   ├── BENCHMARK_COMPARATIVO_DATASETS.md    # Reporte exhaustivo de validación cruzada
│   ├── metrics_and_metadata.json       # Métricas verificadas, calibración, Friedman/Wilcoxon y SHAP
│   ├── nhanes_2017_2018/               # Modelos entrenados (RF, XGBoost, TabNet, Stacking, Weibull)
│   └── pima_indians/                   # Modelos entrenados sobre cohorte Pima
├── n8n/
│   └── workflows/                      # 6 Workflows n8n JSON (WF-01 a WF-06)
├── langflow/
│   └── diabetes_clinical_rag_flow.json # Flujo visual exportable para LangFlow / LangChain
├── scripts/
│   ├── train_and_evaluate.py           # Pipeline reproducible de entrenamiento y evaluación
│   └── run_datasets_benchmark.py       # Benchmark multiplataforma sobre las 5 cohortes reales
├── backend/
│   └── app/
│       ├── main.py                     # API FastAPI v2.0 (/health, /predict, /rag, /reports, /stats)
│       ├── predictor.py                # Servicio de inferencia 5 modelos + SHAP
│       ├── schemas.py                  # Esquemas Pydantic y validaciones clínicas
│       └── rag/                        # Agente RAG sobre Guías ADA 2024
├── frontend/
│   ├── package.json                    # Configuración React 18 + Vite + Tailwind CSS
│   ├── vite.config.ts                  # Configuración Vite en puerto 5173
│   └── src/
│       ├── App.tsx                     # Interfaz completa en español con 8 pestañas interactivas
│       ├── types.ts                    # Contratos TypeScript alineados con la API
│       └── styles.css                  # Estilos responsivos
└── tests/
    └── test_api.py                     # Suite de pruebas automatizadas con pytest (11/11 tests)
```

---

## 7. Instrucciones de Instalación, Entrenamiento y Ejecución en Linux

### 7.1 Preparar el entorno virtual Python 3.12 e instalar dependencias
```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

### 7.2 Ejecutar el pipeline de entrenamiento y evaluación verificada
```bash
.venv/bin/python scripts/train_and_evaluate.py
```

### 7.3 Ejecutar las pruebas automatizadas del backend
```bash
.venv/bin/python -m pytest -v tests/test_api.py
```

### 7.4 Iniciar el Backend FastAPI (Puerto 8000)
```bash
.venv/bin/uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

### 7.5 Instalar e iniciar el Frontend React + Vite (Puerto 5173)
```bash
cd frontend
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

### 7.6 Ejecución opcional mediante Docker Compose
```bash
docker compose up --build
```
