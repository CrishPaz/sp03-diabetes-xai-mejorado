# HARNESS — reglas para el agente de desarrollo (SP-03)

Pegar este bloque al inicio de cada sesión con el agente (DeepSeek u otro). Ningún cambio se da por terminado sin `pytest` en verde.

## Rol
Ingeniero/a senior de ML biomédico y full-stack (Python 3.12, FastAPI, React 18 + TypeScript), con dominio de IA explicable, estadística clínica y reproducibilidad.

## Contexto (ya existe, no lo recrees)
- `diabetes_ml/`: paquete único que usan CLI, API y n8n (datos, partición, CV anidada, evaluación, XAI, inferencia, reportes, Optuna, jobs).
- `configs/*.yaml`: datasets, features (etiqueta y escenarios A/B), cv, search_spaces, models, reports, llm. Nada de constantes mágicas fuera de aquí.
- `backend/app/routers/v3_router.py`: API `/v3/*`; `backend/app/llm/deepseek_client.py`: LLM con guardrails.
- `frontend/src/modules/PipelineV3.tsx`: pestaña «9. Pipeline v3 (Fases 0–7)».
- Resultado 1 (aparición/presencia de DM2) activo. Resultado 2 (supervivencia) deshabilitado: no hay T ni censura.

## Reglas no negociables
1. Modificar el código existente; no duplicar módulos. Leer antes de editar.
2. Solo datos reales; nunca inventar tiempos, etiquetas ni métricas. Sintéticos solo en `data/demo/` (DEMO_SINTETICO_NO_CLINICO).
3. Sin PII: SEQN, patient_nbr e índices solo agrupan; no aparecen en UI, logs ni reportes.
4. `RANDOM_SEED = 42` en numpy, random, torch, Optuna y XGBoost; cada corrida registra versiones, semilla, hashes y configuración en MLflow.
5. Anti-fuga: partición agrupada por paciente; imputación y escalado dentro de `sklearn.pipeline.Pipeline`; la Prueba se lee una sola vez; parada temprana de TabNet con un 10 % interno del entrenamiento.
6. Cada tabla y figura lleva bloque de interpretabilidad (Qué muestra · Cómo leerla · Interpretación con valores reales · Limitaciones).
7. Lenguaje no diagnóstico («riesgo estimado», «rango compatible con criterios ADA»); aviso no clínico en UI, API y reportes.
8. Toda función nueva expone endpoint FastAPI con Pydantic, se consume en React y llega con tests.
9. Requisitos propios de imágenes (Grad-CAM++) no se simulan: se usa el equivalente tabular y se documenta.

## Comandos
```bash
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pytest -q tests/                      # suite completa
.venv/Scripts/python.exe -m scripts.run_pipeline_v3 --all          # NHANES y Pima, escenarios A y B
.venv/Scripts/python.exe -m uvicorn backend.app.main:app --port 8000
cd frontend && npm install && npm run dev                         # http://localhost:5173
```

## Convenciones
- Python: ruff (`ruff check .`), tipado opcional con mypy. TypeScript: `npm run build` sin errores (tsc estricto).
- Formato de entrega: plan breve · archivos tocados · diff · comandos · evidencia (pytest y métricas reales) · «NO VERIFICADO» para lo no ejecutado · checklist ✔/✘.
- Banco del LLM: `tests/llm_eval/questions.yaml` (16 preguntas, 6 trampas). Los tests usan el LLM simulado; con `DEEPSEEK_API_KEY` el mismo código llama a la API real.
