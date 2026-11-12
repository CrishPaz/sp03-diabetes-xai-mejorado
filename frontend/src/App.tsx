import React, { useEffect, useMemo, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  BarChart3,
  BookOpen,
  Bot,
  CheckCircle2,
  Database,
  Download,
  ExternalLink,
  FileSpreadsheet,
  FileText,
  GitBranch,
  HeartPulse,
  Info,
  Layers,
  MessageSquare,
  Play,
  RefreshCw,
  ShieldCheck,
  Sliders,
  Sparkles,
  Upload,
} from 'lucide-react';
import {
  BatchPredictResponse,
  CohortData,
  DatasetId,
  FeatureMeta,
  ModelId,
  ModelInfoResponse,
  N8nAlert,
  N8nWorkflow,
  PredictResponse,
  RagQueryResponse,
  StatisticalEvaluation,
} from './types';
import PipelineV3 from './modules/PipelineV3';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

type TabKey =
  | 'inicio'
  | 'prediccion'
  | 'asistente_rag'
  | 'evaluacion_estadistica'
  | 'orquestacion_n8n'
  | 'reportes'
  | 'lotes'
  | 'metodologia'
  | 'pipeline_v3';

const NHANES_PRESETS: Record<string, { label: string; values: Record<string, number> }> = {
  bajo_riesgo: {
    label: 'Perfil Metabólico Normoglucémico',
    values: {
      edad: 34,
      imc: 22.8,
      glucosa: 88,
      hba1c: 5.1,
      presion_sistolica: 114,
      presion_diastolica: 72,
      colesterol_total: 174,
      colesterol_hdl: 58,
      trigliceridos: 92,
      antecedente_familiar_diabetes: 0,
      antecedente_hipertension: 0,
      tratamiento_antihipertensivo: 0,
    },
  },
  sindrome_metabolico: {
    label: 'Riesgo Intermedio / Prediabetes',
    values: {
      edad: 52,
      imc: 30.8,
      glucosa: 114,
      hba1c: 6.1,
      presion_sistolica: 136,
      presion_diastolica: 84,
      colesterol_total: 215,
      colesterol_hdl: 42,
      trigliceridos: 188,
      antecedente_familiar_diabetes: 1,
      antecedente_hipertension: 1,
      tratamiento_antihipertensivo: 0,
    },
  },
  alto_riesgo: {
    label: 'Alto Riesgo Cardiometabólico',
    values: {
      edad: 63,
      imc: 35.2,
      glucosa: 162,
      hba1c: 7.2,
      presion_sistolica: 148,
      presion_diastolica: 88,
      colesterol_total: 232,
      colesterol_hdl: 36,
      trigliceridos: 255,
      antecedente_familiar_diabetes: 1,
      antecedente_hipertension: 1,
      tratamiento_antihipertensivo: 1,
    },
  },
};

const PIMA_PRESETS: Record<string, { label: string; values: Record<string, number> }> = {
  bajo_riesgo: {
    label: 'Perfil Normoglucémico (Pima)',
    values: {
      embarazos: 1,
      glucosa: 95,
      presion_diastolica: 68,
      pliegue_cutaneo: 22,
      insulina: 85,
      imc: 24.2,
      pedigri_diabetes: 0.25,
      edad: 25,
    },
  },
  moderado: {
    label: 'Intolerancia Oral a la Glucosa',
    values: {
      embarazos: 3,
      glucosa: 138,
      presion_diastolica: 78,
      pliegue_cutaneo: 32,
      insulina: 145,
      imc: 32.5,
      pedigri_diabetes: 0.58,
      edad: 37,
    },
  },
  alto_riesgo: {
    label: 'Alto Riesgo Hiperglucémico',
    values: {
      embarazos: 6,
      glucosa: 176,
      presion_diastolica: 86,
      pliegue_cutaneo: 39,
      insulina: 220,
      imc: 37.8,
      pedigri_diabetes: 1.12,
      edad: 49,
    },
  },
};

export function App() {
  const [activeTab, setActiveTab] = useState<TabKey>('inicio');
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [loadingInfo, setLoadingInfo] = useState<boolean>(true);
  const [infoError, setInfoError] = useState<string | null>(null);

  const [datasetId, setDatasetId] = useState<DatasetId>('nhanes_2017_2018');
  const [modelId, setModelId] = useState<ModelId>('stacking_ensemble');

  const [formValues, setFormValues] = useState<Record<string, number>>(
    NHANES_PRESETS.sindrome_metabolico.values
  );
  const [predicting, setPredicting] = useState<boolean>(false);
  const [predictError, setPredictError] = useState<string | null>(null);
  const [predictionResult, setPredictionResult] = useState<PredictResponse | null>(null);

  // Estados para RAG
  const [ragQuery, setRagQuery] = useState<string>('¿Cuáles son los criterios de diagnóstico de diabetes según ADA 2024?');
  const [ragLoading, setRagLoading] = useState<boolean>(false);
  const [ragResult, setRagResult] = useState<RagQueryResponse | null>(null);
  const [ragError, setRagError] = useState<string | null>(null);

  // Estados para n8n
  const [n8nWorkflows, setN8nWorkflows] = useState<N8nWorkflow[]>([]);
  const [n8nAlerts, setN8nAlerts] = useState<N8nAlert[]>([]);
  const [n8nLoading, setN8nLoading] = useState<boolean>(false);
  const [triggeringWf, setTriggeringWf] = useState<string | null>(null);
  const [n8nMessage, setN8nMessage] = useState<string | null>(null);

  // Estados para Estadística Formal
  const [statData, setStatData] = useState<StatisticalEvaluation | null>(null);
  const [statLoading, setStatLoading] = useState<boolean>(false);
  const [crossBenchmarks, setCrossBenchmarks] = useState<Record<string, any> | null>(null);
  const [selectedBenchmarkCohort, setSelectedBenchmarkCohort] = useState<string>('ALL');

  // Estados para Reportes
  const [reportLoading, setReportLoading] = useState<string | null>(null);
  const [patientName, setPatientName] = useState<string>('Paciente Clínico de Referencia');
  const [patientIdInput, setPatientIdInput] = useState<string>('PX-2026-001');

  // Estados para Lotes CSV
  const [batchFile, setBatchFile] = useState<File | null>(null);
  const [batchLoading, setBatchLoading] = useState<boolean>(false);
  const [batchError, setBatchError] = useState<string | null>(null);
  const [batchResult, setBatchResult] = useState<BatchPredictResponse | null>(null);

  useEffect(() => {
    async function fetchModelInfo() {
      setLoadingInfo(true);
      setInfoError(null);
      try {
        const res = await fetch(`${API_BASE}/model-info`);
        if (!res.ok) {
          throw new Error(`Error HTTP ${res.status} al consultar /model-info`);
        }
        const data: ModelInfoResponse = await res.json();
        setModelInfo(data);
      } catch (err: any) {
        setInfoError(err.message || 'No se pudo conectar con el servidor FastAPI.');
      } finally {
        setLoadingInfo(false);
      }
    }
    fetchModelInfo();
  }, []);

  // Carga inicial de n8n y estadísticas
  useEffect(() => {
    if (activeTab === 'orquestacion_n8n') {
      loadN8nData();
    } else if (activeTab === 'evaluacion_estadistica') {
      loadStatisticalData();
    }
  }, [activeTab]);

  const loadN8nData = async () => {
    setN8nLoading(true);
    try {
      const [wfRes, alertsRes] = await Promise.all([
        fetch(`${API_BASE}/n8n/workflows`),
        fetch(`${API_BASE}/n8n/alerts`),
      ]);
      if (wfRes.ok) {
        const data = await wfRes.json();
        setN8nWorkflows(data.workflows || []);
      }
      if (alertsRes.ok) {
        const data = await alertsRes.json();
        setN8nAlerts(data.alerts || []);
      }
    } catch (err) {
      console.error('Error cargando n8n:', err);
    } finally {
      setN8nLoading(false);
    }
  };

  const loadStatisticalData = async () => {
    setStatLoading(true);
    try {
      const [statRes, benchRes] = await Promise.all([
        fetch(`${API_BASE}/statistics/evaluation`),
        fetch(`${API_BASE}/datasets/comparative-benchmark`),
      ]);
      if (statRes.ok) {
        const data = await statRes.json();
        setStatData(data);
      }
      if (benchRes.ok) {
        const dataBench = await benchRes.json();
        setCrossBenchmarks(dataBench.benchmarks || null);
      }
    } catch (err) {
      console.error('Error cargando estadísticas y benchmark:', err);
    } finally {
      setStatLoading(false);
    }
  };

  const triggerWorkflow = async (workflowId: string) => {
    setTriggeringWf(workflowId);
    setN8nMessage(null);
    try {
      const res = await fetch(`${API_BASE}/n8n/workflows/${workflowId}/trigger`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          patient_id: patientIdInput,
          riesgo_pct: predictionResult ? predictionResult.percentage : 82.5,
        }),
      });
      const data = await res.json();
      setN8nMessage(`Workflow ${workflowId} ejecutado con éxito. ID: ${data.execution_id}`);
      await loadN8nData();
    } catch (err: any) {
      setN8nMessage(`Error ejecutando workflow ${workflowId}: ${err.message}`);
    } finally {
      setTriggeringWf(null);
    }
  };

  const currentCohort: CohortData | null = useMemo(() => {
    if (!modelInfo) return null;
    return modelInfo.cohorts[datasetId];
  }, [modelInfo, datasetId]);

  const handleDatasetChange = (newDataset: DatasetId) => {
    setDatasetId(newDataset);
    setPredictError(null);
    setPredictionResult(null);
    if (newDataset === 'nhanes_2017_2018') {
      setFormValues(NHANES_PRESETS.sindrome_metabolico.values);
    } else {
      setFormValues(PIMA_PRESETS.moderado.values);
    }
  };

  const handleFieldChange = (fmeta: FeatureMeta, rawVal: string) => {
    const parsed = parseFloat(rawVal);
    setFormValues((prev) => ({
      ...prev,
      [fmeta.name]: Number.isNaN(parsed) ? 0 : parsed,
    }));
  };

  const runPrediction = async (customValues?: Record<string, number>) => {
    if (!currentCohort) return;
    setPredicting(true);
    setPredictError(null);
    const valuesToSend = customValues || formValues;

    // Validación clínica en cliente
    for (const fmeta of currentCohort.features) {
      const val = valuesToSend[fmeta.name];
      if (val === undefined || Number.isNaN(val)) {
        setPredictError(`El parámetro "${fmeta.label}" es obligatorio.`);
        setPredicting(false);
        return;
      }
      if (val < fmeta.min || val > fmeta.max) {
        setPredictError(
          `"${fmeta.label}" está fuera del rango fisiológico permitido (${fmeta.min} - ${fmeta.max} ${fmeta.unit}).`
        );
        setPredicting(false);
        return;
      }
    }

    if (datasetId === 'nhanes_2017_2018') {
      if (valuesToSend.presion_diastolica >= valuesToSend.presion_sistolica) {
        setPredictError('Incoherencia fisiológica: La presión diastólica no puede ser mayor o igual que la sistólica.');
        setPredicting(false);
        return;
      }
      if (valuesToSend.colesterol_hdl >= valuesToSend.colesterol_total) {
        setPredictError('Incoherencia fisiológica: El colesterol HDL no puede superar al colesterol total.');
        setPredicting(false);
        return;
      }
    }

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: datasetId,
          model_id: modelId,
          features: valuesToSend,
          patient_id: patientIdInput,
        }),
      });
      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || `Error HTTP ${res.status}`);
      }
      const data: PredictResponse = await res.json();
      setPredictionResult(data);
    } catch (err: any) {
      setPredictError(err.message || 'Error en la inferencia del modelo.');
    } finally {
      setPredicting(false);
    }
  };

  const executeRagQuery = async (customQuestion?: string) => {
    const q = customQuestion || ragQuery;
    if (!q.trim()) return;
    setRagLoading(true);
    setRagError(null);
    try {
      const res = await fetch(`${API_BASE}/rag/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: q,
          patient_context: formValues,
          max_sources: 3,
        }),
      });
      if (!res.ok) {
        throw new Error(`Error HTTP ${res.status}`);
      }
      const data: RagQueryResponse = await res.json();
      setRagResult(data);
    } catch (err: any) {
      setRagError(err.message || 'Error al consultar al agente RAG.');
    } finally {
      setRagLoading(false);
    }
  };

  const downloadReport = async (format: 'pdf' | 'docx' | 'xlsx') => {
    setReportLoading(format);
    try {
      const res = await fetch(`${API_BASE}/reports/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          format,
          patient_name: patientName,
          patient_id: patientIdInput,
          model_id: modelId,
          features: formValues,
          prediction_result: predictionResult,
        }),
      });
      if (!res.ok) {
        throw new Error(`Error HTTP ${res.status} al generar reporte`);
      }
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `reporte_diabetes_${patientIdInput}.${format}`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err: any) {
      alert(`No se pudo descargar el reporte: ${err.message}`);
    } finally {
      setReportLoading(null);
    }
  };

  const handleBatchSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!batchFile) {
      setBatchError('Seleccione un archivo CSV para procesar.');
      return;
    }
    setBatchLoading(true);
    setBatchError(null);
    setBatchResult(null);

    const formData = new FormData();
    formData.append('file', batchFile);
    formData.append('dataset_id', datasetId);
    formData.append('model_id', modelId);

    try {
      const res = await fetch(`${API_BASE}/predict-batch`, {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || `Error HTTP ${res.status}`);
      }
      const data: BatchPredictResponse = await res.json();
      setBatchResult(data);
    } catch (err: any) {
      setBatchError(err.message || 'Error al procesar el archivo CSV en lote.');
    } finally {
      setBatchLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Barra de Navegación Superior */}
      <header className="border-b border-slate-800 bg-slate-900/90 backdrop-blur sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-sky-600 via-indigo-600 to-emerald-500 p-0.5 shadow-lg shadow-sky-500/20">
              <div className="h-full w-full bg-slate-950 rounded-[10px] flex items-center justify-center">
                <HeartPulse className="h-5 w-5 text-sky-400" />
              </div>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-base tracking-tight text-white">SP-03 | XAI Diabetes</span>
                <span className="text-[10px] font-semibold tracking-wider uppercase px-2 py-0.5 rounded-full bg-sky-500/10 text-sky-400 border border-sky-500/20">
                  Arquitectura 2.0 (5 Modelos + n8n + RAG)
                </span>
              </div>
              <p className="text-xs text-slate-400 hidden sm:block">
                CDC NHANES & NIDDK Pima · Optuna · SHAP · Batería Friedman/Wilcoxon · Reportes
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {loadingInfo ? (
              <span className="text-xs text-slate-400 flex items-center gap-1.5">
                <RefreshCw className="h-3 w-3 animate-spin text-sky-400" /> Conectando...
              </span>
            ) : infoError ? (
              <span className="text-xs text-rose-400 flex items-center gap-1 bg-rose-950/60 border border-rose-800/60 px-2 py-1 rounded">
                <AlertTriangle className="h-3 w-3" /> Backend desconectado
              </span>
            ) : (
              <span className="text-xs text-emerald-400 flex items-center gap-1 bg-emerald-950/60 border border-emerald-800/60 px-2.5 py-1 rounded-full">
                <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse"></span>
                FastAPI Activo (v2.0)
              </span>
            )}
          </div>
        </div>

        {/* Pestañas de Navegación */}
        <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex space-x-1 overflow-x-auto scrollbar-none py-1.5 border-t border-slate-800/60">
          {[
            { key: 'inicio', label: '1. Arquitectura & Flujo', icon: Layers },
            { key: 'prediccion', label: '2. Calculadora 5 Modelos & SHAP', icon: Sliders },
            { key: 'asistente_rag', label: '3. Asistente RAG Clínico (ADA/OMS)', icon: Bot },
            { key: 'evaluacion_estadistica', label: '4. Batería Estadística (Friedman/Wilcoxon)', icon: BarChart3 },
            { key: 'orquestacion_n8n', label: '5. Orquestador n8n (WF-01 a 06)', icon: GitBranch },
            { key: 'reportes', label: '6. Reportes PDF / Word / Excel', icon: FileText },
            { key: 'lotes', label: '7. Lotes CSV & DEMO', icon: FileSpreadsheet },
            { key: 'metodologia', label: '8. Metodología F1-F12 & Trazabilidad', icon: BookOpen },
            { key: 'pipeline_v3', label: '9. Pipeline v3 (Fases 0–7)', icon: ShieldCheck },
          ].map((tab) => {
            const Icon = tab.icon;
            const isSelected = activeTab === tab.key;
            return (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key as TabKey)}
                className={`flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-medium whitespace-nowrap transition-colors ${
                  isSelected
                    ? 'bg-sky-500/15 text-sky-300 border border-sky-500/30'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                }`}
              >
                <Icon className={`h-4 w-4 ${isSelected ? 'text-sky-400' : 'text-slate-500'}`} />
                {tab.label}
              </button>
            );
          })}
        </nav>
      </header>

      {/* Contenedor Principal */}
      <main className="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 w-full">
        {/* ==================================================== */}
        {/* PESTAÑA 1: INICIO Y ARQUITECTURA GENERAL            */}
        {/* ==================================================== */}
        {activeTab === 'inicio' && (
          <div className="space-y-6">
            <div className="rounded-2xl border border-sky-500/20 bg-gradient-to-br from-slate-900 via-slate-900 to-sky-950/40 p-6 sm:p-8 shadow-xl">
              <div className="max-w-3xl space-y-3">
                <span className="text-xs font-semibold uppercase tracking-wider text-sky-400">
                  Protocolo Común de Investigación y Aplicación Inteligente
                </span>
                <h1 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">
                  Predicción Temprana de Diabetes Mellitus Tipo 2 y Riesgo de Complicaciones con IA Explicable
                </h1>
                <p className="text-sm text-slate-300 leading-relaxed">
                  Sistema integral que implementa la arquitectura recomendada por el docente: 
                  <strong className="text-white"> 5 Modelos (3 individuales + 2 híbridos)</strong>, 
                  optimización con <strong className="text-sky-300">Optuna</strong>, explicabilidad con <strong className="text-sky-300">SHAP</strong>, 
                  orquestación automatizada de 6 workflows con <strong className="text-sky-300">n8n</strong>, 
                  asistente de evidencia médica <strong className="text-sky-300">RAG (LangChain / LangFlow)</strong> y 
                  generación de reportes científicos en <strong className="text-sky-300">PDF, Word y Excel</strong>.
                </p>
                <div className="flex flex-wrap gap-3 pt-2">
                  <button
                    onClick={() => setActiveTab('prediccion')}
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-sky-500 text-slate-950 font-semibold text-xs hover:bg-sky-400 transition"
                  >
                    <Sliders className="h-4 w-4" /> Probar Predicción 5 Modelos
                  </button>
                  <button
                    onClick={() => setActiveTab('asistente_rag')}
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-800 text-slate-200 font-semibold text-xs hover:bg-slate-700 border border-slate-700 transition"
                  >
                    <Bot className="h-4 w-4 text-sky-400" /> Consultar Asistente RAG ADA 2024
                  </button>
                  <button
                    onClick={() => setActiveTab('orquestacion_n8n')}
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-800 text-slate-200 font-semibold text-xs hover:bg-slate-700 border border-slate-700 transition"
                  >
                    <GitBranch className="h-4 w-4 text-emerald-400" /> Ver 6 Workflows n8n
                  </button>
                </div>
              </div>
            </div>

            {/* Diagrama de Arquitectura de 6 Capas */}
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-6 space-y-4">
              <h2 className="text-base font-semibold text-white flex items-center gap-2">
                <Layers className="h-5 w-5 text-sky-400" /> Arquitectura Tecnológica Recomendada en 6 Capas
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 text-xs">
                <div className="p-4 rounded-lg bg-slate-950/70 border border-slate-800 space-y-2">
                  <div className="flex items-center justify-between text-sky-400 font-bold">
                    <span>Capa 1: Adquisición de Datos</span>
                    <Database className="h-4 w-4" />
                  </div>
                  <p className="text-slate-300">
                    Cohortes clínicas reales de <strong>CDC NHANES 2017–2018</strong> (4,829 adultos, 12 biomarcadores) y 
                    <strong> Pima Indians NIDDK</strong> (768 pacientes).
                  </p>
                </div>

                <div className="p-4 rounded-lg bg-slate-950/70 border border-slate-800 space-y-2">
                  <div className="flex items-center justify-between text-emerald-400 font-bold">
                    <span>Capa 2: Orquestación n8n</span>
                    <GitBranch className="h-4 w-4" />
                  </div>
                  <p className="text-slate-300">
                    6 workflows automatizados: Ingestión (WF-01), Calidad e Imputación (WF-02), Inferencia programada (WF-03), 
                    Alertas clínicas (WF-04), Reportes (WF-05) y Auditoría MLflow (WF-06).
                  </p>
                </div>

                <div className="p-4 rounded-lg bg-slate-950/70 border border-slate-800 space-y-2">
                  <div className="flex items-center justify-between text-indigo-400 font-bold">
                    <span>Capa 3: LangChain & LangFlow</span>
                    <Bot className="h-4 w-4" />
                  </div>
                  <p className="text-slate-300">
                    Agente RAG sobre directrices <strong>ADA Standards of Care 2024, OMS y CDC</strong>. Genera explicaciones clínicas y 
                    cuenta con flujo exportable para LangFlow.
                  </p>
                </div>

                <div className="p-4 rounded-lg bg-slate-950/70 border border-slate-800 space-y-2">
                  <div className="flex items-center justify-between text-amber-400 font-bold">
                    <span>Capa 4: Motor de IA (5 Modelos)</span>
                    <Sliders className="h-4 w-4" />
                  </div>
                  <p className="text-slate-300">
                    3 individuales (Random Forest, XGBoost, TabNet) + 2 híbridos (Stacking Calibrado y Híbrido Supervivencia Weibull a 5/10 años) + Optuna + SHAP.
                  </p>
                </div>

                <div className="p-4 rounded-lg bg-slate-950/70 border border-slate-800 space-y-2">
                  <div className="flex items-center justify-between text-purple-400 font-bold">
                    <span>Capa 5: Backend & Persistencia</span>
                    <Activity className="h-4 w-4" />
                  </div>
                  <p className="text-slate-300">
                    FastAPI con validación Pydantic estricta, SQLAlchemy (SQLite local / PostgreSQL en producción) y registro de auditoría.
                  </p>
                </div>

                <div className="p-4 rounded-lg bg-slate-950/70 border border-slate-800 space-y-2">
                  <div className="flex items-center justify-between text-rose-400 font-bold">
                    <span>Capa 6: Frontend & Reportes</span>
                    <FileText className="h-4 w-4" />
                  </div>
                  <p className="text-slate-300">
                    Dashboard React con Tailwind CSS, generador multiformato en PDF (ReportLab), Word (.docx) y Excel (.xlsx) con semáforos condicionales.
                  </p>
                </div>
              </div>
            </div>

            {/* Aviso de Integridad y Ética */}
            <div className="rounded-xl border border-amber-500/30 bg-amber-950/20 p-4 text-xs text-amber-300 flex items-start gap-3">
              <ShieldCheck className="h-5 w-5 text-amber-400 shrink-0 mt-0.5" />
              <div>
                <strong className="font-semibold text-amber-200">Aviso Ético, Regulatorio y Metodológico (Normas Q1):</strong>
                <p className="mt-1 text-amber-300/90 leading-relaxed">
                  Este sistema es un prototipo investigativo de inteligencia artificial explicable para soporte a la decisión clínica.
                  No sustituye el criterio médico ni las pruebas diagnósticas confirmatorias de laboratorio (glucemia en ayunas / HbA1c NGSP).
                  Los datos de CDC NHANES y Pima Indians están auditados con partición estratificada por clusters para garantizar cero fuga de información.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* ==================================================== */}
        {/* PESTAÑA 2: PREDICCIÓN CON LOS 5 MODELOS & SHAP       */}
        {/* ==================================================== */}
        {activeTab === 'prediccion' && currentCohort && (
          <div className="space-y-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-4">
              <div>
                <h2 className="text-lg font-bold text-white flex items-center gap-2">
                  <Sliders className="h-5 w-5 text-sky-400" /> Calculadora de Riesgo Multimodelo
                </h2>
                <p className="text-xs text-slate-400">
                  Seleccione cohorte, algoritmo predictivo y ajuste los parámetros clínicos del paciente.
                </p>
              </div>

              {/* Selector de Cohorte y Modelo */}
              <div className="flex flex-wrap items-center gap-2">
                <div className="flex items-center bg-slate-900 border border-slate-800 rounded-lg p-1 text-xs">
                  <button
                    onClick={() => handleDatasetChange('nhanes_2017_2018')}
                    className={`px-3 py-1 rounded-md font-medium transition ${
                      datasetId === 'nhanes_2017_2018'
                        ? 'bg-sky-500 text-slate-950 font-semibold'
                        : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    NHANES 2017-2018 (CDC)
                  </button>
                  <button
                    onClick={() => handleDatasetChange('pima_indians')}
                    className={`px-3 py-1 rounded-md font-medium transition ${
                      datasetId === 'pima_indians'
                        ? 'bg-sky-500 text-slate-950 font-semibold'
                        : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    Pima Indians (NIDDK)
                  </button>
                </div>

                <select
                  value={modelId}
                  onChange={(e) => setModelId(e.target.value as ModelId)}
                  className="bg-slate-900 border border-slate-800 text-slate-200 rounded-lg px-3 py-1.5 text-xs focus:ring-1 focus:ring-sky-500 outline-none"
                >
                  <optgroup label="Modelos Híbridos (Recomendados)">
                    <option value="stacking_ensemble">Stacking Calibrado (Meta-Clasificador)</option>
                    <option value="hybrid_survival" disabled>Supervivencia (deshabilitado: requiere T y censura)</option>
                  </optgroup>
                  <optgroup label="Modelos Individuales">
                    <option value="xgboost">XGBoost Classifier (Optuna Tuned)</option>
                    <option value="random_forest">Random Forest Classifier</option>
                    <option value="tabnet">TabNet (Attentive Tabular Network)</option>
                  </optgroup>
                </select>
              </div>
            </div>

            {/* Presets Rápidos */}
            <div className="flex items-center gap-2 text-xs flex-wrap">
              <span className="text-slate-400 font-medium">Perfiles clínicos de prueba:</span>
              {Object.entries(datasetId === 'nhanes_2017_2018' ? NHANES_PRESETS : PIMA_PRESETS).map(([k, p]) => (
                <button
                  key={k}
                  onClick={() => {
                    setFormValues(p.values);
                    runPrediction(p.values);
                  }}
                  className="px-2.5 py-1 rounded bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 hover:text-white transition"
                >
                  {p.label}
                </button>
              ))}
            </div>

            {predictError && (
              <div className="p-3.5 rounded-xl bg-rose-950/70 border border-rose-800/80 text-rose-300 text-xs flex items-center gap-2">
                <AlertTriangle className="h-4 w-4 shrink-0 text-rose-400" />
                <span>{predictError}</span>
              </div>
            )}

            {/* Formulario y Resultados en 2 Columnas */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              {/* Columna Izquierda: Formulario de Parámetros Clínicos */}
              <div className="lg:col-span-5 rounded-xl border border-slate-800 bg-slate-900/60 p-5 space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                    <Sliders className="h-4 w-4 text-sky-400" /> Parámetros del Paciente
                  </h3>
                  <input
                    type="text"
                    value={patientIdInput}
                    onChange={(e) => setPatientIdInput(e.target.value)}
                    placeholder="Código Paciente (PX-001)"
                    className="bg-slate-950 border border-slate-800 text-xs px-2.5 py-1 rounded text-slate-300 w-32 focus:outline-none focus:border-sky-500"
                  />
                </div>

                <div className="space-y-3 max-h-[580px] overflow-y-auto pr-2 scrollbar-thin">
                  {currentCohort.features.map((fmeta) => {
                    const currentVal = formValues[fmeta.name] ?? fmeta.default;
                    const isBinary = fmeta.min === 0 && fmeta.max === 1 && fmeta.step === 1;

                    return (
                      <div key={fmeta.name} className="p-2.5 rounded-lg bg-slate-950/60 border border-slate-800/80 space-y-1">
                        <div className="flex items-center justify-between text-xs">
                          <label className="font-medium text-slate-200">{fmeta.label}</label>
                          <span className="font-semibold text-sky-400">
                            {isBinary ? (currentVal ? 'Sí (1)' : 'No (0)') : `${currentVal} ${fmeta.unit}`}
                          </span>
                        </div>

                        {isBinary ? (
                          <div className="flex gap-2 pt-1">
                            <button
                              type="button"
                              onClick={() => handleFieldChange(fmeta, '0')}
                              className={`flex-1 py-1 rounded text-xs font-medium border transition ${
                                currentVal === 0
                                  ? 'bg-slate-800 border-sky-500 text-sky-300'
                                  : 'bg-slate-900 border-slate-800 text-slate-400'
                              }`}
                            >
                              No
                            </button>
                            <button
                              type="button"
                              onClick={() => handleFieldChange(fmeta, '1')}
                              className={`flex-1 py-1 rounded text-xs font-medium border transition ${
                                currentVal === 1
                                  ? 'bg-sky-500 text-slate-950 font-bold border-sky-400'
                                  : 'bg-slate-900 border-slate-800 text-slate-400'
                              }`}
                            >
                              Sí
                            </button>
                          </div>
                        ) : (
                          <div className="space-y-1">
                            <input
                              type="range"
                              min={fmeta.min}
                              max={fmeta.max}
                              step={fmeta.step}
                              value={currentVal}
                              onChange={(e) => handleFieldChange(fmeta, e.target.value)}
                              className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-sky-400"
                            />
                            <div className="flex justify-between text-[10px] text-slate-500">
                              <span>Min: {fmeta.min}</span>
                              <span className="text-slate-400 font-mono">Norma: {fmeta.normal_range}</span>
                              <span>Max: {fmeta.max}</span>
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>

                <button
                  type="button"
                  onClick={() => runPrediction()}
                  disabled={predicting}
                  className="w-full py-2.5 rounded-xl bg-gradient-to-r from-sky-500 to-indigo-600 text-slate-950 font-bold text-xs hover:from-sky-400 hover:to-indigo-500 transition shadow-lg shadow-sky-500/20 disabled:opacity-50 flex items-center justify-center gap-2"
                >
                  {predicting ? (
                    <>
                      <RefreshCw className="h-4 w-4 animate-spin" /> Inferencia en Proceso...
                    </>
                  ) : (
                    <>
                      <Play className="h-4 w-4 fill-slate-950" /> Calcular Riesgo y Explicación SHAP
                    </>
                  )}
                </button>
              </div>

              {/* Columna Derecha: Tarjeta de Riesgo, Comparación y SHAP */}
              <div className="lg:col-span-7 space-y-5">
                {predictionResult ? (
                  <>
                    {/* Semáforo de Riesgo */}
                    <div className="p-5 rounded-xl border border-slate-800 bg-slate-900/60 space-y-4">
                      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                        <div>
                          <span className="text-xs text-slate-400">Modelo Activo: {predictionResult.model_name}</span>
                          <h3 className="text-xl font-bold text-white">
                            Probabilidad Estimada: {(predictionResult.probability * 100).toFixed(1)}%
                          </h3>
                        </div>
                        <div
                          className={`px-3 py-1.5 rounded-lg text-xs font-bold uppercase tracking-wider text-center ${
                            predictionResult.risk_stratum === 'Bajo'
                              ? 'bg-emerald-950/80 text-emerald-400 border border-emerald-800'
                              : predictionResult.risk_stratum === 'Moderado'
                              ? 'bg-amber-950/80 text-amber-400 border border-amber-800'
                              : 'bg-rose-950/80 text-rose-400 border border-rose-800'
                          }`}
                        >
                          Estrato: {predictionResult.risk_stratum}
                        </div>
                      </div>

                      {/* Barra de Progreso de Riesgo */}
                      <div className="space-y-1">
                        <div className="w-full h-3 rounded-full bg-slate-950 overflow-hidden flex border border-slate-800">
                          <div
                            className={`h-full transition-all duration-500 ${
                              predictionResult.probability < 0.25
                                ? 'bg-emerald-500'
                                : predictionResult.probability < 0.5
                                ? 'bg-amber-500'
                                : 'bg-rose-500'
                            }`}
                            style={{ width: `${Math.min(100, Math.max(2, predictionResult.percentage))}%` }}
                          />
                        </div>
                        <div className="flex justify-between text-[10px] text-slate-500">
                          <span>0% (Riesgo Mínimo)</span>
                          <span>Umbral Clínico: 50%</span>
                          <span>100% (Alto Riesgo)</span>
                        </div>
                      </div>

                      <p className="text-xs text-slate-300 leading-relaxed bg-slate-950/60 p-3 rounded-lg border border-slate-800">
                        {predictionResult.risk_recommendation}
                      </p>

                      {/* Acciones Rápidas con el Paciente */}
                      <div className="flex flex-wrap items-center gap-2 pt-1 border-t border-slate-800/60">
                        <span className="text-xs text-slate-400">Acciones:</span>
                        <button
                          onClick={() => downloadReport('pdf')}
                          disabled={reportLoading !== null}
                          className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-xs text-slate-200 border border-slate-700 flex items-center gap-1.5 transition"
                        >
                          <Download className="h-3 w-3 text-sky-400" /> Reporte PDF
                        </button>
                        <button
                          onClick={() => downloadReport('docx')}
                          disabled={reportLoading !== null}
                          className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-xs text-slate-200 border border-slate-700 flex items-center gap-1.5 transition"
                        >
                          <Download className="h-3 w-3 text-indigo-400" /> Reporte Word
                        </button>
                        <button
                          onClick={() => {
                            setActiveTab('asistente_rag');
                            executeRagQuery(
                              `Paciente con riesgo ${predictionResult.risk_stratum} (${(
                                predictionResult.probability * 100
                              ).toFixed(1)}%). ¿Qué protocolo diagnóstico y terapéutico recomiendan las guías ADA 2024?`
                            );
                          }}
                          className="px-2.5 py-1 rounded bg-sky-500/10 hover:bg-sky-500/20 text-xs text-sky-400 border border-sky-500/30 flex items-center gap-1.5 transition"
                        >
                          <Bot className="h-3 w-3" /> Explicar con RAG ADA 2024
                        </button>
                      </div>
                    </div>

                    {/* Proyecciones de Supervivencia / Complicaciones (Resultado 2) */}
                    {predictionResult.survival_projections && (
                      <div className="p-4 rounded-xl border border-indigo-500/30 bg-indigo-950/15 space-y-3">
                        <div className="flex items-center justify-between">
                          <h4 className="text-xs font-semibold text-indigo-300 flex items-center gap-2">
                            <Activity className="h-4 w-4 text-indigo-400" /> 
                            Resultado 2: Proyección de Complicaciones a Largo Plazo (Modelo Weibull)
                          </h4>
                          <span className="text-[10px] bg-indigo-500/10 text-indigo-300 px-2 py-0.5 rounded border border-indigo-500/20">
                            Supervivencia Libre de Evento
                          </span>
                        </div>
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                          {predictionResult.survival_projections.map((sp) => (
                            <div key={sp.horizon_years} className="p-3 rounded-lg bg-slate-950/70 border border-indigo-900/40 text-xs space-y-1.5">
                              <div className="flex items-center justify-between text-slate-300">
                                <span className="font-bold text-white">Horizonte a {sp.horizon_years} Años</span>
                                <span className="text-indigo-400 font-mono">Riesgo: {sp.complication_risk_percentage}%</span>
                              </div>
                              <p className="text-[11px] text-slate-400 leading-tight">{sp.projected_event}</p>
                              <div className="text-[10px] text-slate-500">
                                Razón de Riesgo Estimada (HR): <span className="text-slate-300 font-bold">{sp.hazard_ratio_estimate}x</span>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Comparación Simultánea entre los 5 Modelos */}
                    <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 space-y-3">
                      <h4 className="text-xs font-semibold text-slate-200 flex items-center gap-2">
                        <Layers className="h-4 w-4 text-sky-400" /> Comparación Simultánea entre los 5 Algoritmos
                      </h4>
                      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2.5">
                        {predictionResult.all_models_comparison.map((m) => (
                          <div
                            key={m.model_id}
                            className={`p-2.5 rounded-lg border text-xs space-y-1 transition ${
                              m.model_id === modelId || m.model_name === predictionResult.model_name
                                ? 'bg-sky-950/40 border-sky-500/50'
                                : 'bg-slate-950/60 border-slate-800'
                            }`}
                          >
                            <div className="flex justify-between items-center text-slate-300">
                              <span className="font-medium truncate pr-1">{m.model_name}</span>
                              <span className="font-bold text-white">{m.percentage.toFixed(1)}%</span>
                            </div>
                            <div className="w-full bg-slate-850 h-1.5 rounded-full overflow-hidden">
                              <div
                                className="h-full bg-sky-400 rounded-full"
                                style={{ width: `${Math.min(100, Math.max(3, m.percentage))}%` }}
                              />
                            </div>
                            <div className="text-[10px] text-slate-400 text-right">Estrato: {m.risk_stratum}</div>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Descomposición Explicable SHAP */}
                    <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 space-y-3">
                      <div className="flex items-center justify-between">
                        <h4 className="text-xs font-semibold text-slate-200 flex items-center gap-2">
                          <BarChart3 className="h-4 w-4 text-emerald-400" /> Descomposición de Factores SHAP (TreeExplainer)
                        </h4>
                        <span className="text-[10px] text-slate-400 font-mono">
                          Valor Base E[f(x)]: {predictionResult.base_value.toFixed(3)}
                        </span>
                      </div>

                      <div className="space-y-2 max-h-72 overflow-y-auto pr-1 scrollbar-thin">
                        {predictionResult.shap_contributions.map((c) => {
                          const isRisk = c.direction === 'incrementa_riesgo';
                          const isProtective = c.direction === 'reduce_riesgo';
                          return (
                            <div key={c.feature} className="p-2.5 rounded-lg bg-slate-950/70 border border-slate-800 text-xs space-y-1">
                              <div className="flex items-center justify-between">
                                <span className="font-medium text-slate-200">
                                  {c.label} ({c.value} {c.unit})
                                </span>
                                <span
                                  className={`font-mono font-bold ${
                                    isRisk ? 'text-rose-400' : isProtective ? 'text-emerald-400' : 'text-slate-400'
                                  }`}
                                >
                                  {c.shap_value > 0 ? `+${c.shap_value.toFixed(4)}` : c.shap_value.toFixed(4)}
                                </span>
                              </div>
                              <p className="text-[11px] text-slate-400 leading-tight">{c.clinical_interpretation}</p>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="h-full min-h-[420px] rounded-xl border border-dashed border-slate-800 flex flex-col items-center justify-center p-8 text-center text-slate-500 space-y-3">
                    <Sliders className="h-10 w-10 text-slate-600" />
                    <div>
                      <p className="text-sm font-medium text-slate-300">Esperando ejecución del modelo</p>
                      <p className="text-xs text-slate-500 mt-1">
                        Ajuste las variables clínicas a la izquierda o seleccione un preset para calcular la predicción y el gráfico SHAP.
                      </p>
                    </div>
                    <button
                      onClick={() => runPrediction()}
                      className="px-4 py-2 rounded-xl bg-sky-500/20 text-sky-400 hover:bg-sky-500/30 text-xs font-semibold border border-sky-500/30 transition"
                    >
                      Calcular con Valores Actuales
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ==================================================== */}
        {/* PESTAÑA 3: ASISTENTE RAG CLÍNICO (LANGCHAIN/LANGFLOW)*/}
        {/* ==================================================== */}
        {activeTab === 'asistente_rag' && (
          <div className="space-y-6">
            <div className="rounded-xl border border-indigo-500/20 bg-indigo-950/20 p-5 space-y-2">
              <div className="flex items-center justify-between">
                <h2 className="text-base font-bold text-white flex items-center gap-2">
                  <Bot className="h-5 w-5 text-indigo-400" /> Asistente RAG Clínico & LangChain / LangFlow
                </h2>
                <a
                  href={`${API_BASE}/rag/langflow-schema`}
                  target="_blank"
                  rel="noreferrer"
                  className="px-3 py-1 rounded bg-indigo-600/30 hover:bg-indigo-600/50 text-indigo-300 border border-indigo-500/30 text-xs flex items-center gap-1.5 transition"
                >
                  <Download className="h-3 w-3" /> Descargar Flujo LangFlow (.json)
                </a>
              </div>
              <p className="text-xs text-slate-300 leading-relaxed">
                El LLM actúa estrictamente como <strong>interfaz de consulta y recuperación de evidencia científica</strong> 
                (ADA Standards of Care 2024, OMS, CDC), y nunca como sustituto del modelo predictivo validado.
              </p>
            </div>

            {/* Cuadro de Consulta RAG */}
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 space-y-4">
              <div className="flex gap-2">
                <input
                  type="text"
                  value={ragQuery}
                  onChange={(e) => setRagQuery(e.target.value)}
                  placeholder="Escriba su consulta médica o epidemiológica..."
                  className="flex-1 bg-slate-950 border border-slate-800 rounded-xl px-4 py-2.5 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:border-indigo-500"
                />
                <button
                  onClick={() => executeRagQuery()}
                  disabled={ragLoading}
                  className="px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs flex items-center gap-2 transition disabled:opacity-50"
                >
                  {ragLoading ? <RefreshCw className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
                  Consultar RAG
                </button>
              </div>

              {/* Sugerencias Rápidas de Consultas */}
              <div className="flex items-center gap-2 flex-wrap text-xs">
                <span className="text-slate-500">Consultas frecuentes:</span>
                {[
                  '¿Cuáles son los criterios de diagnóstico de diabetes según ADA 2024?',
                  '¿Qué metas de HbA1c y presión arterial recomienda la ADA?',
                  'Criterios de prediabetes y conducta preventiva',
                  'Vigilancia epidemiológica y tamizaje poblacional de diabetes',
                ].map((sug, idx) => (
                  <button
                    key={idx}
                    onClick={() => {
                      setRagQuery(sug);
                      executeRagQuery(sug);
                    }}
                    className="px-2.5 py-1 rounded bg-slate-950 hover:bg-slate-800 border border-slate-800 text-slate-300 hover:text-white transition"
                  >
                    {sug}
                  </button>
                ))}
              </div>

              {ragError && (
                <div className="p-3 rounded-lg bg-rose-950/70 border border-rose-800 text-rose-300 text-xs">
                  {ragError}
                </div>
              )}

              {/* Respuesta del Agente RAG */}
              {ragResult && (
                <div className="space-y-4 pt-2 border-t border-slate-800/80">
                  <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800 space-y-3">
                    <div className="flex items-center justify-between text-xs text-indigo-400 font-semibold">
                      <span>Síntesis Basada en Guías de Práctica Clínica</span>
                      <span className="font-mono text-slate-500">Latencia: {ragResult.response_latency_ms} ms</span>
                    </div>
                    <div className="text-xs text-slate-200 leading-relaxed whitespace-pre-wrap font-sans">
                      {ragResult.synthesized_answer}
                    </div>
                  </div>

                  {/* Citas Oficiales Recuperadas */}
                  <div className="space-y-2">
                    <h4 className="text-xs font-semibold text-slate-300">Fuentes Científicas y Guías Citadas:</h4>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                      {ragResult.citations.map((c) => (
                        <div key={c.source_id} className="p-3 rounded-lg bg-slate-950/60 border border-slate-800 text-xs space-y-1">
                          <span className="font-semibold text-sky-400 block truncate">{c.guideline}</span>
                          <span className="text-slate-300 font-medium block">{c.topic}</span>
                          <p className="text-[11px] text-slate-400 line-clamp-3 leading-tight">{c.recommendation}</p>
                          <div className="pt-1 flex items-center justify-between text-[10px] text-slate-500">
                            <span>{c.evidence_level.split('(')[0]}</span>
                            <a
                              href={c.doi_or_url}
                              target="_blank"
                              rel="noreferrer"
                              className="text-indigo-400 hover:underline"
                            >
                              Ver DOI
                            </a>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Guardrails Clínicos */}
                  <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 text-[11px] text-slate-400 space-y-1">
                    <strong className="text-slate-300 block">Restricciones y Guardrails de Seguridad Clínica:</strong>
                    <ul className="list-disc list-inside space-y-0.5">
                      {ragResult.clinical_guardrails.map((g, i) => (
                        <li key={i}>{g}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ==================================================== */}
        {/* PESTAÑA 4: BATERÍA DE EVALUACIÓN ESTADÍSTICA (F9)    */}
        {/* ==================================================== */}
        {activeTab === 'evaluacion_estadistica' && (
          <div className="space-y-6">
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 space-y-2">
              <h2 className="text-base font-bold text-white flex items-center gap-2">
                <BarChart3 className="h-5 w-5 text-sky-400" /> Batería de Pruebas Estadísticas Formales (Fase F9)
              </h2>
              <p className="text-xs text-slate-300 leading-relaxed">
                Evaluación no paramétrica rigurosa entre los 5 algoritmos sobre los pliegues de validación cruzada:
                <strong> Prueba de Friedman</strong>, comparaciones post-hoc de <strong>Wilcoxon con corrección de Holm</strong>, 
                <strong> prueba pareada de McNemar</strong> e intervalos de confianza <strong>Bootstrap al 95% (B=1000)</strong>.
              </p>
            </div>

            {statLoading ? (
              <div className="p-8 text-center text-slate-400 flex items-center justify-center gap-2">
                <RefreshCw className="h-4 w-4 animate-spin text-sky-400" /> Cargando evaluación estadística...
              </div>
            ) : statData && statData.nhanes_2017_2018 ? (
              <div className="space-y-5">
                {/* 1. Prueba de Friedman */}
                <div className="p-5 rounded-xl border border-slate-800 bg-slate-900/60 space-y-3">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-bold text-white">1. Prueba no Paramétrica de Friedman (k=5 Modelos)</h3>
                    <span
                      className={`px-2.5 py-0.5 rounded text-xs font-semibold ${
                        statData.nhanes_2017_2018.friedman_test.significant_at_05
                          ? 'bg-emerald-950 text-emerald-400 border border-emerald-800'
                          : 'bg-slate-800 text-slate-300 border border-slate-700'
                      }`}
                    >
                      {statData.nhanes_2017_2018.friedman_test.significant_at_05
                        ? 'Diferencias Significativas (p < 0.05)'
                        : 'Sin Diferencias Globales (p ≥ 0.05)'}
                    </span>
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
                    <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                      <span className="text-slate-400 block">Estadístico Chi-Cuadrado (χ²)</span>
                      <span className="text-base font-bold text-sky-400">
                        {statData.nhanes_2017_2018.friedman_test.statistic.toFixed(4)}
                      </span>
                    </div>
                    <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                      <span className="text-slate-400 block">P-valor asintótico</span>
                      <span className="text-base font-bold text-white">
                        {statData.nhanes_2017_2018.friedman_test.p_value.toExponential(3)}
                      </span>
                    </div>
                    <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                      <span className="text-slate-400 block">Interpretación</span>
                      <span className="text-xs text-slate-300">{statData.nhanes_2017_2018.friedman_test.interpretation}</span>
                    </div>
                  </div>
                </div>

                {/* 2. Pruebas Pareadas de Wilcoxon con Ajuste de Holm */}
                <div className="p-5 rounded-xl border border-slate-800 bg-slate-900/60 space-y-3">
                  <h3 className="text-sm font-bold text-white">2. Comparaciones Pareadas de Wilcoxon (Corrección de Holm-Bonferroni)</h3>
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs border-collapse">
                      <thead>
                        <tr className="border-b border-slate-800 text-slate-400">
                          <th className="py-2 px-3">Modelo A</th>
                          <th className="py-2 px-3">Modelo B</th>
                          <th className="py-2 px-3">Estadístico W</th>
                          <th className="py-2 px-3">P-raw</th>
                          <th className="py-2 px-3">P-Holm (Ajustado)</th>
                          <th className="py-2 px-3 text-right">Significativo (α=0.05)</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/60">
                        {statData.nhanes_2017_2018.wilcoxon_holm_pairwise.slice(0, 8).map((w, i) => (
                          <tr key={i} className="hover:bg-slate-850/50">
                            <td className="py-2 px-3 font-medium text-slate-200">{w.model_a}</td>
                            <td className="py-2 px-3 font-medium text-slate-300">{w.model_b}</td>
                            <td className="py-2 px-3 font-mono text-slate-400">{w.statistic.toFixed(1)}</td>
                            <td className="py-2 px-3 font-mono text-slate-400">{w.p_raw.toFixed(4)}</td>
                            <td className="py-2 px-3 font-mono font-bold text-sky-400">{w.p_holm.toFixed(4)}</td>
                            <td className="py-2 px-3 text-right">
                              <span
                                className={`px-2 py-0.5 rounded text-[10px] font-semibold ${
                                  w.significant_at_05
                                    ? 'bg-emerald-950 text-emerald-400'
                                    : 'bg-slate-800 text-slate-400'
                                }`}
                              >
                                {w.significant_at_05 ? 'Sí' : 'No'}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* 3. Intervalos Bootstrap al 95% */}
                {statData.nhanes_2017_2018.bootstrap_ci_95 && (
                  <div className="p-5 rounded-xl border border-slate-800 bg-slate-900/60 space-y-3">
                    <h3 className="text-sm font-bold text-white">3. Intervalos de Confianza al 95% mediante Remuestreo Bootstrap (B=1000)</h3>
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 text-xs">
                      {Object.entries(statData.nhanes_2017_2018.bootstrap_ci_95).map(([mod, metrics]: any) => (
                        <div key={mod} className="p-3 rounded-lg bg-slate-950 border border-slate-800 space-y-1.5">
                          <span className="font-bold text-white block capitalize truncate">{mod.replace('_', ' ')}</span>
                          {metrics.roc_auc && (
                            <div className="text-[11px] text-slate-300">
                              <span className="text-slate-400">AUC-ROC: </span>
                              <strong className="text-sky-400">
                                {metrics.roc_auc.mean.toFixed(3)} [{metrics.roc_auc.ci_lower.toFixed(3)} - {metrics.roc_auc.ci_upper.toFixed(3)}]
                              </strong>
                            </div>
                          )}
                          {metrics.f1 && (
                            <div className="text-[11px] text-slate-300">
                              <span className="text-slate-400">F1-Score: </span>
                              <span>
                                {metrics.f1.mean.toFixed(3)} [{metrics.f1.ci_lower.toFixed(3)} - {metrics.f1.ci_upper.toFixed(3)}]
                              </span>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* 5. Benchmark Comparativo Multi-Cohorte sobre 5 Datasets Reales */}
                {crossBenchmarks && (
                  <div className="p-5 rounded-xl border border-sky-500/30 bg-gradient-to-br from-slate-900 to-sky-950/20 space-y-4">
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-3">
                      <div>
                        <h3 className="text-sm font-bold text-white flex items-center gap-2">
                          <Layers className="h-4 w-4 text-sky-400" /> 5. Benchmark de Modelos sobre 5 Cohortes Reales Independientes
                        </h3>
                        <p className="text-xs text-slate-400 mt-0.5">
                          Evaluación reproducible con <strong>5-Fold Stratified Cross-Validation</strong> sin fuga de datos.
                        </p>
                      </div>
                      <div className="flex flex-wrap gap-1.5 text-xs">
                        {['ALL', ...Object.keys(crossBenchmarks)].map((key) => (
                          <button
                            key={key}
                            onClick={() => setSelectedBenchmarkCohort(key)}
                            className={`px-2.5 py-1 rounded text-[11px] font-semibold transition ${
                              selectedBenchmarkCohort === key
                                ? 'bg-sky-500 text-slate-950 font-bold'
                                : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                            }`}
                          >
                            {key === 'ALL' ? 'Todas las Cohortes' : crossBenchmarks[key]?.cohort_name || key}
                          </button>
                        ))}
                      </div>
                    </div>

                    <div className="overflow-x-auto">
                      <table className="w-full text-xs text-left">
                        <thead className="bg-slate-950 text-slate-400 font-semibold border-b border-slate-800">
                          <tr>
                            <th className="py-2.5 px-3">Cohorte Real</th>
                            <th className="py-2.5 px-2">Muestra (N)</th>
                            <th className="py-2.5 px-3">Dominio de Variables</th>
                            <th className="py-2.5 px-3">Algoritmo</th>
                            <th className="py-2.5 px-2 text-right">ROC-AUC (Media ± DE)</th>
                            <th className="py-2.5 px-2 text-right">PR-AUC</th>
                            <th className="py-2.5 px-2 text-right">F1-Score</th>
                            <th className="py-2.5 px-2 text-right">Sensibilidad</th>
                            <th className="py-2.5 px-2 text-right">Especificidad</th>
                            <th className="py-2.5 px-2 text-right">Brier Score</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800 text-slate-200">
                          {Object.entries(crossBenchmarks)
                            .filter(([k]) => selectedBenchmarkCohort === 'ALL' || selectedBenchmarkCohort === k)
                            .flatMap(([cohortKey, cohort]: [string, any]) =>
                              Object.entries(cohort.models).map(([modelName, m]: [string, any], idx: number) => (
                                <tr key={`${cohortKey}-${modelName}`} className="hover:bg-slate-800/40 transition">
                                  {idx === 0 ? (
                                    <td rowSpan={Object.keys(cohort.models).length} className="py-2.5 px-3 font-medium text-white border-r border-slate-800 bg-slate-900/40 align-top">
                                      <div className="font-bold text-sky-300">{cohort.cohort_name}</div>
                                      <div className="text-[10px] text-slate-400 mt-0.5">Prev: {(cohort.prevalence * 100).toFixed(1)}%</div>
                                    </td>
                                  ) : null}
                                  {idx === 0 ? (
                                    <td rowSpan={Object.keys(cohort.models).length} className="py-2.5 px-2 font-mono text-slate-300 border-r border-slate-800 bg-slate-900/40 align-top">
                                      {cohort.sample_size.toLocaleString()}
                                    </td>
                                  ) : null}
                                  {idx === 0 ? (
                                    <td rowSpan={Object.keys(cohort.models).length} className="py-2.5 px-3 text-slate-400 border-r border-slate-800 bg-slate-900/40 align-top text-[11px]">
                                      {cohort.category}
                                    </td>
                                  ) : null}
                                  <td className="py-2.5 px-3 font-semibold text-slate-100 flex items-center gap-1.5">
                                    <span className="w-1.5 h-1.5 rounded-full bg-sky-400"></span>
                                    {modelName}
                                  </td>
                                  <td className="py-2.5 px-2 text-right font-mono font-bold text-sky-400">
                                    {m.roc_auc_mean.toFixed(4)} <span className="text-[10px] text-slate-500">±{m.roc_auc_std.toFixed(3)}</span>
                                  </td>
                                  <td className="py-2.5 px-2 text-right font-mono text-emerald-400 font-semibold">{m.pr_auc_mean.toFixed(4)}</td>
                                  <td className="py-2.5 px-2 text-right font-mono text-slate-200">{m.f1_mean.toFixed(4)}</td>
                                  <td className="py-2.5 px-2 text-right font-mono text-slate-300">{m.sensitivity_mean.toFixed(4)}</td>
                                  <td className="py-2.5 px-2 text-right font-mono text-slate-300">{m.specificity_mean.toFixed(4)}</td>
                                  <td className="py-2.5 px-2 text-right font-mono text-amber-400">{m.brier_score_mean.toFixed(4)}</td>
                                </tr>
                              ))
                            )}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="p-6 rounded-xl border border-slate-800 bg-slate-900/40 text-center text-xs text-slate-400">
                La batería estadística está disponible tras ejecutar el pipeline de entrenamiento scripts/train_and_evaluate_v2.py.
              </div>
            )}
          </div>
        )}

        {/* ==================================================== */}
        {/* PESTAÑA 5: ORQUESTADOR N8N & ALERTAS                 */}
        {/* ==================================================== */}
        {activeTab === 'orquestacion_n8n' && (
          <div className="space-y-6">
            <div className="rounded-xl border border-emerald-500/20 bg-emerald-950/20 p-5 space-y-2">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <h2 className="text-base font-bold text-white flex items-center gap-2">
                  <GitBranch className="h-5 w-5 text-emerald-400" /> Orquestación de Workflows n8n (6 Procesos)
                </h2>
                <div className="flex items-center gap-2">
                  <a
                    href="http://localhost:5678"
                    target="_blank"
                    rel="noreferrer"
                    className="px-3 py-1.5 rounded-lg bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold text-xs flex items-center gap-1.5 transition shadow"
                  >
                    <ExternalLink className="h-3.5 w-3.5" /> Abrir Editor n8n en Vivo (Puerto 5678)
                  </a>
                  <button
                    onClick={() => loadN8nData()}
                    className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs text-slate-200 border border-slate-700 flex items-center gap-1.5 transition"
                  >
                    <RefreshCw className="h-3.5 w-3.5" /> Actualizar Estado
                  </button>
                </div>
              </div>
              <p className="text-xs text-slate-300 leading-relaxed">
                Automatización reproducible conforme al requerimiento del docente: 6 flujos de ingestión, validación, 
                predicción programada, alertas epidemiológicas, reportes y monitoreo de deriva con MLflow.
              </p>
            </div>

            {n8nMessage && (
              <div className="p-3 rounded-lg bg-emerald-950/80 border border-emerald-800 text-emerald-300 text-xs">
                {n8nMessage}
              </div>
            )}

            {/* Listado de los 6 Workflows n8n */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {n8nWorkflows.map((wf) => (
                <div key={wf.workflow_id} className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 space-y-2.5">
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-sm text-white">{wf.name}</span>
                    <span className="text-[10px] bg-emerald-950 text-emerald-400 border border-emerald-800 px-2 py-0.5 rounded uppercase font-semibold">
                      {wf.last_status}
                    </span>
                  </div>
                  <p className="text-xs text-slate-400 leading-relaxed">{wf.description}</p>
                  <div className="flex items-center justify-between text-[11px] text-slate-500 pt-1 border-t border-slate-800/60">
                    <span className="font-mono">{wf.schedule}</span>
                    <button
                      onClick={() => triggerWorkflow(wf.workflow_id)}
                      disabled={triggeringWf !== null}
                      className="px-2.5 py-1 rounded bg-slate-800 hover:bg-emerald-600 hover:text-slate-950 text-xs text-slate-200 border border-slate-700 transition flex items-center gap-1"
                    >
                      {triggeringWf === wf.workflow_id ? <RefreshCw className="h-3 w-3 animate-spin" /> : <Play className="h-3 w-3" />}
                      Ejecutar Workflow
                    </button>
                  </div>
                </div>
              ))}
            </div>

            {/* Alertas Clínicas Registradas (WF-04) */}
            <div className="p-5 rounded-xl border border-slate-800 bg-slate-900/60 space-y-3">
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <AlertTriangle className="h-4 w-4 text-rose-400" /> Registro de Alertas Clínicas de Riesgo (WF-04)
              </h3>
              {n8nAlerts.length > 0 ? (
                <div className="space-y-2 max-h-60 overflow-y-auto pr-1 scrollbar-thin">
                  {n8nAlerts.map((al) => (
                    <div key={al.id} className="p-3 rounded-lg bg-slate-950/80 border border-slate-800 text-xs flex items-center justify-between gap-2">
                      <div>
                        <span className="font-bold text-white">{al.message}</span>
                        <div className="text-[11px] text-slate-400 mt-0.5">
                          Paciente: {al.payload?.patient_name || al.payload?.patient_id} · Riesgo: {al.payload?.riesgo_pct}% ({al.payload?.estrato})
                        </div>
                      </div>
                      <span className="text-[10px] text-slate-500 font-mono shrink-0">{al.created_at.slice(0, 19)}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-xs text-slate-500">No hay alertas recientes registradas.</p>
              )}
            </div>
          </div>
        )}

        {/* ==================================================== */}
        {/* PESTAÑA 6: REPORTES MULTIFORMATO                     */}
        {/* ==================================================== */}
        {activeTab === 'reportes' && (
          <div className="space-y-6">
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 space-y-2">
              <h2 className="text-base font-bold text-white flex items-center gap-2">
                <FileText className="h-5 w-5 text-sky-400" /> Centro de Reportes Clínicos y Gerenciales
              </h2>
              <p className="text-xs text-slate-300 leading-relaxed">
                Generación automática multi-formato exigida por el protocolo (WF-05): 
                <strong> PDF médico estructurado</strong> (ReportLab con gráficos y semáforo), 
                <strong> Word editable .docx</strong> (python-docx con tablas académicas) y 
                <strong> Excel con formato condicional .xlsx</strong> (openpyxl).
              </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              {/* Formulario de Datos para el Reporte */}
              <div className="lg:col-span-5 p-5 rounded-xl border border-slate-800 bg-slate-900/60 space-y-4">
                <h3 className="text-sm font-semibold text-white">Datos de Identificación del Paciente</h3>
                <div className="space-y-3 text-xs">
                  <div>
                    <label className="text-slate-400 block mb-1">Nombre Completo o Rótulo Anonimizado</label>
                    <input
                      type="text"
                      value={patientName}
                      onChange={(e) => setPatientName(e.target.value)}
                      className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-sky-500"
                    />
                  </div>
                  <div>
                    <label className="text-slate-400 block mb-1">Código Único de Paciente</label>
                    <input
                      type="text"
                      value={patientIdInput}
                      onChange={(e) => setPatientIdInput(e.target.value)}
                      className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-sky-500"
                    />
                  </div>
                  <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 text-[11px] text-slate-400 space-y-1">
                    <span>Variables que se incluirán en el informe:</span>
                    <p className="font-mono text-slate-300">
                      Edad: {formValues.edad || 50}a · IMC: {formValues.imc || 28} · Glucosa: {formValues.glucosa || 120} mg/dL · HbA1c: {formValues.hba1c || 6.2}%
                    </p>
                  </div>
                </div>
              </div>

              {/* Botones de Descarga de Formatos */}
              <div className="lg:col-span-7 grid grid-cols-1 sm:grid-cols-3 gap-4">
                {/* PDF */}
                <div className="p-5 rounded-xl border border-slate-800 bg-slate-900/60 flex flex-col justify-between space-y-3">
                  <div className="space-y-2">
                    <div className="h-10 w-10 rounded-lg bg-rose-500/10 border border-rose-500/20 flex items-center justify-center text-rose-400">
                      <FileText className="h-5 w-5" />
                    </div>
                    <h4 className="font-bold text-sm text-white">Informe Médico PDF</h4>
                    <p className="text-xs text-slate-400">
                      Documento formal generado con ReportLab. Contiene cabecera asistencial, semáforo, tabla de factores SHAP y guías ADA.
                    </p>
                  </div>
                  <button
                    onClick={() => downloadReport('pdf')}
                    disabled={reportLoading !== null}
                    className="w-full py-2 rounded-lg bg-rose-600 hover:bg-rose-500 text-white font-semibold text-xs flex items-center justify-center gap-1.5 transition disabled:opacity-50"
                  >
                    {reportLoading === 'pdf' ? <RefreshCw className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}
                    Descargar PDF
                  </button>
                </div>

                {/* Word .docx */}
                <div className="p-5 rounded-xl border border-slate-800 bg-slate-900/60 flex flex-col justify-between space-y-3">
                  <div className="space-y-2">
                    <div className="h-10 w-10 rounded-lg bg-sky-500/10 border border-sky-500/20 flex items-center justify-center text-sky-400">
                      <FileText className="h-5 w-5" />
                    </div>
                    <h4 className="font-bold text-sm text-white">Reporte Word (.docx)</h4>
                    <p className="text-xs text-slate-400">
                      Informe editable generado con python-docx. Formato académico con tablas científicas para inclusión en historias clínicas.
                    </p>
                  </div>
                  <button
                    onClick={() => downloadReport('docx')}
                    disabled={reportLoading !== null}
                    className="w-full py-2 rounded-lg bg-sky-600 hover:bg-sky-500 text-white font-semibold text-xs flex items-center justify-center gap-1.5 transition disabled:opacity-50"
                  >
                    {reportLoading === 'docx' ? <RefreshCw className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}
                    Descargar Word
                  </button>
                </div>

                {/* Excel .xlsx */}
                <div className="p-5 rounded-xl border border-slate-800 bg-slate-900/60 flex flex-col justify-between space-y-3">
                  <div className="space-y-2">
                    <div className="h-10 w-10 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
                      <FileSpreadsheet className="h-5 w-5" />
                    </div>
                    <h4 className="font-bold text-sm text-white">Planilla Excel (.xlsx)</h4>
                    <p className="text-xs text-slate-400">
                      Matriz analítica generada con openpyxl. Contiene formato condicional (verde/amarillo/rojo) por estrato de riesgo.
                    </p>
                  </div>
                  <button
                    onClick={() => downloadReport('xlsx')}
                    disabled={reportLoading !== null}
                    className="w-full py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-slate-950 font-bold text-xs flex items-center justify-center gap-1.5 transition disabled:opacity-50"
                  >
                    {reportLoading === 'xlsx' ? <RefreshCw className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}
                    Descargar Excel
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ==================================================== */}
        {/* PESTAÑA 7: INFERENCIA POR LOTES Y DATASETS REALES    */}
        {/* ==================================================== */}
        {activeTab === 'lotes' && (
          <div className="space-y-6">
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 space-y-2">
              <h2 className="text-base font-bold text-white flex items-center gap-2">
                <Database className="h-5 w-5 text-sky-400" /> Datasets Clínicos Reales y Procesamiento por Lotes
              </h2>
              <p className="text-xs text-slate-300 leading-relaxed">
                Descargue los conjuntos de datos clínicos reales completos unificados para su investigación o suba archivos CSV
                para realizar inferencia y estratificación masiva de riesgo con detección automática de archivos DEMO.
              </p>
            </div>

            {/* Tarjetas de Descarga de Datasets Reales Completos (5 Cohortes) */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
              {/* Dataset 1: CDC NHANES 2017-2018 Completo */}
              <div className="p-5 rounded-xl border border-sky-500/30 bg-gradient-to-br from-slate-900 to-sky-950/20 space-y-3 flex flex-col justify-between">
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded-full bg-sky-500/10 text-sky-400 border border-sky-500/20">
                      Cohorte Primaria (CDC/NCHS)
                    </span>
                    <span className="text-xs font-mono text-slate-400">4,829 Pacientes</span>
                  </div>
                  <div>
                    <h3 className="text-sm font-bold text-white">CDC NHANES 2017–2018</h3>
                    <p className="text-xs text-slate-400 mt-1 leading-relaxed">
                      16 columnas: biomarcadores HPLC (glucosa basal, HbA1c, lípidos), PA, IMC y diagnóstico médico confirmado.
                    </p>
                  </div>
                </div>
                <div className="flex flex-wrap items-center gap-2 pt-2 border-t border-slate-800">
                  <a
                    href={`${API_BASE}/sample-csv/nhanes_completo`}
                    download="nhanes_2017_2018_dataset_completo.csv"
                    className="px-3 py-1.5 rounded-lg bg-sky-500 hover:bg-sky-400 text-slate-950 font-bold text-xs flex items-center gap-1.5 transition shadow"
                  >
                    <Download className="h-3.5 w-3.5" /> CSV (415 KB)
                  </a>
                  <a
                    href={`${API_BASE}/sample-csv/nhanes_excel`}
                    download="nhanes_2017_2018_dataset_completo.xlsx"
                    className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 font-medium text-xs flex items-center gap-1.5 border border-slate-700 transition"
                  >
                    <FileSpreadsheet className="h-3.5 w-3.5 text-emerald-400" /> Excel (.xlsx)
                  </a>
                </div>
              </div>

              {/* Dataset 2: NIDDK Pima Indians Completo */}
              <div className="p-5 rounded-xl border border-indigo-500/30 bg-gradient-to-br from-slate-900 to-indigo-950/20 space-y-3 flex flex-col justify-between">
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                      Benchmark Histórico (NIDDK)
                    </span>
                    <span className="text-xs font-mono text-slate-400">768 Pacientes</span>
                  </div>
                  <div>
                    <h3 className="text-sm font-bold text-white">NIDDK Pima Indians</h3>
                    <p className="text-xs text-slate-400 mt-1 leading-relaxed">
                      12 columnas: sobrecarga oral 2h OGTT, insulina sérica, pliegue cutáneo tricipital, pedigrí genético y edad.
                    </p>
                  </div>
                </div>
                <div className="flex flex-wrap items-center gap-2 pt-2 border-t border-slate-800">
                  <a
                    href={`${API_BASE}/sample-csv/pima_completo`}
                    download="pima_indians_dataset_completo.csv"
                    className="px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-xs flex items-center gap-1.5 transition shadow"
                  >
                    <Download className="h-3.5 w-3.5" /> CSV (62 KB)
                  </a>
                  <a
                    href={`${API_BASE}/sample-csv/pima_excel`}
                    download="pima_indians_dataset_completo.xlsx"
                    className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 font-medium text-xs flex items-center gap-1.5 border border-slate-700 transition"
                  >
                    <FileSpreadsheet className="h-3.5 w-3.5 text-emerald-400" /> Excel (.xlsx)
                  </a>
                </div>
              </div>

              {/* Dataset 3: CDC BRFSS 2015 Completo (50/50 split) */}
              <div className="p-5 rounded-xl border border-emerald-500/30 bg-gradient-to-br from-slate-900 to-emerald-950/20 space-y-3 flex flex-col justify-between">
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                      Salud Poblacional (CDC BRFSS)
                    </span>
                    <span className="text-xs font-mono text-slate-400">70,692 Encuestados</span>
                  </div>
                  <div>
                    <h3 className="text-sm font-bold text-white">CDC BRFSS 2015 Indicators</h3>
                    <p className="text-xs text-slate-400 mt-1 leading-relaxed">
                      22 columnas: factores conductuales, dieta, tabaquismo, ejercicio, salud física/mental, comorbilidades (Big Data).
                    </p>
                  </div>
                </div>
                <div className="flex flex-wrap items-center gap-2 pt-2 border-t border-slate-800">
                  <a
                    href={`${API_BASE}/sample-csv/brfss_completo`}
                    download="cdc_brfss_5050_clean.csv"
                    className="px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-xs flex items-center gap-1.5 transition shadow"
                  >
                    <Download className="h-3.5 w-3.5" /> CSV Completo (6.1 MB)
                  </a>
                  <a
                    href={`${API_BASE}/sample-csv/brfss_excel`}
                    download="cdc_brfss_5050_sample_5k.xlsx"
                    className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 font-medium text-xs flex items-center gap-1.5 border border-slate-700 transition"
                  >
                    <FileSpreadsheet className="h-3.5 w-3.5 text-emerald-400" /> Excel (Muestra 5k)
                  </a>
                </div>
              </div>

              {/* Dataset 4: Diabetes 130-US Hospitals (UCI 296) */}
              <div className="p-5 rounded-xl border border-rose-500/30 bg-gradient-to-br from-slate-900 to-rose-950/20 space-y-3 flex flex-col justify-between">
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded-full bg-rose-500/10 text-rose-400 border border-rose-500/20">
                      EHR Hospitalario (UCI 296)
                    </span>
                    <span className="text-xs font-mono text-slate-400">71,515 Pacientes</span>
                  </div>
                  <div>
                    <h3 className="text-sm font-bold text-white">Diabetes 130-US Hospitals</h3>
                    <p className="text-xs text-slate-400 mt-1 leading-relaxed">
                      18 columnas: admisiones, medicaciones (insulina), procedimientos, estancias y reingreso prematuro (&lt;30 días sin fuga).
                    </p>
                  </div>
                </div>
                <div className="flex flex-wrap items-center gap-2 pt-2 border-t border-slate-800">
                  <a
                    href={`${API_BASE}/sample-csv/hospitals_completo`}
                    download="diabetes_130_hospitals_clean.csv"
                    className="px-3 py-1.5 rounded-lg bg-rose-600 hover:bg-rose-500 text-white font-bold text-xs flex items-center gap-1.5 transition shadow"
                  >
                    <Download className="h-3.5 w-3.5" /> CSV Completo (4.8 MB)
                  </a>
                  <a
                    href={`${API_BASE}/sample-csv/hospitals_excel`}
                    download="diabetes_130_hospitals_sample_5k.xlsx"
                    className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 font-medium text-xs flex items-center gap-1.5 border border-slate-700 transition"
                  >
                    <FileSpreadsheet className="h-3.5 w-3.5 text-emerald-400" /> Excel (Muestra 5k)
                  </a>
                </div>
              </div>

              {/* Dataset 5: Early Stage Diabetes (UCI 529 - Sylhet) */}
              <div className="p-5 rounded-xl border border-amber-500/30 bg-gradient-to-br from-slate-900 to-amber-950/20 space-y-3 flex flex-col justify-between">
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/20">
                      Sintomatología Prodrómica (UCI 529)
                    </span>
                    <span className="text-xs font-mono text-slate-400">520 Pacientes</span>
                  </div>
                  <div>
                    <h3 className="text-sm font-bold text-white">Early Stage Diabetes Risk</h3>
                    <p className="text-xs text-slate-400 mt-1 leading-relaxed">
                      17 columnas: signos precoces cardinales (poliuria, polidipsia, pérdida de peso, visión borrosa, astenia, paresias).
                    </p>
                  </div>
                </div>
                <div className="flex flex-wrap items-center gap-2 pt-2 border-t border-slate-800">
                  <a
                    href={`${API_BASE}/sample-csv/early_stage_completo`}
                    download="early_stage_diabetes_sylhet_clean.csv"
                    className="px-3 py-1.5 rounded-lg bg-amber-500 hover:bg-amber-400 text-slate-950 font-bold text-xs flex items-center gap-1.5 transition shadow"
                  >
                    <Download className="h-3.5 w-3.5" /> CSV Completo (34 KB)
                  </a>
                  <a
                    href={`${API_BASE}/sample-csv/early_stage_excel`}
                    download="early_stage_diabetes_sylhet_clean.xlsx"
                    className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 font-medium text-xs flex items-center gap-1.5 border border-slate-700 transition"
                  >
                    <FileSpreadsheet className="h-3.5 w-3.5 text-emerald-400" /> Excel (.xlsx)
                  </a>
                </div>
              </div>
            </div>

            {/* Sección de Carga e Inferencia por Lotes */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              <div className="lg:col-span-5 p-5 rounded-xl border border-slate-800 bg-slate-900/60 space-y-4">
                <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                  <Upload className="h-4 w-4 text-sky-400" /> Procesar Archivo CSV de Pacientes
                </h3>
                <form onSubmit={handleBatchSubmit} className="space-y-4 text-xs">
                  <div>
                    <label className="text-slate-400 block mb-1">Seleccionar Archivo CSV Local</label>
                    <input
                      type="file"
                      accept=".csv"
                      onChange={(e) => setBatchFile(e.target.files ? e.target.files[0] : null)}
                      className="w-full text-slate-300 file:mr-3 file:py-2 file:px-3 file:rounded-lg file:border-0 file:text-xs file:bg-sky-500/20 file:text-sky-300 hover:file:bg-sky-500/30 cursor-pointer"
                    />
                  </div>

                  <div className="flex flex-wrap gap-2 pt-1">
                    <span className="text-[11px] text-slate-500 w-full">Plantillas de prueba rápida:</span>
                    <a
                      href={`${API_BASE}/sample-csv/nhanes_test`}
                      className="px-2.5 py-1.5 rounded bg-slate-950 border border-slate-800 text-[11px] text-slate-300 hover:text-white flex items-center gap-1"
                    >
                      <Download className="h-3 w-3" /> Muestra NHANES (25 filas)
                    </a>
                    <a
                      href={`${API_BASE}/sample-csv/demo_sintetico`}
                      className="px-2.5 py-1.5 rounded bg-amber-950/40 border border-amber-800 text-[11px] text-amber-300 hover:text-amber-200 flex items-center gap-1"
                    >
                      <Download className="h-3 w-3" /> DEMO Sintético
                    </a>
                  </div>

                  {batchError && (
                    <div className="p-3 rounded-lg bg-rose-950/70 border border-rose-800 text-rose-300 text-xs">
                      {batchError}
                    </div>
                  )}

                  <button
                    type="submit"
                    disabled={batchLoading}
                    className="w-full py-2.5 rounded-xl bg-sky-500 hover:bg-sky-400 text-slate-950 font-bold text-xs flex items-center justify-center gap-2 transition disabled:opacity-50"
                  >
                    {batchLoading ? <RefreshCw className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
                    Ejecutar Inferencia por Lotes
                  </button>
                </form>
              </div>

              <div className="lg:col-span-7">
                {batchResult ? (
                  <div className="p-5 rounded-xl border border-slate-800 bg-slate-900/60 space-y-4">
                    <div className="flex items-center justify-between">
                      <h3 className="text-sm font-bold text-white">Resultados del Procesamiento en Lote</h3>
                      <span className="text-xs text-sky-400 font-mono">
                        {batchResult.total_rows_processed} filas procesadas
                      </span>
                    </div>

                    {batchResult.demo_warning && (
                      <div className="p-3 rounded-lg bg-amber-950/70 border border-amber-800 text-amber-300 text-xs flex items-center gap-2">
                        <AlertTriangle className="h-4 w-4 text-amber-400 shrink-0" />
                        <span>{batchResult.demo_warning}</span>
                      </div>
                    )}

                    <div className="overflow-x-auto max-h-80 scrollbar-thin">
                      <table className="w-full text-left text-xs border-collapse">
                        <thead>
                          <tr className="border-b border-slate-800 text-slate-400">
                            <th className="py-2 px-2.5">#</th>
                            <th className="py-2 px-2.5">Probabilidad</th>
                            <th className="py-2 px-2.5">Estrato</th>
                            <th className="py-2 px-2.5">Factor Principal (SHAP)</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800/60">
                          {batchResult.predictions.map((p) => (
                            <tr key={p.row_index} className="hover:bg-slate-850/50">
                              <td className="py-2 px-2.5 text-slate-500">{p.row_index}</td>
                              <td className="py-2 px-2.5 font-mono font-bold text-white">{p.percentage}%</td>
                              <td className="py-2 px-2.5">
                                <span
                                  className={`px-2 py-0.5 rounded text-[10px] font-semibold ${
                                    p.risk_stratum === 'Bajo'
                                      ? 'bg-emerald-950 text-emerald-400'
                                      : p.risk_stratum === 'Moderado'
                                      ? 'bg-amber-950 text-amber-400'
                                      : 'bg-rose-950 text-rose-400'
                                  }`}
                                >
                                  {p.risk_stratum}
                                </span>
                              </td>
                              <td className="py-2 px-2.5 text-slate-300 text-[11px] truncate max-w-[200px]">
                                {p.top_risk_factor}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                ) : (
                  <div className="h-full min-h-[280px] rounded-xl border border-dashed border-slate-800 flex items-center justify-center p-6 text-center text-slate-500 text-xs">
                    Suba un archivo CSV a la izquierda para ver el resumen y listado de predicciones por fila.
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ==================================================== */}
        {/* PESTAÑA 8: METODOLOGÍA F1-F12 & TRAZABILIDAD        */}
        {/* ==================================================== */}
        {activeTab === 'metodologia' && (
          <div className="space-y-6">
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 space-y-2">
              <h2 className="text-base font-bold text-white flex items-center gap-2">
                <BookOpen className="h-5 w-5 text-sky-400" /> Protocolo Experimental Común en 12 Fases (F1 a F12)
              </h2>
              <p className="text-xs text-slate-300 leading-relaxed">
                Estructura metodológica integral para garantizar publicaciones científicas reproducibles en revistas indexadas Q1 (Scopus / WoS).
              </p>
            </div>

            {/* Grid de las 12 Fases */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 text-xs">
              {[
                { f: 'F1', name: 'Revisión Científica', prod: 'PRISMA 2020, PubMed, IEEE Xplore' },
                { f: 'F2', name: 'Comprensión del Problema', prod: 'Población, unidades de análisis y variables' },
                { f: 'F3', name: 'Datasets & Licencias', prod: 'CDC NHANES 2017-2018 (CC0) y Pima NIDDK' },
                { f: 'F4', name: 'EDA Estadístico', prod: 'Detección de outliers, distribuciones y correlaciones' },
                { f: 'F5', name: 'Preprocesamiento & Anti-fuga', prod: 'Imputación por mediana en pipeline de partición' },
                { f: 'F6', name: 'Modelamiento (5 Modelos)', prod: '3 individuales (RF, XGB, TabNet) + 2 híbridos' },
                { f: 'F7', name: 'Optimización Optuna', prod: 'Búsqueda bayesiana TPE Sampler sobre CV' },
                { f: 'F8', name: 'Validación por Clusters', prod: 'Partición estratificada aislada sin fuga' },
                { f: 'F9', name: 'Evaluación Estadística', prod: 'Friedman, Wilcoxon-Holm, McNemar, Bootstrap' },
                { f: 'F10', name: 'Interpretabilidad XAI', prod: 'SHAP (TreeExplainer) global e individual' },
                { f: 'F11', name: 'Prototipo & n8n', prod: 'FastAPI + React + 6 Workflows de automatización' },
                { f: 'F12', name: 'Publicación Científica', prod: 'Articulo_SP03.docx, 6 figuras y 6 tablas' },
              ].map((item) => (
                <div key={item.f} className="p-3 rounded-lg bg-slate-900 border border-slate-800 space-y-1">
                  <div className="flex items-center justify-between text-sky-400 font-bold">
                    <span>{item.f}. {item.name}</span>
                    <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
                  </div>
                  <p className="text-[11px] text-slate-400 leading-tight">{item.prod}</p>
                </div>
              ))}
            </div>

            {/* Matriz de Trazabilidad */}
            {modelInfo && modelInfo.traceability_matrix && (
              <div className="p-5 rounded-xl border border-slate-800 bg-slate-900/60 space-y-3">
                <h3 className="text-sm font-bold text-white">Matriz de Trazabilidad con Caso03 y Especificaciones</h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs border-collapse">
                    <thead>
                      <tr className="border-b border-slate-800 text-slate-400">
                        <th className="py-2 px-3">Req ID</th>
                        <th className="py-2 px-3">Especificación Docente</th>
                        <th className="py-2 px-3">Estado de Cumplimiento</th>
                        <th className="py-2 px-3">Ubicación de Código / Módulo</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60">
                      {modelInfo.traceability_matrix.map((t, idx) => (
                        <tr key={idx} className="hover:bg-slate-850/50">
                          <td className="py-2 px-3 font-mono text-sky-400 font-bold">{t.requirement_id}</td>
                          <td className="py-2 px-3 text-slate-300 font-medium">{t.caso03_specification}</td>
                          <td className="py-2 px-3">
                            <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-950 text-emerald-400 border border-emerald-800">
                              {t.implementation_status}
                            </span>
                          </td>
                          <td className="py-2 px-3 text-slate-400 text-[11px]">{t.module_location}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}
        {activeTab === 'pipeline_v3' && <PipelineV3 apiBase={API_BASE} />}
      </main>

      {/* Pie de Página */}
      <footer className="border-t border-slate-900 bg-slate-950 py-4 text-center text-xs text-slate-500">
        <p>Caso SP-03 | Multimodal Explainable Artificial Intelligence for Early Prediction of Type 2 Diabetes</p>
        <p className="text-[10px] text-slate-600 mt-1">
          Arquitectura: 5 Modelos · Optuna · SHAP · n8n · LangChain · LangFlow · FastAPI · React · ReportLab · OpenPyXL
        </p>
      </footer>
    </div>
  );
}
