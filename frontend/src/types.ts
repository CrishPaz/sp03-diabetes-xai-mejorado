export type DatasetId = 'nhanes_2017_2018' | 'pima_indians';
export type ModelId =
  | 'xgboost'
  | 'random_forest'
  | 'tabnet'
  | 'stacking_ensemble'
  | 'hybrid_survival'
  | 'logistic_regression'
  | string;

export interface FeatureMeta {
  name: string;
  source_col: string;
  label: string;
  unit: string;
  category: string;
  type?: string;
  min: number;
  max: number;
  step: number;
  default: number;
  normal_range: string;
  description: string;
}

export interface ConfusionMatrix {
  tn: number;
  fp: number;
  fn: number;
  tp: number;
}

export interface RocPoint {
  fpr: number;
  tpr: number;
}

export interface CalibrationPoint {
  prob_pred: number;
  prob_true: number;
}

export interface BinaryMetrics {
  threshold: number;
  confusion_matrix: ConfusionMatrix;
  sensitivity: number;
  specificity: number;
  precision: number;
  npv: number;
  f1: number;
  accuracy: number;
  balanced_accuracy: number;
  roc_auc: number;
  pr_auc: number;
  brier_score: number;
  roc_curve: RocPoint[];
  calibration_curve: CalibrationPoint[];
}

export interface SubgroupMetric {
  dimension: string;
  subgroup: string;
  n: number;
  prevalence: number;
  predicted_positive_rate: number;
  roc_auc: number;
  sensitivity: number;
  specificity: number;
  brier_score: number;
}

export interface ModelCatalogEntry {
  model_id: ModelId;
  name: string;
  family: string;
  role: string;
  hyperparameters: Record<string, number | string>;
  cv_5fold: {
    roc_auc_mean: number;
    roc_auc_std: number;
    f1_mean: number;
    f1_std: number;
    sensitivity_mean: number;
    brier_mean: number;
  };
  test_metrics: BinaryMetrics;
  subgroup_fairness: SubgroupMetric[];
}

export interface CohortData {
  dataset_id: DatasetId;
  audit: {
    dataset_id: string;
    name: string;
    period: string;
    license: string;
    source_url: string;
    raw_participants_count: number;
    adults_evaluated_count: number;
    final_analytic_cohort_count: number;
    positive_cases: number;
    negative_cases: number;
    prevalence_rate: number;
    duplicate_patient_ids: number;
    duplicate_feature_rows: number;
    missing_values_documented: Record<string, number>;
    imputation_strategy: string;
    leakage_prevention_notes: string[];
    target_definition: string;
  };
  split_summary: {
    strategy: string;
    train_patients: number;
    train_positive_cases: number;
    test_patients_isolated: number;
    test_positive_cases: number;
    patient_overlap_count: number;
  };
  features: FeatureMeta[];
  global_shap_importance: Array<{
    feature: string;
    label: string;
    mean_abs_shap: number;
  }>;
  models: Record<string, ModelCatalogEntry>;
}

export interface ModelInfoResponse {
  project_code: string;
  scientific_title: string;
  generated_at_utc?: string;
  updated_at_utc?: string;
  training_duration_seconds?: number;
  random_seed: number;
  default_dataset: DatasetId;
  environment_versions?: Record<string, string>;
  datasets_evaluated_feasibility?: Array<{
    name: string;
    status: string;
    license: string;
    access: string;
    strengths: string;
    limitations: string;
  }>;
  outcomes_architecture?: {
    outcome_1_diabetes_onset: {
      title: string;
      status: string;
      models_implemented: string[];
      explainability: string;
    };
    outcome_2_complications_survival: {
      title: string;
      status: string;
      planned_hybrid_models: string[];
      justification: string;
      required_longitudinal_schema: string[];
    };
  };
  traceability_matrix?: Array<{
    requirement_id: string;
    caso03_specification: string;
    implementation_status: string;
    module_location: string;
  }>;
  cohorts: Record<DatasetId, CohortData>;
  demo_synthetic_info?: {
    file_path: string;
    rows: number;
    label: string;
  };
  clinical_disclaimer: string;
}

export interface ShapContribution {
  feature: string;
  label: string;
  value: number;
  unit: string;
  shap_value: number;
  abs_importance: number;
  direction: 'incrementa_riesgo' | 'reduce_riesgo' | 'neutro';
  clinical_interpretation: string;
}

export interface SurvivalProjection {
  horizon_years: number;
  complication_risk_probability: number;
  complication_risk_percentage: number;
  projected_event: string;
  hazard_ratio_estimate: number;
}

export interface PredictResponse {
  dataset_id: DatasetId;
  dataset_name: string;
  model_id: ModelId;
  model_name: string;
  outcome_evaluated: string;
  complications_note: string;
  probability: number;
  percentage: number;
  prediction: number;
  threshold: number;
  risk_stratum: 'Bajo' | 'Moderado' | 'Alto' | 'Muy Alto';
  risk_recommendation: string;
  base_value: number;
  shap_contributions: ShapContribution[];
  all_models_comparison: Array<{
    model_id: string;
    model_name: string;
    probability: number;
    percentage: number;
    risk_stratum: string;
  }>;
  survival_projections?: SurvivalProjection[];
  clinical_evidence_summary?: string;
  disclaimer: string;
}

export interface BatchPredictResponse {
  dataset_id: string;
  model_id: string;
  is_synthetic_demo: boolean;
  demo_warning: string | null;
  total_rows_processed: number;
  high_risk_count: number;
  mean_probability: number;
  predictions: Array<{
    row_index: number;
    probability: number;
    percentage: number;
    prediction: number;
    risk_stratum: string;
    top_risk_factor: string;
  }>;
  disclaimer: string;
}

// ==========================================
// Tipos para RAG Clínico y LangChain / LangFlow
// ==========================================

export interface RagCitation {
  source_id: string;
  guideline: string;
  section: string;
  topic: string;
  recommendation: string;
  evidence_level: string;
  doi_or_url: string;
  similarity_score: number;
}

export interface RagQueryResponse {
  query: string;
  synthesized_answer: string;
  citations: RagCitation[];
  clinical_guardrails: string[];
  disclaimer: string;
  response_latency_ms: number;
}

// ==========================================
// Tipos para n8n Orquestación y Alertas
// ==========================================

export interface N8nWorkflow {
  workflow_id: string;
  file_name: string;
  name: string;
  phase: string;
  schedule: string;
  description: string;
  active: boolean;
  last_status: 'success' | 'running' | 'failed' | 'pending';
  last_run: string;
  file_exists?: boolean;
}

export interface N8nAlert {
  id: number;
  workflow_id: string;
  event_type: string;
  severity: string;
  message: string;
  payload: {
    patient_id?: string;
    patient_name?: string;
    edad?: number;
    riesgo_pct?: number;
    estrato?: string;
    top_factor?: string;
    accion_recomendada?: string;
  };
  status: string;
  created_at: string;
}

// ==========================================
// Tipos para Evaluación Estadística Formal
// ==========================================

export interface StatisticalEvaluation {
  status?: string;
  nhanes_2017_2018?: {
    friedman_test: {
      statistic: number;
      p_value: number;
      significant_at_05: boolean;
      interpretation: string;
    };
    wilcoxon_holm_pairwise: Array<{
      model_a: string;
      model_b: string;
      statistic: number;
      p_raw: number;
      p_holm: number;
      significant_at_05: boolean;
    }>;
    mcnemar_pairwise: Array<{
      model_a: string;
      model_b: string;
      statistic: number;
      p_value: number;
      significant_at_05: boolean;
      b_discordant: number;
      c_discordant: number;
    }>;
    bootstrap_ci_95: Record<
      string,
      Record<
        string,
        {
          mean: number;
          ci_lower: number;
          ci_upper: number;
        }
      >
    >;
  };
  pima_indians?: any;
  battery_components?: string[];
}
