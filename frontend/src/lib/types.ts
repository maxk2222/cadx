export interface Project {
  id: string;
  name: string;
  description: string;
  settings: Record<string, unknown>;
  created_at: string;
  updated_at: string;
  version_count: number;
  latest_version_status: VersionStatus | null;
  latest_thumbnail_url: string | null;
}

export interface Asset {
  id: string;
  project_id: string;
  kind: "image" | "reference_step" | "reference_stl" | "reference_dxf";
  filename: string;
  mime: string;
  width: number | null;
  height: number | null;
  caption: string;
  created_at: string;
  url: string;
}

export type VersionStatus =
  | "queued"
  | "running"
  | "waiting_for_input"
  | "succeeded"
  | "needs_review"
  | "failed"
  | "cancelled";

export type GenMode = "auto" | "cad" | "mesh" | "hybrid";

export interface VersionSummary {
  id: string;
  project_id: string;
  parent_id: string | null;
  number: number;
  kind: "generate" | "edit" | "params";
  prompt: string;
  status: VersionStatus;
  score: number | null;
  iterations: number;
  error: string | null;
  gen_mode: GenMode;
  resolved_mode: "cad" | "mesh" | "hybrid" | null;
  created_at: string;
  thumbnail_url: string | null;
}

export interface Artifact {
  id: string;
  kind: string;
  view: string | null;
  path: string;
  size: number;
  url: string;
  meta: Record<string, unknown>;
}

export interface MeshInfo {
  generation?: {
    model?: string;
    backend?: string;
    seed?: number;
    steps?: number | null;
    duration_s?: number;
    triangles?: number | null;
  };
  orientation?: { up?: string; front?: string };
  orientation_source?: string;
  post?: Record<string, unknown>;
  triangles?: number;
  scale_factor?: number;
  solid_ok?: boolean;
  step_tessellated?: boolean;
  bbox_size?: number[];
  best_iteration?: number;
  dim_issues?: string[];
  reused_solid?: boolean;
  [key: string]: unknown;
}

export interface ParamDef {
  name: string;
  value: number | string | boolean;
  unit: string;
  min: number | null;
  max: number | null;
  description: string;
}

export interface DfmWarning {
  severity: "high" | "medium" | "low";
  code: string;
  message: string;
}

export interface SpecDim {
  value?: number;
  unit?: string;
  source?: "stated" | "inferred" | "assumed" | string;
}

export interface SpecFeature {
  name?: string;
  type?: string;
  description?: string;
  dims?: Record<string, number>;
  position?: string;
  count?: number;
  source?: string;
  standard?: string;
}

export interface OpenQuestion {
  question: string;
  why_it_matters?: string;
  default_assumption?: string;
  blocking?: boolean;
}

export interface PartSpec {
  name?: string;
  summary?: string;
  units?: string;
  length_x?: SpecDim;
  width_y?: SpecDim;
  height_z?: SpecDim;
  orientation_note?: string;
  features?: SpecFeature[];
  symmetry?: string;
  process?: string;
  material?: string;
  tolerances_note?: string;
  assumptions?: string[];
  open_questions?: OpenQuestion[];
  must_keep?: string[];
}

export interface ModelingPlan {
  approach?: string;
  approach_reason?: string;
  mesh_subject?: string;
  mesh_source_images?: string[];
  mesh_up_hint?: string;
  hybrid_cad_features?: string[];
  steps?: string[];
  params?: ParamDef[];
  standard_parts?: string[];
  notes?: string;
}

export interface VersionDetail extends VersionSummary {
  spec: PartSpec | null;
  plan: ModelingPlan | null;
  code: string | null;
  params: Record<string, unknown> | null;
  params_schema: ParamDef[] | null;
  analysis: Record<string, unknown> | null;
  critique: { score: number; passed: boolean; discrepancies: { what: string; where: string; severity: string; fix_hint: string }[] } | null;
  dfm: { warnings: DfmWarning[]; metrics?: Record<string, unknown> } | null;
  mesh: MeshInfo | null;
  usage: Record<string, number>;
  asset_ids: string[];
  artifacts: Artifact[];
}

export interface Job {
  id: string;
  version_id: string;
  type: string;
  status: string;
  step: string | null;
  iteration: number;
  progress: number;
  log: string;
  questions: { question: string; why_it_matters: string; default_assumption: string; blocking: boolean }[] | null;
  cancel_requested: boolean;
  started_at: string | null;
  finished_at: string | null;
  created_at: string;
}

export interface Settings {
  llm_base_url: string;
  llm_model_vision: string;
  llm_model_code: string;
  llm_json_mode: string;
  llm_api_key_set: boolean;
  data_dir: string;
  agent_max_iterations: number;
  agent_max_repairs: number;
  agent_pass_score: number;
  agent_clarification_mode: string;
  exec_timeout_s: number;
  exec_memory_mb: number;
  agent_default_gen_mode: GenMode;
  mesh_base_url: string;
  mesh_model: string;
}

export interface MeshModelInfo {
  id: string;
  name: string;
  device: string;
  backend: string;
  supports_multiview: boolean;
  max_images: number;
  license_note: string;
  loaded: boolean;
  available: boolean;
}

export interface MeshHealth {
  ok: boolean;
  device: string;
  backend: string;
  error: string | null;
  models: MeshModelInfo[];
}
