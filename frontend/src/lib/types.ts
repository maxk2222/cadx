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
  latest_version_id: string | null;
  running_version_id: string | null;
}

export interface Asset {
  id: string;
  project_id: string;
  kind: "image" | "generated_image" | "reference_step" | "reference_stl" | "reference_dxf";
  filename: string;
  mime: string;
  width: number | null;
  height: number | null;
  caption: string;
  derived_from: string | null;
  /** provenance for generated images (prompt/model/seed/…) */
  gen_params: Record<string, unknown> | null;
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

export type GenMode = "auto" | "cad" | "mesh" | "hybrid" | "topopt";

export type ProcessType = "fdm" | "sla" | "cnc3" | "cnc2d";

/** Per-run overrides from the options panel (merged over project settings). */
export interface RunOptions {
  process?: ProcessType;
  material?: string;
  include_fasteners?: boolean;
  fem?: boolean;
  fem_note?: string;
  topopt?: boolean;
  topopt_volfrac?: number;
  max_iterations?: number;
  max_repairs?: number;
  pass_score?: number;
}

export interface VersionSummary {
  id: string;
  project_id: string;
  parent_id: string | null;
  number: number;
  kind: "generate" | "edit" | "params" | "optimize" | "refine" | "mesh_edit" | "organic";
  prompt: string;
  status: VersionStatus;
  score: number | null;
  iterations: number;
  error: string | null;
  gen_mode: GenMode;
  resolved_mode: "cad" | "mesh" | "hybrid" | "topopt" | null;
  created_at: string;
  thumbnail_url: string | null;
  /** newest non-terminal job on this version — live step/iteration state */
  active_job?: {
    id: string;
    type: string;
    status: string;
    step: string | null;
    iteration: number;
    progress: number;
    max_iterations: number | null;
  } | null;
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
  organic?: OrganicReport;
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
  load_cases?: LoadCase[];
  optimize?: OptimizeSpec | null;
}

export interface RegionSpec {
  kind?:
    | "face"
    | "surface"
    | "area"
    | "cylinder"
    | "box"
    | "hole"
    | "anchor"
    | "bore"
    | "points"
    | "all";
  axis?: string;
  center?: number[];
  diameter_mm?: number | null;
  margin_mm?: number;
  polygon?: number[][];
  x_range?: number[];
  y_range?: number[];
  hole_index?: number | null;
  /** anchor kind filter for anchor/bore regions (hole|thread|insert|bearing_seat). */
  anchor_kind?: string;
  /** fixed regions only: "all" = encastre, "normal" = pin only the axis DOF (roller). */
  fix_mode?: "all" | "normal";
  /** keep regions only: material override inside the region (insert zones). */
  material?: string;
  /** GUI brush: painted surface points [x,y,z] mm + brush radius + avg normal. */
  points?: number[][];
  radius_mm?: number | null;
  normal?: number[];
  description?: string;
}

export interface LoadSpec {
  region: RegionSpec;
  force_n?: number[];
  pressure_mpa?: number | null;
  /** cosine-distributed shaft/bearing load (N) on a bore wall region. */
  bearing_n?: number | null;
  bearing_dir?: number[];
  /** pure couple [mx,my,mz] in N·m. */
  moment_nm?: number[];
  /** bolt pretension along the bore axis (N). */
  preload_n?: number | null;
  /** apply force_n at this point — adds the induced moment to the region. */
  remote_point?: number[];
  /** body acceleration in g units, e.g. [0,0,-1] = gravity. */
  g_load?: number[];
  /** robust TopOpt: sample force directions within this cone (degrees). */
  direction_cone_deg?: number | null;
}

export interface LoadCase {
  name: string;
  fixed: RegionSpec[];
  loads: LoadSpec[];
  /** relative objective weight in TopOpt (default 1). */
  weight?: number;
}

export interface OptimizeSpec {
  enabled?: boolean;
  target_volume_fraction?: number;
  safety_factor_min?: number;
  keep_regions?: RegionSpec[];
  /** regions forced EMPTY (forbidden material). */
  void_regions?: RegionSpec[];
  /** "auto" adds a keep ring around every detected anchor. */
  keep_mode?: "manual" | "auto";
  min_member_mm?: number | null;
  density_threshold?: number | null;
  /** re-optimize an already-optimized version instead of refusing. */
  allow_reopt?: boolean;
  /** auto-enqueue feature finishing (exact bores/threads) after optimize. */
  auto_refine_features?: boolean;
  /** 45° AM support filter — no unsupported overhangs (print_direction). */
  self_supporting?: boolean;
  print_direction?: "+x" | "-x" | "+y" | "-y" | "+z" | "-z";
  /** density filter — heaviside gives crisp 0/1 fields. */
  filter?: "cone" | "heaviside";
  /** coarse 2x-pitch SIMP warm-starts the fine grid (~3x faster). */
  multires?: boolean;
  /** union original keep-region surfaces back during refine (default on). */
  restore_planes?: boolean;
  /** auto-detect dominant planar base faces and restore them after TopOpt. */
  auto_plane_restore?: boolean;
  /** refine: cut real helical thread grooves (experimental, FDM). */
  helical_threads?: boolean;
  /** optimizer: "oc" (default) or "mma" — enables constraint support. */
  solver?: "oc" | "mma";
  /** "minmax" optimizes the worst load case (smooth P-norm, mma only). */
  case_objective?: "weighted" | "minmax";
  /** displacement cap per load case, mm (mma only). */
  max_displacement_mm?: number | null;
  /** von-Mises cap, MPa — P-norm + qp-relaxation (mma only). */
  max_stress_mpa?: number | null;
  /** penal continuation start exponent (< penal ⇒ enabled, e.g. 1.5). */
  penal_init?: number | null;
  /** warn when void gaps below this size survive extraction, mm. */
  min_gap_mm?: number | null;
  note?: string;
}

/* ---------- pseudo-topology "organic" restyle ---------- */

export type OrganicStyle = "gyroid" | "trabecular" | "voronoi" | "branches";

/** OrganicParams — identical to the `organic` mesh-op args and the
 *  POST /versions/{id}/organic request body, so live preview == commit. */
export interface OrganicParams {
  style?: OrganicStyle;
  volfrac?: number;
  seed?: number;
  period_mm?: number;
  /** fraction of the skin the field may carve: 0=solid shell, 1=open skeleton. */
  surface_open?: number;
  /** skin depth carved by surface_open; 0/omitted = auto. */
  shell_mm?: number;
  floor_mm?: number;
  min_member_mm?: number | null;
  min_gap_mm?: number | null;
  density_threshold?: number | null;
  auto_keep?: boolean;
  auto_keep_margin_mm?: number;
  keep_regions?: RegionSpec[];
  void_regions?: RegionSpec[];
  voxel_mm?: number | null;
  max_elements?: number | null;
  smooth_iterations?: number | null;
}

/** MeshInfo.organic — the report stored on an organic child version. */
export interface OrganicReport {
  style?: string;
  seed?: number;
  period_mm?: number;
  volfrac_target?: number;
  volfrac_achieved?: number;
  threshold?: number;
  pitch_mm?: number;
  solid_elements?: number;
  n_components?: number;
  floor_mm?: number;
  /** fraction of the skin carved away (0=solid shell, 1=open skeleton). */
  surface_open?: number;
  shell_mm?: number;
  skin_cells?: number;
  /** share of the original skin that survived — recognizability metric. */
  surface_coverage?: number | null;
  min_member_mm?: number;
  keep_regions?: number;
  void_regions?: number;
  triangles?: number;
  watertight?: boolean;
  warnings?: string[];
  disclaimer?: string;
  density_relpath?: string;
  [key: string]: unknown;
}

/** Measured feature anchor (hole/thread/insert/bearing seat), any axis. */
export interface FeatureAnchor {
  index: number;
  kind: string;
  axis: string;
  x: number;
  y: number;
  z?: number | null;
  dia: number;
  depth_mm?: number | null;
  bolt_size?: string;
  name?: string;
  source?: string;
  /** full 3D position for non-+z anchors (axial coord may be null). */
  pos?: (number | null)[] | null;
}

export interface AnchorsResult {
  ok: boolean;
  anchors: FeatureAnchor[];
}

/** One recorded mesh transform (server-side OPS_REGISTRY). */
export interface MeshOp {
  op: string;
  args: Record<string, unknown>;
}

export interface MeshPreviewResult {
  ok: boolean;
  glb_url: string | null;
  stats: {
    faces?: number;
    watertight?: boolean;
    bbox_size?: number[];
    ops_applied?: number;
    /** 0-based index of the op that failed (error responses). */
    failed_op?: number;
    /** per-op report: faces in/out + watertight (up to and excluding the failed op). */
    report?: {
      op: string;
      args?: Record<string, unknown>;
      faces_in?: number;
      faces_out?: number;
      watertight?: boolean;
      salvaged?: boolean;
      reverted?: boolean;
      /** op-stashed metrics (organic restyle: recognizability + warnings). */
      organic?: {
        surface_coverage?: number | null;
        volfrac_achieved?: number;
        n_components?: number;
        shell_mm?: number;
        warnings?: string[];
      };
    }[];
  };
  error: string | null;
}

export interface FemPreviewResult {
  ok: boolean;
  pitch_mm: number | null;
  dims: number[];
  solid_elements: number;
  cases: {
    case: string;
    fixed_nodes: number;
    loaded_nodes: number;
    total_force_n: number;
    /** resolved applied moment about the support centroid (N·mm). */
    applied_moment_nmm?: number[] | null;
    warnings: string[];
  }[];
  /** measured feature anchors (same table as GET /anchors). */
  anchors?: FeatureAnchor[];
  /** resolved selection on the voxel domain — subsampled cell centres (mm)
   *  per case, for the 3D overlay in the structural editor. */
  cells: {
    case: string;
    fixed: number[][];
    load: number[][];
    keep: number[][];
  }[];
  pngs: string[];
  error: string | null;
}

export interface FemCaseResult {
  name: string;
  max_vm_mpa: number;
  max_disp_mm: number;
  safety_factor: number | null;
  compliance_j: number;
  converged: boolean;
  solver: string;
  /** resultant reaction force at the fixed supports (N) — should balance loads. */
  reactions?: number[] | null;
  residual?: number | null;
  warnings: string[];
  vonmises_png?: string | null;
  deformed_png?: string | null;
}

export interface FemReport {
  material: Record<string, unknown>;
  grid: Record<string, unknown>;
  cases: FemCaseResult[];
  warnings: string[];
  note?: string;
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
  options: RunOptions;
  spec: PartSpec | null;
  plan: ModelingPlan | null;
  code: string | null;
  params: Record<string, unknown> | null;
  params_schema: ParamDef[] | null;
  analysis: Record<string, unknown> | null;
  critique: { score: number; passed: boolean; discrepancies: { what: string; where: string; severity: string; fix_hint: string }[] } | null;
  dfm: { warnings: DfmWarning[]; metrics?: Record<string, unknown> } | null;
  bom?: BomEntry[] | null;
  issues: { source: string; severity: string; what: string; where?: string; fix_hint?: string }[] | null;
  mesh: MeshInfo | null;
  fem: FemReport | null;
  usage: Record<string, number>;
  asset_ids: string[];
  library_part_ids: string[];
  artifacts: Artifact[];
}

/** Structured live progress state persisted on Job.live. */
export interface JobLive {
  max_iterations?: number;
  scores?: { it: number; score: number }[];
  step?: string;
  step_started_ts?: string;
  step_history?: { step: string; it: number; dt_s: number }[];
  sub_progress?: number;
  sub_label?: string;
  tokens?: Record<string, number>;
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
  live: JobLive | null;
  questions: { question: string; why_it_matters: string; default_assumption: string; blocking: boolean }[] | null;
  cancel_requested: boolean;
  finish_requested: boolean;
  started_at: string | null;
  finished_at: string | null;
  created_at: string;
}

/** Job row with version/project context — GET /jobs. */
export interface JobListItem extends Job {
  version_number: number;
  version_status: VersionStatus;
  prompt: string;
  project_id: string;
  project_name: string;
}

/** Available live render views of a version — GET /versions/{id}/live-renders. */
export interface LiveRenders {
  iteration: number | null;
  views: string[];
  mtime: number;
  final: boolean;
}

export interface Settings {
  llm_base_url: string;
  llm_model_vision: string;
  llm_model_code: string;
  llm_json_mode: string;
  llm_api_key_set: boolean;
  llm_temperature_code: number;
  llm_max_tokens: number;
  llm_timeout_s: number;
  llm_stream: boolean;
  llm_idle_timeout_s: number;
  llm_total_timeout_s: number;
  llm_degenerate_stall_s: number;
  llm_extra_body: string;
  llm_tools: "auto" | "on" | "off";
  agent_max_iterations: number;
  agent_max_repairs: number;
  agent_pass_score: number;
  agent_dim_tolerance_pct: number;
  agent_dim_tolerance_mm: number;
  agent_clarification_mode: "assume" | "ask";
  agent_max_input_images: number;
  agent_max_tokens_per_job: number;
  agent_max_wall_s: number;
  agent_default_gen_mode: GenMode;
  agent_param_autofix: boolean;
  agent_patch_mode: "auto" | "always" | "never";
  agent_patch_retries: number;
  agent_feature_feedback: boolean;
  agent_critique_reask: boolean;
  agent_tool_max_calls: number;
  agent_tool_max_calls_info: number;
  agent_tool_verify: boolean;
  agent_tool_verify_max: number;
  agent_mesh_tools: boolean;
  agent_mesh_max_ops: number;
  agent_mesh_preview_max: number;
  agent_mesh_refine_rounds: number;
  agent_mesh_hard_fail_score: number;
  agent_mesh_hard_fail_limit: number;
  chat_model_role: "vision" | "code";
  chat_tool_max_calls: number;
  chat_max_turns: number;
  chat_spawn_timeout_s: number;
  chat_history_messages: number;
  chat_image_max: number;
  fem_voxel_mm: number;
  fem_max_elements: number;
  fem_timeout_s: number;
  topopt_timeout_s: number;
  topopt_max_iterations: number;
  topopt_default_volfrac: number;
  topopt_default_rmin_mm: number;
  mesh_base_url: string;
  mesh_model: string;
  image_model: string;
  image_edit_model: string;
  image_default_strength: number;
  image_steps: number | null;
  image_guidance: number | null;
  image_max_side: number;
  image_max_count: number;
  image_autogen_mesh: boolean;
  image_tool_budget: number;
  image_timeout_s: number;
  mesh_timeout_s: number;
  mesh_stall_timeout_s: number;
  mesh_min_free_mb: number;
  mesh_decimate_faces: number;
  mesh_max_faces_for_cad: number;
  exec_timeout_s: number;
  exec_mem_limit_mb: number;
  data_dir: string;
}

/** PUT /settings payload — all fields optional; key is write-only. */
export interface SettingsUpdate extends Partial<Omit<Settings, "llm_api_key_set" | "data_dir">> {
  llm_api_key?: string;
  clear_llm_api_key?: boolean;
}

export interface ProviderTestResult {
  ok: boolean;
  vision_ok: boolean;
  latency_ms: number;
  error: string | null;
  vision_error?: string;
}

export interface MeshModelInfo {
  id: string;
  name: string;
  device: string;
  backend: string;
  /** "mesh" | "image" — the sidecar hosts both adapter families */
  kind: string;
  supports_multiview: boolean;
  max_images: number;
  license_note: string;
  loaded: boolean;
  available: boolean;
  last_error: string | null;
  default_size: number | null;
  max_count: number | null;
  /** accepts an init image (img2img / instruction edit) */
  supports_edit: boolean;
  /** edit-only adapter — cannot do text-to-image */
  requires_image: boolean;
}

export interface MeshHealth {
  ok: boolean;
  device: string;
  backend: string;
  error: string | null;
  models: MeshModelInfo[];
}

export interface CatalogEntry {
  kind: "normed" | "parametric";
  category: string;
  doc: string;
  types?: Record<string, string[]>;
  params?: string[];
  holes?: string[];
}

export interface CatalogSearchHit {
  class: string;
  category: string;
  kind: string;
  score: number;
  matched_sizes: string[];
}

export interface BomEntry {
  part_class: string;
  type: string | null;
  size: string | null;
  count: number;
}

export interface LibraryPart {
  id: string;
  name: string;
  kind: string; // step | stl | template
  filename: string;
  caption: string;
  params: Record<string, unknown> | null;
  source_version_id: string | null;
  created_at: string;
  url: string;
}

/* ---------- /stats ---------- */

export type NamedCount = {
  name: string;
  count: number;
};

export interface StatsDayBucket {
  date: string; // YYYY-MM-DD
  versions: number;
  jobs: number;
  tokens: number;
}

export interface ModelStat {
  model: string;
  calls: number;
  prompt_tokens: number;
  completion_tokens: number;
  avg_latency_ms: number;
}

export interface StepStat {
  step: string;
  calls: number;
  tokens: number;
  avg_latency_ms: number | null;
  wall_seconds: number;
}

export interface JobTypeStat {
  type: string;
  count: number;
  failed: number;
  avg_duration_s: number | null;
}

export interface ProjectStat {
  id: string;
  name: string;
  versions: number;
  succeeded: number;
  avg_score: number | null;
  tokens: number;
  last_activity: string | null;
}

export interface Stats {
  days: number;
  generated_at: string;
  totals: {
    projects: number;
    versions: number;
    jobs: number;
    assets: number;
    artifacts: number;
    library_parts: number;
    artifact_bytes: number;
    prompt_tokens: number;
    completion_tokens: number;
    llm_calls: number;
    compute_seconds: number;
  };
  versions_by_status: NamedCount[];
  versions_by_kind: NamedCount[];
  versions_by_mode_requested: NamedCount[];
  versions_by_mode_resolved: NamedCount[];
  jobs_by_status: NamedCount[];
  job_types: JobTypeStat[];
  score: {
    avg: number | null;
    best: number | null;
    pass_rate: number | null;
    buckets: NamedCount[];
  };
  iterations_avg: number | null;
  activity: StatsDayBucket[];
  models: ModelStat[];
  steps: StepStat[];
  top_errors: NamedCount[];
  failed_steps: NamedCount[];
  mesh: {
    generations: number;
    cached: number;
    avg_duration_s: number | null;
    models: NamedCount[];
  };
  artifacts_by_kind: NamedCount[];
  projects: ProjectStat[];
}

/** Chat thread — GET /projects/{pid}/chats. */
export interface Chat {
  id: string;
  project_id: string | null;
  title: string;
  turn_running: boolean;
  created_at: string;
  updated_at: string;
}

/** Chat row with the bound project's name — GET /chats. */
export interface ChatListItem extends Chat {
  project_name: string | null;
}

/** Tool-call record stored in assistant message meta.tool_log. */
export interface ToolLogEntry {
  name: string;
  args?: Record<string, unknown>;
  result?: Record<string, unknown>;
}

export interface SpawnedRef {
  /** tool that spawned the job */
  tool?: string;
  kind?: string;
  job_id?: string;
  version_id?: string;
  version_number?: number;
  status?: string;
  questions?: { question: string }[];
}

export interface ChatMessage {
  id: string;
  chat_id: string;
  role: "user" | "assistant" | "system";
  content: string;
  asset_ids: string[];
  meta: {
    streaming?: boolean;
    tool_log?: ToolLogEntry[];
    spawned?: SpawnedRef[];
    assets?: { id: string; url?: string; caption?: string; kind?: string }[];
    error?: string;
    cancelled?: boolean;
    exhausted?: boolean;
  };
  pending: boolean;
  created_at: string;
}

/** GET /chats/{id} — thread + full message history. */
export interface ChatState {
  chat: Chat;
  messages: ChatMessage[];
}
