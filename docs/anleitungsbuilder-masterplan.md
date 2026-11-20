# Masterplan: Anleitungsbuilder (PDF instruction manual → Unity AR guide)

> **Purpose.** A self-contained, implementation-ready plan for a weaker LLM to
> build the "Anleitungsbuilder" feature. Every claim below was verified against
> the code — file paths, symbols, and line numbers are real. Implement phases
> strictly in order (§16). Do not deviate from §3 without recording it in §19.

Reference implementation (READ-ONLY template):
`/Users/maxkoch/Documents/GitKraken/KIARA/kiara_web_suite/backend/ar_guide_generator/ar_guide_agent_v4/`
— cited as `KIARA:v4/<rel path>`.
Target branch: `feature/anleitungsbuilder`

---

## 1. Feature definition

User flow (all inside ONE KIARA chat):

1. User uploads a PDF instruction manual + zero or more CAD part files
   (`.step/.stp/.stl/.glb/.gltf`) as project assets; says "Erstelle eine
   AR-Anleitung".
2. The chat agent calls `start_guide_builder` → spawns a durable `guide` job
   bound to the chat's project.
3. Pipeline missions (each a bounded LLM tool loop):
   `doc → steps → parts → assembly → review(loop) → annotate → export`
4. Step extraction: LLM gets OCR page text AND page images (vision role).
   Each step carries `text` (detailed internal) + `display_text` (AR label).
5. CAD port features (cylindrical interfaces) extracted from uploaded parts.
6. Missing parts generated via the existing `spawn_generate` codegen job.
7. AssemblyAgent places parts step-by-step via a deterministic solver;
   pyvista renders go back to the VLM for verification.
8. ReviewAgent scores steps+assembly vs PDF evidence; bounded fix loop.
9. `unity_guide.json` + GLBs + renders exported; attached to the guide
   version as downloadable artifacts.
10. The agent asks clarifying questions via `waiting_for_input` ↔
    `answer_questions`; user mid-run messages land in `state.user_notes`.

Hard requirements: all decisions via tool calls; transforms ONLY from the
solver; state file rewritten atomically after every tool; resume via
`missions_done`.

---

## 2. Reuse map

| Capability | Status | Source / action |
|---|---|---|
| Tool loop (budgets, compaction, `_images` attach) | **exists** — `agent/toolloop.py` `run_session`, `llm/client.py` | reuse |
| Chat orchestration + `answer_questions` | **exists** — `agent/chat_worker.py`, `chat_tools.py`, `spawn.enqueue_resume` | reuse, one generalization (§5) |
| PDF → page images + text | missing | port `KIARA:pdf_data_extraction/pdf_img_extraction/pdf_to_images.py` |
| Typed blackboard state | missing | port `KIARA:v4/state/models.py` → `guide/state.py` |
| Step tools (dual text) | missing | port `KIARA:v4/tools/step_tools.py` |
| CAD port extraction | partial — `cad/features.py` hole/boss recognition | reimplement slim (`guide/cadmeta.py` + executor task §8a) |
| Placement solver (Kabsch, multi-attach, free/env) | missing | port `KIARA:ar_guide_generation_v3/phases/p5_placement/solver.py` + `geometry.py` |
| Collision check | **dep exists** — `manifold3d` via `cad/mesh_boolean.py` | implement boolean-volume check |
| Scene render → VLM | **render engine exists** — `cad/render.py` pyvista offscreen (executor only) | add `render_scene()` + `render_assembly` exec task |
| Review agent + fix loop | missing | port `KIARA:v4/agents/review_agent.py` + orchestrator loop |
| `unity_guide.json` schema | missing | port `KIARA:v4/tools/output_tools.py`, `common/guide_object_builders.py` — **structure only, NOT the transform math** (unit/frame differs, §11) |
| Missing-part generation | **exists** — `spawn_generate` → worker codegen → STEP/STL | wire via `generate_missing_part` tool |
| Scratchpad `plan.md` | missing | port `KIARA:v4/tools/plan_tools.py` (4 tiny tools) |

Non-goals v1: Unity app code, in-browser AR preview, instruction editor UI,
multi-pipeline selection, non-GLB CAD passthrough.

---

## 3. Verified platform facts (do NOT re-derive — trust these)

- **Jobs**: `Job.version_id` is a **NOT NULL FK** (`db/models.py:234`) and
  `Job.type` is a **native PG enum** `jobtype` (`generate, edit, params,
  reexport, resume, optimize, resample, refine, mesh_edit, fem, organic,
  code, meshgen`). `VersionKind`, `AssetKind`, `JobStatus` are PG enums too.
  → New members need an **Alembic migration** with
  `ALTER TYPE <enum> ADD VALUE IF NOT EXISTS '<v>'` — pattern:
  `backend/alembic/versions/i9c0d1e2f3a4_editor_enums.py`.
  **There is no free-form job kind.**
- **Every job needs a Version.** Guide runs therefore get a companion
  `Part("AR-Guide …")` + `Version(kind="guide")` — this also gives us the
  artifact attachment point (`Artifact.version_id`) and makes `enqueue_resume`
  work unchanged (it requires `job.version_id`).
- **`enqueue_resume`** (`spawn.py:1160`) creates `Job(type=resume,
  resume_state=job.resume_state)`, marks the parked job `succeeded`, and
  **hardcodes `func="run_agent_job"`** → must be generalized (§5.4).
- **`Job.payload` = `payload_envelope(func, args)`** (`api/deps.py:86`) —
  `{func, args}`, JSON-safe or NULL; the sweeper replays it verbatim. Guide
  payloads MUST include `"job_id"` (also drives the deterministic arq dedup
  key, `deps.py:52-75`).
- **Queues**: agent worker `AgentWorkerSettings` (`queue.py:430`),
  `max_jobs=8` (`KIARA_AGENT_MAX_JOBS`), `job_timeout=
  agent_job_timeout_s=14400` (4h). A guide job waiting on a child
  `generate` job does NOT deadlock (8 slots) — but **document
  `KIARA_AGENT_MAX_JOBS≥2`** as a hard requirement.
- **Executor dispatch**: `QueueExecutor._run_task(func, payload, timeout_s)`
  (`agent/executor_client.py:317`); func names in `_LIGHT_FUNCS`
  (`executor_client.py:41`) route to `cad-light` (2 slots), else `cad`.
  New light tasks = add func to `tasks` module + `CadLightWorkerSettings.
  functions` (`queue.py:534`) + `_LIGHT_FUNCS` + a `QueueExecutor` facade
  method. Registration also needs `queue.py` import line.
- **Rendering**: pyvista offscreen works **only in the executor** —
  `cad/render.py` `render_available()` probe, `render_views()` used from
  `cad/tasks.py:1477`. The worker container must be assumed GL-less →
  ALL guide renders go through a `render_assembly` exec task.
- **LLM roles**: `Role = Literal["vision","code"]` (`llm/client.py:46`).
  Chat resolves `runtime.chat_model_role` (`runtime_config.py:210`,
  default `"vision"`). Guide missions default `role="vision"` (page images,
  renders); fall back cleanly when the vision model is absent
  (`ctx.image_available`-style check — mirror `chat_worker.py:417-418`).
- **Tool result images**: return `{"_images": [path, …]}` — `run_session`
  moves them into vision message parts (`toolloop.py:607-630`). The key is
  `_images` (underscore), NOT `images`. OpenAI tool messages can't carry
  images, so they land on the NEXT user/tool message — design tool results
  expecting a one-call delay.
- **`ToolSession`** (`toolspecs.py:1901`): `kind` is a closed `Literal` —
  use the `ProjectToolSession` trick: `super().__init__(ctx, kind="info")`
  then `self.kind = "guide"` (`project_tools.py:512-513`). Dispatch is
  `getattr(self, f"_tool_{name}")` (`_dispatch`, `project_tools.py:601`).
  `tools()` returns the mission-scoped schema list; `tool_log` and
  `submitted` are managed by the base/loop.
- **`run_session`** signature (`toolloop.py:653`):
  `run_session(ctx, role, messages, *, session, json_schema=None, max_calls,
  step, iteration=0, …)` — same call shape `chat_worker` uses minus
  `on_delta`/`drain_fn`.
- **`wait_job_terminal`** (`spawn.py:1333`) polls a job to
  `TERMINAL_JOB_STATES` (incl. `waiting_for_input`) with cancel/interrupt
  checks — reuse for `generate_missing_part` child waits.
- **`spawn_generate`** (`spawn.py:177`):
  `spawn_generate(db, project, prompt, gen_mode, part_id, part_name,
  asset_ids, enqueue_fn) -> (Version, Job)`.
- **Prompts**: `read_prompt(name)` / `system_prompt(name)`
  (`agent/context.py:23-28`) load `agent/prompts/*.md`. AGENTS rule: prompts
  NEVER inline in Python.
- **Storage**: all paths via `kiara/storage.py` — `project_dir`,
  `asset_path`, `version_dir` (`storage.py:53-69`), `resolve`/`relpath`.
- **`emit`** callback signature (worker_jobs.py): `await emit(step_name,
  progress_int_or_float, label, sub_label=…)`; job progress surface is
  `job.step` (str) + `job.progress` (float) + `job.log`.
- **Uploads**: `assets.py` `_REF_EXT` map (`:26-30`) → `reference_step/
  reference_stl/reference_dxf`; images via `_IMAGE_EXT`;
  `MAX_ASSET_BYTES = 25MB` is a single global cap.
- **Frontend composer**: `frontend/src/components/chat/Composer.tsx:132`
  has `accept="image/*"`.
- **Units**: KIARA is mm, Z-up RH, X=length/Y=width/Z=height (AGENTS frame
  contract). KIARA assembly space is **cm, Y-up RH** — do NOT copy its
  transform math blindly (§11 has the verified KIARA→Unity formula).
- **`waiting_for_input` is terminal** in `TERMINAL_JOB_STATES` — a child
  `generate` job that parks on questions returns to the waiting caller;
  the guide mission must forward that as `ask_user`.
- **`ProjectSettings.eval_mode`** suppresses `waiting_for_input` parking;
  `clarification_mode: "assume"|"ask"` is the per-project knob the
  `ask_user` tool MUST honor (`assume` → pick the documented default,
  record it in state, never park).

---

## 4. Data model — `backend/kiara/guide/state.py`

Port KIARA `state/models.py` field-for-field where noted; pydantic v2.

```python
class PortInfo(BaseModel):          # KIARA PortInfo
    name: str                        # "port_0"
    pos: list[float]                 # mm, part-local
    axis: list[float]                # unit, outward normal
    radius: float                    # mm
    gender: str                      # male | female | flat | unknown
    kind: str = "cylinder"           # cylinder | plane | slot
    depth: float = 0.0               # female ports: usable depth mm

class GuidePart(BaseModel):          # KIARA PartInfo
    id: str                          # slug
    name: str
    source: str                      # uploaded | generated
    asset_id: str | None = None
    version_id: str | None = None    # generated Version id
    mesh_path: str | None = None     # normalized GLB (mm) for scene/solver
    step_path: str | None = None
    ports: list[PortInfo] = []
    bbox_min: list[float] = [0, 0, 0]
    bbox_max: list[float] = [0, 0, 0]

class GuideSubStep(BaseModel):       # KIARA SubStep
    action: str                      # place | attach | orient | insert | fasten | check
    part: str                        # GuidePart.id or free label
    target: str = ""
    text: str = ""
    side: str = ""

class GuideStep(BaseModel):          # KIARA GuideStepDraft
    number: int
    title: str
    text: str                        # INTERNAL detail (drives tools)
    display_text: str                # AR-facing one/two-liner
    action_type: str = "assembly"    # assembly | check | safety | prep
    required_parts: list[str] = []
    tools: list[str] = []
    safety_hint: str = ""
    media_image: str | None = None
    source_pages: list[int] = []
    sub_steps: list[GuideSubStep] = []
    parts_delta: list[str] = []      # parts newly placed this step

class PageSummary(BaseModel):        # KIARA PageSummary
    page: int
    image_path: str
    text: str
    classification: str = "unknown"  # cover | toc | steps | parts_list | appendix
    figures: list[dict] = []
    vlm_summary: str = ""

class DocumentIndex(BaseModel):      # KIARA DocumentIndex
    pdf_path: str
    pdf_asset_id: str
    dpi: int = 200
    page_count: int = 0
    language: str = "de"
    doc_type: str = "assembly_manual"
    pages: list[PageSummary] = []
    detected_part_names: list[str] = []

class AssemblyInstance(BaseModel):   # KIARA AssemblyInstance
    part_id: str
    transform: list[list[float]]     # 4x4 row-major, mm, CAD frame
    placed_in_step: int
    substep_idx: int = 0

class AssemblyState(BaseModel):      # KIARA AssemblyState
    instances: list[AssemblyInstance] = []
    joints: list[dict] = []          # [{a_part,a_port,b_part,b_port,residual_mm}]
    step_instances: dict[str, list[str]] = {}
    status: str = "empty"            # empty | partial | complete

class ReviewFinding(BaseModel):
    severity: str                    # blocker | warning | note
    mission: str                     # mission that must fix it
    step: int | None = None
    text: str
    evidence: str = ""

class ReviewReport(BaseModel):       # KIARA ReviewReport
    score: float = 0.0               # 0..10
    findings: list[ReviewFinding] = []
    geometry_ok: bool | None = None
    iteration: int = 0

class GuideState(BaseModel):         # KIARA Blackboard
    project_id: str
    guide_version_id: str            # state dir key — NOT job_id (§5.5)
    language: str = "de"
    user_notes: list[str] = []
    pending_answers: list[str] = []  # answers injected on resume
    document: DocumentIndex | None = None
    steps: list[GuideStep] = []
    parts: list[GuidePart] = []
    assembly: AssemblyState = AssemblyState()
    review: ReviewReport = ReviewReport()
    annotations: list[dict] = []
    missions_done: list[str] = []
    missions_failed: list[str] = []
    output_dir: str = ""
    unity_guide_path: str | None = None
```

`save()`: write `state.json.tmp` → `os.replace` (atomic). `load(dir)`:
returns None if absent. Every state-changing tool calls `save()` before
returning — never leave mutations in memory only.

---

## 5. Job plumbing (corrected vs v1 — read §3 first)

### 5.1 Alembic migration

`backend/alembic/versions/<rev>_guide_enums.py` — one migration, pattern
copied from `i9c0d1e2f3a4_editor_enums.py`:

```python
op.execute("ALTER TYPE jobtype ADD VALUE IF NOT EXISTS 'guide'")
op.execute("ALTER TYPE versionkind ADD VALUE IF NOT EXISTS 'guide'")
op.execute("ALTER TYPE assetkind ADD VALUE IF NOT EXISTS 'pdf'")
op.execute("ALTER TYPE assetkind ADD VALUE IF NOT EXISTS 'cad_glb'")
```

Then in `db/models.py`: `JobType.guide`, `VersionKind.guide`,
`AssetKind.pdf`, `AssetKind.cad_glb`.

### 5.2 Upload — `api/routers/assets.py`

- `_REF_EXT` += `{".glb": cad_glb, ".gltf": cad_glb, ".pdf": pdf}`
  (`.step/.stp`→`reference_step`, `.stl`→`reference_stl` already exist).
- Replace the single `MAX_ASSET_BYTES` check with per-kind caps:
  pdf 100 MB, cad 50 MB, image 25 MB (keep global default 25 MB).
- PDF magic-byte check `%PDF` in the upload handler (not in storage).
- Frontend: `Composer.tsx:132` `accept="image/*"` →
  `image/*,.pdf,.step,.stp,.stl,.glb,.gltf,.dxf`; `AssetManager` accept
  list likewise (grep `accept=`).

### 5.3 `spawn_guide` — `kiara/spawn.py`

```python
async def spawn_guide(db, project, pdf_asset_id, cad_asset_ids, chat_id) -> tuple[Version, Job]
```

1. `Part` (name `"AR-Guide {pdf filename stem}"`) + `Version`
   (`kind=guide`, `status=queued`, `prompt` = user request text) in the
   chat's project.
2. `Job(version_id=v.id, type=JobType.guide,
   resume_state={"guide_dir": str(guide_dir)})` where
   `guide_dir = storage.version_dir(project.id, v.id) / "guide"`.
3. `job.payload = payload_envelope("run_guide_job",
   {"job_id": str(job.id), "version_id": str(v.id)})` — REQUIRED or the
   sweeper can't recover it.
4. `enqueue("run_guide_job", args)` via `api/deps.enqueue` (deterministic
   arq id from `payload["job_id"]`).
5. Guard: refuse when a `type=guide` job for the same version is
   `queued|running|waiting_for_input` → return the existing job instead.

### 5.4 `enqueue_resume` generalization — `spawn.py`

Current signature hardcodes `func="run_agent_job"` and copies
`resume_state` verbatim. Change to:

```python
async def enqueue_resume(db, job, answers_text, *, func=None, resume_extra=None, enqueue_fn=None):
    func = func or (job.payload or {}).get("func") or "run_agent_job"
    resume_state = {**(job.resume_state or {}), **(resume_extra or {}),
                    "answers_text": answers_text}
    # ... rest identical: Job(version_id=v.id, type=resume,
    #     resume_state=resume_state), payload_envelope(func, args)
```

`jobs.py:213` + `chat_tools._tool_answer_questions` keep calling it with
defaults — reading `func` from `job.payload["func"]` makes resume
automatically re-enqueue the right task for ANY parked job type.

### 5.5 `run_guide_job` — `agent/guide_runner.py`

```python
MISSIONS = ["doc", "steps", "parts", "assembly", "review", "annotate", "export"]

async def run_guide_job(ctx: dict, payload: dict) -> dict:
    job, version, project = load via session_scope
    guide_dir = Path(job.resume_state["guide_dir"])  # key: version, not job
    state = GuideState.load(guide_dir) or GuideState(project_id=…,
            guide_version_id=str(version.id), output_dir=str(guide_dir))
    if answers := (job.resume_state or {}).get("answers_text"):
        state.pending_answers.append(answers); state.save()
    executor = QueueExecutor(ctx["arq_redis"], runtime_settings=…, emit_cb=emit)

    for mission in MISSIONS:
        if mission in state.missions_done: continue
        if mission == "review":
            await _review_loop(ctx, state, emit, executor); continue
        await _run_mission(ctx, job, state, mission, executor, emit)
        if check_cancel(): raise CancelledError-ish path (mirror worker_jobs)
        if job.status == waiting: return early  # ask_user parked us
    return {"ok": not state.missions_failed, "unity_guide": state.unity_guide_path}
```

`_run_mission` = build `GuideToolSession` + `AgentCtx` (`mode="guide"`,
`version_id=guide_version.id`, `job_id=str(job.id)`) + seed messages →
`run_session(ctx, role, msgs, session=session, max_calls=BUDGET[mission],
step=f"guide:{mission}")`; wrap in `asyncio.timeout(guide_mission_timeout_s)`;
on success the `done_mission` tool already recorded the mission in
`missions_done`; verify before proceeding (`missions_failed` else).

Progress: `job.step = f"guide:{mission}"`, `job.progress = done/len(MISSIONS)`;
`emit` at mission boundaries + after each `place_instance`.

### 5.6 `ask_user` mechanics

`ask_user(question)` tool → `job.questions=[{text}]`,
`job.status=waiting_for_input`, `job.save`, mission returns sentinel
`{"parked": True}` → runner returns early. Chat `answer_questions(job_id,
answers)` → `enqueue_resume` → new `run_guide_job` whose
`resume_state.answers_text` lands in `state.pending_answers` → missions
read it at start. The tool docstring MUST say the job suspends — the LLM
should batch related questions in one call.

### 5.7 `GuideToolSession` — `guide/session.py`

```python
class GuideToolSession(ToolSession):
    def __init__(self, ctx, *, mission, state, state_dir, executor, publish_fn=None):
        super().__init__(ctx, kind="info")          # Literal placeholder (§3)
        self.kind = "guide"
        self.mission = mission
        self.state = state; self.state_dir = state_dir
        self.executor = executor
        self.attach_images = role == "vision"        # passed in by runner
        self.image_left = settings.guide_image_budget
        self._preview_cache: dict[str, dict] = {}    # candidate_id → transform

    def tools(self):
        return [SCHEMAS[n] for n in MISSION_TOOLS[self.mission]]

    async def _dispatch(self, name, args):           # copy ProjectToolSession
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None: return {"ok": False, "error": f"unknown tool '{name}'"}
        res = handler(args)
        res = await res if inspect.iscoroutinefunction(handler) else res
        if name in STATE_CHANGING: self.state.save()  # safety net — handlers save too
        return res
```

Tool schemas live in `guide/tools.py` as a `MISSION_TOOLS: dict[str,
list[str]]` + `SCHEMAS: dict[str, dict]` registry (OpenAI
`{"type":"function","function":{name,description,parameters}}` shape —
same `_fn` helper style as `project_tools.py`).

---

## 6. Mission `doc` (DocAnalyst)

Files:

- `guide/pdfdoc.py` — port `KIARA:pdf_data_extraction/pdf_img_extraction/
  pdf_to_images.py`:
  ```python
  def render_pdf(pdf_path, out_dir, dpi=200) -> list[Path]   # pypdfium2→PyMuPDF→pdf2image
  def extract_page_text(pdf_path) -> list[str]               # pypdf per page
  def extract_figures(page, pdf_path, out_dir) -> list[dict] # port KIARA crop logic; [] ok v1
  ```
  Deps: add `pypdfium2` (+`PyMuPDF` optional) to `backend/pyproject.toml`.
- `guide/doc_tools.py` — session methods:
  `prepare_pdf` (render+text → `state.document`), `detect_language`,
  `get_page_text(page)` (≤2k chars, truncation note), `get_page_image(page)`
  → `{"_images":[png]}`, `list_figures(page?)`, `classify_pages(labels)`,
  `set_detected_part_names(names)`, `done_mission`.
- `agent/prompts/guide_doc.md` — port KIARA DocAnalyst prompt: owns
  `document` only; `prepare_pdf` FIRST; `done_mission` LAST; classify all
  pages; extract part names from parts-list pages + step headers.

Seeding: after `prepare_pdf`, attach the first ≤4 step-classified page
images to the seed message; agent pulls more via `get_page_image`
(`guide_max_pages` cap = 80, truncate + warn).

Acceptance: `tests/guide/test_pdfdoc.py` — fixture PDF (copy
`Montageanleitung_Demonstrator_KIARA_v2.pdf` from KIARA uploads →
`tests/fixtures/guide/`), page_count>0, text non-empty, images exist,
state round-trip.

---

## 7. Mission `steps` (StepAuthor)

`guide/step_tools.py` — port `KIARA:tools/step_tools.py`:
`create_step(number,title,text,display_text,action_type,required_parts,
tools,safety_hint,source_pages)`, `update_step`, `add_sub_step`,
`assign_media(step, page, figure_idx?)` (crop PNG → `media_image`),
`reorder_steps(order)`, `finalize_steps()` (renumber 1..N, require ≥1
step, every `required_parts` non-empty for action_type=assembly),
`done_mission`.

`agent/prompts/guide_steps.md` — port StepAuthor prompt; MUST carry the
contract verbatim:

> `text` = vollständige interne Schrittbeschreibung für die
> Weiterverarbeitung (parts, tools, spatial hints, tolerances).
> `display_text` = kurze Anweisung für den AR-Nutzer (≤140 Zeichen,
> imperativ, Sprache des Manuals).

Every step requires `source_pages`; sub-steps mandatory when a step places
parts; `finalize_steps` last.

Context: all page texts injected (truncated); `get_page_image` on demand.
`pending_answers` injected at mission start.

Acceptance: `test_step_tools.py` validations + media crop; e2e on fixture:
≥3 steps, both texts non-empty, `display_text` ≪ `text`.

---

## 8. Mission `parts` (inventory + generation)

### 8a. Feature extraction — `guide/cadmeta.py` + executor task

Do NOT port KIARA's 1.9k-line `cad_preprocessing_router.py`. Slim
extractor producing `PortInfo`:

- **STEP input** → executor (build123d lives there, sandbox boundary):
  new arq func `extract_ports` in `cad/guide_exec.py` (imported by
  `queue.py`, added to `CadLightWorkerSettings.functions` +
  `_LIGHT_FUNCS`): import STEP → iterate BRep faces → `Cylinder` faces →
  port (`pos`=axis point at outer extreme, `axis`, `radius`, `gender`
  from hole-vs-boss heuristic — reuse `cad/features.py` classification);
  large planar faces → `flat` ports (area ≥ 4% bbox cross-section).
  Returns `{ports, bbox, mesh_glb_path}` (tessellates + writes GLB to a
  DATA_DIR path the worker reads back).
- **Mesh input** (`.glb/.stl`) → in-process trimesh: bbox + cylinder
  detection via `cad/features.py` mesh path; fallback = 6 bbox-face
  `flat` ports so placement can always proceed.
- `QueueExecutor.extract_ports(payload, timeout_s)` facade method added
  next to `export_all` etc. (`executor_client.py`).

### 8b. Tools — `guide/part_tools.py`

- `register_uploaded_part(asset_id, name)` → normalize+extract → `parts[]`.
- `match_required_parts()` → deterministic token-overlap scorer (NOT LLM):
  normalize (`lower`, strip `-_.,()[]`, split camel/digits); returns
  `{matched, missing, ambiguous}`. `alias_part(required_name, part_id)`
  for LLM-resolved fuzz ("Rohr 500" ↔ `Pipe_l500.step`).
- `generate_missing_part(required_name, spec_text, reference_pages)` →
  `spawn_generate(db, project, prompt, gen_mode="cad", part_name=name)`
  + `wait_job_terminal` → register returned Version STEP via
  `extract_ports`; source="generated". Child job parking on
  `waiting_for_input` → return its questions to the mission (it calls
  `ask_user`).
- `ask_user(question)` — §5.6. Honors `clarification_mode`/`eval_mode`.
- `list_user_notes()` → `state.user_notes` + `state.pending_answers`.
- `done_mission` — refuses while unresolved `missing` remain (must be
  generated, aliased, or user-acknowledged).

`agent/prompts/guide_parts.md`: inventory → match → resolve missing via
generate or `ask_user`; NEVER guess a critical dimension the manual
doesn't state — ask.

Acceptance: `test_part_matching.py` (fuzzy cases), `test_cadmeta.py`
(STEP fixture → expected port count/axis). `guide_gen_missing_max=8` cap.

---

## 9. Mission `assembly` (AssemblyAgent)

- `guide/solver.py` — port `KIARA:p5_placement/solver.py` (+ `geometry.py`
  helpers). **Unit port, not blind copy**: KIARA is cm/Y-up — ours is
  mm/Z-up. Keep every input/output in mm/CAD-frame; if KIARA's code
  normalizes via `model_scale`, re-derive for mm (its `model_scale` only
  converts for rendering — check `geometry.py:model_scale`).
  ```python
  def solve_attachment(a_part, a_port, b_part, b_port, clearance=0.0) -> list[list[float]]  # 4x4 candidates
  def solve_multi_attachment(moving, pairs) -> list[float] | None   # Kabsch
  def solve_free_placement(part, placed, env_hint) -> list[float]
  def solve_env_placement(part, placed, step_text) -> list[float]
  ```
- `guide/scene.py` — trimesh scene: load `mesh_path` once per part,
  instance = mesh+transform. `collisions()` → manifold3d boolean
  intersection volume (`cad/mesh_boolean.py` proves the dep works):
  `{pairs:[{a,b,intersection_volume_mm3,overlap_bbox}]}`, AABB pre-filter.
  **No FCL dependency** — do NOT use `trimesh.collision.CollisionManager`.
- `cad/render.py` — add `render_scene(meshes=[(path, matrix4x4, rgba)],
  out_path, view="iso", highlight_idx=None)` reusing `_camera_for`,
  `_load_for_render`, single-Plotter pattern from `render_views`.
- `cad/guide_exec.py` — `render_assembly(ctx, payload)` arq func:
  payload `{meshes, view, out_path}` → `render_scene` → returns path.
  Register per §3 (queue functions + `_LIGHT_FUNCS` + facade).
- `guide/assembly_tools.py` — session methods (port KIARA `cad_tools.py`
  tool *semantics*, not its 4342 lines):
  `list_parts()`, `get_part_features(part_id)`,
  `find_matching_ports(part_id)` (deterministic radius/gender/axis
  scoring → ranked JSON),
  `solve_attachment_preview(moving_part, moving_port, fixed_part,
  fixed_port)` → `{candidate_id, collision_report}` (transforms cached
  server-side — **the LLM never sees the matrix**),
  `place_instance(part_id, candidate_id, step, substep)` → commits
  instance+joints+`step_instances` → `state.save()`,
  `rotate_instance(part_id, euler_deg)` (escape hatch, logged),
  `render_assembly(view?, highlight_part?)` → exec task →
  `{"_images":[png]}`,
  `check_collisions()`, `validate_assembly_geometry()` (joint residuals
  <1mm, no collision vol >1mm³, all required placed),
  `validate_assembly_completeness()` (each `required_parts` placed once,
  ascending step order), `reset_step(step)` (fix-loop support),
  `done_mission`.
- `agent/prompts/guide_assembly.md` — port KIARA AssemblyAgent prompt
  verbatim where applicable: solver output is a CANDIDATE — verify via
  `render_assembly` + compare against `media_image`/PDF figure BEFORE
  `place_instance`. Never invent transforms.

Acceptance: `test_solver.py` — two-cylinder mate (axis flip, pos
coincidence <1e-6, depth offset), Kabsch 2-point fixture, "on top" free
placement. `test_scene.py` — two overlapping boxes report positive
intersection volume; `render_scene` produces PNG in executor (Docker
smoke check).

---

## 10. Mission `review` (ReviewAgent + fix loop)

- `guide/review_tools.py` — port `KIARA:agents/review_agent.py` tools:
  read-only on steps/assembly + `write_review_report(score, findings)` +
  plan.md tools (`read_plan/append_plan/write_plan/clear_plan` — port
  `KIARA:v4/tools/plan_tools.py`; file = `state_dir/plan.md`, ≤200KB).
- `agent/prompts/guide_review.md` — port prompt + scoring rubric;
  findings cite `source_pages` evidence; NEVER edit state.
- Runner loop (`guide_runner.py::_review_loop`) — deterministic, NOT in
  the LLM:
  ```python
  for it in range(1, guide_review_max_iters + 1):          # default 3
      _run_mission("review"); state.review.iteration = it
      if state.review.score >= guide_review_pass: break    # default 8.0
      blockers = [f for f in findings if f.severity == "blocker"]
      if not blockers and it >= 1: break
      if same_finding_texts_as_prev_iter: break            # stuck detect
      for step in {f.step for f in blockers}: reset_step(step)
      _run_mission("assembly")                             # targeted fix
  ```
  Loop exit without pass → `missions_failed += ["review"]` but continue
  to annotate/export (soft fail — a 6.5 guide still ships, flagged).

---

## 11. Missions `annotate` + `export` (Unity output)

- `guide/annotate_tools.py` — port KIARA AnnotationAgent essentials:
  `add_arrow(step, from_part|pos, to_part|pos, style)` → `annotations`.
  Small mission (`max_calls ≈ 3×steps`).
- `guide/unity.py` — port `output_tools.py`/`guide_object_builders.py`
  **schema + prefab conventions**, NOT their transform math.
  **Verified KIARA→Unity conversion** (mm/Z-up/RH → m/Y-up/LH):
  ```
  pos_u   = (cad_x, cad_z, cad_y) * 0.001        # x right, z up→y up, y front→z fwd
  R_u     = M · R_c · Mᵀ ,  M = [[1,0,0],[0,0,1],[0,1,0]]  (det = −1)
  scale_u = cad_scale · 0.001 per axis
  ```
  Prefab names (KIARA compat): models `"P_<slug>(Clone)"`, arrows
  `"P_ArrowStraightLP"`, text `"P_MovableTextFieldWithButton(Clone)"`.
  ```json
  {"format":"unity_guide/v1",
   "steps":[{"number","display_text","text","media_image",
             "objects":[{"prefab_name","type","transform_unity"}],
             "arrows":[...]}],
   "models":[{"part_id","prefab_name","file":"<name>.glb"}],
   "source":{"pdf":"<asset>","generated_by":"kiara-anleitungsbuilder"}}
  ```
  Export dir `guide_dir/export/`: `unity_guide.json`, part GLBs, per-step
  renders, media crops. `validate_unity_guide()` — every referenced file
  exists; every step has ≥1 visible object or media; pydantic
  `UnityGuide` schema check. Register output as `Artifact` rows on the
  **guide version** (unity guide = `ArtifactKind.render`? — check
  ArtifactKind members first; fall back to storing the dir + returning
  paths in job result; do NOT invent a kind without the enum migration).
- `guide_export` prompt for `annotate`; `export` mission is pure code
  (no LLM needed — runner calls `unity.emit_all()` directly).

Acceptance: `test_unity.py` golden file — synthetic state (2 parts, 2
steps, 1 arrow): a +100mm CAD-Z offset lands at Unity y=+0.1; rotation
golden pair verified; schema valid.

---

## 12. Chat integration

`project_tools.py` + `chat_tools.py` — add to `CHAT_ACTION_TOOLS` (all
return `_spawn_and_wait`/`job_status` shapes):

- `start_guide_builder(pdf_asset_id?, cad_asset_ids?, notes?)` →
  `spawn_guide` → `wait=False` always (runs minutes–hours).
- `guide_status(job_id?)` → job progress + `missions_done` + current step.
- `attach_guide_note(text)` → resolves the project's active guide job
  (`type=guide` in queued/running/waiting) → appends to its
  `state.json.user_notes` (or `pending_notes.json` pre-start — runner
  merges on mission start).
- `answer_questions` — unchanged; §5.4 `func` default makes it work for
  guide jobs automatically.

`agent/prompts/chat_system.md` — add paragraph: guide jobs are
long-running background tasks; relay progress via `guide_status`; parked
jobs surface questions → answer with `answer_questions`; user text during
a run → `attach_guide_note`, NEVER restart or duplicate `spawn_guide`.

---

## 13. Frontend (minimal)

- `Composer.tsx:132` — extend `accept` (§5.2).
- Job progress card: `kind="guide"` already renders; add `t()` labels
  for `guide:<mission>` step names in all 9 locale files
  (`frontend/src/i18n/locales/*/common.json`).
- `GuideResultCard` component: on `type=guide` job success → download
  links (`/api/versions/{id}/artifacts/...` pattern — check existing
  artifact download endpoint) for `unity_guide.json` + `export.zip` +
  first renders.

---

## 14. Settings (`runtime_config.py`)

```
guide_model_role: Literal["vision","code"] = "vision"
guide_pdf_dpi=200, guide_max_pages=80, guide_page_text_max_chars=2000
guide_image_budget=24           # _images per mission
guide_doc_budget=12, guide_steps_budget=40, guide_parts_budget=40,
guide_assembly_budget=120, guide_annotate_budget=40  # max_calls per mission
guide_review_max_iters=3, guide_review_pass=8.0
guide_mission_timeout_s=1200    # asyncio.timeout inside the 4h job budget
guide_gen_missing_max=8
guide_render_max_px=1600, guide_render_size=1024
```

Security: PDF `%PDF` magic + page cap; per-kind size caps (§5.2); all
generated code only via the existing sandbox path; no `exec()` of LLM
output in the worker (solver/tools are ours).

---

## 15. Testing

| Layer | File | Covers |
|---|---|---|
| unit | `tests/guide/test_pdfdoc.py` | render/text on fixture (xfail if pypdfium2 absent; Docker must have it) |
| unit | `tests/guide/test_state.py` | save/load atomicity, missions_done resume |
| unit | `tests/guide/test_step_tools.py` | dual-text validation, finalize ordering |
| unit | `tests/guide/test_part_matching.py` | name matcher incl. "Rohr 500"↔`Pipe_l500.step` |
| unit | `tests/guide/test_cadmeta.py` | STEP fixture → ports/bbox; mesh fallback → bbox ports |
| unit | `tests/guide/test_solver.py` | Kabsch, attachment, free placement (mm) |
| unit | `tests/guide/test_scene.py` | collision volume via manifold3d; render_scene PNG |
| unit | `tests/guide/test_review_loop.py` | bounded fix loop, stuck detect |
| unit | `tests/guide/test_unity.py` | schema + frame conversion golden file |
| unit | `tests/guide/test_spawn_resume.py` | guide job spawn, park, resume via generalized enqueue_resume |
| integration | `tests/guide/test_pipeline_smoke.py` | scripted tool-call stubs → full state → unity_guide.json |
| e2e (manual) | `scripts/guide_eval.py` | fixture PDF + 4 CAD files, real LLM → steps>0, assembly complete, unity valid |

Fixtures: copy `Montageanleitung_Demonstrator_KIARA_v2.pdf` + 2–3 CADs
from `KIARA/uploads/project_data/d03b92a9-…/` → `backend/tests/fixtures/guide/`.

Run: `cd backend && ../.venv/bin/uv run pytest tests/guide -x -q`;
`make lint` (ruff, 100 cols, type hints).

---

## 16. Implementation order (do not reorder)

1. **Scaffold**: `guide/` package, `state.py`, `pdfdoc.py`, deps
   (`pypdfium2`), tests for state+pdf. `make test` green.
2. **Plumbing**: Alembic migration + enum members, `_REF_EXT`/caps,
   `spawn_guide`, `run_guide_job` (empty mission runner),
   `enqueue_resume` generalization, `GuideToolSession` skeleton,
   `composer accept`. Job spawns/cancels/resumes cleanly.
3. **doc mission**: `doc_tools` + `guide_doc.md`. Manual run on fixture →
   `state.document` populated.
4. **steps mission**: `step_tools` + `guide_steps.md`. doc+steps produce
   real steps (inspect `state.json`).
5. **Solver+scene**: `solver.py` + `scene.py` + `render_scene` +
   `render_assembly`/`extract_ports` exec tasks + unit tests (no LLM).
6. **parts mission**: `cadmeta.py` + `part_tools` + generate bridge +
   `ask_user` plumbing.
7. **assembly mission**: `assembly_tools` + prompt. Verify render→VLM
   loop with the dev model.
8. **review loop**: `review_tools` + `plan.md` tools + runner loop.
9. **annotate+export**: `annotate_tools`, `unity.py`, export dir,
   Artifact registration.
10. **chat tools**: `start_guide_builder`/`guide_status`/`attach_guide_note`
    + `chat_system.md` paragraph.
11. **frontend**: accept list, mission labels ×9 locales, `GuideResultCard`.
12. **docs**: `docs/anleitungsbuilder.md`, AGENTS.md section,
    `scripts/guide_eval.py`, §19 deviations.

Each step: `make lint && make test` green + one commit.

---

## 17. Pitfalls (verified — port these deliberately)

1. **`Job.version_id` NOT NULL** — guide jobs need a companion
   Part+Version; don't try to make Job.version-less.
2. **Resume spawns a NEW job_id** — state dir keyed by
   `guide_version_id`, never `job_id`; `resume_state["guide_dir"]` is the
   load key.
3. **`enqueue_resume` hardcoded `run_agent_job`** — §5.4 generalization
   is mandatory or `answer_questions` enqueues codegen on a guide job.
4. **Agent queue needs ≥2 slots** (`max_jobs=8` today) — a guide job
   blocking on its child `generate` would deadlock a 1-slot worker;
   assert on startup or document in `.env.example`.
5. **Renders only work executor-side** — worker has no GL; ALL scene
   images via `render_assembly` cad-light task. Never `import pyvista`
   in worker code paths.
6. **`payload` must be JSON-safe + carry `job_id`** — else the sweeper
   fails the job "lost" on restart and dedup breaks (`deps.py:55-86`).
7. **`_images` key, not `images`** — and images land on the NEXT message,
   not inside the tool result (`toolloop.py:607-630`); size render calls
   accordingly (`image_left` budget).
8. **cm vs mm / Y-up vs Z-up** — KIARA is cm/Y-up; we are mm/Z-up. Port
   solver *math* but re-verify every constant (`model_scale` in KIARA is
   a cm→m render factor — meaningless here). The ONLY frame conversion
   lives in `unity.py` (§11 golden test).
9. **`waiting_for_input` IS terminal** — `wait_job_terminal` returns it;
   `generate_missing_part` MUST handle a parked child and forward the
   questions (don't mark missing resolved).
10. **Tool results stay small** — page text ≤2k chars, port lists capped,
    `list_figures` summarized; the `agent_context_budget_frac` compactor
    drops content mid-mission otherwise.
11. **Dual text contract** — `text`≠`display_text` required fields;
    merging them reintroduces KIARA v3's "internal notes shown in AR" bug.
12. **LLM never sees a matrix** — `place_instance` takes `candidate_id`
    from `_preview_cache`, not a transform argument (stricter than
    KIARA on purpose).
13. **`eval_mode`/`clarification_mode`** — `ask_user` must honor
    `assume` (record the assumption in state, don't park) or evalkit-style
    runs hang forever.
14. **`guide_version` shows in the part tree** — `VersionKind.guide`
    versions are containers for guide artifacts; ensure list endpoints
    don't break on `code=None` (they don't — check `Version` nullable
    fields, but smoke it).

---

## 18. Open questions (defaults chosen — override if user disagrees)

1. Unity prefab compatibility: keep KIARA `P_*(Clone)` names (default YES).
2. Generated parts appear as normal project Parts/Versions: YES (via
   `spawn_generate`).
3. AR asset delivery: files in `export/` + Artifact rows + zip (all three).
4. Guide version kind `guide` is user-visible in the part tree — acceptable
   (it IS the deliverable); alternative would be a hidden flag — not worth
   a schema change.

---

## 19. Deviations log

*(append every divergence from this plan during implementation, with why)*

- v1→v2 corrections folded in: JobType/VersionKind/AssetKind are PG enums
  needing a migration (v1 claimed no schema change); `enqueue_resume`
  generalization required; state keyed by version not job; roles are
  vision|code only; KIARA cm/Y-up → KIARA mm/Z-up conversion re-derived
  and golden-tested; renders executor-side only; collision via
  manifold3d (no FCL); frontend accept list + settings file name
  (`runtime_config.py`) corrected.
