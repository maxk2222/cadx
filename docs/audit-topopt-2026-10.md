# TopOpt Audit 2026-10 — Defect-Class Review (benchmark: webcae audit)

**Scope:** `backend/cadx/cad/fem/topopt.py`, `solver.py`, `extract.py`, `analyze.py`,
`regions.py`, `loads.py`, `backend/cadx/cad/engine.py` (`run_topopt_pipeline`
L.981–1465, `run_topopt_resample` L.1468–1648), `backend/cadx/llm/schemas.py`
(`OptimizeSpec` L.274), plus `agent/steps/optimize.py`, `agent/toolspecs.py`,
`cad/fem/voxelize.py` as context. Branch `dev`, READ-ONLY — verified code
findings only, no speculation.

> **Addendum (fix status):** the findings have since been addressed — the audit
> below documents the state at the time:
>
> - **BUG-1/BUG-2 fixed:** `_one_pass` passes `void_cells | ~domain.solid` to
>   `density_to_mesh` (engine.py:~1615) and `density.npz` persists `voids`
>   (engine.py:~1788); the resample consumes the key or re-resolves
>   `void_regions` (engine.py:~1958–1977).
> - **RISK-1 fixed:** `solver=oc` now warns explicitly when MMA constraints
>   are set (topopt.py:~896).
> - **RISK-2 fixed:** `OptimizeSpec` carries `Field(ge/le)` bounds
>   (schemas.py:~300–355).
> - **RISK-3 fixed:** `spn` is 1/N-normalized (`/ n_sol`, topopt.py:~1425).
> - **RISK-4 fixed:** `change_box` shares `last_change` with `_run_mma` — the
>   CG-rtol reads the live value (topopt.py:~787, ~1267–1295).
> - **Item-10 note fixed:** pass-2 renders get the `_p2` suffix
>   (engine.py:~1587), no more overwriting.
> - **Item-11 telemetry implemented:** `res.topopt`/`verification` now carries
>   `compliance_export_j`, `max_disp_export_mm`, `max_vm_export_mpa`,
>   `gray_fraction`, `volfrac_mask`, `solver_used`, `beta_final`, `sf_gap`,
>   `case_sfs_export` etc. (engine.py:~1716–1755); `_run_summary` uses
>   `sf_export` (optimize.py:~61–81).
> - **Open:** item 4c (heuristic objective gradient `oc_filter=True` in the
>   MMA weighted path, topopt.py:~1353) and the remaining notes.

## Findings table

| # | Item | Status | file:line | Finding |
|---|------|--------|-----------|---------|
| 1 | Re-analysis on export geometry | **OK** (notes) | engine.py:1327–1338, 1611–1621 | `verify_exp` runs on `domain_exp = voxelize(opt_mesh)` — NOT on `final_mask`. Both verifications exist (`sf_mask` on the exact mask L.1312–1321, `sf_export` on the re-voxelized export mesh). The retry trigger uses `sf_export`. The webcae defect is avoided. |
| 2 | Signed displacement projection | **OK** | topopt.py:857–871 | The constraint uses `max(|u|.max())` across all cases — magnitude norm, both directions bounded. Adjoint RHS uses `np.sign(u[dof_star])` — correct subgradient of \|u\|. |
| 3 | P-norm without volume/N weighting | **BUG** | topopt.py:889 | `spn = (Σ srel^P)^(1/P)` without 1/N normalization → the constraint is N^(1/P)-scaled, i.e. mesh-dependent (finer grid ⇒ implicitly stricter limit; at P=8 doubling the resolution ≈ +30 %). The gradient is consistent with the unnormalized spn — only the normalization is missing. |
| 4a | Constraint sensitivities `oc_filter=False` | **OK** | topopt.py:845, 869, 904, 819 | Vol/disp/stress constraints and the minmax objective use `oc_filter=False` (true chain rule). |
| 4b | Volume constraint on physical density | **OK** | topopt.py:558–561, 847 | OC bisection evaluates `_physical(cand)` once heaviside/AM is active; MMA `_g_vol` uses `p[solid_idx]` (post-AM physical). Consistent. |
| 4c | Heuristic objective gradient in MMA | **RISK** | topopt.py:828 | `solver=mma` + `case_objective=weighted` feeds the classic OC sensitivity filter (`oc_filter=True`) to nlopt-MMA as the "gradient" — heuristic, not the true derivative; constraints are exact. Mixed consistency (documented, questionable for MMA). The minmax path (L.819) correctly uses `False`. |
| 5 | Degenerate min-mass objective | **N/A** + RISK | topopt.py:309–315, 373–379; engine.py:1147–1149; schemas.py:278 | No mass-minimization path exists — the objective is always compliance (weighted/minmax p-norm), volume is an upper bound. Guards exist (ndesign==0, free==0, "no solvable load case", passive_frac>volfrac warning). **RISK:** `target_volume_fraction`, `density_threshold`, `penal`, `max_iterations` etc. had NO `Field(ge/le)` bounds — a comment claimed 0.1–0.9 but the schema validated nothing (volfrac=0.02 → empty extraction; volfrac>1 → solid part). |
| 6 | Parameter passthrough / boundary freeze | **BUG** (2×) + RISK | see field table below | `void_cells` was NOT passed to `density_to_mesh` in `_one_pass` (engine.py:1287–1295); `density.npz` stored no `voids` key (engine.py:1397–1407) → the resample guard at engine.py:1562 was dead for topopt npz. `solver=oc` silently ignored mma fields. |
| 7 | Stress-path coverage | **OK** (caveat) | solver.py:271–284 (`_B0`); topopt.py:656–694 | Von-Mises evaluated at the element centroid (B at the centre) — consistent in constraint AND verification. No corner recovery → peaks at material boundaries/load introduction are undervalued; covered by the "10–20 % accuracy" disclaimer (analyze.py:49–52). Caveat: `solve()` scales D with a single `E_ref` for all cells — insert material zones get base-E stress (solver.py:271). |
| 8 | AM filter chain rule | **OK** | topopt.py:152–197 | Forward `p=min(x, mean(5 supports))`; backward distributes `g/5` to valid support slots, `own=(x<=m)` consistent subgradient at the kink; top-down accumulation over layers recursively correct. Deposits on air cells are dead (never read) — correct, since their p does not depend on x. |
| 9 | Heaviside beta doubling | **OK** (note) | topopt.py:596–604 (OC), 930–939 (MMA) | OC: stall → beta×2 up to beta_max, warning + `history[].beta`. MMA: restart loop ≤4 restarts, cap beta_max. The objective jump appears implicitly as a history discontinuity — no dedicated metric field. With non-default beta0/beta_max span >4 doublings, MMA stops before beta_max. |
| 10 | sf_export retry picks the better result | **OK** (note) | engine.py:1350–1368 | `if sf_exp2 is not None and sf_exp2 > sf_exp: best = p2` — the strictly better pass wins; `best` bundle (result/opt_mesh/final_mask/verify/verify_export) consistent. `retry` block logs both SFs. Note: FEM renders of both passes wrote the same filenames in `out_dir/fem` — pass-2 renders overwrote pass-1 even when p1 won (render/report mismatch). |
| 11 | Telemetry gaps | see below | engine.py:1376–1395 | See "Telemetry recommendation" section. |
| 12 | Multires verification on the fine domain | **OK** | engine.py:1208–1255, 1312–1342 | The coarse pass only provides the `x0` warmstart; `verify`/`verify_exp` always run on `domain` (fine) / `final_mask` / `voxelize(opt_mesh)`. Note: coarse `pc` (L.1227–1236) omits solver/constraints/penal_init/beta — by design ("cheap"); the fine pass applies them. |

## Detailed findings

### BUG-1: `void_regions` lost during extraction (`_one_pass`)

- `void_cells` is computed (`engine.py:1161`) and only passed to `simp` as
  `passive_void` (`engine.py:1276`) — the density there is RHO_MIN, correct.
- `density_to_mesh` in `_one_pass` (`engine.py:1287–1295`) received **no**
  `void_cells=`. `density_to_mask` (extract.py:217–287) explicitly documents
  that `void_cells` must win against re-growth through `_enforce_min_member`
  (dilation, extract.py:152–177) and `_regularize_mask` (diagonal fill,
  extract.py:40–77).
- Consequences: (a) declared void_regions could be refilled by min-member
  thickening/diagonal fill; (b) `mask |= passive_solid` (extract.py:241) let
  keep/load overlap win over the void region; (c) dilation/regularization
  could grow material **outside** `domain.solid` — the resample side therefore
  ORs `~domain.solid` into the void mask (engine.py:1562, comment:
  "or dilation grows material past the part surface").
- **Fix:** in `_one_pass` pass
  `void_cells=void_cells.reshape(domain.dims) | ~domain.solid` to `density_to_mesh`.

### BUG-2: `density.npz` stored no `voids` → resample guard dead

- `np.savez_compressed` (engine.py:1397–1407) wrote keys: density, mask, solid,
  passive, origin, pitch, dims, flat_bottom — **no `voids`**.
- `run_topopt_resample` reads `void_cells` only when `"voids" in npz`
  (engine.py:1561–1563) → for topopt-produced npz `void_cells=None` → no void
  protection AND no `~domain.solid` guard in the resample. `opt.void_regions`
  was not re-resolved in the resample (only `keep_list`/`passive`,
  engine.py:1539–1544).
- **Fix:** include `voids=void_cells.reshape(domain.dims)` in the npz dump
  (or re-resolve `void_regions` in the resample via `keep_element_mask` and
  OR with `~domain.solid`).

### RISK-1: `solver=oc` silently swallowed MMA constraints

- `max_displacement_mm`, `max_stress_mpa`, `case_objective=minmax` were passed
  through to `TopOptParams` (engine.py:1180–1183) but the OC loop
  (topopt.py:527–633) never referenced them. No warning path like for
  `minmax`-with-1-case (topopt.py:744) or the MMA fallback (topopt.py:505).
- The agent could set `max_stress_mpa` via the tool without `solver="mma"` →
  the result violated the requested constraint without any notice.
- **Fix:** in `simp` (or engine) warn `"solver=oc ignores
  max_displacement_mm/max_stress_mpa/case_objective=minmax"` (or auto-upgrade
  to mma when constraints are set).

### RISK-2: Missing schema bounds on OptimizeSpec

- schemas.py:278–311: `target_volume_fraction` (comment "0.1..0.9"),
  `density_threshold` ("0.2..0.9"), `penal`, `penal_init`, `max_iterations`,
  `min_member_mm`, `min_gap_mm`, `max_displacement_mm`, `max_stress_mpa` — no
  `Field(ge=…, le=…)` constraints. LLM output reached the solver unvalidated.
- **Fix:** add bounds (e.g. `volfrac ∈ [0.05, 0.95]`, `threshold ∈ [0.2, 0.9]`).

### RISK-3: P-norm stress constraint mesh-dependent (see item 3)

- `spn = (Σ_{solid} srel^P)^(1/P)` (topopt.py:889) — with uniform voxels,
  volume weighting ≡ mean; the missing 1/N normalization made the limit
  depend on the cell count (P=8, resolution ×2 ⇒ spn ×~1.3).
- **Fix:** `spn = (np.mean(srel[solid_idx]**P))**(1.0/P)`; divide `coef` /
  the direct dgdp term by `solid_idx.size` accordingly in the gradient.

### RISK-4: CG accuracy in the MMA path fixed at 1e-3

- `_eval` selects `cg_rtol` from `change` (topopt.py:424). `change` is a
  `simp`-scope variable that was never updated in the MMA path — `_run_mma`
  tracked `last_change` separately (topopt.py:770, 786–788) and only wrote
  back on return. All MMA FEM solves (primary + constraints on CG domains
  > `spsolve_dofs`) ran with `cg_rtol=1e-3` → noisy objective/constraint
  values and gradients for MMA.
- **Fix:** feed `change` for MMA evals from `last_change`, or use a fixed
  `1e-5` for MMA.

### Notes (no bug)

- **Item 1:** the agent run table (`_run_summary`, optimize.py:60–62) used
  `res.fem` (mask!) as `worst_sf` — runs were compared by the more optimistic
  mask verification, not `sf_export`. Individual cases can silently end
  without an SF in `verify_exp` when fix/load regions no longer match on the
  export domain (analyze.py:262–264 — the case is appended with
  `safety_factor=None`, `worst_safety_factor` ignores None).
  Export voxelize can coarsen the pitch adaptively (voxelize.py:155–168).
- **Item 8 (observation):** the AM filter can push passive cells below 1;
  `density_to_mask` re-adds `passive_solid` (extract.py:241) — density and
  exported mask diverge on pinned cells (verification uses the mask —
  consistent with the printed part; noted for the record).
- **Item 10:** `volfrac_target` in `res.topopt` showed the raised volfrac on
  retry win (the original only in `retry.volfrac_first`); pass-2 renders
  overwrote same-named pass-1 renders.
- **minmax ignores `case.weight`** (smooth p-norm across cases unweighted,
  topopt.py:817) — semantically defensible, worth noting.

## OptimizeSpec → consumer (item 6 table)

| Field (schemas.py:274–323) | Consumed where | Status |
|---|---|---|
| `enabled` | orchestrator.py:4794+ (pipeline dispatch) | OK |
| `target_volume_fraction` | engine.py:1170 → `params.volfrac` | OK (no bound) |
| `max_iterations` | engine.py:1173 → `params.max_iter` | OK |
| `safety_factor_min` | engine.py:1353 (retry gate) | OK |
| `keep_regions` | engine.py:1152 → keep_list → passive | OK |
| `void_regions` | engine.py:1161 → `passive_void` | **BUG:** extraction (BUG-1) + resample (BUG-2) |
| `keep_mode` / `auto_keep_margin_mm` | engine.py:1153–1155 | OK |
| `min_member_mm` | engine.py:1167 → extraction + rmin | OK |
| `min_gap_mm` | engine.py:1168 → `_min_gap_warning` | OK (warning only, by design) |
| `density_threshold` | engine.py:1174 → `params.threshold` | OK (no bound) |
| `warm_density_relpath` | engine.py:1193–1203 | OK |
| `multires` | engine.py:1208 | OK |
| `self_supporting` / `print_direction` | engine.py:1176–1177 | OK |
| `filter` | engine.py:1175 | OK |
| `penal` / `penal_init` | engine.py:1178–1179 | OK |
| `solver` | engine.py:1180 | OK (RISK-1: oc silently swallowed constraints) |
| `case_objective` | engine.py:1181 | OK (oc/minmax warning only in mma) |
| `max_displacement_mm` / `max_stress_mpa` | engine.py:1182–1183 | OK value, RISK-1 |
| `auto_refine_features` | worker_jobs.py:939–940 → refine enqueue | OK |
| `restore_planes` / `auto_plane_restore` / `helical_threads` | worker_jobs.py:1458–1469 → engine.py:1786–1848 (refine path) | OK |
| `note` | no code consumer (LLM context) | by design (semi-dead) |

`TopOptParams` not exposed via `OptimizeSpec` (defaults only):
`tol`, `converge_patience`, `move`, `beta0`, `beta_max`, `stress_pnorm`,
`stress_relax_q`, `penal_ramp_iters`, `rmin_mm` (derived from
runtime+min_member).

**"Boundary-layer freeze":** fix/load cells are permanently pinned at ρ=1 via
`passive_solid_mask` (topopt.py:953–971) — deliberate design ("functional
features preserved by construction"), quantified via `passive_frac` +
warning when >volfrac (topopt.py:509–517, 617–625). OK.

## Telemetry recommendation (item 11)

Already computed but NOT in `res.topopt` (only in `res.fem`/`res.fem_export`
model_dumps or not at all):

| Metric | Source | Status |
|---|---|---|
| `compliance_export_j` (per case) | `verify_export.cases[].compliance_j` | computed, not propagated |
| `displacement_export_mm` | `verify_export.cases[].max_disp_mm` | computed, not propagated |
| `max_vm_export_mpa` | `verify_export.cases[].max_vm_mpa` | computed, not propagated |
| `compliance_pred_mj` | `history[-1].compliance_mj` (SIMP objective; watch the unit mJ vs J) | buried in history |
| `volfrac_mask` | `final_mask.sum()/nsolid` — extraction volfrac vs `volfrac_achieved` (density integral) exposes thickening/erosion effects | not computed |
| `gray_fraction` | share of design cells with 0.05 < ρ < 0.95 (filter/extraction quality) | **never computed** |
| `beta_final`, `penal_final` | history (MMA recs L.790–797 didn't even carry `beta`) | partial |
| `solver_used` / `mma_fallback` | only visible in warnings | not propagated |
| `sf_gap` | `sf_mask − sf_export` (erosion indicator of the export chain) | both values present, difference not |
| `dofs` | `report.grid["dofs"]` | only in res.fem |
| `n_cases_effective` | count of resolved cases incl. cone expansion | not propagated |
| `simp_s` / `verify_s` | `res.timings` | not in topopt |
| `case_sfs_mask` / `case_sfs_export` | per-case SF arrays of both verifications | only implicit in res.fem/fem_export |

Recommendation: add flat summary fields in `res.topopt.update`
(engine.py:1376) — otherwise the agent compares runs by `res.fem` (mask SF)
instead of `sf_export`.

## Item 12 — multires

`verify` (L.1312) and `verify_exp` (L.1327) both run on the fine domain;
the coarse pass only produces `x0` (`_upsample_density`, engine.py:1917–1935).
Note: `pc` (engine.py:1227–1236) does not set `solver`/`case_objective`/
`max_displacement_mm`/`max_stress_mpa`/`penal_init`/`beta` control —
OC+cone in the coarse run is the documented "cheap" heuristic; user
constraints only take effect in the fine pass (warm-started).
