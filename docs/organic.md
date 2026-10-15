# Organic structures — pseudo topology optimization

The **Organische Struktur** action on a finished version opens the organic
editor: pick a procedural style, tune parameters against a live preview, and
apply to create a child version of kind `organic`.

> **This is a visual styling feature, not a structural optimization.** No FEM
> or SIMP solver runs; the field is generated deterministically and the result
> is explicitly *not* structurally verified. For real topology optimization use
> the [structural editor](fem-topopt.md) instead.

## How it works

`backend/cadx/cad/fem/organic.py` voxelizes the source mesh into the same
`VoxelDomain` used by FEM/TopOpt, builds a synthetic **density field** inside
the design space, and reuses `density_to_mesh` (`cad/fem/extract.py`) for
extraction — so the output is watertight, connectivity-pruned and respects
`min_member_mm` like a real TopOpt result. A floor-pinned Taubin pass
(`smooth_preserve`) keeps a flat stand base flat.

| Style | Field |
|-------|-------|
| `gyroid` | TPMS sheet — smooth flowing minimal-surface walls |
| `trabecular` | bone-like spongy core (two-octave smoothed noise) |
| `voronoi` | organic cell walls from a seeded Voronoi diagram |
| `branches` | floor-rooted struts growing into a canopy |

Everything is deterministic: same params + `seed` + same input mesh produce
the identical part.

## Parameters (`OrganicParams`)

| Field | Range | Meaning |
|-------|-------|---------|
| `style` | see table | procedural field |
| `volfrac` | 0.05–0.9 | fraction of the design space kept solid |
| `seed` | int | deterministic variation — dice button in the GUI |
| `period_mm` | >0 | characteristic feature size (cell/blob/strut) |
| `shell_mm` | 0–50 | solid skin preserved at the part surface (0 = off) |
| `floor_mm` | 0–100 | always-solid stand band above the part's min Z |
| `min_member_mm` | 0–50 | printable-feature floor (default: runtime setting) |
| `density_threshold` | 0.2–0.9 | explicit iso override (default: quantile for `volfrac`) |
| `auto_keep` | bool | pin a keep ring around every detected feature anchor |
| `auto_keep_margin_mm` | >0 | auto-keep ring margin |
| `keep_regions` | `RegionSpec[]` | manual keeps — face/box/cylinder/hole/painted `points` |
| `void_regions` | `RegionSpec[]` | regions forced empty |
| `voxel_mm`, `max_elements` | | resolution of the design domain |
| `smooth_iterations` | 0–30 | extraction smoothing (default: `topopt_smooth_iterations`) |

## GUI

![Organic editor — style select, parameter live preview, keep-region brush](screenshots/organic-editor.png)

The editor (`frontend/src/components/OrganicEditor.tsx`) shows a live preview:
parameter changes fire a debounced `POST /versions/{id}/mesh-preview` carrying
the `organic` mesh op — the **same** `OPS_REGISTRY` code path the commit runs,
so preview and result cannot drift. A keep-region brush paints `points`
regions on the source surface while painting (the preview is suppressed while
the brush is armed so picks land on the real geometry).

**Apply** → `POST /versions/{id}/organic` (`OrganicRequest` = `OrganicParams`,
`extra="forbid"`) creates an `organic` child version + `JobType.organic`
enqueued as `run_organic_job` — LLM-free, like `optimize`/`refine`.

## Executor pipeline

`run_organic_pipeline` (`cad/engine.py`, arq task `organic` on the `cad`
queue): load parent STL → `organic_transform` → `smooth_preserve` →
STL/GLB/3MF exports, renders, thumbnail, contact sheet, `mesh_analysis`.

The density field is persisted as `organic/density.npz` in the same layout as
`topopt/density.npz` — including the resolved `passive` pin mask — so
`POST /versions/{id}/topopt-resample` can re-extract it at a different
`density_threshold`/`min_member_mm` without regenerating the field.

## Failure semantics

- Extraction has a fallback cascade (same style at `volfrac + 0.15`, then
  `trabecular` at 0.5); a fully empty result fails the job with a clear error.
- `n_components > 1` → the child lands in `needs_review` (floating islands are
  unprintable); this is a manufacturability flag, never a strength verdict.
- The stored report (`version.mesh.organic`) carries a `warnings` list plus a
  `disclaimer` string — surfaces should keep it visible.

## See also

- [FEM & TopOpt](fem-topopt.md) — the *structural* optimization this feature
  visually resembles
- [Web GUI](webgui-ux.md) — editors and version actions
- [HTTP API](api.md) — `POST /versions/{id}/organic`, `mesh-preview`,
  `topopt-resample`
