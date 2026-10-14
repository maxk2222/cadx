# HTTP API

The FastAPI backend serves all endpoints under `/api` (nginx proxies `/api/*`
to `api:8000`; the same prefix carries the SSE job stream). Routers live in
`backend/cadx/api/routers/`. Interactive docs are available at `/docs` on the
api service in dev mode.

## Projects

```
GET|POST      /api/projects                              list / create
GET|PATCH|DELETE /api/projects/{id}                      detail / rename+settings / delete
GET           /api/projects/{id}/assets                  list assets
POST          /api/projects/{id}/assets                  upload images/STEP/STL/DXF (multipart)
POST          /api/projects/{id}/images                  local text-to-image → generated_image assets
GET           /api/projects/{id}/versions                version tree
POST          /api/projects/{id}/generate                {prompt, asset_ids, library_part_ids, mode, options}
```

## Assets

```
GET           /api/assets/{id}/file                      raw file download
POST          /api/assets/{id}/derivative                save an edited image (PNG, keeps alpha) as a new asset
POST          /api/assets/{id}/remove-bg                 rembg subject cutout → PNG bytes
PATCH         /api/assets/{id}                           caption / role_hint (e.g. "sketch")
DELETE        /api/assets/{id}
```

## Versions

```
GET|DELETE    /api/versions/{id}                         full detail (BOM/FEM/artifacts) / delete
POST          /api/versions/{id}/edit                    {prompt, asset_ids, library_part_ids, mode, options}
POST          /api/versions/{id}/params                  {params} — re-run code/ops without an LLM call
POST          /api/versions/{id}/optimize                manual FEM/SIMP TopOpt child version
POST          /api/versions/{id}/topopt-resample         re-extract the stored density field at a new threshold
POST          /api/versions/{id}/refine                  post-TopOpt feature finishing (bores/threads/inserts, keep-surface restore)
POST          /api/versions/{id}/reexport                re-run export pipeline
GET           /api/versions/{id}/jobs                    jobs for this version
GET           /api/versions/{id}/code                    generated model.py (text/plain)
GET           /api/versions/{id}/live-renders            render manifest for in-progress preview
GET           /api/versions/{id}/preview?view=…          live render image (`iso`, `front`, `top`, `fem:*`, …)
```

## Editors

Endpoints behind the mesh and structural editors — see
[Web GUI](webgui-ux.md) and [FEM & TopOpt](fem-topopt.md).

```
POST          /api/versions/{id}/mesh-preview            replay a MeshOp stack → preview GLB + report
POST          /api/versions/{id}/mesh-edit               apply an op stack as a new mesh_edit child version
GET           /api/versions/{id}/scratch/{path}          whitelisted preview artifact (ops GLBs, region PNGs)
PATCH         /api/versions/{id}/editor-state            persist editor drafts (merged into options.ui_editor)
POST          /api/versions/{id}/fem-preview             resolve regions / voxel preview (preview_only)
POST          /api/versions/{id}/fem                     standalone FEM check on this version
GET           /api/versions/{id}/anchors                 merged measured-anchor table
```

## Jobs

```
GET           /api/jobs                                  global job list (?active=true)
GET           /api/jobs/{id}                             job detail incl. live state + log tail
GET           /api/jobs/{id}/events                      SSE progress stream
POST          /api/jobs/{id}/answer                      clarification answers → resume a waiting job
POST          /api/jobs/{id}/cancel                      abort
POST          /api/jobs/{id}/finish                      stop early, keep the best candidate
POST          /api/jobs/{id}/restart                     replay a terminal job
```

## Library

```
GET|POST      /api/library                               list / upload reusable STEP/STL parts
GET|PATCH|DELETE /api/library/{id}                       detail / edit metadata / remove
GET           /api/library/{id}/file                     download a library body
POST          /api/library/from-version/{id}             save a version as STEP body or code template
```

## Catalog

```
GET           /api/catalog                               bd_warehouse types + sizes
GET           /api/catalog/search?q=…                    name/type/size search
GET           /api/catalog/lookup                        instantiate a part → real published dims
GET           /api/catalog/{category}/{class_name}       class detail
```

## Settings, mesh, images, health

```
GET|PUT       /api/settings                              runtime settings (see configuration.md)
POST          /api/settings/test                         probe code + vision models without exposing the key
GET           /api/mesh/health                           mesh-server reachability + device
GET           /api/mesh/models                           installed adapter/model info + licenses
GET           /api/images/models                         installed image adapters
GET           /api/health                                liveness
GET           /api/artifacts/{id}/file                   artifact downloads (step/stl/3mf/glb/dxf/svg/png/json)
```

## Conventions

- Requests/responses are JSON (`pydantic` v2); uploads are `multipart/form-data`.
- Errors are `{"detail": …}` with the usual status codes (`422` validation,
  `404` unknown id, `409` wrong state, `413` payload over the nginx/API cap).
- The SSE stream emits `step` / `status` / `question` / `snapshot` events; see
  `backend/cadx/events.py` for the payload shapes.
- The mesh-server sidecar has its own API — documented in
  [mesh-server-api.md](mesh-server-api.md).
