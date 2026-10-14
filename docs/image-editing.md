# Image editing before generation

Reference images can be cleaned up in the WebGUI before they are sent to the
agent — crop them, isolate a region, or remove the background. Edits are
always saved as a **new derived asset** (`derived_from` links it to the
source); the original upload is never modified.

## Opening the editor

Hover an image card in *Reference assets* and click the pencil icon.
Derived images carry a small **edited** badge.

## Tools

| Tool | Behaviour |
|------|-----------|
| **crop** | Drag a rectangle; corners resize, dragging inside moves. Aspect presets: free / 1:1 / 4:3 / 3:2 / 16:9. Confirm with *apply crop*. |
| **brush keep** | Paint the region to keep (radius slider, erase mode to un-paint). The kept area stays bright, everything else dims. |
| **lasso** | Click polygon points, double-click or *close selection* to finish. Adds to the keep-mask like the brush. |
| **remove background** | Server-side rembg (u2net) subject cutout — result lands back on the canvas so you can still crop/mask afterwards. |
| **undo** | Steps back through applied operations (crop/mask/bg). |

*keep selected, remove rest* applies the brush/lasso mask: pixels outside the
mask become transparent and the image is autocropped to the content bounding
box (small margin). *save as copy* persists the canvas as a derived PNG asset;
when the source was selected for generation, the derived image replaces it in
the selection automatically.

## Alpha / file format

Edited images are stored as **PNG with alpha** — transparent regions are
meaningful downstream:

- the mesh pipeline (`mesh_server/preprocess.py`) keeps RGBA and pads on
  transparency instead of white;
- the LLM vision path (`images.prepare`) flattens alpha onto **white**, so
  cut-outs do not appear on black for the model.

## Background removal details

- Implemented via `rembg[cpu]` in the backend (`POST /api/assets/{id}/remove-bg`),
  which returns PNG bytes; saving the edit still goes through
  `POST /api/assets/{id}/derivative` so cancelling never creates an orphan asset.
- The lightweight **u2net** model (~170 MB) is used (not rembg's ~1 GB default)
  and downloaded on **first use** — the first click can take a while; the UI
  shows a progress hint. Docker Compose sets `U2NET_HOME=/data/rembg` so the
  model survives container recreates; native installs use `~/.u2net`.
- The endpoint returns `503` when rembg is not installed (e.g. a slim custom
  install without the `[cpu]` extra).

## See also

- [Web GUI](webgui-ux.md) — the asset manager around this editor
- [Image generation](image-generation.md) — generating references locally
- [HTTP API](api.md) — `derivative` / `remove-bg` endpoints
