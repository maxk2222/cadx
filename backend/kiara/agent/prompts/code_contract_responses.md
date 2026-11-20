## CODEGEN response (CodeResponse)

Return your answer as JSON: {"reasoning": "...", "params": [...], "expected_bbox":
    [x, y, z], "code": "..."}. `params` must list exactly the keys of PARAMS with
    unit/min/max/description; `expected_bbox` is the bounding box your code should produce.

## REPAIR / IMPROVE responses (PatchResponse)

When asked to FIX or IMPROVE existing code (not to write new code), respond with
a PatchResponse JSON instead — `{"reasoning": ..., "params": [...],
"expected_bbox": [...], "edits": [{"find": "...", "replace": "..."}],
"full_code": null}`:

- `edits`: exact-text replacements. Every `find` must match the current file
  EXACTLY ONCE — include 2-4 surrounding lines verbatim to be unique.
- `full_code`: the complete corrected file — only when the change is structural
  (> ~40 % of lines). Leave `null` when `edits` is used.
- Update `params`/`expected_bbox` to describe the part AFTER the fix.
