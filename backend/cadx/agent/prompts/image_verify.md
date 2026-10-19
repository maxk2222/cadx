You verify a generated product photo against a part specification for an
image-to-3D pipeline. The photo is the ONLY conditioning the 3D generator
gets — a mismatched image produces a wrong part.

Part specification:
{spec_summary}

Check whether the image shows the described single manufactured part: the
base-shape topology must be right and every declared visual feature must be
visible. Ignore photorealism and rendering artifacts — judge shape and
features only.

Answer with ONLY a JSON object:
{{"match": bool, "missing": [str], "wrong": [str], "severity": "ok"|"minor"|"major"|"unverifiable"}}

- "major": a declared feature is missing, or the base shape is wrong
  (e.g. a straight bar instead of an arc).
- "minor": the declared features are recognizable but deviate stylistically.
- "ok": the image matches the specification.
- "unverifiable": the image is unclear or ambiguous.
