The generated code is inspectable via tools: `read_lines` and `search_code`
show the candidate implementation (line-numbered), `lookup_part` returns the
real catalog dimensions of standard parts. Use them to cross-check features
you cannot judge from the renders — e.g. whether a hole diameter in the code
matches the spec's screw. Then answer with the final Critique JSON as usual.

Inside generated code, `list_edges(shape, axis=None)` returns an edge
inventory for fillet/chamfer targeting:
`{"total": n, "by_axis": {"+z": m, ...}, "concave": [...], "convex": [...],
"circles": [...]}` — each list up to 20 entries of
`{"center": [x, y, z], "length": mm, "axis": ax}`. `axis="x"|"y"|"z"` (signs
ignored) restricts the report to edges running along that axis; `by_axis`
always shows the full split, and circles are labelled by rim-plane normal
(the bore direction). Use it to verify a fillet/chamfer hit the intended
edges — or to find the right `edges_at`/`hole_rim_edges` arguments.
