## CODEGEN response (tool session)

Submit the finished code with the `submit_code` tool — pass the COMPLETE file
as the `code` argument plus `expected_bbox` ([L, W, H] mm). Before submitting
you may check your draft with `check_snippet`. `params` are derived from your
PARAMS dict automatically — keep every tunable value in PARAMS.

## REPAIR / IMPROVE sessions

When fixing or improving existing code you work on a shared buffer — use the
edit tools (`set_param`, `replace_lines`, `insert_after`/`insert_before`,
`find_replace`), then `check_code`, optionally `verify`, and finish with
`done`. Never paste the whole file back.
