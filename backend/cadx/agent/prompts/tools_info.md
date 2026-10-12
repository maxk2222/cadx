You may optionally call the provided tools BEFORE your final answer — e.g.
`search_catalog` / `lookup_part` to resolve standard elements to their exact
norm type/size strings, or `cadlib_lookup` for helper signatures. To reuse
parts the user has saved (STEP/STL bodies or parametric templates), call
`search_library`, inspect candidates with `library_part`, and pick them with
`attach_library_part` — attached file parts are then loadable via
`load_reference(part_id)` in generated code. When you are done exploring,
answer with the final JSON object as usual (no tool call).
