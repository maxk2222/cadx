You may optionally call the provided tools BEFORE your final answer — e.g.
`search_catalog` / `lookup_part` to resolve standard elements to their exact
norm type/size strings, or `cadlib_lookup` for helper signatures. To reuse
parts the user has saved (STEP/STL bodies or parametric templates), call
`search_library`, inspect candidates with `library_part`, and pick them with
`attach_library_part` — attached file parts are then loadable via
`load_reference(part_id)` in generated code. You may also call
`generate_concept_image` to render a concept/reference image with the local
image model: use it when the part is organic or sculptural but NO reference
images were attached (the image->3D mesh generator needs at least one view),
or to produce a sketch/img2img variation of an existing asset
(`reference_asset_id` + `strength`). Generated images are stored as
`generated_image` assets — the returned `asset_id` can go straight into
`mesh_source_images`. Budget is limited to a few calls. When you are done
exploring, answer with the final JSON object as usual (no tool call).
