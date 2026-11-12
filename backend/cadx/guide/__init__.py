"""Anleitungsbuilder — PDF instruction manual -> Unity AR guide pipeline.

Ported from KIARA's ar_guide_agent_v4 architecture into the CADX
chat/job/tool-use stack (see docs/anleitungsbuilder-masterplan.md).

Layout:
    state.py        GuideState blackboard (typed pydantic, atomic JSON save)
    pdfdoc.py       PDF -> page images + text + figures
    session.py      GuideToolSession (mission-scoped tool surface)
    tools.py        tool schemas + per-mission allow lists
    doc_tools.py    doc-mission tool handlers
    step_tools.py   step-mission tool handlers
    cadmeta.py      CAD part feature/port extraction orchestration
    part_tools.py   parts-mission tool handlers (incl. missing-part generation)
    solver.py       deterministic placement math (ported from KIARA p5)
    scene.py        trimesh assembly scene + manifold3d collisions
    assembly_tools.py   assembly-mission tool handlers
    review_tools.py review-mission tools + plan.md scratchpad
    annotate_tools.py   arrow annotation tools
    unity.py        GuideState -> unity_guide.json emission
    guide_exec.py   (in cad/) executor-side tasks: extract_ports/render_assembly
    guide_runner.py arq-facing mission loop (run_guide_job → run_guide)
"""
