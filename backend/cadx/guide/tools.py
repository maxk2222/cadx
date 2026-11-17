"""Guide tool registry — OpenAI function schemas + per-mission allow lists.

Each tool module (doc_tools, step_tools, part_tools, assembly_tools,
review_tools, annotate_tools) exports ``SCHEMAS: dict[str, dict]`` and
``HANDLERS: dict[str, Handler]`` where ``Handler`` is
``async (GuideToolSession, args) -> dict``. ``SCHEMAS``/``HANDLERS`` are
merged here; ``GuideToolSession.tools()`` filters to the mission's list;
``_dispatch`` resolves session ``_tool_*`` methods first (shared/lifecycle
tools) then the module registry.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cadx.guide.session import GuideToolSession

Handler = Callable[["GuideToolSession", dict[str, Any]], Awaitable[dict[str, Any]]]


# -- schema helpers (mirror toolspecs._fn/_s/_i — private there, kept local) -- #


def _fn(name: str, desc: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def _s(desc: str) -> dict[str, Any]:
    return {"type": "string", "description": desc}


def _i(desc: str) -> dict[str, Any]:
    return {"type": "integer", "description": desc}


def _num(desc: str) -> dict[str, Any]:
    return {"type": "number", "description": desc}


def _arr(item: dict[str, Any], desc: str) -> dict[str, Any]:
    return {"type": "array", "items": item, "description": desc}


def _b(desc: str) -> dict[str, Any]:
    return {"type": "boolean", "description": desc}


# -- mission allow lists ------------------------------------------------------ #
#
# Shared lifecycle/context tools are implemented as GuideToolSession._tool_*
# methods (job wiring lives on the session) and are listed per mission here.

SHARED_PAGE_TOOLS = [
    "get_page_image",
    "get_page_text",
    "get_all_page_text",
    "list_figures",
    "get_figure_image",
]
SHARED_PLAN_TOOLS = ["read_plan", "append_plan", "write_plan", "clear_plan"]
SHARED_STATE_TOOLS = [
    "list_steps",
    "get_step",
    "list_parts",
    "get_part_features",
    "list_assets",
    "add_master_note",
]
MISSION_TOOLS: dict[str, list[str]] = {
    "doc": [
        "prepare_pdf",
        "detect_language",
        "classify_pages",
        "set_detected_part_names",
        "ask_user",
        "add_master_note",
        *SHARED_PAGE_TOOLS,
        *SHARED_PLAN_TOOLS,
        "done_mission",
    ],
    "steps": [
        "create_step",
        "create_steps",
        "update_step",
        "add_sub_step",
        "update_sub_step",
        "delete_sub_step",
        "assign_media",
        "reorder_steps",
        "finalize_steps",
        *SHARED_PAGE_TOOLS,
        *SHARED_STATE_TOOLS,
        *SHARED_PLAN_TOOLS,
        "list_user_notes",
        "ask_user",
        "done_mission",
    ],
    "parts": [
        "register_uploaded_part",
        "match_required_parts",
        "alias_part",
        "generate_missing_part",
        "mark_context_part",
        "rename_required_name",
        # a step's required_parts entry can carry a name the alias guards
        # refuse — fixing it where it lives is the parts mission's job
        "update_step",
        "ask_user",
        *SHARED_STATE_TOOLS,
        *SHARED_PLAN_TOOLS,
        "list_user_notes",
        "done_mission",
    ],
    "assembly": [
        "find_matching_ports",
        "solve_attachment_preview",
        "solve_free_preview",
        "solve_env_preview",
        "place_instance",
        "rotate_instance",
        "unplace_instance",
        "request_repair",
        "reset_step",
        "render_assembly",
        "check_collisions",
        "validate_assembly_geometry",
        "validate_assembly_completeness",
        *SHARED_STATE_TOOLS,
        *SHARED_PLAN_TOOLS,
        "get_page_image",
        "get_page_text",
        "list_user_notes",
        "ask_user",
        "done_mission",
    ],
    # blender-mode build mission — replaces parts+assembly entirely: the
    # agent models/places/renders freely in Blender; registration and
    # scene read-back go through the thin blender_* tools
    "build": [
        "blender_ping",
        "blender_exec",
        "blender_screenshot",
        "blender_register_part",
        "blender_sync_scene",
        "blender_render_step",
        "blender_save",
        "blender_export_scene",
        "register_uploaded_part",
        "match_required_parts",
        "alias_part",
        "mark_context_part",
        "list_user_notes",
        "ask_user",
        *SHARED_PAGE_TOOLS,
        *SHARED_STATE_TOOLS,
        *SHARED_PLAN_TOOLS,
        "done_mission",
    ],
    "review": [
        "render_assembly",
        "check_collisions",
        "validate_assembly_geometry",
        "validate_assembly_completeness",
        "write_review_report",
        *SHARED_STATE_TOOLS,
        *SHARED_PAGE_TOOLS,
        *SHARED_PLAN_TOOLS,
        "done_mission",
    ],
    "annotate": [
        "add_arrow",
        "add_text",
        "list_arrows",
        "remove_arrow",
        "render_assembly",
        "ask_user",
        *SHARED_STATE_TOOLS,
        "done_mission",
    ],
}


def merged_schemas() -> dict[str, dict[str, Any]]:
    """All tool schemas: session-owned (SHARED_*) + every tool module's."""
    from cadx.guide import (
        annotate_tools,
        assembly_tools,
        blender_tools,
        doc_tools,
        part_tools,
        review_tools,
        step_tools,
    )

    out: dict[str, dict[str, Any]] = {}
    for mod in (
        doc_tools,
        step_tools,
        part_tools,
        assembly_tools,
        review_tools,
        annotate_tools,
        blender_tools,
    ):
        out.update(mod.SCHEMAS)
    return out


def merged_handlers() -> dict[str, Handler]:
    from cadx.guide import (
        annotate_tools,
        assembly_tools,
        blender_tools,
        doc_tools,
        part_tools,
        review_tools,
        step_tools,
    )

    out: dict[str, Handler] = {}
    for mod in (
        doc_tools,
        step_tools,
        part_tools,
        assembly_tools,
        review_tools,
        annotate_tools,
        blender_tools,
    ):
        out.update(mod.HANDLERS)
    return out


# mission-completion guards: tool modules register one via
# ``DONE_GUARDS[mission] = fn(session) -> error_dict | None``; the session's
# done_mission refuses to close the mission while a guard returns an error.
DONE_GUARDS: dict[str, Callable[[GuideToolSession], dict[str, Any] | None]] = {}
