"""CADX MCP server — the deterministic tool surface for external agents.

Exposes sandbox execution, FEM/topopt jobs, assets and the catalog over
the Model Context Protocol — strictly LLM-free: no MCP tool may trigger
the WebGUI's LLM (the session's `ctx.llm` stays None and the LLM-touching
action tools are never registered).
"""
