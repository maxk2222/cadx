def test_build_issues_solids_gate():
    """Disconnected solids are a major deterministic issue regardless of
    what the critique model scored."""
    from cadx.agent.orchestrator import build_issues

    issues = build_issues(None, [], [], None, {"solid_count": 3})
    assert issues[0]["source"] == "solids"
    assert issues[0]["severity"] == "major"
    assert "3 disconnected solids" in issues[0]["what"]

    assert build_issues(None, [], [], None, {"solid_count": 1}) == []
    assert build_issues(None, [], [], None, None) == []
