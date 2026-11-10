"""find_or_create_project must PATCH case settings onto a re-run project."""

from __future__ import annotations

import json

import httpx
import pytest

from cadx.evalkit.http import HttpDriver


@pytest.fixture
def driver(monkeypatch: pytest.MonkeyPatch) -> HttpDriver:
    return HttpDriver("http://testserver")


def _handler(captured: dict):
    def handle(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/api/projects":
            return httpx.Response(
                200,
                json=[
                    {
                        "id": "11111111-1111-1111-1111-111111111111",
                        "name": "eval:helical-pinion-shaft",
                        "settings": {
                            "gen_mode": "cad",
                            "clarification_mode": "assume",
                            "min_wall_mm": 0.8,
                        },
                    }
                ],
            )
        if request.method == "PATCH":
            captured["patch_body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "id": "11111111-1111-1111-1111-111111111111",
                    "name": "eval:helical-pinion-shaft",
                    "settings": captured["patch_body"]["settings"],
                },
            )
        return httpx.Response(404)

    return handle


@pytest.mark.asyncio
async def test_existing_project_gets_case_settings_patch(driver, monkeypatch):
    captured: dict = {}
    driver._client = httpx.AsyncClient(
        transport=httpx.MockTransport(_handler(captured)), base_url="http://testserver"
    )
    p = await driver.find_or_create_project(
        "eval:helical-pinion-shaft",
        {"_eval": True, "clarification_mode": "assume", "min_wall_mm": 0.1},
    )
    assert captured.get("patch_body"), "case settings must be PATCHed onto an existing project"
    patched = captured["patch_body"]["settings"]
    assert patched["min_wall_mm"] == 0.1
    # untouched case keys merge over, existing project keys survive
    assert patched["gen_mode"] == "cad"
    assert p["settings"]["min_wall_mm"] == 0.1


@pytest.mark.asyncio
async def test_optimize_body_forwards_allow_reopt(driver, monkeypatch):
    """escalate/simp_warm regression: the local driver forwards
    options.allow_reopt but the http body dropped it — a reopt stage
    after a successful optimize hits the 409 already-optimized gate."""
    from cadx.evalkit.http import DriverCtx
    from cadx.evalkit.suite import EvalStage

    async def _no_fallback(s, c):
        return None, None

    monkeypatch.setattr(driver, "_parent_fallback_lcs", _no_fallback)
    stage = EvalStage(
        id="simp_warm",
        type="optimize",
        options={"allow_reopt": True},
        optimize={"enabled": True, "structure_mode": "simp"},
        material="aluminum",
    )
    ctx = DriverCtx()
    ctx.project_id = "p"
    ctx.parent_version_id = "v"
    _path, body, _meta = await driver._spawn_body(stage, ctx)
    assert body["allow_reopt"] is True


@pytest.mark.asyncio
async def test_optimize_body_defaults_allow_reopt_false(driver, monkeypatch):
    from cadx.evalkit.http import DriverCtx
    from cadx.evalkit.suite import EvalStage

    async def _no_fallback(s, c):
        return None, None

    monkeypatch.setattr(driver, "_parent_fallback_lcs", _no_fallback)
    stage = EvalStage(id="infill", type="optimize", optimize={"enabled": True})
    ctx = DriverCtx()
    _path, body, _meta = await driver._spawn_body(stage, ctx)
    assert body["allow_reopt"] is False


@pytest.mark.asyncio
async def test_optimize_dispatch_passes_remaining_stage_timeout(driver, monkeypatch):
    from cadx.evalkit.http import DriverCtx
    from cadx.evalkit.suite import EvalStage

    async def _no_fallback(s, c):
        return None, None

    captured: dict = {}

    async def _post(path, *, json):
        captured["path"] = path
        captured["body"] = json
        return {"version_id": "child", "job_id": "job"}

    async def _wait_job(job_id, timeout_s):
        captured["wait_timeout_s"] = timeout_s
        return {"status": "succeeded"}

    async def _version(version_id):
        return {"status": "succeeded", "score": 90}

    async def _trace(job_id):
        return []

    monkeypatch.setattr(driver, "_parent_fallback_lcs", _no_fallback)
    monkeypatch.setattr(driver, "_post", _post)
    monkeypatch.setattr(driver, "wait_job", _wait_job)
    monkeypatch.setattr(driver, "version", _version)
    monkeypatch.setattr(driver, "trace", _trace)
    stage = EvalStage(
        id="topopt",
        type="optimize",
        timeout_s=60.0,
        optimize={"enabled": True},
        # dispatch fail-fast requires resolvable cases for load-bearing
        # stages (posting null only earns a server 422)
        load_cases=[{"name": "lc", "loads": [], "supports": []}],
    )
    ctx = DriverCtx()
    ctx.parent_version_id = "parent"
    ctx.project_id = "project"

    outcome = await driver.dispatch(stage, ctx)

    assert outcome.status == "succeeded"
    assert 0 < captured["body"]["stage_timeout_s"] <= 60.0
    assert 0 <= captured["wait_timeout_s"] <= 60.0


@pytest.mark.asyncio
async def test_existing_project_same_settings_no_patch(driver, monkeypatch):
    captured: dict = {}
    driver._client = httpx.AsyncClient(
        transport=httpx.MockTransport(_handler(captured)), base_url="http://testserver"
    )
    await driver.find_or_create_project(
        "eval:helical-pinion-shaft",
        {"gen_mode": "cad", "clarification_mode": "assume", "min_wall_mm": 0.8},
    )
    assert "patch_body" not in captured


@pytest.mark.asyncio
async def test_optimize_body_merges_stage_keep_regions(driver, monkeypatch):
    """Stage-level keep_regions follow the resample/refine convention —
    the driver merges them into the OptimizeSpec the endpoint forwards."""
    from cadx.evalkit.http import DriverCtx
    from cadx.evalkit.suite import EvalStage

    async def _no_fallback(s, c):
        return None, None

    monkeypatch.setattr(driver, "_parent_fallback_lcs", _no_fallback)
    pad = {"kind": "cylinder", "axis": "z", "center": [42, 32, 0], "diameter_mm": 16}
    stage = EvalStage(
        id="topopt",
        type="optimize",
        optimize={"enabled": True, "keep_regions": [dict(pad, description="declared")]},
        keep_regions=[dict(pad, description="stage")],
    )
    ctx = DriverCtx()
    _path, body, _meta = await driver._spawn_body(stage, ctx)
    keeps = body["optimize"]["keep_regions"]
    assert [k["description"] for k in keeps] == ["declared", "stage"]


@pytest.mark.asyncio
async def test_req_retries_transient_5xx(driver, monkeypatch):
    """A 502/503/504 during a container restart must not kill a stage —
    idempotent calls retry with bounded backoff."""
    calls = {"n": 0}

    def handle(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] <= 2:
            return httpx.Response(502, text="Bad Gateway")
        return httpx.Response(200, json={"status": "succeeded"})

    driver._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handle), base_url="http://testserver", timeout=5.0
    )
    import cadx.evalkit.http as http_mod

    async def fast_sleep(_s):
        return None

    # keep the test fast — real backoff is 1+2+4+8+16s
    monkeypatch.setattr(http_mod.asyncio, "sleep", fast_sleep)
    out = await driver._req("GET", "/jobs/j1")
    assert out == {"status": "succeeded"}
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_req_no_retry_on_post_or_4xx(driver):
    calls = {"n": 0}

    def handle(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, text="unavailable")

    driver._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handle), base_url="http://testserver", timeout=5.0
    )
    from cadx.evalkit.http import EvalApiError

    with pytest.raises(EvalApiError) as exc:
        await driver._req("POST", "/projects")
    assert exc.value.status == 503
    assert calls["n"] == 1  # POST never retries — double-enqueue risk

    calls["n"] = 0

    def handle_404(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404, text="nope")

    driver._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handle_404), base_url="http://testserver", timeout=5.0
    )
    with pytest.raises(EvalApiError) as exc:
        await driver._req("GET", "/versions/v1")
    assert exc.value.status == 404
    assert calls["n"] == 1
