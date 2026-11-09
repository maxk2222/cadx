"""webresearch — failover search chain, SSRF guard, text extraction."""

from __future__ import annotations

import httpx

from cadx.agent import webresearch


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), headers={})


DDG_LITE_HTML = """
<html><body><table>
<tr><td><a class="result-link" href="https://intel.example/i7-spec">i7-14700K specs</a></td></tr>
<tr><td><a class="result-link" href="/l/?uddg=https%3A%2F%2Fark.intel.com%2Fi7">ark page</a></td></tr>
<tr><td><a class="result-link" href="http://192.168.0.1/x">bad host</a></td></tr>
</table></body></html>
"""

BING_HTML = """
<html><body><ol>
<li class="b_algo"><h2><a href="https://nema.example/mount">NEMA 23 dims</a></h2></li>
<li class="b_algo"><h2><a href="https://wiki.example/nema">NEMA wiki</a></h2></li>
</ol></body></html>
"""


async def test_ddg_lite_parses_and_filters_hosts():
    async def handler(req):
        assert "lite.duckduckgo.com" in str(req.url)
        return httpx.Response(200, text=DDG_LITE_HTML)

    rows = await webresearch.search("i7-14700K", _client(handler))
    assert len(rows) == 2  # private IP result dropped by the SSRF guard
    assert rows[0]["url"] == "https://intel.example/i7-spec"
    assert rows[0]["backend"] == "ddg_lite"
    assert rows[1]["url"] == "https://ark.intel.com/i7"  # uddg unwrapped


async def test_failover_falls_through_to_next_backend():
    calls = []

    async def handler(req):
        calls.append(str(req.url))
        if "lite.duckduckgo.com" in str(req.url):
            return httpx.Response(500)
        if "html.duckduckgo.com" in str(req.url):
            return httpx.Response(200, text="<html><body></body></html>")
        if "bing.com" in str(req.url):
            return httpx.Response(200, text=BING_HTML)
        return httpx.Response(404)

    rows = await webresearch.search("nema 23 mount", _client(handler))
    assert rows and rows[0]["backend"] == "bing"
    assert any("lite.duckduckgo" in c for c in calls)
    assert any("bing.com" in c for c in calls)


async def test_all_backends_empty_returns_empty():
    async def handler(req):
        return (
            httpx.Response(200, text="<html><body></body></html>")
            if "duckduckgo" in str(req.url) or "bing" in str(req.url)
            else httpx.Response(200, json=[[""], [], [], []])
        )

    assert await webresearch.search("nothing", _client(handler)) == []


async def test_safe_url_blocks_internal():
    for bad in (
        "http://localhost/x",
        "http://127.0.0.1/x",
        "http://192.168.1.1/",
        "http://10.0.0.5/",
        "http://printer.local/",
        "file:///etc/passwd",
        "ftp://x/y",
        "not a url",
        # single-label hosts — docker/k8s service names resolve internally
        "http://cadx-api:8000/api/system/stack",
        "http://redis:6379/",
        "http://worker.internal/x",
    ):
        assert webresearch._safe_url(bad) is None
    assert webresearch._safe_url("https://intel.com/spec") == "https://intel.com/spec"


async def test_fetch_revalidates_redirect_target():
    """A public URL redirecting to an internal host must not be followed
    blindly — r.url is re-checked after follow_redirects."""

    async def handler(req):
        if "bait.example" in str(req.url):
            return httpx.Response(302, headers={"location": "http://localhost/admin"})
        return httpx.Response(200, text="<html>internal</html>")

    out = await webresearch.fetch("https://bait.example/spec", _client(handler))
    assert out["ok"] is False
    assert "redirect" in out["error"]


async def test_fetch_rejects_oversized_page():
    async def handler(req):
        return httpx.Response(
            200,
            text="<html></html>",
            headers={
                "content-type": "text/html",
                "content-length": str(webresearch._MAX_BODY + 1),
            },
        )

    out = await webresearch.fetch("https://big.example/dump", _client(handler))
    assert out["ok"] is False
    assert "too large" in out["error"]


FETCH_HTML = """
<html><head><title>Spec sheet</title><style>body{color:red}</style></head>
<body>
<nav><a href="/">Home</a><a href="/login">Login</a></nav>
<article><h1>i7-14700K</h1>
<table><tr><td>Socket</td><td>LGA1700</td></tr>
<tr><td>Base power</td><td>125 W</td><td>Turbo</td><td>253 W</td></tr></table>
</article>
<script>track();</script>
<footer>Copyright Intel</footer>
</body></html>
"""


async def test_fetch_extracts_main_text():
    async def handler(req):
        return httpx.Response(200, text=FETCH_HTML, headers={"content-type": "text/html"})

    out = await webresearch.fetch("https://intel.example/spec", _client(handler))
    assert out["ok"] is True
    assert "LGA1700" in out["text"] and "125 W" in out["text"]
    assert "Home" not in out["text"] and "Login" not in out["text"]
    assert "track" not in out["text"] and "Copyright" not in out["text"]
    assert out["title"] == "Spec sheet"


async def test_fetch_rejects_non_html_and_bad_url():
    async def handler(req):
        return httpx.Response(
            200, content=b"\x00\x01", headers={"content-type": "application/octet-stream"}
        )

    out = await webresearch.fetch("https://x.example/blob", _client(handler))
    assert out["ok"] is False
    out2 = await webresearch.fetch("http://169.254.0.1/meta", _client(handler))
    assert out2["ok"] is False
