"""webresearch — free-backend web search + page fetch for the research step.

Failover chain of keyless backends (DuckDuckGo lite → DDG html → Bing →
Wikipedia opensearch): first non-empty result wins, URLs deduped across
backends. `fetch` extracts readable text with a stdlib tag-strip — no
bs4 dependency. All outbound URLs pass the SSRF guard (public DNS names
only) — the worker fetches whatever a model emits.
"""

from __future__ import annotations

import ipaddress
import re
from html.parser import HTMLParser
from typing import Any
from urllib.parse import quote_plus, urlparse

import httpx

from cadx.logging import get_logger

log = get_logger(__name__)

_UA = {"User-Agent": "CADX-research/1.0 (+https://devin.ai) CAD spec research"}
_TIMEOUT = 15.0
_MAX_BODY = 8_000_000  # bytes — datasheets/pages beyond this are PDFs/dumps


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(headers=_UA)


def _safe_url(url: str) -> str | None:
    """http(s) public-host URLs only — blocks the internal-network scan
    vector when a model emits a crafted fetch target."""
    try:
        u = urlparse(url.strip())
    except Exception:
        return None
    if u.scheme not in ("http", "https") or not u.hostname:
        return None
    host = u.hostname.lower()
    if (
        host in ("localhost",)
        or host.endswith((".local", ".internal", ".lan", ".corp"))
        or "." not in host  # single-label — docker/k8s service names
    ):
        return None
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return None
    except ValueError:
        pass  # DNS name — allowed (resolution happens inside httpx)
    return url


# ---------------------------------------------------------------- search ---


class _LinkTextParser(HTMLParser):
    """Grabs <a> elements matching a predicate → (href, text) pairs."""

    def __init__(self, match) -> None:
        super().__init__(convert_charrefs=True)
        self._match = match
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a" and self._match(dict(attrs)):
            self._href = dict(attrs).get("href") or None
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._href:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._href:
            self.links.append((self._href, " ".join(self._text).strip()))
            self._href = None


def _class_has(name: str):
    def m(attrs: dict[str, str | None]) -> bool:
        return name in (attrs.get("class") or "").split()

    return m


def _ddg_unwrap(href: str) -> str:
    """DDG result links are /l/?uddg=<target> redirects."""
    if "uddg=" in href:
        from urllib.parse import parse_qs

        real = parse_qs(urlparse(href).query).get("uddg")
        if real:
            return real[0]
    return href


async def _ddg_lite(query: str, client: httpx.AsyncClient) -> list[dict[str, Any]]:
    r = await client.get(
        f"https://lite.duckduckgo.com/lite/?q={quote_plus(query)}", timeout=_TIMEOUT
    )
    r.raise_for_status()
    p = _LinkTextParser(_class_has("result-link"))
    p.feed(r.text)
    return [{"title": t, "url": _ddg_unwrap(h), "snippet": ""} for h, t in p.links]


async def _ddg_html(query: str, client: httpx.AsyncClient) -> list[dict[str, Any]]:
    r = await client.get(
        f"https://html.duckduckgo.com/html/?q={quote_plus(query)}", timeout=_TIMEOUT
    )
    r.raise_for_status()
    p = _LinkTextParser(_class_has("result__a"))
    p.feed(r.text)
    return [{"title": t, "url": _ddg_unwrap(h), "snippet": ""} for h, t in p.links]


async def _bing(query: str, client: httpx.AsyncClient) -> list[dict[str, Any]]:
    r = await client.get(f"https://www.bing.com/search?q={quote_plus(query)}", timeout=_TIMEOUT)
    r.raise_for_status()
    out: list[dict[str, Any]] = []
    for m in re.finditer(
        r'<li class="b_algo"[^>]*>.*?<h2[^>]*><a[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        r.text,
        re.S,
    ):
        title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
        out.append({"title": title, "url": m.group(1), "snippet": ""})
    return out


async def _wikipedia(query: str, client: httpx.AsyncClient) -> list[dict[str, Any]]:
    r = await client.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "opensearch",
            "search": query,
            "limit": 6,
            "namespace": 0,
            "format": "json",
        },
        timeout=_TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()
    return [
        {"title": t, "url": u, "snippet": s}
        for t, s, u in zip(data[1], data[2], data[3], strict=False)
    ]


SEARCH_BACKENDS: list[tuple[str, Any]] = [
    ("ddg_lite", _ddg_lite),
    ("ddg_html", _ddg_html),
    ("bing", _bing),
    ("wikipedia", _wikipedia),
]


async def search(
    query: str, client: httpx.AsyncClient, max_results: int = 5
) -> list[dict[str, Any]]:
    """Failover chain — first non-empty backend wins; SSRF-guarded results."""
    for name, backend in SEARCH_BACKENDS:
        try:
            rows = await backend(query, client)
        except Exception as e:
            log.info("webresearch.backend_failed", backend=name, error=str(e)[:120])
            continue
        out = []
        seen: set[str] = set()
        for row in rows:
            url = _safe_url(str(row.get("url") or ""))
            if not url or url in seen:
                continue
            seen.add(url)
            out.append(
                {
                    "title": str(row.get("title") or "")[:200],
                    "url": url,
                    "snippet": str(row.get("snippet") or "")[:300],
                    "backend": name,
                }
            )
            if len(out) >= max_results:
                break
        if out:
            return out
    return []


# ----------------------------------------------------------------- fetch ---

_STRIP_TAGS = re.compile(
    r"<(script|style|nav|header|footer|aside|form|noscript|svg)[^>]*>.*?</\1>",
    re.S | re.I,
)
_ALL_TAGS = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t]+")
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.S | re.I)
_MAIN_RE = re.compile(r"<(main|article)[^>]*>(.*?)</\1>|<body[^>]*>(.*?)</body>", re.S | re.I)


async def fetch(url: str, client: httpx.AsyncClient, max_chars: int = 6000) -> dict[str, Any]:
    """Readable-text extraction — main/article preferred, nav/script stripped."""
    safe = _safe_url(url)
    if not safe:
        return {"ok": False, "error": "url not fetchable (scheme/host blocked)"}
    try:
        r = await client.get(safe, timeout=_TIMEOUT, follow_redirects=True)
        r.raise_for_status()
    except Exception as e:
        return {"ok": False, "error": f"fetch failed: {e}"}
    if _safe_url(str(r.url)) is None:
        return {"ok": False, "error": "redirect target blocked (scheme/host)"}
    try:
        if int(r.headers.get("content-length") or 0) > _MAX_BODY:
            return {"ok": False, "error": f"page too large (>{_MAX_BODY // 1_000_000} MB)"}
    except ValueError:
        pass
    ctype = (r.headers.get("content-type") or "").lower()
    if "html" not in ctype and "text" not in ctype:
        return {"ok": False, "error": f"unsupported content-type {ctype or '?'}"}
    html = r.text[:2_000_000]
    title_m = _TITLE_RE.search(html)
    body_m = _MAIN_RE.search(html)
    chunk = body_m.group(2) or body_m.group(3) if body_m else html
    text = _STRIP_TAGS.sub(" ", chunk)
    text = _ALL_TAGS.sub(" ", text)
    text = re.sub(r"&nbsp;?", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;|&gt;", " ", text)
    text = re.sub(r"&#39;|&quot;", '"', text)
    text = _WS.sub(" ", text)
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 2]
    out = "\n".join(lines)
    truncated = len(out) > max_chars
    return {
        "ok": True,
        "url": str(r.url),
        "title": (title_m.group(1).strip() if title_m else "")[:200],
        "text": out[:max_chars],
        "truncated": truncated,
    }
