"""Guide translation + step-semantic inference on top of `kiara.llm.client`.

Ports the suite's `OllamaClient` guide methods (`translate_guide`,
`translate_texts_batch`, `infer_semantic_for_step`) and
`ar_guide_service.translate_unity_guide` — the prompts and response shapes are
identical, only the transport differs: the suite POSTs to Ollama `/api/generate`,
we use the configured LLM backend via `make_llm(runtime).chat("code", …)`.

All public methods are SYNC: the instructions router calls them through
`run_in_threadpool` and `retranslate_changed_texts_task` is a sync background
task. Each call spins up `asyncio.run` to load runtime settings (DB) and run
the chat call — there is never a running loop in these contexts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import UTC, datetime
from typing import Any

from kiara.guides.paths import PROJECT_DATA_DIR
from kiara.guides.persistence import atomic_write_json

log = logging.getLogger(__name__)


def _run(coro):
    """Run a coroutine from a sync threadpool context."""
    return asyncio.run(coro)


async def _chat_json(prompt: str, *, max_tokens: int = 8192) -> dict[str, Any]:
    """One JSON-mode chat call against the configured backend."""
    from kiara.db.session import session_scope
    from kiara.llm.client import make_llm
    from kiara.runtime_config import load_runtime_settings

    async with session_scope() as db:
        runtime = await load_runtime_settings(db)
    client = make_llm(runtime)
    try:
        resp = await client.chat(
            "code",
            [{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=max_tokens,
        )
    finally:
        await client.aclose()
    return _parse_json_response(resp.text or "")


def _parse_json_response(text: str) -> Any:
    """Parse the model's JSON reply — tolerates ``` fences and prose around it."""
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}|\[.*\]", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return None


class GuideTranslator:
    """Suite `OllamaClient` guide-method surface, backed by our LLM client."""

    @property
    def model(self) -> str:
        return "configured"

    def infer_semantic_for_step(self, step_text: str) -> dict[str, Any]:
        """
        Annotiert einen einzelnen Schritt mit Semantik für das AR-System.
        """
        prompt = f"""
Du annotierst einzelne Montageschritte für ein AR-Assistenzsystem.

Gegeben ist genau EIN Montageschritt auf Deutsch.

Extrahiere:

- actionType: einer der folgenden Werte:
  - "prepare"  (Teile/Werkzeuge bereitlegen)
  - "position" (Teil ausrichten/platzieren, ohne zu befestigen)
  - "assemble" (Teile zusammenstecken/-bauen)
  - "tighten"  (verschrauben, festziehen)
  - "check"    (prüfen, kontrollieren)
  - "test"     (funktional testen)
  - "other"    (wenn nichts passt)

- focusParts: Liste der wichtigsten Bauteile/Komponenten als kurze deutsche Nomen,
              z.B. ["Grundplatte", "Center Platte"].

- tools: Liste der verwendeten Werkzeuge, z.B.
         ["Innensechskantschlüssel 2,5", "Schraubenschlüssel SW 5,5"].
         Wenn keine Werkzeuge vorkommen, gib eine leere Liste [] zurück.

- safetyHint: kurzer Sicherheitshinweis, falls im Schritt Sicherheitsaspekte vorkommen
              (Verletzungsgefahr, scharfe Kanten, heiße Teile, elektrische Spannung, etc.).
              Wenn keine Sicherheitsaspekte vorkommen, gib einen leeren String "" zurück.

Antwortformat:
Gib ausschließlich gültiges JSON mit genau dieser Struktur zurück:

{{
  "actionType": "prepare",
  "focusParts": ["..."],
  "tools": ["..."],
  "safetyHint": "..."
}}

Montageschritt:
\"\"\"{step_text}\"\"\""""

        parsed = _run(_chat_json(prompt))
        if not isinstance(parsed, dict):
            raise RuntimeError(f"Unerwartetes JSON-Format von infer_semantic_for_step: {parsed!r}")
        return parsed

    def translate_guide(self, guide: dict[str, Any], target_language: str) -> dict[str, Any]:
        """
        Übersetzt Titel und alle Textfelder einer Anleitung in die Zielsprache.
        """
        title = str(guide.get("title") or "")
        steps = guide.get("steps") or []
        if not isinstance(steps, list):
            steps = []

        source_steps = []
        for step in steps:
            step_texts = [
                str(obj.get("text") or "")
                for obj in step.get("objects", [])
                if obj.get("isTextField")
            ]
            source_steps.append({"texts": step_texts})

        prompt = f"""Du bist ein Übersetzer für technische Montageanleitungen.
Übersetze den folgenden Titel und alle Schritt-Texte ins {target_language}.
Behalte die genaue Anzahl und Reihenfolge der Schritte und Textfelder bei.
Ändere keine IDs, keine Medienpfade und keine technischen Begriffe, wenn sie im Zielsprachraum üblich sind.
Antworte ausschließlich als gültiges JSON mit genau dieser Struktur:

{{
  "title": "...",
  "steps": [
    {{"texts": ["..."]}},
    ...
  ]
}}

Titel:
\"\"\"{title}\"\"\"

Schritte:
{json.dumps(source_steps, ensure_ascii=False, indent=2)}"""

        parsed = _run(_chat_json(prompt))
        if not isinstance(parsed, dict):
            raise RuntimeError(f"Unerwartetes JSON-Format von translate_guide: {parsed!r}")

        translated_title = str(parsed.get("title") or title)
        translated_steps = parsed.get("steps") or []
        if len(translated_steps) != len(source_steps):
            raise RuntimeError(
                f"Anzahl übersetzter Schritte stimmt nicht: "
                f"{len(translated_steps)} != {len(source_steps)}"
            )

        snapshot_steps = []
        for step, tstep in zip(steps, translated_steps, strict=False):
            ttexts = tstep.get("texts") or []
            translated_objects = []
            text_idx = 0
            for obj in step.get("objects", []):
                if obj.get("isTextField"):
                    try:
                        new_text = str(ttexts[text_idx])
                    except IndexError:
                        new_text = str(obj.get("text") or "")
                    text_idx += 1
                    translated_objects.append({**obj, "text": new_text})
                else:
                    translated_objects.append(obj)
            snapshot_steps.append({**step, "objects": translated_objects})

        return {
            "title": translated_title,
            "steps": snapshot_steps,
            "language": target_language,
            "translatedAt": datetime.now(UTC).isoformat(),
            "model": self.model,
        }

    def translate_texts_batch(self, texts: list[str], target_language: str) -> list[str]:
        """Übersetzt eine Liste von Einzeltexten; gleiche Reihenfolge/Länge."""
        if not texts:
            return []

        prompt = f"""Du bist ein Übersetzer für technische Montageanleitungen.
Übersetze die folgenden Texte ins {target_language}.
Behalte technische Begriffe bei, wenn sie im Zielsprachraum üblich sind.
Antworte ausschließlich als gültiges JSON mit genau dieser Struktur:

{{
  "translations": ["übersetzter Text 1", "übersetzter Text 2", ...]
}}

Die Anzahl der Übersetzungen muss exakt mit der Anzahl der Eingabetexte übereinstimmen.

Texte:
{json.dumps(texts, ensure_ascii=False, indent=2)}"""

        parsed = _run(_chat_json(prompt))
        if not isinstance(parsed, dict):
            raise RuntimeError(f"Unerwartetes JSON-Format von translate_texts_batch: {parsed!r}")

        translations = parsed.get("translations") or []
        if not isinstance(translations, list):
            raise RuntimeError(f"translations-Feld ist keine Liste: {translations!r}")
        if len(translations) != len(texts):
            raise RuntimeError(
                f"Anzahl übersetzter Texte stimmt nicht: "
                f"{len(translations)} != {len(texts)}"
            )
        return [str(t) for t in translations]


def translate_unity_guide(
    guide_id: str,
    language: str,
    translation_snapshot: dict[str, Any],
) -> str | None:
    """
    Erzeugt eine übersetzte Unity-Guide-Variante, indem nur die Textfelder
    im bestehenden unity_guide.json ausgetauscht werden (atomar geschrieben).
    """
    unity_guide_path = (
        PROJECT_DATA_DIR / guide_id / "output" / "ar_guide_generation" / "unity_guide.json"
    )
    if not unity_guide_path.exists():
        log.warning(
            "[UNITY-TRANSLATE] unity_guide.json nicht gefunden für %s, überspringe.",
            guide_id,
        )
        return None

    try:
        unity_guide = json.loads(unity_guide_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.error("[UNITY-TRANSLATE] Fehler beim Lesen von unity_guide.json: %s", exc)
        return None

    translated_title = translation_snapshot.get("title")
    if translated_title:
        unity_guide["title"] = str(translated_title)

    translated_steps = translation_snapshot.get("steps") or []
    unity_steps = unity_guide.get("steps") or []

    if len(translated_steps) != len(unity_steps):
        log.warning(
            "[UNITY-TRANSLATE] Schrittanzahl stimmt nicht überein (%s vs %s), überspringe Text-Ersetzung.",
            len(unity_steps),
            len(translated_steps),
        )
    else:
        for ustep, tstep in zip(unity_steps, translated_steps, strict=False):
            u_text_objs = [o for o in ustep.get("objects", []) if o.get("isTextField")]
            t_text_objs = [o for o in tstep.get("objects", []) if o.get("isTextField")]
            if len(u_text_objs) != len(t_text_objs):
                log.warning(
                    "[UNITY-TRANSLATE] Textfeldanzahl in Schritt %s unterschiedlich, überspringe.",
                    ustep.get("step"),
                )
                continue
            for uobj, tobj in zip(u_text_objs, t_text_objs, strict=False):
                uobj["text"] = str(tobj.get("text") or uobj.get("text") or "")

    output_path = unity_guide_path.with_name(f"unity_guide_{language}.json")
    try:
        atomic_write_json(output_path, unity_guide)
        log.info("[UNITY-TRANSLATE] Übersetzter Unity-Guide geschrieben: %s", output_path)
        return f"/api/assets/{guide_id}/output/ar_guide_generation/unity_guide_{language}.json"
    except Exception as exc:
        log.error("[UNITY-TRANSLATE] Fehler beim Schreiben von %s: %s", output_path, exc)
        return None
