#!/usr/bin/env python3
"""capture_screenshots.py — regenerate every screenshot/GIF in docs/screenshots/.

Drives the live web GUI (docker stack on :8080 by default) with Playwright,
captures PNGs + GIF frames, assembles the GIFs with Pillow and installs the
results into docs/screenshots/.

Requirements
------------
    pip install playwright pillow && playwright install chromium
    docker compose stack running (`make up`) with the eval/demo data present

Usage
-----
    python3 docs/capture_screenshots.py            # full run: capture + assemble + install
    python3 docs/capture_screenshots.py png        # only the PNG pass
    python3 docs/capture_screenshots.py wt hero    # only those GIF passes
    python3 docs/capture_screenshots.py --list     # dump projects/versions/jobs for CONFIG upkeep
    python3 docs/capture_screenshots.py --base http://host:8080 --no-rename --no-seed

Demo content
------------
The screenshots feature configured demo projects/versions (see CONFIG below —
run `--list` to pick new ids when the database content changes). To keep the
captures presentable the script:

- temporarily renames the configured projects to DISPLAY_NAMES (restored in a
  finally block — `eval:*` names stay load-bearing for `make eval`);
- seeds a demo chat bound to the TopOpt project (POST /api/chats + direct psql
  inserts — there is no message API that skips the agent turn), deleted again
  afterwards; skipped entirely when docker/psql is unreachable;
- deletes the 0-version draft projects the wizard passes create (snapshot
  diff, not name matching);
- masks the LLM Base URL field on the settings page — never screenshot a real
  endpoint host.

Selector notes: the wizard advances on a button named "Continue", version
tabs render with role="tab", and the editor entry points on the version page
are the actions-panel texts "Edit mesh visually" / "Structural editor (FEM /
TopOpt)" / "Organic structure". The init script forces English
(cadx:lang=en) — if the UI language file changes these labels, update
SELECTORS below.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

DOCS = Path(__file__).resolve().parent
SHOTS_DEST = DOCS / "screenshots"

# --------------------------------------------------------------------------- #
# CONFIG — demo content used by the captures. Run `--list` to find new ids.   #
# --------------------------------------------------------------------------- #

# project id -> name shown in the screenshots (original names restored after)
DISPLAY_NAMES = {
    "823b1441-9063-423a-b553-1483772baf5b": "520 Sprocket 45T",
    "065d895d-8bba-4f06-ad59-d702f1089ce5": "Robot Arm Mount",
    "97e6e728-36de-4dee-b31c-f2bc646312fe": "Helical Pinion Shaft",
    "ffacfa7a-4372-44d1-afb8-8333e76945a3": "L-Bracket with Rib",
    "097c1d04-3abe-4771-a9a0-0db1fa8dfd15": "Spur Gear — Hybrid",
}

P_CAD_VIEW = "97e6e728-36de-4dee-b31c-f2bc646312fe"   # pinion — project.png's richer sibling
V_CAD_VIEW = "1e91cd34-1aa5-4d3b-bf39-af13f6b6fac5"  # pinion v4 — version-cad + mesh editor
V_FEM = "bb80cd44-7855-4654-8daa-10dd5b906b3e"       # arm mount v16 — FEM code+quality, editors
V_TOPOPT = "ab669d40-99aa-4a18-98b0-5d56d29b4feb"    # arm mount v18 — TopOpt refine (chat panel)
P_FEM = "065d895d-8bba-4f06-ad59-d702f1089ce5"       # Robot Arm Mount — project.png + chat home

# prompts typed into the wizard (the wizard's mode suggestion follows content)
PROMPT_WIZARD = ("L-bracket 60x40x30 mm with two M5 holes per leg, carrying "
                 "200 N vertically on the short leg - topology-optimize to 40% volume.")
PROMPT_HERO = ("Helical pinion shaft - module 2, 14 helical teeth at 25 deg, "
               "18 mm shaft diameter with a keyway.")

CHAT_TITLE = "Robot arm mount — TopOpt run"
CHAT_MESSAGES = [  # (role, content, meta)
    ("user",
     "I need a mounting bracket for a small robot arm — four M4 bolt holes "
     "in the base plate, and it must carry 50 N on the top face. Generate it, "
     "verify the stress, and topology-optimize it down to ~40% volume.",
     {}),
    ("assistant",
     "Understood — I drafted the spec (L=70 mm base, four M4 anchors, 50 N "
     "downward on the top pad, PLA) and spawned a generation job. Once v16 "
     "lands I'll check the anchors and run the FEM verification.",
     {"tool_log": [
         {"name": "ask_questions", "args": {"n": 0},
          "result": {"answers": "none — spec was unambiguous"}},
         {"name": "spawn_generate", "args": {"mode": "cad"},
          "result": {"version_id": V_FEM}},
     ],
      # {tool, kind, job_id, version_id, version_number, status} — job_id is
      # filled in at seed time from the jobs API
      "spawned": [{"tool": "spawn_generate", "kind": "job",
                   "version_id": V_FEM, "version_number": 16, "status": "succeeded"}]}),
    ("assistant",
     "Generation finished — v16 verified PASS (score 93, all four M4 anchors "
     "measured). FEM: SF 2.4 under the 50 N case, worst hot-spot at the bore "
     "rim. I've spawned a TopOpt refine targeting 40% volume with the load "
     "case pinned — it remeshes v16 into the organic bracket.",
     {"spawned": [{"tool": "spawn_edit", "kind": "job",
                   "version_id": V_TOPOPT, "version_number": 18, "status": "needs_review"}]}),
]

# output geometry — CSS viewport x deviceScaleFactor = pixel size
PNG_SIZE = (1920, 1200, 2)     # → 3840x2400 retina stills
GIF_SIZE = (1600, 1000, 1)     # → 1600x1000 gif frames
GIF_COLORS = 192


# --------------------------------------------------------------------------- #
# small helpers                                                                #
# --------------------------------------------------------------------------- #

def api(base: str, method: str, path: str, body: dict | None = None) -> dict | list:
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode() if body is not None else None,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=15).read())


def psql(container: str, user: str, db: str, sql: str) -> str:
    return subprocess.run(
        ["docker", "exec", "-i", container, "psql", "-U", user, "-d", db,
         "-v", "ON_ERROR_STOP=1", "-tAc", sql],
        capture_output=True, text=True, timeout=30, check=True).stdout


def find_job(base: str, version_id: str) -> str | None:
    for j in api(base, "GET", "/api/jobs"):
        if j.get("version_id") == version_id:
            return j["id"]
    return None


def project_ids(base: str) -> dict[str, int]:
    return {p["id"]: p.get("version_count", 0) for p in api(base, "GET", "/api/projects")}


def id_ok(base: str, _id: str) -> bool:
    for kind in ("projects", "versions"):
        try:
            api(base, "GET", f"/api/{kind}/{_id}")
            return True
        except Exception:
            continue
    return False


# --------------------------------------------------------------------------- #
# housekeeping — renames, seeded chat, wizard drafts                            #
# --------------------------------------------------------------------------- #

def rename_projects(base: str) -> dict[str, str]:
    """Apply DISPLAY_NAMES; returns {id: original_name} for restore."""
    originals = {}
    for pid, nice in DISPLAY_NAMES.items():
        try:
            cur = api(base, "GET", f"/api/projects/{pid}")
            originals[pid] = cur.get("name", "")
            api(base, "PATCH", f"/api/projects/{pid}", {"name": nice})
        except Exception as e:
            print(f"  ! rename {pid[:8]}: {e}")
    return originals


def restore_projects(base: str, originals: dict[str, str]) -> None:
    for pid, name in originals.items():
        try:
            api(base, "PATCH", f"/api/projects/{pid}", {"name": name})
        except Exception as e:
            print(f"  ! restore {pid[:8]}: {e}")


def seed_chat(base: str, pg: tuple[str, str, str]) -> str | None:
    """Create a chat + messages pointing at real jobs; returns chat id."""
    try:
        chat = api(base, "POST", "/api/chats",
                   {"project_id": P_FEM, "title": CHAT_TITLE})
        cid = chat["id"]
        container, user, db = pg
        for i, (role, content, meta) in enumerate(CHAT_MESSAGES):
            meta = json.loads(json.dumps(meta))  # deep copy per row
            for s in meta.get("spawned", []):
                s["job_id"] = find_job(base, s["version_id"])
            # dollar-quoting keeps apostrophes/unicode intact
            psql(container, user, db,
                 "INSERT INTO chatmessage (id, chat_id, role, content, asset_ids, meta, "
                 f"pending, created_at) VALUES (gen_random_uuid(), '{cid}', '{role}', "
                 f"$${content}$$, '[]'::jsonb, $${json.dumps(meta)}$$::jsonb, false, "
                 f"now() + interval '{i} minute')")
        print(f"  seeded chat {cid[:8]} ({len(CHAT_MESSAGES)} messages)")
        return cid
    except Exception as e:
        print(f"  ! chat seeding failed ({e}) — chat shots fall back to /chat index")
        return None


def delete_chat(base: str, pg: tuple[str, str, str] | None, cid: str) -> None:
    """DELETE /api/chats/{id} hits an FK ordering bug — remove via psql."""
    if pg:
        try:
            container, user, db = pg
            psql(container, user, db, f"DELETE FROM chatmessage WHERE chat_id='{cid}'")
            psql(container, user, db, f"DELETE FROM chat WHERE id='{cid}'")
            print(f"  removed seeded chat {cid[:8]}")
            return
        except Exception as e:
            print(f"  ! chat cleanup via psql failed: {e}")
    try:
        api(base, "DELETE", f"/api/chats/{cid}")
    except Exception as e:
        print(f"  ! chat cleanup failed: {e}")


def cleanup_drafts(base: str, before: dict[str, int]) -> None:
    """Delete 0-version projects created during this run (wizard drafts)."""
    after = project_ids(base)
    for pid in set(after) - set(before):
        if after[pid] == 0:
            try:
                api(base, "DELETE", f"/api/projects/{pid}")
                print(f"  removed wizard draft {pid[:8]}")
            except Exception as e:
                print(f"  ! draft cleanup {pid[:8]}: {e}")


# --------------------------------------------------------------------------- #
# playwright helpers                                                           #
# --------------------------------------------------------------------------- #

async def settle(page, ms: int = 400) -> None:
    await page.wait_for_timeout(ms)


async def wait_model(page, extra: int = 2500) -> None:
    """Wait for the 3D canvas + loaders to finish."""
    try:
        await page.wait_for_selector("canvas", timeout=20000)
    except Exception:
        return
    try:
        await page.wait_for_selector(".animate-spin", state="detached", timeout=30000)
    except Exception:
        pass
    await settle(page, extra)


async def new_ctx(pw, size: tuple[int, int, int]):
    w, h, dsf = size
    browser = await pw.chromium.launch(args=[
        "--use-gl=angle", "--enable-unsafe-swiftshader",
    ])
    ctx = await browser.new_context(viewport={"width": w, "height": h},
                                    device_scale_factor=dsf)
    # English UI + no leftover wizard state
    await ctx.add_init_script("localStorage.setItem('cadx:lang','en');"
                              "localStorage.removeItem('wizard:draft');")
    return browser, ctx


async def wizard_to(page, base: str, step: int, prompt: str | None = None) -> None:
    """Advance the new-part wizard. step: 1=typed prompt, 2=images, 3=mode, 4=review."""
    await page.goto(base + "/", wait_until="domcontentloaded")
    await settle(page, 1200)
    if prompt is not None:
        await page.locator("textarea").first.fill(prompt)
    await settle(page, 400)
    for _ in range(step - 1):
        await page.get_by_role("button", name="Continue").first.click()
        await settle(page, 1500)


async def drop_wizard_draft(page, base: str) -> None:
    """Delete the draft project the wizard persisted (wizard:draft.projectId)
    so it never leaks into projects.png / the walkthrough projects frames."""
    raw = await page.evaluate("localStorage.getItem('wizard:draft')")
    try:
        pid = json.loads(raw or "{}").get("projectId")
    except Exception:
        pid = None
    if pid:
        try:
            api(base, "DELETE", f"/api/projects/{pid}")
        except Exception:
            pass  # the end-of-run snapshot diff still catches it


async def _editor_shot(page, out_png: str) -> None:
    """Screenshot an editor dialog after its preview/mesh is actually rendered.
    The preview spinner may mount late (executor roundtrip) and run >30s —
    wait for it to appear AND disappear before shooting."""
    await wait_model(page, 2000)
    try:  # give a late-mounting preview spinner a chance to appear
        await page.wait_for_selector(".animate-spin", state="attached", timeout=5000)
    except Exception:
        pass
    try:  # then wait it out
        await page.wait_for_selector(".animate-spin", state="detached", timeout=120000)
    except Exception:
        pass
    await settle(page, 1000)
    # demand-frameloop: the environment cubemap resolves over several frames
    # after mount and nothing invalidates afterwards in the dialog — a tiny
    # orbit drag forces a post-env re-render, else the model stays unlit.
    try:
        box = await page.locator("canvas").last.bounding_box()
        if box:
            cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
            await page.mouse.move(cx, cy)
            await page.mouse.down()
            await page.mouse.move(cx + 6, cy + 3, steps=3)
            await page.mouse.up()
            await settle(page, 900)
    except Exception:
        pass
    await page.screenshot(path=out_png)


async def open_structural(page, out_png: str) -> None:
    await page.get_by_text("Structural editor (FEM / TopOpt)").first.click()
    await settle(page, 800)
    await _editor_shot(page, out_png)
    await page.keyboard.press("Escape")
    await settle(page, 500)


async def open_organic(page, out_png: str) -> None:
    await page.get_by_text("Organic structure").first.click()
    await settle(page, 800)
    await _editor_shot(page, out_png)
    await page.keyboard.press("Escape")
    await settle(page, 500)


# --------------------------------------------------------------------------- #
# capture passes                                                               #
# --------------------------------------------------------------------------- #

async def pngs(pw, base: str, out: Path, chat_id: str | None) -> None:
    """All PNG stills. The wizard shot runs LAST — it creates a draft project
    which the snapshot-diff cleanup removes, keeping projects.png clean."""
    browser, ctx = await new_ctx(pw, PNG_SIZE)
    page = await ctx.new_page()
    png = out / "png"
    png.mkdir(parents=True, exist_ok=True)

    await page.goto(base + "/projects", wait_until="domcontentloaded")
    await settle(page, 2000)
    await page.screenshot(path=png / "projects.png")

    await page.goto(base + f"/projects/{P_FEM}", wait_until="domcontentloaded")
    await settle(page, 2500)
    await page.screenshot(path=png / "project.png")

    await page.goto(base + f"/versions/{V_CAD_VIEW}", wait_until="domcontentloaded")
    await wait_model(page, 3500)
    await page.screenshot(path=png / "version-cad.png")

    await page.goto(base + "/jobs", wait_until="domcontentloaded")
    await settle(page, 2000)
    await page.screenshot(path=png / "jobs.png")

    await page.goto(base + f"/chat/{chat_id}" if chat_id else base + "/chat",
                    wait_until="domcontentloaded")
    await settle(page, 2500)
    await wait_model(page, 3000)
    await page.screenshot(path=png / "chat.png")

    await page.goto(base + "/stats", wait_until="domcontentloaded")
    await settle(page, 3000)
    await page.screenshot(path=png / "stats.png")

    await page.goto(base + "/library", wait_until="domcontentloaded")
    await settle(page, 2000)
    await page.screenshot(path=png / "library.png")

    await page.goto(base + "/settings", wait_until="domcontentloaded")
    await settle(page, 2500)
    # never screenshot a real endpoint host
    mask_urls = """() => {
        for (const el of document.querySelectorAll('input')) {
            const v = el.value || '';
            if (/^https?:\\/\\//.test(v) && !/example\\.com|localhost|127\\./.test(v)) {
                const setter = Object.getOwnPropertyDescriptor(
                    HTMLInputElement.prototype, 'value').set;
                setter.call(el, 'https://vllm.internal.example.com/v1');
                el.dispatchEvent(new Event('input', {bubbles: true}));
            }
        }
    }"""
    await page.evaluate(mask_urls)
    await settle(page, 400)
    await page.screenshot(path=png / "settings.png")

    # models tab: HF-token card + local-model download catalog
    await page.goto(base + "/settings?tab=models", wait_until="domcontentloaded")
    await settle(page, 2500)
    await page.evaluate(mask_urls)
    await settle(page, 400)
    await page.screenshot(path=png / "settings-models.png")

    await page.goto(base + f"/versions/{V_CAD_VIEW}/edit", wait_until="domcontentloaded")
    await wait_model(page, 4000)
    await page.screenshot(path=png / "editor.png")

    await page.goto(base + f"/versions/{V_FEM}", wait_until="domcontentloaded")
    await wait_model(page, 3000)
    await open_structural(page, str(png / "structural-editor.png"))
    await open_organic(page, str(png / "organic-editor.png"))

    # wizard LAST — its draft project is cleaned up right after
    await wizard_to(page, base, step=3, prompt=PROMPT_WIZARD)
    await page.screenshot(path=png / "start-wizard.png")
    await drop_wizard_draft(page, base)
    await browser.close()


async def gif_wizard(pw, base: str, out: Path) -> None:
    """wizard.gif — describe → images → mode → review (held)."""
    browser, ctx = await new_ctx(pw, GIF_SIZE)
    page = await ctx.new_page()
    frames = out / "frames"
    frames.mkdir(parents=True, exist_ok=True)

    await wizard_to(page, base, step=1, prompt=PROMPT_WIZARD)
    await settle(page, 500)
    await page.screenshot(path=frames / "wizard-1.png")
    for i in range(3):
        await page.get_by_role("button", name="Continue").first.click()
        await settle(page, 1600)
        await page.screenshot(path=frames / f"wizard-{i + 2}.png")
    await page.screenshot(path=frames / "wizard-5.png")  # review hold
    await drop_wizard_draft(page, base)
    await browser.close()


async def gif_tabs(pw, base: str, out: Path) -> None:
    """version-tabs.gif — Model → Spec & Plan → Code → Quality → Files.
    Needs a version WITH generated code (refine/topopt versions hide the tab)."""
    browser, ctx = await new_ctx(pw, GIF_SIZE)
    page = await ctx.new_page()
    frames = out / "frames"
    frames.mkdir(parents=True, exist_ok=True)

    await page.goto(base + f"/versions/{V_FEM}", wait_until="domcontentloaded")
    await wait_model(page, 3500)
    await page.screenshot(path=frames / "tabs-1.png")
    for i, tab in enumerate(["Spec & Plan", "Code", "Quality"], start=2):
        await page.get_by_role("tab", name=tab).first.click()
        await settle(page, 1400)
        await page.screenshot(path=frames / f"tabs-{i}.png")
    await page.get_by_role("tab", name="Files").first.click()
    await settle(page, 1400)
    await page.screenshot(path=frames / "tabs-5.png")
    await browser.close()


async def gif_hero(pw, base: str, out: Path) -> tuple[list, Path]:
    """hero-demo.gif — wizard flow, cut to the finished pinion version, orbit."""
    browser, ctx = await new_ctx(pw, GIF_SIZE)
    page = await ctx.new_page()
    frames = out / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    seq: list[tuple[str, int]] = []

    async def snap(name: str, dur: int) -> None:
        p = frames / f"hero-{name}.png"
        await page.screenshot(path=p)
        seq.append((str(p), dur))

    await wizard_to(page, base, step=1, prompt=None)
    await page.locator("textarea").first.type(PROMPT_HERO, delay=12)
    await settle(page, 300)
    await snap("01", 480)
    for i in range(3):
        await page.get_by_role("button", name="Continue").first.click()
        await settle(page, 1500)
        await snap(f"0{i + 2}", 640 if i < 2 else 960)
    await snap("05", 960)

    await page.goto(base + f"/versions/{V_CAD_VIEW}", wait_until="domcontentloaded")
    await wait_model(page, 3500)
    await snap("06", 800)
    await snap("07", 960)

    box = await page.locator("canvas").first.bounding_box()
    cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    await page.mouse.move(cx, cy)
    await page.mouse.down()
    for i in range(14):
        await page.mouse.move(cx + 26 * (i + 1), cy + 6 * (i + 1), steps=5)
        await settle(page, 120)
        await snap(f"{i + 8:02d}", 160)
    await page.mouse.up()
    await snap("99", 160)
    await drop_wizard_draft(page, base)
    await browser.close()
    # persist so a standalone `assemble` target works without re-capturing
    (out / "hero_seq.json").write_text(json.dumps(seq))
    return seq, frames


async def gif_walkthrough(pw, base: str, out: Path, chat_id: str | None) -> list:
    """walkthrough.gif — navigate the whole GUI across several projects."""
    browser, ctx = await new_ctx(pw, GIF_SIZE)
    page = await ctx.new_page()
    frames = out / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    seq: list[tuple[str, int]] = []

    async def snap(name: str, dur: int = 1000) -> None:
        p = frames / f"wt-{name}.png"
        await page.screenshot(path=p)
        seq.append((str(p), dur))

    await page.goto(base + "/", wait_until="domcontentloaded")
    await settle(page, 1300)
    await snap("01-start", 1100)

    await page.get_by_role("link", name="Projects").first.click()
    await settle(page, 1800)
    await snap("02-projects", 1100)

    await page.goto(base + f"/projects/{P_CAD_VIEW}", wait_until="domcontentloaded")
    await settle(page, 2500)
    await snap("03-project", 1100)

    await page.goto(base + f"/versions/{V_CAD_VIEW}", wait_until="domcontentloaded")
    await wait_model(page, 3500)
    await snap("04-version", 1100)

    box = await page.locator("canvas").first.bounding_box()
    cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    await page.mouse.move(cx, cy)
    await page.mouse.down()
    for i in range(3):
        await page.mouse.move(cx + 30 * (i + 1), cy + 8 * (i + 1), steps=5)
        await settle(page, 120)
        await snap(f"05-orbit{i}", 160)
    await page.mouse.up()
    await settle(page, 300)
    await snap("06-version-after-orbit", 800)

    for tab, name, dur in [("Spec & Plan", "07-spec", 1000),
                           ("Quality", "08-quality", 1000),
                           ("Files", "09-files", 900)]:
        await page.get_by_role("tab", name=tab).first.click()
        await settle(page, 1300)
        await snap(name, dur)

    await page.get_by_text("Edit mesh visually").first.click()
    await wait_model(page, 4000)
    await snap("10-mesh-editor", 1100)

    await page.goto(base + f"/versions/{V_FEM}", wait_until="domcontentloaded")
    await wait_model(page, 3000)
    await open_structural(page, str(frames / "wt-11-structural.png"))
    seq.append((str(frames / "wt-11-structural.png"), 1200))
    await open_organic(page, str(frames / "wt-12-organic.png"))
    seq.append((str(frames / "wt-12-organic.png"), 1200))

    await page.goto(base + "/jobs", wait_until="domcontentloaded")
    await settle(page, 1800)
    await snap("13-jobs", 1000)

    await page.goto(base + f"/chat/{chat_id}" if chat_id else base + "/chat",
                    wait_until="domcontentloaded")
    await settle(page, 2200)
    await wait_model(page, 2500)
    await snap("14-chat", 1000)

    await page.goto(base + "/library", wait_until="domcontentloaded")
    await settle(page, 1800)
    await snap("15-library", 1000)
    await page.goto(base + "/stats", wait_until="domcontentloaded")
    await settle(page, 2800)
    await snap("16-stats", 1200)

    # finish on a second project so the tour shows more than one chain
    await page.goto(base + "/projects", wait_until="domcontentloaded")
    await settle(page, 1500)
    await snap("17-projects2", 800)
    await page.goto(base + f"/projects/{P_FEM}", wait_until="domcontentloaded")
    await settle(page, 2500)
    await snap("18-project-arm", 1600)

    await browser.close()
    (out / "wt_seq.json").write_text(json.dumps(seq))
    return seq


# --------------------------------------------------------------------------- #
# assemble + install                                                           #
# --------------------------------------------------------------------------- #

def make_gif(frames_durs: list[tuple[str, int]], out: Path) -> None:
    from PIL import Image
    frames_durs = [(p, d) for p, d in frames_durs if Path(p).exists()]
    if not frames_durs:
        print(f"  ! no frames for {out.name} — run the capture pass first")
        return
    ims = [Image.open(p).convert("RGB") for p, _ in frames_durs]
    q = [im.quantize(colors=GIF_COLORS, method=Image.Quantize.MEDIANCUT,
                     dither=Image.Dither.FLOYDSTEINBERG) for im in ims]
    q[0].save(out, save_all=True, append_images=q[1:],
              duration=[d for _, d in frames_durs], loop=0, optimize=True)
    print(f"  {out.name}: {out.stat().st_size / 1e6:.2f} MB, {len(q)} frames")


def _load_seq(out: Path, name: str, seq: list) -> list:
    """Prefer the in-memory seq; fall back to the json persisted at capture."""
    if seq:
        return seq
    p = out / f"{name}.json"
    return [tuple(x) for x in json.loads(p.read_text())] if p.exists() else []


def assemble(out: Path, hero_seq: list, wt_seq: list) -> None:
    frames = out / "frames"
    make_gif([(str(frames / f"wizard-{i}.png"), 1320 if i == 5 else 880)
              for i in range(1, 6)], out / "wizard.gif")
    make_gif([(str(frames / f"tabs-{i}.png"), 1440) for i in range(1, 6)],
             out / "version-tabs.gif")
    make_gif(_load_seq(out, "hero_seq", hero_seq), out / "hero-demo.gif")
    make_gif(_load_seq(out, "wt_seq", wt_seq), out / "walkthrough.gif")


def install(out: Path, dest: Path) -> None:
    import shutil
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted((out / "png").glob("*.png")):
        shutil.copy2(p, dest / p.name)
        n += 1
    for gif in ("wizard.gif", "version-tabs.gif", "hero-demo.gif", "walkthrough.gif"):
        src = out / gif
        if src.exists():
            shutil.copy2(src, dest / gif)
            n += 1
    print(f"  installed {n} files → {dest}")


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #

async def run(args) -> None:
    base = args.base.rstrip("/")
    out = Path(args.work_dir)
    out.mkdir(parents=True, exist_ok=True)
    pg = (args.pg_container, args.pg_user, args.pg_db) if args.seed else None

    if args.list:
        print("projects:")
        for p in api(base, "GET", "/api/projects"):
            print(f"  {p['id']}  {p['name']}  (versions: {p.get('version_count', '?')})")
        print("\nversions (succeeded/needs_review, best first per project):")
        for p in api(base, "GET", "/api/projects"):
            vs = api(base, "GET", f"/api/projects/{p['id']}/versions")
            vs = [v for v in vs if v["status"] in ("succeeded", "needs_review")]
            for v in sorted(vs, key=lambda v: -(v.get("score") or 0))[:3]:
                print(f"  {v['id']}  v{v['number']} {v['kind']:9s} "
                      f"{v['status']:12s} score={v.get('score')}  {p['name']}")
        return

    try:
        from playwright.async_api import async_playwright
    except ImportError:
        sys.exit("playwright not installed — pip install playwright && "
                 "playwright install chromium")

    # verify configured ids resolve before burning a capture run
    missing = [i for i in (*DISPLAY_NAMES, P_CAD_VIEW, V_CAD_VIEW, V_FEM, V_TOPOPT, P_FEM)
               if not id_ok(base, i)]
    if missing:
        sys.exit(f"configured ids not found — run --list and update CONFIG: {missing}")

    before = project_ids(base)
    originals = rename_projects(base) if args.rename else {}
    chat_id = seed_chat(base, pg) if args.seed else None
    hero_seq: list = []
    wt_seq: list = []
    try:
        async with async_playwright() as pw:
            targets = args.targets or ["all"]
            want = lambda t: "all" in targets or t in targets
            if want("png"):
                await pngs(pw, base, out, chat_id)
            if want("wt"):
                wt_seq = await gif_walkthrough(pw, base, out, chat_id)
            if want("tabs"):
                await gif_tabs(pw, base, out)
            if want("wizard"):
                await gif_wizard(pw, base, out)
            if want("hero"):
                hero_seq, _ = await gif_hero(pw, base, out)
            if want("assemble") or want("all"):
                assemble(out, hero_seq, wt_seq)
            if want("install") or want("all"):
                install(out, SHOTS_DEST)
    finally:
        cleanup_drafts(base, before)
        if chat_id:
            delete_chat(base, pg, chat_id)
        if originals:
            restore_projects(base, originals)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="\n".join(__doc__.splitlines()[2:16]))
    valid = ["png", "wizard", "tabs", "hero", "wt", "assemble", "install", "all"]
    ap.add_argument("targets", nargs="*", metavar="target",
                    help=f"passes to run ({', '.join(valid)}; default: all)")
    ap.add_argument("--base", default=os.environ.get("CADX_BASE", "http://localhost:8080"))
    ap.add_argument("--work-dir", default=tempfile.gettempdir() + "/cadx_capture",
                    help="staging dir for frames + assembled gifs")
    ap.add_argument("--list", action="store_true",
                    help="print projects/versions for CONFIG upkeep and exit")
    ap.add_argument("--no-rename", dest="rename", action="store_false",
                    help="don't apply DISPLAY_NAMES")
    ap.add_argument("--no-seed", dest="seed", action="store_false",
                    help="don't seed the demo chat (chat shots use /chat index)")
    ap.add_argument("--pg-container", default="cadx-postgres-1")
    ap.add_argument("--pg-user", default="cadx")
    ap.add_argument("--pg-db", default="cadx")
    args = ap.parse_args()
    bad = [t for t in args.targets if t not in valid]
    if bad:
        ap.error(f"unknown target(s): {bad} — pick from {valid}")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
