"""KIARA web-suite parity — the AR sync surface, end to end.

Covers the wire contract a Unity/headset client depends on: CAS push
(428/409/conflict body), no-op idempotence, pull-selection, conflict-copy,
upload-attempt lifecycle (manifest → raw-body media → atomic commit →
authoritative URL rewrite), media-reference delete protection, the model
store + model-changes feed, app-update 404 fail-soft, the retired
global_assets routes (must stay 404) and the deterministic GuideSchema
emitter (cumulative objects, prefab/modelLocation conventions).
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import uuid

import pytest
from fastapi.testclient import TestClient

from kiara.db.session import create_all


@pytest.fixture(scope="module", autouse=True)
def _db():
    asyncio.run(create_all())
    yield


@pytest.fixture(scope="module")
def client():
    from kiara.main import app

    with TestClient(app) as c:
        yield c


def _png_bytes(color=(200, 30, 30)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, "PNG")
    return buf.getvalue()


def _doc(gid: str, *, title: str = "T", step: int = 1) -> dict:
    return {
        "guideId": gid,
        "title": title,
        "formatVersion": 2,
        "steps": [
            {
                "objects": [],
                "semantic": {
                    "actionType": "assemble",
                    "focusParts": [],
                    "tools": [],
                    "safetyHint": "",
                },
                "step": step,
            }
        ],
    }


# ── CAS matrix ────────────────────────────────────────────────────────────────


def test_cas_create_then_preconditions(client: TestClient):
    gid = f"cas-{uuid.uuid4().hex[:8]}"
    doc = _doc(gid)
    doc["customField"] = {"keep": "me"}

    # creation requires NO expected base (unknown fields must survive + hash)
    r = client.post("/api/sync/guide", json={"guide": doc})
    assert r.status_code == 200
    g = r.json()
    assert g["version"] == 1
    assert g["customField"] == {"keep": "me"}  # extra=allow round-trips

    # a write to an existing guide without a base → 428
    r = client.post("/api/sync/guide", json={"guide": g})
    assert r.status_code == 428
    assert r.json()["detail"]["error"] == "expected_base_version_required"

    # stale base → 409 carrying the authoritative server guide
    r = client.post("/api/sync/guide", json={"guide": g, "expectedBaseVersion": 99})
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["guideId"] == gid
    assert detail["expectedBaseVersion"] == 99
    assert detail["currentVersion"] == g["version"]
    assert detail["serverGuide"]["guideId"] == gid

    # creation WITH a base is a 404, not silent creation
    r = client.post(
        "/api/sync/guide",
        json={"guide": _doc(f"gone-{uuid.uuid4().hex[:8]}"), "expectedBaseVersion": 3},
    )
    assert r.status_code == 404


def test_cas_update_and_noop_idempotence(client: TestClient):
    gid = f"noop-{uuid.uuid4().hex[:8]}"
    g = client.post("/api/sync/guide", json={"guide": _doc(gid)}).json()

    # first round-trip adopts server-serialized fields (source_file_id etc.)
    r = client.post("/api/sync/guide", json={"guide": g, "expectedBaseVersion": g["version"]})
    assert r.status_code == 200
    g = r.json()

    # identical content + correct base → same version, same hash (§7f.1)
    r = client.post("/api/sync/guide", json={"guide": g, "expectedBaseVersion": g["version"]})
    assert r.status_code == 200
    again = r.json()
    assert again["version"] == g["version"]
    assert again["contentHash"] == g["contentHash"]

    # real edit advances exactly once
    g["title"] = "Changed"
    r = client.post("/api/sync/guide", json={"guide": g, "expectedBaseVersion": g["version"]})
    assert r.status_code == 200
    assert r.json()["version"] == g["version"] + 1


def test_force_is_inert(client: TestClient):
    gid = f"force-{uuid.uuid4().hex[:8]}"
    g = client.post("/api/sync/guide", json={"guide": _doc(gid)}).json()
    r = client.post(
        "/api/sync/guide?force=true", json={"guide": g, "expectedBaseVersion": 99}
    )
    assert r.status_code == 409  # force must never bypass CAS


def test_meta_pull_checkconflict(client: TestClient):
    gid = f"meta-{uuid.uuid4().hex[:8]}"
    g = client.post("/api/sync/guide", json={"guide": _doc(gid)}).json()

    r = client.get(f"/api/sync/guide/{gid}/meta")
    assert r.status_code == 200
    meta = r.json()
    assert meta["guideId"] == gid and meta["version"] == g["version"]
    assert meta["contentHash"] == g["contentHash"]

    # pull-selection: an outdated client copy is offered for re-download,
    # an up-to-date one is not (version comparison wins over updatedAt)
    g["title"] = "bumped"
    g2 = client.post(
        "/api/sync/guide", json={"guide": g, "expectedBaseVersion": g["version"]}
    ).json()
    assert g2["version"] > g["version"]
    r = client.post(
        "/api/sync/guides_from_db",
        json={
            "localGuides": [
                {"guideId": gid, "updatedAt": "2000-01-01T00:00:00.000Z", "version": 1}
            ]
        },
    )
    assert r.status_code == 200
    assert gid in {d["guideId"] for d in r.json()}
    r = client.post(
        "/api/sync/guides_from_db",
        json={
            "localGuides": [
                {"guideId": gid, "updatedAt": g2["updatedAt"], "version": g2["version"]}
            ]
        },
    )
    assert r.status_code == 200
    assert gid not in {d["guideId"] for d in r.json()}

    r = client.post("/api/sync/guide/check_conflict", json={"guide": _doc(gid)})
    assert r.status_code == 200
    body = r.json()
    assert body["hasConflict"] is False
    assert body["recommendedAction"] in ("upload", "download_before_upload")


def test_conflict_copy_gets_fresh_identity(client: TestClient):
    gid = f"src-{uuid.uuid4().hex[:8]}"
    g = client.post("/api/sync/guide", json={"guide": _doc(gid, title="Orig")}).json()
    r = client.post(f"/api/sync/guide/{gid}/conflict-copy", json={"guide": g})
    assert r.status_code == 201
    copy = r.json()
    assert copy["guideId"] != gid
    assert copy["databaseId"] != g["databaseId"]
    assert copy["version"] == 1


def test_delete_with_precondition(client: TestClient):
    gid = f"del-{uuid.uuid4().hex[:8]}"
    g = client.post("/api/sync/guide", json={"guide": _doc(gid)}).json()
    r = client.delete(f"/api/sync/guide/{gid}", params={"expectedBaseVersion": 99})
    assert r.status_code == 409
    r = client.delete(f"/api/sync/guide/{gid}", params={"expectedBaseVersion": g["version"]})
    assert r.status_code == 204
    assert client.get(f"/api/sync/guide/{gid}").status_code == 404


def test_hash_canonicalization(client: TestClient):
    """Unknown top-level fields must participate in contentHash (extra=allow)."""
    gid = f"hash-{uuid.uuid4().hex[:8]}"
    doc = _doc(gid)
    doc["x_unknown"] = {"z": 1}
    g1 = client.post("/api/sync/guide", json={"guide": doc}).json()
    # server-owned keys are excluded; content keys stay — a differing unknown
    # field is a real authoring change
    g1["x_unknown"] = {"z": 2}
    r = client.post("/api/sync/guide", json={"guide": g1, "expectedBaseVersion": g1["version"]})
    assert r.status_code == 200 and r.json()["version"] == 2


# ── upload attempts ───────────────────────────────────────────────────────────


def test_attempt_lifecycle_and_authoritative_urls(client: TestClient):
    gid = f"att-{uuid.uuid4().hex[:8]}"
    g = client.post("/api/sync/guide", json={"guide": _doc(gid)}).json()

    payload = _png_bytes()
    sha = hashlib.sha256(payload).hexdigest()
    oid = str(uuid.uuid4())

    r = client.post(
        "/api/sync/guide-upload-attempts",
        json={
            "guideId": gid,
            "expectedBaseVersion": g["version"],
            "media": [
                {
                    "mediaKey": "m1",
                    "objectId": oid,
                    "assetType": "images",
                    "fileName": "a.png",
                    "sha256": sha,
                    "length": len(payload),
                    "required": True,
                }
            ],
        },
    )
    assert r.status_code == 201
    aid = r.json()["attemptId"]

    # commit before media arrived → 409 incomplete, nothing published
    g2 = dict(g)
    r = client.post(f"/api/sync/guide-upload-attempts/{aid}/commit", json={"guide": g2})
    assert r.status_code == 409

    # wrong sha → refused at PUT
    r = client.put(
        f"/api/sync/guide-upload-attempts/{aid}/media/m1",
        content=b"WRONGBYTES",
        headers={"Content-Type": "image/png"},
    )
    assert r.status_code == 400

    r = client.put(
        f"/api/sync/guide-upload-attempts/{aid}/media/m1",
        content=payload,
        headers={"Content-Type": "image/png"},
    )
    assert r.status_code == 200 and r.json()["state"] == "uploaded"

    # commit: client-predicted URL must be rewritten to the authoritative one
    g["title"] = "with-media"
    g["steps"][0]["objects"] = [
        {
            "objectId": oid,
            "step": 1,
            "objectNumber": 1,
            "isPhoto": True,
            "photoLocation": "/client/predicted/a.png",
            "prefabName": "P_Photograph",
        }
    ]
    r = client.post(f"/api/sync/guide-upload-attempts/{aid}/commit", json={"guide": g})
    assert r.status_code == 200
    stored = r.json()
    loc = stored["steps"][0]["objects"][0]["photoLocation"]
    assert loc.startswith(f"/api/sync/assets/{gid}/images/")
    assert "sha256" in loc  # content-addressed immutable final

    fetched = client.get(loc)
    assert fetched.status_code == 200 and fetched.content == payload

    # committed guide doc carries the new title
    assert stored["title"] == "with-media"

    # media referenced by the guide → delete is 409
    from kiara.guides.media_index import media_repository

    rows = media_repository.find_project_media_by_guide_id(gid)
    assert rows
    mid = rows[0]["id"]
    r = client.delete(f"/api/sync/media/{mid}")
    assert r.status_code == 409 and r.json()["detail"]["reason"] == "referenced"


def test_attempt_abort(client: TestClient):
    gid = f"abort-{uuid.uuid4().hex[:8]}"
    r = client.post(
        "/api/sync/guide-upload-attempts",
        json={"guideId": gid, "media": []},
    )
    assert r.status_code == 201
    aid = r.json()["attemptId"]
    r = client.delete(f"/api/sync/guide-upload-attempts/{aid}")
    assert r.status_code == 204
    r = client.delete(f"/api/sync/guide-upload-attempts/{aid}")
    assert r.status_code == 204  # abort is idempotent


# ── store + model changes + retired routes ────────────────────────────────────


def test_global_assets_stays_404(client: TestClient):
    for path in (
        "/api/sync/global_assets/tools/Arrow.glb",
        "/sync/global_assets",
        "/api/sync/global_assets",
    ):
        r = client.get(path)
        assert r.status_code == 404, path


def test_model_changes_and_store_surface(client: TestClient):
    r = client.get("/api/sync/model-changes", params={"guideId": "missing"})
    assert r.status_code in (200, 404)
    r = client.get("/api/sync/store")
    assert r.status_code == 200
    assert isinstance(r.json(), list)
    r = client.get("/api/sync/store/nonexistent")
    assert r.status_code in (400, 404)
    r = client.get(f"/api/sync/store/{uuid.uuid4()}")
    assert r.status_code == 404


def test_app_update_failsoft(client: TestClient):
    for path in (
        "/api/app/latest?platform=ml2",
        "/api/app/latest?platform=quest",
        "/api/app/download/ml2",
        "/api/app/latest?platform=bogus",
    ):
        r = client.get(path)
        assert r.status_code == 404, path  # never a 500 — the headset reads
        # non-200 as "no update available"


# ── instructions CRUD + derived-only translation ──────────────────────────────


def test_instructions_crud_cas(client: TestClient):
    gid = f"ins-{uuid.uuid4().hex[:8]}"
    r = client.post("/api/instructions", json=_doc(gid, title="Web"))
    assert r.status_code == 201
    g = r.json()

    r = client.patch(
        f"/api/instructions/{gid}/title",
        json={"title": "Web2", "expectedBaseVersion": g["version"]},
    )
    assert r.status_code == 200 and r.json()["title"] == "Web2"

    # stale web edit → same 409 discipline
    r = client.patch(
        f"/api/instructions/{gid}/title",
        json={"title": "Web3", "expectedBaseVersion": 99},
    )
    assert r.status_code == 409


# ── the deterministic emitter ─────────────────────────────────────────────────


def _fake_state(tmp_path) -> object:
    """Minimal GuideState-shaped object: two steps, one instance each."""
    from kiara.guide.state import (
        AssemblyInstance,
        AssemblyState,
        GuidePart,
        GuideState,
        GuideStep,
    )

    state = GuideState(project_id="p1", guide_version_id="v1")
    state.steps = [
        GuideStep(number=1, display_text="Setze Rohr A", required_parts=["Rohr A"]),
        GuideStep(
            number=2,
            display_text="Setze Bogen B",
            required_parts=["Bogen B"],
            safety_hint="Scharfe Kante",
        ),
    ]
    state.parts = [
        GuidePart(
            id="pipe_a",
            name="Rohr A",
            mesh_path=str(tmp_path / "pipe_a.glb"),
            bbox_min=[0, 0, 0],
            bbox_max=[100, 20, 20],
        ),
        GuidePart(
            id="elbow_b",
            name="Bogen B",
            mesh_path=str(tmp_path / "elbow_b.glb"),
            bbox_min=[0, 0, 0],
            bbox_max=[40, 40, 40],
        ),
    ]
    state.assembly = AssemblyState(
        instances=[
            AssemblyInstance(
                part_id="pipe_a",
                instance_id="pipe_a#1",
                transform=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                placed_in_step=1,
            ),
            AssemblyInstance(
                part_id="elbow_b",
                instance_id="elbow_b#1",
                transform=[[1, 0, 0, 50], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                placed_in_step=2,
            ),
        ],
        step_instances={"1": ["pipe_a#1"], "2": ["elbow_b#1"]},
    )
    return state


def test_guide_schema_emission(tmp_path):
    from kiara.guide.unity_schema import build_guide_doc

    state = _fake_state(tmp_path)
    doc = build_guide_doc(state, "guide-x", title="Emit")

    assert doc["guideId"] == "guide-x"
    assert doc["formatVersion"] == 2
    assert len(doc["steps"]) == 2

    s1, s2 = doc["steps"]

    # step 1: text + cad(pipe_a, new) + photo + hero(pipe_a)
    s1_cad = [o for o in s1["objects"] if o.get("is3DModel")]
    assert any(o["prefabName"] == "pipe_a(Clone)" for o in s1_cad)
    new_objs = [o for o in s1_cad if o.get("isNew") and not o.get("isHeroPreview")]
    assert [o["instanceId"] for o in new_objs] == ["pipe_a#1"]
    assert new_objs[0]["color"] == "#FFD700" and new_objs[0]["opacity"] == 0.85
    assert new_objs[0]["modelLocation"].endswith("/api/assets/guide-x/cad_models/pipe_a.glb")
    assert new_objs[0]["sourceUnits"] == "m"
    assert new_objs[0]["poseOrigin"] == "web" and new_objs[0]["localOverride"] is False

    # heroes: isHeroPreview + hero color, shared instance lineage
    heroes = [o for o in s1_cad if o.get("isHeroPreview")]
    assert [o["instanceId"] for o in heroes] == ["pipe_a#1"]
    assert heroes[0]["color"] == "#ff6600"

    # step 2 is CUMULATIVE: pipe_a context (dimmed) + elbow new
    s2_cad = [o for o in s2["objects"] if o.get("is3DModel") and not o.get("isHeroPreview")]
    by_iid = {o["instanceId"]: o for o in s2_cad}
    assert set(by_iid) == {"pipe_a#1", "elbow_b#1"}
    assert by_iid["pipe_a#1"]["isNew"] is False
    assert by_iid["pipe_a#1"]["color"] == "#808080"
    assert by_iid["elbow_b#1"]["isNew"] is True
    assert by_iid["pipe_a#1"]["introducedAtStep"] == 1
    assert by_iid["elbow_b#1"]["introducedAtStep"] == 2

    # stable lineage: same instance → same originalObjectId across steps
    s1_pipe = next(o for o in s1_cad if o["instanceId"] == "pipe_a#1" and not o.get("isHeroPreview"))
    assert s1_pipe["originalObjectId"] == by_iid["pipe_a#1"]["originalObjectId"]

    # text field conventions (P_MovableTextFieldWithButton, poseOrigin unity)
    texts = [o for o in s2["objects"] if o.get("isTextField")]
    assert texts and texts[0]["prefabName"] == "P_MovableTextFieldWithButton(Clone)"
    assert texts[0]["poseOrigin"] == "unity" and texts[0]["localOverride"] is True
    assert texts[0]["text"] == "Setze Bogen B"

    # photo object: shared objectId across both steps, P_Photograph
    p1 = next(o for o in s1["objects"] if o.get("prefabName") == "P_Photograph")
    p2 = next(o for o in s2["objects"] if o.get("prefabName") == "P_Photograph")
    assert p1["objectId"] == p2["objectId"]

    # semantic step metadata
    assert s2["semantic"]["actionType"] == "assemble"
    assert s2["semantic"]["safetyHint"] == "Scharfe Kante"
    assert s2["semantic"]["focusParts"] == ["Bogen B"]

    # determinism: same state → identical doc
    assert build_guide_doc(state, "guide-x", title="Emit") == doc


def test_emit_yaw_compensation_toggle(tmp_path):
    from kiara.guide.unity_schema import build_guide_doc

    state = _fake_state(tmp_path)
    on = build_guide_doc(state, "g", emit_yaw_compensation=True)
    off = build_guide_doc(state, "g", emit_yaw_compensation=False)
    cad_on = next(o for o in on["steps"][0]["objects"] if o.get("is3DModel"))
    cad_off = next(o for o in off["steps"][0]["objects"] if o.get("is3DModel"))
    # yaw compensation changes the emitted rotation only, never position
    assert cad_on["pos"] == cad_off["pos"]
    assert cad_on["rot"] != cad_off["rot"]


# ── publish pipeline ──────────────────────────────────────────────────────────


def test_publish_materializes_and_links(tmp_path):
    from kiara.guides.publish import guide_id_for_project, publish_guide

    state = _fake_state(tmp_path)
    for p in state.parts:
        p.mesh_path = str(tmp_path / f"{p.id}.glb")
        (tmp_path / f"{p.id}.glb").write_bytes(b"GLB-FAKE")
    pid = uuid.uuid4()
    vid = uuid.uuid4()
    stored = publish_guide(state, project_id=pid, version_id=vid, title="Pub")
    gid = guide_id_for_project(pid)
    assert stored["guideId"] == gid
    # second publish upserts the SAME guide (project↔guide 1:1)
    stored2 = publish_guide(state, project_id=pid, version_id=uuid.uuid4(), title="Pub2")
    assert stored2["guideId"] == gid
    assert stored2["version"] >= stored["version"]
    # cad models landed under the guide tree
    from kiara.guides.paths import guide_dir

    assert (guide_dir(gid) / "cad_models" / "pipe_a.glb").is_file()


def test_needs_review_publishes_fehler(tmp_path):
    from kiara.guides.publish import publish_run_dir

    state = _fake_state(tmp_path)
    state_dir = tmp_path / "st"
    state_dir.mkdir()
    state.save(state_dir)
    r = publish_run_dir(
        state_dir, project_id=uuid.uuid4(), version_id=uuid.uuid4(), status="needs_review"
    )
    assert r is not None and r["title"].startswith("[FEHLER]")
    assert publish_run_dir(state_dir, project_id=uuid.uuid4(), version_id=uuid.uuid4(), status="failed") is None
