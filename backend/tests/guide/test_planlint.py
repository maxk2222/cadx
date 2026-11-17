"""plan_lint: deterministic plan-vs-registry contradiction checks."""

from __future__ import annotations

from types import SimpleNamespace

from cadx.guide.planlint import plan_lint
from cadx.guide.state import AssemblyState, GuidePart, GuideState, GuideStep, GuideSubStep


def _part(pid: str) -> GuidePart:
    return GuidePart(id=pid, name=pid, bbox_min=[0, 0, 0], bbox_max=[1, 1, 1])


def _sess(parts, steps) -> SimpleNamespace:
    state = GuideState(output_dir="/tmp/x")
    state.parts = parts
    state.steps = steps
    state.assembly = AssemblyState()
    return SimpleNamespace(state=state, _aliases_store={}, state_dir="/tmp/x")


def _step(no, subs, atype="assembly") -> GuideStep:
    return GuideStep(
        number=no,
        title=f"s{no}",
        text="",
        action_type=atype,
        sub_steps=[GuideSubStep(**s) for s in subs],
    )


def test_clean_plan_no_blockers() -> None:
    sess = _sess(
        [_part("casing"), _part("adapter"), _part("pipe")],
        [
            _step(1, [{"action": "place", "part": "adapter", "target": "casing"}]),
            _step(2, [{"action": "place", "part": "pipe", "target": "adapter"}]),
        ],
    )
    assert not [f for f in plan_lint(sess) if f["severity"] == "blocker"]


def test_unresolvable_target_routes_to_parts() -> None:
    """Unresolvable placement targets route to ``parts`` — the steps
    mission runs BEFORE parts exist and cannot alias; name-landscape
    fixes (alias_part / register) live in the parts mission."""
    sess = _sess(
        [_part("adapter"), _part("pipe")],
        [_step(1, [{"action": "place", "part": "pipe", "target": "mystery mount"}])],
    )
    fs = plan_lint(sess)
    b = [f for f in fs if f["severity"] == "blocker"]
    assert b and all(f["mission"] == "parts" for f in b)
    assert "mystery mount" in b[0]["text"]


def test_unresolvable_part_routes_to_parts() -> None:
    sess = _sess(
        [_part("pipe")],
        [_step(1, [{"action": "place", "part": "gizmoid 3000", "target": "pipe"}])],
    )
    fs = plan_lint(sess)
    b = [f for f in fs if f["severity"] == "blocker"]
    assert any(f["mission"] == "parts" for f in b)


def test_target_placed_later_is_ordering_blocker() -> None:
    sess = _sess(
        [_part("adapter"), _part("pipe")],
        [
            _step(1, [{"action": "place", "part": "pipe", "target": "adapter"}]),
            _step(3, [{"action": "place", "part": "adapter", "target": ""}]),
        ],
    )
    b = [f for f in plan_lint(sess) if f["severity"] == "blocker"]
    assert any("first placed in step 3" in f["text"] for f in b)


def test_anchor_target_is_no_ordering_blocker() -> None:
    """The casing is never placed — preassembly anchors it; referencing it
    from step 1 is the normal pattern, not an ordering bug."""
    sess = _sess(
        [_part("casing"), _part("adapter")],
        [_step(1, [{"action": "place", "part": "adapter", "target": "casing"}])],
    )
    assert not [f for f in plan_lint(sess) if f["severity"] == "blocker"]


def test_same_step_chain_is_fine() -> None:
    """Pipe onto adapter placed in the SAME step — the first placement
    exists by the time the second attaches."""
    sess = _sess(
        [_part("adapter"), _part("pipe")],
        [
            _step(
                2,
                [
                    {"action": "place", "part": "adapter", "target": ""},
                    {"action": "place", "part": "pipe", "target": "adapter"},
                ],
            )
        ],
    )
    assert not [f for f in plan_lint(sess) if f["severity"] == "blocker"]


def test_empty_assembly_step_warns_only() -> None:
    sess = _sess(
        [_part("pipe")],
        [_step(1, [{"action": "check", "part": "", "target": ""}])],
    )
    fs = plan_lint(sess)
    assert not [f for f in fs if f["severity"] == "blocker"]
    assert any(f["severity"] == "warning" for f in fs)


def test_quantity_overflow_routes_to_steps() -> None:
    """Kiara bare failure shape: the plan mapped the 2-piece elbow onto
    five steps — no geometry repair can satisfy it, the steps mission owns
    the fix (wrong part mapping / phantom occurrences)."""
    elbow = _part("elbow_ss")
    elbow.quantity = 2
    sess = _sess(
        [elbow],
        [
            _step(n, [{"action": "attach", "part": "elbow_ss", "target": ""}])
            for n in (1, 2, 3, 4, 5)
        ],
    )
    b = [f for f in plan_lint(sess) if f["severity"] == "blocker"]
    assert any(
        f["mission"] == "steps" and "5x 'elbow_ss'" in f["text"] and "only 2" in f["text"]
        for f in b
    ), b


def test_quantity_none_unbounded() -> None:
    """Unknown quantity must not lint — many manuals list no counts."""
    sess = _sess(
        [_part("pipe")],
        [
            _step(n, [{"action": "attach", "part": "pipe", "target": ""}])
            for n in (1, 2, 3, 4, 5, 6)
        ],
    )
    assert not [f for f in plan_lint(sess) if "requires" in f["text"]]


def test_host_without_incoming_mates_is_blocker() -> None:
    """The largest registered part is the assembly's base — a plan mating
    nothing to it produces a satellite disconnected from the host (kiara
    heatpump: adapters stacked onto each other, zero casing joints)."""
    from cadx.guide.planlint import plan_lint
    from cadx.guide.state import GuidePart

    big = GuidePart(id="casing", name="Gehäuse", bbox_min=[0, 0, 0], bbox_max=[10, 10, 10])
    small = GuidePart(id="adapter", name="Adapter", bbox_min=[0, 0, 0], bbox_max=[1, 1, 1])
    sess = _sess(
        [big, small],
        [_step(1, [{"action": "attach", "part": "adapter", "target": "adapter"}])],
    )
    fs = plan_lint(sess)
    anchor = [f for f in fs if "host/base" in f["text"]]
    assert anchor and anchor[0]["mission"] == "parts", fs
    # once a placement targets the big part the flag clears
    sess2 = _sess(
        [big, small],
        [_step(1, [{"action": "attach", "part": "adapter", "target": "casing"}])],
    )
    fs2 = plan_lint(sess2)
    assert not [f for f in fs2 if "host/base" in f["text"]], fs2


def test_context_target_routes_to_parts_for_aliasing() -> None:
    """A placement target that IS context (a nozzle/port feature name) is
    fixed by alias_part onto the enclosing part — only the parts mission
    has that tool."""
    from cadx.guide.planlint import plan_lint

    sess = _sess(
        [_part("casing"), _part("adapter")],
        [_step(1, [{"action": "attach", "part": "adapter", "target": "Anschlussrohr"}])],
    )
    sess.state.context_names = ["Anschlussrohr"]
    fs = plan_lint(sess)
    ctx = [f for f in fs if "Anschlussrohr" in f["text"]]
    assert ctx and ctx[0]["mission"] == "parts", fs


def test_nested_context_target_is_blocked_not_guessed() -> None:
    """A target containing a declared context name denotes a feature of
    the device — its qualifier tokens must never guess a kit part
    ('oberes Anschlussrohr' bound pipe_l500 through a single-owner
    alias token and spawned phantom premounts). Once the feature is
    aliased onto the host the same reference resolves cleanly."""
    pipe = _part("pipe_l500")
    pipe.name = "Rohr DN22, l = 500 mm"
    sess = _sess(
        [_part("casing"), _part("adapter"), pipe],
        [
            _step(
                1,
                [
                    {
                        "action": "attach",
                        "part": "adapter",
                        "target": "vormontiertes linkes oberes Anschlussrohr "
                        "in der Verkleidung",
                    }
                ],
            )
        ],
    )
    sess.state.context_names = ["Verkleidung", "Anschlussrohr"]
    # the trap alias: 'oberes' is owned only by pipe_l500 — token scoring
    # would bind the device stub to a 500mm pipe
    sess._aliases_store = {"freies oberes ende des linken rohrs": "pipe_l500"}
    fs = plan_lint(sess)
    b = [
        f
        for f in fs
        if f["severity"] == "blocker" and "Anschlussrohr" in f["text"]
    ]
    assert b and b[0]["mission"] == "parts", fs
    # the repair path: alias the context feature onto the host part —
    # the reference then resolves to the host, not a kit part
    sess._aliases_store["anschlussrohr"] = "casing"
    fs = plan_lint(sess)
    assert not [
        f
        for f in fs
        if f["severity"] == "blocker" and "Anschlussrohr" in f["text"]
    ], fs


def test_detected_qty_bounds_plan_when_register_omits() -> None:
    """Register calls that omit `quantity` still get bounded by the
    Stückliste numbers the doc mission captured (heatpump: qty=None →
    5x l500 placed for a 2-piece kit)."""
    from cadx.guide.planlint import plan_lint
    from cadx.guide.state import DocumentIndex

    p = _part("pipe_l500")
    p.name = "Rohr DN22, l = 500 mm"
    sess = _sess(
        [p],
        [
            _step(i, [{"action": "insert", "part": "pipe_l500", "target": "elbow"}])
            for i in (1, 2, 3, 4)
        ],
    )
    sess.state.document = DocumentIndex(
        detected_part_names=["Rohr DN22, l = 500 mm"],
        detected_part_qty={"rohr dn22 l 500 mm": 2},
    )
    fs = plan_lint(sess)
    qf = [f for f in fs if "ships only" in f["text"]]
    assert qf and "4x" in qf[0]["text"] and qf[0]["mission"] == "steps", fs


def test_split_name_qty_variants() -> None:
    from cadx.guide.doc_tools import _split_name_qty

    assert _split_name_qty("2x Rohr DN22") == ("Rohr DN22", 2, 0)
    assert _split_name_qty("Rohr DN22 x2") == ("Rohr DN22", 2, 0)
    assert _split_name_qty("Rohr DN22 (2x)") == ("Rohr DN22", 2, 0)
    assert _split_name_qty("Adapter DN22 — 2 Stück") == ("Adapter DN22", 2, 0)
    assert _split_name_qty({"name": "Rohr", "qty": 3}) == ("Rohr", 3, 0)
    assert _split_name_qty("Rohr DN22, l = 500 mm") == ("Rohr DN22, l = 500 mm", None, 0)
    # "N + M vormontiert" split: kit copies + device-mounted copies
    assert _split_name_qty("Adapter DN22, 2 + 2 (vormontiert)") == ("Adapter DN22", 4, 2)
    assert _split_name_qty("Adapter DN22 2 + 2 premounted") == ("Adapter DN22", 4, 2)
    # plain "N vormontiert" — the whole count is device-side already
    assert _split_name_qty("Halterung 2 vormontiert") == ("Halterung", 2, 2)


def _sess_with_assets(parts, steps, assets) -> SimpleNamespace:
    s = _sess(parts, steps)
    s.guide_assets = assets
    return s


def _upart(pid: str, aid: str) -> GuidePart:
    p = _part(pid)
    p.asset_id = aid
    return p


def test_unregistered_cad_asset_is_parts_blocker() -> None:
    """An uploaded CAD file mapped to no registered part means the plan
    silently dropped real geometry — observed: the host casing marked
    context, leaving nothing to anchor to (satellite assembly)."""
    sess = _sess_with_assets(
        [_upart("pipe", "a-pipe"), _upart("elbow", "a-elbow")],
        [_step(1, [{"action": "place", "part": "elbow", "target": "pipe"}])],
        [
            {"id": "a-pipe", "filename": "Pipe.glb", "kind": "model"},
            {"id": "a-elbow", "filename": "Elbow.glb", "kind": "model"},
            {"id": "a-casing", "filename": "Casing.step", "kind": "model"},
            {"id": "a-pdf", "filename": "manual.pdf", "kind": "document"},
        ],
    )
    fs = plan_lint(sess)
    b = [f for f in fs if "Casing.step" in f["text"]]
    assert b and b[0]["mission"] == "parts" and b[0]["severity"] == "blocker", fs
    assert not [f for f in fs if "manual.pdf" in f["text"]], fs


def test_all_cad_assets_registered_clean() -> None:
    sess = _sess_with_assets(
        [_upart("pipe", "a-pipe"), _upart("casing", "Casing.step")],
        [_step(1, [{"action": "place", "part": "pipe", "target": "casing"}])],
        [
            {"id": "a-pipe", "filename": "Pipe.glb", "kind": "model"},
            {"id": "a-casing", "filename": "Casing.step", "kind": "model"},
        ],
    )
    # filename also counts as the registered key (forgiving asset lookup
    # accepts filenames, not only uuids)
    assert not [f for f in plan_lint(sess) if "maps to no registered" in f["text"]]


def test_plan_lint_reopen_uncapped_returns_fatal() -> None:
    """Persistent blockers after the reopen cap must not reach assembly —
    only the capped owning missions can fix them, so they are fatal."""
    from pathlib import Path
    from tempfile import mkdtemp

    from cadx.guide.guide_runner import _plan_lint_reopen

    sess = _sess(
        [_part("pipe")],
        [_step(1, [{"action": "place", "part": "pipe", "target": "ghost mount"}])],
    )
    sess.state.missions_done = ["doc", "steps", "parts"]
    sdir = Path(mkdtemp())
    notes, fatal = _plan_lint_reopen(sess, sess.state, sdir, allow_reopen=False)
    assert notes is None and fatal and "ghost mount" in fatal[0]
    assert sess.state.missions_done == ["doc", "steps", "parts"]

    notes, fatal = _plan_lint_reopen(sess, sess.state, sdir, allow_reopen=True)
    assert notes and "ghost mount" in notes and not fatal
    assert "parts" not in sess.state.missions_done  # owning mission reopened


def test_side_coverage_blocker_for_uncovered_side() -> None:
    """'Adapter an der unteren Anschlussstelle' while the plan only
    places adapters at the upper connections — the reference has no
    matching instance; without a marker it must block, not silently
    bind the upper adapter (observed: chains built top-side, verdict
    'succeeded' on the wrong topology)."""
    sess = _sess(
        [_part("casing"), _part("adapter"), _part("elbow")],
        [
            _step(1, [{"action": "place", "part": "adapter",
                       "target": "linker oberer Anschluss am Gehäuse"}]),
            _step(2, [{"action": "place", "part": "adapter",
                       "target": "rechter oberer Anschluss am Gehäuse"}]),
            _step(3, [{"action": "place", "part": "elbow",
                       "target": "Adapter an der linken unteren Anschlussstelle"}]),
        ],
    )
    fs = plan_lint(sess)
    b = [f for f in fs if f["severity"] == "blocker" and "Anschlussstelle" in f["text"]]
    assert b and b[0]["mission"] == "steps", fs
    # the repair hint must name the resolved part — a literal '{tpid}'
    # placeholder slipped through baseline4-r3 and left the model
    # without the concrete id to place
    assert "'adapter'" in b[0]["text"] and "{tpid}" not in b[0]["text"], b[0]


def test_side_coverage_ok_when_matched() -> None:
    """'(unteres Ende) des linken Rohrs' references the placed LEFT pipe —
    end-feature words are stripped, {links} ⊆ {links,oben} → no blocker."""
    sess = _sess(
        [_part("casing"), _part("pipe"), _part("elbow")],
        [
            _step(1, [{"action": "place", "part": "pipe",
                       "target": "linker oberer Anschluss am Gehäuse"}]),
            _step(2, [{"action": "place", "part": "pipe",
                       "target": "rechter oberer Anschluss am Gehäuse"}]),
            _step(3, [{"action": "place", "part": "elbow",
                       "target": "linkes Rohr (unteres Ende)"}]),
        ],
    )
    fs = plan_lint(sess)
    assert not [f for f in fs if "no matching instance" in f["text"]
                or "locates a" in f["text"]], fs


def test_side_coverage_exempt_with_premount_marker() -> None:
    """'vormontierter Adapter am linken unteren Anschluss' — the marker
    declares a device-side piece; preassembly spawns it, lint exempt."""
    sess = _sess(
        [_part("casing"), _part("adapter"), _part("elbow")],
        [
            _step(1, [{"action": "place", "part": "adapter",
                       "target": "linker oberer Anschluss am Gehäuse"}]),
            _step(2, [{"action": "place", "part": "adapter",
                       "target": "rechter oberer Anschluss am Gehäuse"}]),
            _step(3, [{"action": "place", "part": "elbow",
                       "target": "vormontierter Adapter am linken unteren Anschluss"}]),
        ],
    )
    fs = plan_lint(sess)
    assert not [f for f in fs if "locates a" in f["text"]], fs


def _named_part(pid: str, name: str) -> GuidePart:
    return GuidePart(id=pid, name=name, bbox_min=[0, 0, 0], bbox_max=[1, 1, 1])


def test_canonicalize_binds_pids() -> None:
    """canonicalize_refs binds every substep reference once — consumers
    read *_pid instead of re-running fuzzy resolution."""
    sess = _sess(
        [_part("casing"), _part("adapter")],
        [_step(1, [{"action": "place", "part": "adapter", "target": "casing"}])],
    )
    plan_lint(sess)
    ss = sess.state.steps[0].sub_steps[0]
    assert ss.part_pid == "adapter"
    assert ss.target_pid == "casing"
    assert ss.target2_pid is None


def test_ambiguous_name_is_blocker_not_guess() -> None:
    """The heatpump failure: 'Rohrbogen' contains in BOTH elbow names —
    silently best-of picking mated the bridge elbow where the premount
    elbow was meant. Ties surface as a steps blocker with candidates."""
    sess = _sess(
        [
            _named_part("elbow_io", "Rohrbogen DN22 1x aussen 1x innen"),
            _named_part("elbow_oo", "Rohrbogen DN22 2x aussen"),
            _part("casing"),
        ],
        [
            _step(
                1,
                [
                    {
                        "action": "place",
                        "part": "Rohrbogen",
                        "target": "casing",
                    }
                ],
            )
        ],
    )
    fs = plan_lint(sess)
    b = [f for f in fs if f["severity"] == "blocker" and "ambiguous" in f["text"]]
    assert b, fs
    assert all(f["mission"] == "steps" for f in b)
    assert "elbow_io" in b[0]["text"] and "elbow_oo" in b[0]["text"]
    assert sess.state.steps[0].sub_steps[0].part_pid is None


def test_ambiguous_target_reports_candidates() -> None:
    """Replay of the failed run: the manual's generic kind-name 'Rohrbogen
    DN22' (no 1x/2x discriminator) is a substring of BOTH elbow names —
    it must block, not silently bind either kind."""
    sess = _sess(
        [
            _named_part("elbow_io", "Rohrbogen DN22 1x aussen 1x innen"),
            _named_part("elbow_oo", "Rohrbogen DN22 2x aussen"),
            _named_part("pipe", "Rohr DN22 l=500"),
            _part("casing"),
        ],
        [
            _step(1, [{"action": "place", "part": "Rohr DN22 l=500", "target": "casing"}]),
            _step(
                2,
                [
                    {
                        "action": "place",
                        "part": "Rohrbogen DN22",
                        "target": "Rohr DN22 l=500",
                    }
                ],
            ),
        ],
    )
    fs = plan_lint(sess)
    assert any(
        f["severity"] == "blocker" and "ambiguous" in f["text"]
        and "elbow_io" in f["text"] and "elbow_oo" in f["text"]
        for f in fs
    ), fs


def test_exact_part_id_bypasses_ambiguity() -> None:
    """The repair path: once the name is qualified (detected part name or
    part_id), resolution binds exactly — no ambiguity remains."""
    sess = _sess(
        [
            _named_part("elbow_io", "Rohrbogen DN22 1x aussen 1x innen"),
            _named_part("elbow_oo", "Rohrbogen DN22 2x aussen"),
            _part("casing"),
        ],
        [
            _step(
                1,
                [
                    {
                        "action": "place",
                        "part": "Rohrbogen DN22 1x aussen 1x innen",
                        "target": "casing",
                    }
                ],
            )
        ],
    )
    fs = plan_lint(sess)
    assert not [f for f in fs if f["severity"] == "blocker" and "ambiguous" in f["text"]], fs
    assert sess.state.steps[0].sub_steps[0].part_pid == "elbow_io"


def _adapter_plan(pin_target: str, step2_target: str = "linker oberer Anschluss") -> SimpleNamespace:
    """Steps mimicking the observed mislabel: adapters placed at plan
    steps 2/3 but referenced as '(Schritt 1)/(Schritt 2)'."""
    return _sess(
        [_part("casing"), _part("adapter_dn22"), _part("pipe_l70")],
        [
            _step(1, [], atype="prep"),
            _step(
                2,
                [
                    {
                        "action": "fasten",
                        "part": "adapter_dn22",
                        "target": step2_target,
                    }
                ],
            ),
            _step(
                3,
                [
                    {
                        "action": "fasten",
                        "part": "adapter_dn22",
                        "target": "rechter oberer Anschluss",
                    }
                ],
            ),
            _step(
                4,
                [{"action": "insert", "part": "pipe_l70", "target": pin_target}],
            ),
        ],
    )


def test_step_ref_pin_on_wrong_step_blocks() -> None:
    """'Bohrung des Adapters (Schritt 1)' — adapters live in plan steps
    2/3; pinning an empty step yields an honest blocker instead of a
    runtime free-fall onto any placed instance."""
    sess = _adapter_plan("Bohrung des linken oberen Adapters (Schritt 1)")
    fs = plan_lint(sess)
    assert any(
        f["severity"] == "blocker" and "Schritt 1" in f["text"] and "step 4" in f["text"]
        for f in fs
    ), fs


def test_step_ref_side_contradiction_blocks() -> None:
    """'rechter Adapter (Schritt 2)' — step 2 places the 'linker'
    adapter: pin and side word contradict, deterministically."""
    sess = _adapter_plan("Bohrung des rechten oberen Adapters (Schritt 2)")
    fs = plan_lint(sess)
    assert any(
        f["severity"] == "blocker" and "Schritt 2" in f["text"] and "side" in f["text"]
        for f in fs
    ), fs


def test_step_ref_consistent_pin_passes() -> None:
    sess = _adapter_plan("Bohrung des linken oberen Adapters (Schritt 2)")
    fs = plan_lint(sess)
    assert not [f for f in fs if "pins '(Schritt" in f["text"]], fs


def test_step_ref_blocker_names_resolved_sides() -> None:
    """The blocker must name the RESOLVED sides — 'places that kind at
    right/top while the ref says left/top' — not a bare 'different
    side' (the vague text survived three repair rounds in the wild
    because the model never learned which side the pin step places)."""
    sess = _adapter_plan("Bohrung des linken oberen Adapters (Schritt 3)")
    fs = plan_lint(sess)
    side = [f for f in fs if "Schritt 3" in f["text"] and "side" in f["text"]]
    assert side, fs
    assert "right" in side[0]["text"] and "left" in side[0]["text"], side[0]


def test_add_sub_step_refuses_pin_side_contradiction() -> None:
    """Same check at write time: 'links (Schritt 3)' while step 3 places
    the right instance is refused immediately, naming both sides."""
    import asyncio

    from cadx.guide.step_tools import add_sub_step

    sess = _adapter_plan("irrelevant")
    r = asyncio.run(
        add_sub_step(
            sess,
            {
                "step": 4,
                "action": "insert",
                "part": "pipe_l70",
                "target": "linker oberer Adapter (Schritt 3)",
            },
        )
    )
    assert not r["ok"], r
    assert "right" in r["error"] and "left" in r["error"], r["error"]
    # refused write leaves no half-written sub-step
    assert len(sess.state.step_by_number(4).sub_steps) == 1


def test_add_sub_step_consistent_pin_and_forward_pin_pass() -> None:
    import asyncio

    from cadx.guide.step_tools import add_sub_step

    sess = _adapter_plan("irrelevant")
    r = asyncio.run(
        add_sub_step(
            sess,
            {
                "step": 4,
                "action": "insert",
                "part": "pipe_l70",
                "target": "rechter oberer Adapter (Schritt 3)",
            },
        )
    )
    assert r["ok"], r
    # forward pin — step 9 does not exist yet; lint judges, write tolerates
    r = asyncio.run(
        add_sub_step(
            sess,
            {
                "step": 4,
                "action": "insert",
                "part": "pipe_l70",
                "target": "ein Adapter (Schritt 9)",
            },
        )
    )
    assert r["ok"], r


def test_update_sub_step_refuses_pin_contradiction_without_mutating() -> None:
    import asyncio

    from cadx.guide.step_tools import update_sub_step

    sess = _adapter_plan("irrelevant")
    step4 = sess.state.step_by_number(4)
    before = step4.sub_steps[0].target
    r = asyncio.run(
        update_sub_step(
            sess,
            {
                "step": 4,
                "index": 0,
                "target": "linker oberer Adapter (Schritt 3)",
            },
        )
    )
    assert not r["ok"], r
    assert step4.sub_steps[0].target == before  # no half-edit


def _u_plan(partner_bore: str):
    """Two-loop plan: anchor → adapter at a bore → elbow bends 'nach
    unten' → l500 descends → closing elbow bridges pipe + partner bore."""
    parts = [_part(p) for p in ("casing", "adapter", "elbow", "pipe", "stub", "socket")]
    steps = [
        _step(1, [{"action": "place", "part": "casing", "target": "device"}]),
        _step(
            2,
            [
                {
                    "action": "insert",
                    "part": "adapter",
                    "target": "casing, untere linke Bohrung",
                    "text": "Adapter in die untere linke Bohrung einsetzen",
                }
            ],
        ),
        _step(
            3,
            [
                {
                    "action": "insert",
                    "part": "stub",
                    "target": "casing, obere linke Bohrung",
                    "text": "Stutzen in die obere linke Bohrung stecken",
                }
            ],
        ),
        _step(
            4,
            [
                {
                    "action": "insert",
                    "part": "elbow",
                    "target": "adapter links",
                    "text": "Bogen auf den Adapter, biegt nach unten ab",
                }
            ],
        ),
        _step(
            5,
            [
                {
                    "action": "insert",
                    "part": "pipe",
                    "target": "elbow links",
                    "text": "Rohr in den Bogen, vertikale Verbindung nach unten",
                }
            ],
        ),
        _step(
            6,
            [
                {
                    "action": "place",
                    "part": "socket",
                    "target": "pipe links (Schritt 5)",
                    "target2": f"stub links {partner_bore} (Schritt 3)",
                    "text": "Muffe auf das Rohr, freie Seite auf den Stutzen",
                }
            ],
        ),
    ]
    return _sess(parts, steps)


def test_bridge_away_from_partner_blocks() -> None:
    """The chain grows 'nach unten' from the LOWER bore while the
    declared mate sits at the UPPER bore — a 90° fitting at the chain's
    far end can never reach back past its own anchor (KIARA run: l500
    descended 500mm and was declared to bridge up to the l70 stubs)."""
    sess = _u_plan("oben")
    fs = plan_lint(sess)
    assert any(
        f["severity"] == "blocker"
        and "grows away" in f["text"]
        and "step 6" in f["text"]
        for f in fs
    ), fs


def test_bridge_toward_partner_passes() -> None:
    """Chain grows down from the LOWER bore and the partner is the
    'unten' stub BELOW the root — descending reaches it, no finding."""
    parts = [_part(p) for p in ("casing", "adapter", "elbow", "pipe", "stub", "socket")]
    steps = [
        _step(1, [{"action": "place", "part": "casing", "target": "device"}]),
        _step(
            2,
            [
                {
                    "action": "insert",
                    "part": "adapter",
                    "target": "casing, obere linke Bohrung",
                    "text": "Adapter in die obere linke Bohrung einsetzen",
                }
            ],
        ),
        _step(
            3,
            [
                {
                    "action": "insert",
                    "part": "stub",
                    "target": "casing, untere linke Bohrung",
                    "text": "Stutzen in die untere linke Bohrung stecken",
                }
            ],
        ),
        _step(
            4,
            [
                {
                    "action": "insert",
                    "part": "elbow",
                    "target": "adapter links",
                    "text": "Bogen auf den Adapter, biegt nach unten ab",
                }
            ],
        ),
        _step(
            5,
            [
                {
                    "action": "insert",
                    "part": "pipe",
                    "target": "elbow links",
                    "text": "Rohr in den Bogen, vertikale Verbindung nach unten",
                }
            ],
        ),
        _step(
            6,
            [
                {
                    "action": "place",
                    "part": "socket",
                    "target": "pipe links (Schritt 5)",
                    "target2": "stub links unten (Schritt 3)",
                    "text": "Muffe auf das Rohr, freie Seite auf den Stutzen",
                }
            ],
        ),
    ]
    sess = _sess(parts, steps)
    fs = plan_lint(sess)
    assert not [f for f in fs if "grows away" in f["text"]], fs


def test_premount_context_part_needs_no_marker() -> None:
    """A part that is never kit-placed is pure context (the host device)
    — every reference to it IS a premount reference; demanding the
    literal 'vormontiert' word adds nothing and deadlocks the repair
    loop (observed: 'Gehäuse linker Port' targets blocked because they
    lacked the marker while 'gehause' was BOM-premounted)."""
    from cadx.guide.state import DocumentIndex

    sess = _sess(
        [GuidePart(id="casing", name="gehause", bbox_min=[0, 0, 0], bbox_max=[1, 1, 1]),
         _part("adapter"), _part("pipe")],
        [
            _step(1, [{"action": "place", "part": "adapter", "target": "gehause linker Port"}]),
            _step(2, [{"action": "place", "part": "pipe", "target": "adapter"}]),
        ],
    )
    sess.state.document = DocumentIndex(detected_premount_qty={"gehause": 1})
    fs = plan_lint(sess)
    assert not [f for f in fs if "premounted target" in f["text"]], fs


def test_premount_kit_kind_still_needs_marker() -> None:
    """A kind that ships BOTH kit and device-mounted copies keeps the
    marker requirement — otherwise premount references silently bind the
    kit instance instead of spawning the mounted one."""
    from cadx.guide.state import DocumentIndex

    sess = _sess(
        [_part("casing"), _part("adapter"), _part("pipe")],
        [
            _step(1, [{"action": "place", "part": "adapter", "target": "casing"}]),
            _step(2, [{"action": "place", "part": "pipe", "target": "adapter"}]),
        ],
    )
    sess.state.document = DocumentIndex(detected_premount_qty={"adapter": 1})
    hits = [f for f in plan_lint(sess) if "premounted target" in f["text"]]
    assert hits, "adapter is kit-placed — the unmarked 'adapter' target cannot count"


def test_side_ref_finer_than_plan_sig_no_blocker() -> None:
    """'freie obere Öffnung des Rohrbogens links oben' resolves to the
    {l}-placed elbow even though the plan's signature is coarser —
    consistency on shared axes, not subset: the ref's extra 'u' word
    qualifies the feature end, it does not contradict the {l} slot.
    (R3R4: 2 reopen cycles died here while the {l} instance was the
    correct mate all along.)"""
    elbow = GuidePart(
        id="elbow_double_socket",
        name="Rohrbogen DN22, 2x außen",
        bbox_min=[0, 0, 0],
        bbox_max=[1, 1, 1],
    )
    sess = _sess(
        [elbow, _part("pipe")],
        [
            _step(1, [{"action": "place", "part": "elbow_double_socket",
                       "target": "linker Anschluss am Gehäuse"}]),
            _step(2, [{"action": "place", "part": "elbow_double_socket",
                       "target": "rechter Anschluss am Gehäuse"}]),
            _step(3, [{"action": "place", "part": "pipe",
                       "target": "freie obere Öffnung des Rohrbogens links oben"}]),
        ],
    )
    fs = plan_lint(sess)
    assert not [f for f in fs if "locates a" in f["text"]], fs


def test_side_ref_contradiction_still_blocks() -> None:
    """The consistency rule must keep real wrong-side targets a blocker:
    'unterer Adapter' can never bind the instance the plan places 'oben'."""
    sess = _sess(
        [_part("adapter"), _part("pipe")],
        [
            _step(1, [{"action": "place", "part": "adapter",
                       "target": "oberer Anschluss am Gehäuse"}]),
            _step(2, [{"action": "place", "part": "pipe",
                       "target": "unterer Adapter am Gehäuse"}]),
        ],
    )
    fs = plan_lint(sess)
    assert [f for f in fs if "locates a" in f["text"]], fs


def test_genitive_instance_ref_warns_not_blocks() -> None:
    """'des linken vertikalen Rohrs aus Schritt 7' — the genitive 'Rohrs'
    still names a placed instance by position; a warning, never a
    'matches no registered part' blocker (R3R2 parts re-death)."""
    sess = _sess(
        [_part("rohr"), _part("adapter")],
        [
            _step(1, [{"action": "place", "part": "rohr",
                       "target": "linker Anschluss"}]),
            _step(2, [{"action": "place", "part": "adapter",
                       "target": "freie obere Seite des linken vertikalen Rohrs aus Schritt 1"}]),
        ],
    )
    fs = plan_lint(sess)
    hits = [f for f in fs if "matches no registered part" in f["text"]
            and "Rohrs" in f["text"]]
    assert not hits, fs
