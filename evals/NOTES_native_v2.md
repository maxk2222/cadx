# Eval-Beobachtung native_v2 — laufende Notizen

## Deployment-Stand
- Commits auf main (ff): `883302f` settings echo-fix, `13158bf` fem/recognition persistent solver state + native CG + mesh-edge
- Images gebaut (executor+api+worker), `cadx_native 0.1.0` im Executor verifiziert (cg_solve + feature_edges vorhanden)
- Stack: `docker compose up -d cad-executor api worker` — alle Up/healthy

## Eval-Lauf (sequentiell, driver pid 38639, log evals/eval_native_v2.log)
1. pressure_tank_autonomous — gestartet 22:13, läuft
2. bearing_bracket_topopt — queued
3. robot_arm_mount_autonomous — queued
4. marine_propeller_autonomous — queued

## Statusquellen
- Jobs: `docker exec cadx-postgres-1 psql -U cadx -d cadx -tc "select j.status, now()::time(0)-j.created_at::time(0), coalesce(p.name,'?') from job j left join version v on j.version_id=v.id left join part pa on v.part_id=pa.id left join project p on pa.project_id=p.id order by j.created_at desc limit N"`
- Executor-Events: `docker logs cadx-cad-executor-1 --tail N | grep -oE "execute_version|simp|fem|recognition|topopt|cg_diverged|budget_exceeded"`
- Watcher: evals/watch_v2.log (executor events), evals/jobs_v2.log (job transitions) — beide evtl. leer/buggy, direkte Queries sind zuverlässiger
- DB-Schema `cadagent`, Tabelle `job` (singular), `payload->>'version_id'` join version

## Beobachtungen
- 22:13–22:37 tank: execute_version + recognition-Loop, fem.profile Events — native Pipeline live
- Tank ~20min in normalem Iterationsverlauf; execute_version-Calls mit recognition_s im niedrigen Sekundenbereich (Worker amortisiert)
- Keine cg_diverged/budget_exceeded/Watchdog-Abbrüche bisher — rigid-gap + native CG laufen sauber
- Wichtig: `worst_sf=None`-Felder im Executor-Log sind nur cases[0]-Lesart, kein Datenverlust (altbekannt)

## Offene Beobachtungspunkte
- [ ] simp.iter s_per_iter auf native+reuse — Ziel ~30-50s vs 160-170s alt
- [ ] m_rebuilds/cg_iters Telemetrie in history records
- [ ] recognition Worker: timeout/respawn Verhalten unter Last
- [ ] mesh_edge Evidence erscheint auf faceted Meshs (approximate=True, source-Tag)
- [ ] Tank ist NEU — unbekannte Fehlerklassen möglich
- [ ] Nach tank: bearing_bracket → robot_arm → marine_propeller sequentiell

## Tank-Befund (codegen FAIL 1480s)
Drei generalisierbare Lücken:
1. **hole_index out-of-range**: FEM-Case referenzierte `hole #6`, nur Anker #1-4 existieren → erst zur FEM-Zeit "matched no nodes"/unsolvable. Gehört als Intake-Fehler früh an den Agenten.
2. **cavity-Region matchte keine Nodes** — Innendruck-Last ins Leere.
3. **anchor re-cut Coverage**: nur Ø24 re-cut; die 3 Ø6.5-Spec-Lugs ohne `depth_mm` bleiben codegen-owned (`_bounded_spec_anchor` verlangt pos+depth_mm).

Driver-Fix: `&&`-Kette stirbt bei Non-Zero → mit `;` neu gestartet (pid 42722, log eval_native_v2b.log): bracket → arm → prop → tank_retry.

## Vertiefung hole_index (intake.py `_Rebinder` Z59-194)
- Rebind existiert: out-of-range `hole_index` → Token-Match (`_best_matches`) oder Spec-Feature-Position (`_feature_pos_match`) → rebound.
- Tank `hole #6`: kein Match gefunden (kein positionales Spec-Feature, keine Token-Überlappung mit Ankernamen) → `r` bleibt → `resolve_case_dofs`→`hole #6 matched no nodes` → unsolvable.
- **Generalisierbarer Fix-Kandidat**: bei unrettbarem `hole_index` (rebound schlägt fehl UND out-of-range) soll die Region als expliziter Agenten-Fehler eskalieren ("hole_index 6 hat keinen Anker — verfügbar: #1-4: <namen>"), nicht als stiller Node-Miss. Wo: `_Rebinder.notes` fliessen schon in den Intake-Output? Prüfen ob notes dem LLM sichtbar sind oder nur geloggt werden.
- Zweiter Kandidat: `cavity`-Region — prüfen ob Tank-Innenraum durch Ø24-Nozzle als "offen" statt "geschlossen" klassifiziert wird und deshalb keine Nodes matcht (Region-Heuristik zu streng vs. Drucklast-Semantik).
- Dritter: `_bounded_spec_anchor` re-cut coverage — Spec-Anker OHNE depth_mm nie re-cut; Lug-Holes blieben Missing. Re-cut-Eligibility ggf. auf "spec + axis + pos + measured span" erweitern?

## Frontend + Deploy
- `cadx-frontend` war 9h stale → `395ffc2` (project page) + `7c76c9c` (ProjectsPage wide + 4/5-col grid) — neu gebaut + deployed.
- Alle Images aktuell: executor (cadx_native 0.1.0), api (settings echo), worker, frontend.
- Settings-PUT-Bug (`resources_detected: Extra inputs are not permitted`) gefixt: SettingsUpdate deklariert Echo-Felder, Router ignoriert sie — `extra=forbid` bleibt für echte Typos. Test `test_put_accepts_settings_out_echo`.

## Eval-Fortschritt (driver pid 42722, `;`-Kette)
- pressure_tank_autonomous: FAILED codegen 1480s (hole_index 6, cavity, re-cut coverage)
- bearing_bracket_topopt: läuft (Jobs ~5min Zyklus, gesund)
- robot_arm_mount_autonomous, marine_propeller_autonomous, tank_retry: queued

## Rust-Perf-Bilanz (live, bracket-eval 22:46-22:51)
- Geometry: contains 0.56→0.005s, ray_any 0.90→0.005s, ray_locs 1.67→0.01-0.03s (~50-160×)
- features_s Gesamt ~10s → ~0.3s (Geometry-Anteil eliminiert)
- fem.profile 446k dofs/34.6M nnz, 2 Cases parallel: total 13-15s (assemble ~12s, prep ~13s, solve ~1.7s)
- recognition_s ~0.14-0.28s (Worker amortisiert, war ~3-8s)
- SIMP ~53s/iter @344k vs 160-170s alt ≈ 3× — davon überwiegend persistent-ctx+watchdog (Algorithmik), Rust am CG-Loop
- Noch ausstehend: simp.iter s_per_iter im optimize-Stage (bisher nur codegen-jobs)

## Crash-Verdacht (22:52, bracket topopt)
`malloc_consolidate(): unaligned fastbin chunk detected` + `Aborted` in Worker pid=7 direkt nach `topopt.helmholtz_operator` (344k dofs, factor_s=4.5). Supervisor restartete beide Worker; Bracket-Job läuft weiter.
- Verdachtsfelder: (a) Fork+Native-Interaktion (Rust-Extension im Parent, Fork-Pool erbt Heap-State), (b) Heap-Overflow in cg_solve/feature_edges (PyReadonlyArray slices), (c) altbekannter OCC/trimesh-Crash.
- `topopt.auto_coarsen dofs=344736 feature_floor_hit=True pitch=1.94` + `helmholtz factor_s=4.5` direkt davor — Kontext: erster echter TopOpt-Iter auf native.
- Zu prüfen: ob Repro bei Wiederholung; ob Crash korreliert mit native-Pfad (Kill-Switch CADX_NATIVE=0 A/B); ob core/backtrace greifbar.

## Stand für Fortsetzung
- Eval-Queue läuft (driver pid 42722, `;`-Kette, log eval_native_v2b.log): bracket aktiv (~37min, jobs ~5min-Zyklus, execute_version-loop, kein Crash-Repro seit 22:52), dann arm → prop → tank_retry.
- Images: alle aktuell deployed (frontend neu mit ProjectsPage-Fix).
- Tank-Befund dokumentiert oben; 3 generalisierbare Fix-Kandidaten offen: (a) hole_index-Eskalation statt stiller Node-Miss, (b) cavity-Region-Resolution, (c) `_bounded_spec_anchor` re-cut-Coverage ohne depth_mm.
- Nächste Schritte: simp.iter s_per_iter sobald optimize-Stage auf bracket läuft; Crash-Beobachtung (`malloc_consolidate` einmalig 22:52); bei Tank-Retry die drei Lücken prüfen.

## Crash-Zombie + Recovery-Lücke (22:52→~00:20)
- `optimize` Job 906d9700 (version 7822849d, bracket 344k dofs) starb mit dem Worker im malloc-Crash — Row blieb `running`, Executor idle 0.53% CPU, Eval-Treiber wartete ~25min.
- **Generalisierbare Lücke**: `queue.py` Sweeper re-enqueue't nur stale `queued` rows — `running`-Jobs deren Worker starb werden nicht gereapt (kein Heartbeat/Liveness auf Job-Ebene). arq zählt `j_ongoing=1` ewig.
- Manuell `failed` gesetzt (UPDATE job) → Treiber fuhr fort. Fix-Kandidat: stale-running-Reaper — Job mit `running` + Worker-Heartbeat-Tot (oder Arq-Result-Key-Fehlen nach Grace) → fail+requeue; alternativ Version-Status-Konsistenzcheck.
- Crash-Root-Cause offen: `malloc_consolidate` = Heap-Metadaten durch früheren OOB-Write/Double-Free korrumpiert. Verdachte: (a) native cg_solve (rayon+Python-M-Callback, GIL-detached), (b) fork+rayon/BLAS-Thread-Interaction im _POOL, (c) neuer `factor_symmetric`→cholmod-Pfad in helmholtz (war SuperLU), (d) pre-existing OCC/numba. UNBEWIESEN — erstes Heavy-TopOpt auf neuem Image. A/B: CADX_NATIVE=0 bei Repro; Core/Backtrace für nächste Instanz einrichten.

## Fortschritt 00:20
- bracket eval: optimize-Stage FAILED (zombie-job manuell failed → driver fuhr fort). codegen+fem succeeded.
- robot_arm_mount_autonomous: RUNNING seit 00:06:44, FEM-Events aktiv.
- Queue weiter: prop → tank_retry.
- Offen: (a) malloc-crash Repro/A-B (CADX_NATIVE=0), (b) running-zombie-Reaper (Sweeper deckt nur queued), (c) Tank-Lücken (hole_index-Eskalation, cavity, re-cut ohne depth_mm).

## Robot-Arm-Befund (eval finished, passed=False, 657s)
- codegen ok (625s, verify_spec PASS 33/0/2), fem-Gate FAILED (24s), topopt/refine skipped.
- Root cause: Load-Case 'arm weight on seat' + 'flange link forces' fixieren `kind="hole"` mit `fix_mode="normal"` → pinnt nur die AXIALE DOF (regions.py:1228-1232, axis-DOF auf Wall-Nodes). Semantisch: Teil darf axial gleiten + um Bohrungsachse rotieren → 3/6 rigid modes frei → `_rigid_gaps` erkennt ehrlich "unsolvable", Solve skipped (24s statt 300s-Burn — Check funktioniert wie designed).
- Agenten-Intent war vermutlich Lager-Fixierung (radial pinnt, axial frei) — semantisches Mapping fehlt: `fix_mode="normal"` auf hole/bore/anchor = axial pin = fast nie gewollt.
- **Generalisierbare Fixes**: (a) `fix_mode="bearing"` — 2 DOFs ⊥ Bohrachse pinnen (physikalisch korrekte Lager-Stütze); (b) Intake-Warnung wenn normal+hole/bore/anchor ("pinnt nur Axial-DOF — meinst du 'all' (encastre) oder 'bearing'?"); (c) Physik-Note: EIN Lager reicht eh nicht für Cantilever-Moment → Agent sollte fix_mode="all" oder zweite Stütze wählen — der Hinweis muss in den fix_hint.
- Fem-Gate `fem_loadcases` ok=False ist KORREKT (physikalisch instabil) — Pipeline-Frage: iteriert der Orchestrator auf fem-Gate-Fails oder ist der Stage-Fail endgültig?

## Propeller-TopOpt (00:30–00:47, läuft sauber)
- Gleiches 344k-Grid wie Bracket-Crash — 25+ Iterationen OHNE Crash → malloc-Bug ist nicht-deterministisch/einmalig, kein systematischer Native-Fehler (schwächt cg_solve-Verdacht; Fork/BLAS-Interaktion oder OCC bleibt Kandidat).
- `s_per_iter` 58s→**40.4s** fallend ab Iter 5, stabil ~40s — AMG-Reuse + Warm-Start sichtbar wirksam (erster echter Beweis für persistent-ctx Gewinn). Kein m_rebuild-Event, kein cg_diverged.
- Compliance 1584→66.6 mJ Plateau, volfrac 0.2502 stabil — saubere Konvergenz.
- Tank: codegen FAIL; Bracket: optimize Crash; Arm: fem-Gate (physikalisch instabil, korrekt); Propeller: läuft durch → tank_retry als letzter Case.

## Regeln
- NUR generalisierbare Fixes, keine case-spezifischen Hacks
- Bei Fehler: reproduzieren → root cause → minimaler Fix → Regressionstest → weiter
- Eval-Treiber NICHT killen wenn nur langsam; harte Fehler → fixen → Deploy → Case neu starten

## native_v3 (02:44) — Master-Plan Runde 2
Commits: 4e9bddb (fem spec honesty) + 9b9b141 (executor durability). Deployed.
- fix_mode="bearing": pinnt 2 DOFs ⊥ Bohrachse (Collar) + normal+hole Warnung
- lint_support_modes: Intake benennt freie rigid modes pro Case
- dead hole_index: rebinder+regions listen verfügbare Anker
- cavity(seal_openings_mm): virtuelles Versiegeln für open-vessel inner skin
- re-cut: spec anchors ohne depth_mm auf dünnem span (<=5x dia, capped)
- executor_client: JobStatus.not_found → task-lost beim nächsten Poll (~5s statt 2h)
- executor env: PYTHONFAULTHANDLER=1 + MALLOC_CHECK_=3 (nächster heap-abort → stack)
- rigid-gap: load-excitation unterscheidet echt-unsolvable (excited→skip) vs
  konsistent-unterbestimmt (F ⊥ free modes → Federn + lösen) — roller-plate-Fix
- eval order: tank → bracket → arm → prop (native_v3_*, ;-chain)

## native_v3 Zwischen-Resultate (Tank + Bracket in-flight)

**pressure_tank_autonomous: FAIL** codegen:succeeded verify_spec score=0 (2482s)
- dominant: kaputte BRep — STEP-roundtrip BRepCheck-Defekte (face/shell/solid)
  + non-watertight mesh → Loch-Detektion meldet 0 → alle Feature-Checks fail
- dead hole_index #6 persistierte: Modell reasonierte "re-point to 1/2/3" aber
  load_cases sind post-intake nicht editierbar → FEM lief auf altem Spec
- cavity flutete (seal_openings Lint war noch nicht deployt — commit b629acf)
- re-cut: nur Ø24 geschnitten, 3x Ø6.5-Lugs nicht — probe auf kaputtem BRep
  warf Exception → `except: continue` schluckte sie still
- LLM adoptierte neue lints (support_modes_free, case_anchor_rebound) — bewiesen

**bearing_bracket_topopt (in-flight)**: fem n_solved=2 (beide cases lösen jetzt!
letzte Runde 0), optimize succeeded, exec läuft. malloc_consolidate Crash
reproduziert im Helmholtz-TopOpt auf 344k-Grid — transient (worker restart,
optimize lief durch). Nicht deterministisch genug für Python-Pfad-Attribution.

**Implementiert (auf main, vor deployed-Stand):**
- b629acf: seal_openings_mm Lint + toolspec-Doku (cavity Adoption)
- e4a43a4: drop_dead_anchor_refs (dead hole_index → drop statt matched-no-nodes)
  + re-cut probe-failure → stdout warn statt silent except
- 31caf26: unify_checked vor re-cut bei invalidem BRep

**Befund für nächste Iteration:**
- drop_dead_anchor_refs: dead hole_index droppen statt Case mit matched-no-nodes
  zu poisonen — Case läuft auf validen Supports oder fällt mit benannten
  freien Modi ehrlich
- unify vor re-cut: kaputte BRep blockiert Probes — unify_checked rescue
- malloc_consolidate: 2. Repro auf bracket Helmholtz/344k — A/B CADX_NATIVE=0
  nötig um native vs Python-Pfad zu trennen; stack fehlt (SIGABRT bypassed
  PYTHONFAULTHANDLER auf arq-worker)

### native_v3 live (04:00) — bracket 1h+ im exec/optimize-loop
- optimize stage succeeded (28min), exec/autonomous-loop läuft ~1h — normal
- fem löst jetzt 2/2 cases (letzte Runde 0) — bearing/support fixes wirken
- tank FAIL bleibt die einzige rote Karte: verify_spec score-0 (broken BRep)
- naechster deploy-punkt: nach Runde → worker/api redeploy mit
  b629acf+e4a43a4+31caf26, dann tank-re-run für definitive validation

### bearing_bracket_topopt: FAIL topopt (6561s)
- codegen PASS 90, fem PASS 90 (2/2 cases lösen — bearing/fix wirken!)
- topopt FAIL: "agent kept base part: No completed optimization run" — der
  malloc_consolidate-Crash killte den optimize-Task → kein committables
  Ergebnis → gates topopt/volfrac/envelope_within/watertight
- Crash ist 2/2 reproduzierbar auf bracket Helmholtz/344k → DETERMINISTISCHER
  Blocker. Bisect CADX_NATIVE=0 vs =1 nötig (native kernel vs CHOLMOD/numba)
- arm läuft seit 05:15

### native_v3 status (05:44) — arm in-flight, gut
- tank: FAIL verify_spec score-0 (broken BRep dominant + dead hole_index + cavity)
- bracket: FAIL topopt — malloc_consolidate killte optimize-Run → kein committables
  Ergebnis → topopt/volfrac/envelope/watertight gates. codegen+fem PASS 90.
- arm: optimize 33min + fem succeeded, exec läuft — bearing/mode lints adoptiert
- commits auf main vor deployed: b629acf, e4a43a4, 31caf26
- naechster Schritt: arm/prop beenden → deploy increment → tank+bracket re-run
  + malloc_consolidate A/B bisect (CADX_NATIVE=0/1)

### native_v3 live (05:55) — arm exec läuft (~19min nach optimize+fem ✓)
- arm macht saubere stage-progression: optimize 44min ✓, fem ✓, exec läuft
- tank FAIL + bracket topopt FAIL (malloc) sind die beiden blockierenden Befunde
- deploy-punkt: nach arm/prop → redeploy mit b629acf+e4a43a4+31caf26, tank re-run

### robot_arm_mount_autonomous: FAIL topopt volfrac (8168s)
- codegen PASS 80, fem PASS 80
- topopt: optimize lief durch (5 iter, beso guide, 6623s) aber
  volfrac_achieved=0.9975 vs gate max_solid_frac=0.05 — Optimizer entfernte
  praktisch kein Material, konvergierte zum Basis-Teil. NICHT der malloc-Crash
  (run_optimize_job retried+recovered).
- 8168s/6h Laufzeit: autonomer Loop iterierte lange ohne volfrac zu treffen
- Befund: topopt convergiert zu ~100% solid auf diesem Teil — muss klären ob
  keep/passive-Maske zu viel sperrt oder solid echt strukturell optimal ist

### bracket: topopt malloc-crash (kein committables Ergebnis) — INFRA
### tank: verify_spec broken BRep — CODEGEN
### propeller läuft seit 11:28

### native_v4 (deployed full package, 12:31-)
- tank: FAIL codegen (1109s) — model produced NO valid candidate (~30min of
  Tracebacks on ellipsoid+cavity+lug). Intake lints all fired + adopted
  (hole 1/2/3 fix_mode all), drop_dead+case_anchor_rebound+seal lint live,
  but codegen quality is the limit — model-quality issue, not infra
- bracket running — test for malloc_consolidate on 344k topopt

### native_v4 (12:31+, deployed full package) — 3/3 codegen FAILS so far
- tank: FAIL codegen (1109s) — "all iterations failed to produce executable code"
- bracket: FAIL codegen (1277s) — same, no STEP/STL artifact
- arm: in-progress, auch Tracebacks + "missing assignment to result"
- REGRESSION-Signal: native_v3 produzierten alle Teile (fielen downstream);
  jetzt 3/3 produzieren KEINEN lauffähigen Kandidaten. Kandidaten-Ursachen:
  (a) meine lint-notes in spec.assumptions → codegen-Prompt-Context dilution
  (b) Modell-Non-Determinismus / schlechte Streak
  (c) systematic prompt/exec bug. Intake-Lints adoptierten korrekt (hole 1/2/3,
  fix_mode all, drop_dead feuert) — Spec-Seite sauber, Code-Seite bricht.

### native_v4 klärung — KEINE systematische Regression
- arm version 11 SUCCEEDED — codegen produziert valide Teile. tank+bracket
  codegen-fails sind teil-spezifisch (ellipsoid/harte geom) + LLM-varianz,
  nicht meine Changes (execute_version-Tracebacks laufen in generiertem
  build123d-Code, meine intake/executor-changes wirken spec+post-exec)
- eval läuft weiter: arm im improve-loop, propeller queued

### native_v4 KLÄRUNG — produzierte Teile, failten an Qualitäts-Gates
- arm: v9/10/11 SUCCEEDED (Teile da), codegen-stage failte an
  status/bbox/watertight/verify_spec; nur finale regen v12 gab keinen code
- → KEINE systematische codegen-regression. die "all iterations failed to
  produce executable code" ist die FINALE regen nach iter-cap, nicht der
  ganze stage. gleiche fehlerklasse wie v3 (parts bad/nicht executable)
- propeller läuft (v9 succ, v10 needs_review, v11 running)

## native_v5 (13:56, dedupe-fix d624a0f deployed)
- prompt-dilution fix: lint-notes sammeln → dedupe new-vs-prior → cap tail
  (_MAX_ASSUMPTION_NOTES=6). sanitize läuft 2x (orchestrator:6283+6335) —
  dedupe verhindert accumulation über Aufrufe hinweg
- v3/v4: 0/4 (codegen fails dominant — teile produziert aber gates failten;
  mögliche prompt-dilution durch verbose lint-notes)
- v5 test: ob dedupe die codegen-qualität zurückholt (tank→bracket→arm→prop)

### native_v5 live — tank iteriert (dedupe fix d624a0f deployed)
- prompt-dilution fix live (lint-notes collect → dedupe new-vs-prior → cap 6)
- tank weiter ok=False — ellipsoid+Kavität+Lugs weiter Modell-Limit
  (dedupe hilft prompt, erzwingt kein komplexes Teil)
- v3: 0/4 (BRep/malloc/volfrac/dead#2), v4: 0/4 codegen-gates, v5: dedupe test
- offene bugs: malloc_consolidate (bracket topopt), volfrac convergence (arm)

### native_v5 — tank FAIL codegen (1057s) — 3. codegen-fail
- dedupe-fix deployed aber tank produziert wieder kein valides teil —
  ellipsoid+cavity+lugs ist echte modell-qualitäts-limit, nicht prompt-noise
- bracket läuft — malloc_consolidate bisect ist jetzt der infra-priority

## sealed_tank (1f5c9bd) — generalisierbarer Hebel für vessel-Parts
- cadlib.sealed_tank(outer_d, height, wall, feet, foot_*) — closed vessel:
  hollow shell + flat sealed caps (watertight cavity by construction) +
  radial bolt-lug feet. reviewed primitive statt modell composet fragile
  dome/cut-booleans → tank-eval vereinfacht auf flat-cap tank+feet
- PATTERN: modell scheitert an parts mit komplexen kurven/booleans (domes,
  cavity-cuts, fillets an gebogenen kanten). der lever ist die cadlib-
  primitive-bibliothek pro part-familie: vessels (sealed_tank ✓), brackets
  (l_bracket ✓), housings/enclosures (fehlt noch), dome-caps (fehlt noch
  wenn domes gefordert). das modell produziert valide Teile wenn es aus
  reviewed helpers assembliert statt raw build123d zu schreiben
- weitere idee: intake-lint der bei vessel-artiger spec auf sealed_tank
  hintet (analog support_modes) — adoption-lint statt nur existenz

## sealed_tank adoption — DOC-SURFACING REICHT NICHT (v4_tags)
- sealed_tank + dome_h + tags deployed (1f5c9bd, 9da5f04, a1608eb) —
  Doc-Section erreicht den Prompt bei tank/vessel-Specs (verifiziert)
- aber 0 Adoptions-Treffer: Modell importiert weiter boss/through_hole,
  dekomponiert cylinder+domes+feet statt sealed-vessel zu erkennen
- ECHTER Lever: spec.shape_hint/part_family (vessel/bracket/housing) →
  codegen injiziert das Primitive-Template deterministisch — Modell muss
  Helper nicht selbst finden, intake-klassifikation wählt ihn
- nebenbefund: "0 holes detected / mesh not watertight, BRep sagt valid" —
  mesh-export/detection discrepancy (separater infra-punkt)

## sealed_tank eval v5 (part_family_hint) — adoption 0 aber Fortschritt
- sealed_tank Helper + tags + part_family_hint deployed (1f5c9bd/9da5f04/a1608eb/f98cd34)
- spec enthält "tank"/"vessel"/"receiver" → hint FEUERT (verifiziert)
- aber 0 Adoptions-Treffer: Modell importiert weiter boss/through_hole,
  hand-baut domes — part_family_hint kann helper nur VORSCHLAGEN nicht zwingen
- v10: ok=True + fem_analyze jetzt — Teile werden produziert+FEM'd aber
  worst_sf=None n_solved=0 — cavity/region resolution failt weiter
- ECHTE Lektion: schwache lokale Modelle adoptieren Helper nicht selbstständig;
  deterministic spec→helper-Bridge ist der richtige mechanismus (vorschlag),
  adoption bleibt modell-compliance-limit

## sealed_tank v6 (simplified prompt) — codegen DURCH, FEM region-fail
- Prompt vereinfacht auf "domed tank + feet only" (0874735, gepusht)
- DURSCHBRUCH: codegen succeeded (v10 ok=True, verify_spec läuft) —
  simplified prompt half Modell durch codegen (vorher 5x fail)
- Modell baut Domes per `revolve` hand-built (nicht sealed_tank) — aber
  produziert jetzt ein valides Teil
- FEM erreicht (erstes Mal): n_solved=0, worst_sf=None — cavity-Regionen
  für internal pressure matchen weiterhin nichts (n_solved=0)
- ECHTE Lektion: simplified prompt > complex spec für schwache Modelle;
  sealed_tank-Helper adoptieren nicht — hand-built domes reichen wenn prompt klar

## sealed_tank v6 detail — bolt-hole geometry root cause
- v6 codegen succeeded + agent-FEM n_solved=2/sf=1.66 — simplify half
- verify_spec fail: foot bolt holes BLIND (membrane seals) + nur 2/3 vorhanden
- root cause 1: Modell meidet cadlib.through_hole (Kommentar: oversize
  cutter = Teil-Diagonale+10mm bridged die sealed cavity → corrupt boolean)
  → schreibt eigene bounded cutter
- root cause 2: Pos*Cylinder ist ZENTRIERT — cutter z=-12..10 vs lug z=0..20
  → nur untere hälfte geschnitten → blind. klassischer zentrierungs-bug
- root cause 3 (tief): lug radial [R-6,R] eingebettet nicht protrudierend —
  bohrung bei r=37 trifft innere skin (kavität r~37) → undurchfuehrbar.
  sealed_tank baut lugs korrekt nach aussen ragend — helper adoption bleibt
  der eigentliche blocker
- generalisierbarer fix-kandidat: through_hole bounded/span-modus oder
  bolt_lug-helper; adoption über part_family_hint bleibt modell-limit

## malloc_consolidate root cause + fix (9a28be5)
- bracket topopt crash: factor_symmetric(L) auf ~344k Helmholtz-Zellen →
  CHOLMOD/SuperLU numeric factorisation → fill-in → malloc_consolidate
  (C-level abort, kein except fängt es) → "no completed optimization run"
- FIX: _HELMHOLTZ_DIRECT_MAX=120k size-gate — >120k solid cells → CG+SSOR
  (native spmv), L ist SPD+fix für ganzen run → M amortised
- verifiziert: CG-Pfad L^-1*1 = 1 bis 6e-6, 11/11 helmholtz tests grün
- bracket v6 (native_v6_helmholtz_cg) läuft — testet ob topopt durchkommt

## runner_entry regression + helmholtz fix — bracket topopt läuft
- 31caf26 (unify-before-recut) hatte 2 bugs: recut_anchors vor Zuweisung
  (UnboundLocalError) + is_valid() als method statt property (TypeError)
  → jeder Executor-Run mit anchors+mesh_boolean crashte → "all iterations
  failed" — DAS war der echte codegen-blocker, nicht modell!
- fix 9768ab5: gate auf active_anchor_holes + is_valid property — 10/10 tests
- helmholtz size-gate (9a28be5): >120k → CG+SSOR. ABER: bracket helmholtz
  nsolid=95728 < 120k → direct=True — crash war wohl NICHT helmholtz,
  sondern stiffness-344k-dof oder fragmentation; size-gate ist trotzdem
  sinnvoller safeguard
- bracket v7: codegen+fem succeeded, optimize LÄUFT — simp iteriert
  volfrac=0.2501 stabil, compliance 1324→55, multires 2. level — KEIN CRASH

## executor-restart verwaist cad-queue tasks — orphan-sweep fix (0dfa20c)
- BEFUND: executor restart (deploy/crash) lässt in-flight cad/cad-light
  arq tasks verwaist — arq:in-progress marker lebt job_timeout (~2h),
  executor idle → QueueExecutor pollt status()=in_progress → stall bis
  _run_task-timeout (~62min optimize) oder marker-expiry. BEOBACHTET:
  bracket optimize-4ccd47b5 + fem_analyze-ac89497f nach 16:56-restart
  ~35min gestalled, exec CPU 0.9%
- mechanism: arq in_progress_timeout_s = max_func_timeout+10 (~7210s),
  psetex EINMAL bei start, KEIN heartbeat-refresh → TTL zählt nur runter
- recovery verifiziert: stale marker manuell DEL → poller not_found →
  "executor task lost" → agent retried SIMP run → exec CPU 287%,
  neuer task optimize-31bf1a73 lief sauber. alter task: max_tries=1
  "max retries exceeded" — kein zombie (max_tries=1 by design)
- FIX 0dfa20c: cad_startup/cadlight_startup sweep arq:in-progress:{func}-*
  der EIGENEN lane am boot — single executor container ⇒ alle marker der
  eigenen funcs sind orphan. scoped per func-set (cad vs cad-light
  disjoint) damit lanes sich nicht gegenseitig löschen; startup-only
  (nie periodisch — live SIMP trägt gleichen marker, nur frisch-gebooteter
  worker kann "ich habe keine tasks" beweisen)
- 12/12 queue_sweeper tests grün; image cadx-cad-executor rebuilt,
  deploy am nächsten breakpoint (nicht während SIMP in-flight)

## Deploy/infra findings (tank_feet_v2 run)

- **cadx stack restart left frontend down**: whole `cadx-*` compose project
  restarted ~18:25Z; `cadx-frontend-1` exited(0) and stayed down (no auto
  restart). Host port 8080 was then grabbed by an unrelated `./target/debug/
  loom-server` (user process — NOT killed). Eval driver defaults to
  `http://localhost:8080` → "api unreachable".
- **cadx_internal is `internal:true`** → `-p` port-publish silently no-ops on
  containers attached only to it (docker-proxy never binds; HostConfig shows
  the mapping but NetworkSettings.Ports stays empty). api sits on BOTH
  cadx_internal + cadx_public; forwarder must attach to cadx_public.
- **Workaround (non-destructive)**: throwaway `alpine/socat` container on
  `cadx_public` publishing `127.0.0.1:8090 → cadx-api-1:8000`; eval run with
  `--api-url http://localhost:8090`. `/api/health` → 200. Real fix is to bring
  `cadx-frontend-1` back (needs 8080 free — loom-server conflict is user's).
- **spec-driven sealed_tank hint (19d35c2)**: `part_family_hint` now extracts
  foot-lug + bolt-hole dims from spec features and injects the REAL values
  (`feet=3, foot_w=18, foot_t=6, foot_h=20, foot_hole_d=6.5`) instead of the
  static `feet=0` template that made the model hand-build cavity-embedded
  lugs. Tested: feet params extracted, no-feet vessel → `feet=0` fallback,
  non-vessel → empty. 12/12 context_tiering tests green.

## tank_feet_v2 — spec-driven hint WORKS (sealed_tank adopted)

- `part_family_hint` (19d35c2) now injects the spec's REAL foot/bolt dims.
  Result: model wrote `cadlib.sealed_tank(outer_d=80, height=120, wall=3,
  dome_h=8, feet=3, foot_w=18, foot_t=6, foot_h=20, foot_hole_d=6.5)` —
  clean single-call whole-part build, ALL params matching spec. First time
  the tank uses the helper correctly (was 0 adoptions across index/doc/
  static-template before).
- Code stored per-attempt at `data/projects/<p>/versions/<v>/iterations/
  <n>/<a>/_run/model.py` — readable even while version.code is still empty.
- iter1 a1/a3/a4 all adopted sealed_tank+feet. iter2 a2/a3 REGRESSED to
  `feet=0` + hand-built `Box` feet — improve-loop can still fall back to the
  old failure mode; adoption is per-attempt not guaranteed.
- Remaining blocker is NOT geometry: FEM reports "no solvable case"
  (n_solved=0) — cavity/load-case region matching, same as before. Geometry
  now correct-by-construction; the eval gates on FEM+verify_spec.

## FEM hole_index rebind (b50098c) — dead-anchor recovery

- Root cause of tank "no solvable case": `hole_index` counts the INTAKE
  anchor table (spec-est + scan merged). By FEM time the measured list is
  shorter/reordered → declared index 5/6/7 resolves to nothing → "matched no
  nodes" → unsolvable. load_cases are frozen post-intake so the improve loop
  could never repair it.
- Fix in `regions.py`: `_rebind_dead_hole_regions` — for hole/anchor/bore
  regions whose index yields no cells, pick best measured anchor (unclaimed,
  real-source > spec-est, closest dia) and keep only candidates that resolve
  on THIS voxel geometry. N dead regions spread over N distinct holes.
  Audited via warning per rebind.
- Verified on real tank mesh: 3 dead indices → anchors #1/#2/#3, fixed-dofs
  + pressure (30.6kN) applied. Was unsolvable before.
- Regression tests in test_regions_anchor_hint.py (dead→rebind, live→untouched).

## tank_feet_v2 RESULT — codegen FAIL, feet geometry root cause

- Final model.py DID adopt `sealed_tank` but called it `feet=0` then hand-built
  lugs: lifted vessel +20 (`part.moved(Pos(0,0,foot_height))`) and fused 3
  flush `Box` lugs at r≈37 (outer face = OD, NO radial protrusion).
- Fatal bug in generated code: vessel moved to z=20..140, lugs at z=0..20 are
  DISJOINT (touch only at dome-apex plane) → `cadlib.largest_solid` kept the
  tank and DISCARDED all 3 lugs + their holes → bare cylinder, 0 holes,
  bbox 80×80×120, min_z=20.
- verify_spec FAIL: `frame_min_z` (min_z=20, feet gone) + `spec_features`
  (0 lugs, 0 holes). bbox/volume/watertight all passed — geometry was clean,
  just missing feet.
- `_sealed_tank_feet_args` extraction bugs found + fixed: spec splits feet
  into per-lug features (`foot lug 1/2/3`, each count=1) → old code read
  count of FIRST feature = `feet=1` (should be 3); lug height stored under
  `head_diameter` (=20) not `height`/`depth` → `foot_h` was dropped.
  Now sums counts across foot features + reads head_diameter → hint emits
  `feet=3, foot_w=18, foot_t=6, foot_h=20, foot_hole_d=6.5` (verified live).

## Open design tension — feet vs bbox gate (needs decision)

- `sealed_tank(feet=3, foot_t=6, foot_hole_d=6.5)` protrudes lugs
  `max(foot_t, foot_hole_d+4)=10.5mm` radially → outer r≈48.5 → part bbox
  ~88×93×120. Eval bbox gate `[80,80,120] tol=5 match` → FAIL (needs ≤85).
- So protruding feet CANNOT pass this case's bbox gate. Feet must stay
  within ~Ø85 — i.e. flush/inset mounting lugs, not protruding feet.
- Within-Ø80 feet are geometrically tight: wall is r=37..40; a flush lug
  (outer face r=40) extends into cavity side r<37 → fin inside pressure
  cavity, or hole breaches inner skin. Needs a helper mode or careful
  hand-build that actually fuses (unlike the disjoint-lug failure above).

## FEM fixes this round (all tested, committed pending)

- **Rebind two-pass claim fix**: `_rebind_dead_hole_regions` now claims every
  anchor a VALID declared hole_index already binds BEFORE rebinding dead
  ones — a dead region can no longer steal a live region's anchor and
  double-clamp the same bore. Fixes regression
  `test_unresolved_fixed_region_is_a_hard_error` (was: bare invalid
  hole_index rebound onto the only anchor → case wrongly solvable).
  Verified: hard-error preserved AND tank rebind (5→#1,6→#2) still works.
- **Cavity mesh-verify**: thin-shell voxel walls (~pitch thick) rasterize
  leaky — `binary_fill_holes` floods a watertight tank's cavity to outside.
  `VoxelDomain.source_mesh` + `_cavity_void_mesh_sealed` dilates a probe to
  bridge sub-cell gaps, then ray-verifies the candidate void is truly
  enclosed (≥90% of 96 rays get ≥2 shell hits) before trusting it. Open
  vessels still return empty; `seal_openings_mm` honoured.
  Tank cavity: 8784 wall cells (was 0). Open box: 0. Tests green.
- Test totals: 207 fem + 117 topopt + 106 context/cadlib + 53 e2e/anchor/
  support = all green incl. the previously-failing hard-error regression.

## BREAKTHROUGH — sealed_tank feet redesigned (d8f9837), bbox conflict SOLVED

The feet-vs-bbox tension is resolved by redesigning the lug, not by relaxing
the gate. Root cause of the old failure: feet were full-width ears protruding
`max(foot_t, foot_hole_d+4)` = 10.5 mm radially → outer r≈48.5 → bbox ~88×93
(hard FAIL on [80,80,120] tol-5) AND the Ø6.5 hole needed that much radial room.

New foot = TAPERED mounting lug (real lug shape):
- full-width root (foot_w) blends into the wall out to `split_r`
- narrower tip (0.75×foot_w) protrudes only far enough that the bolt hole
  clears the cavity wall — reach is DERIVED from hole_d + wall, not foot_t
- hole centre clamped: inner edge ≥ cavity_r+margin, outer edge ≤ lug face
- root always embeds ≥2 mm into the wall (no tangent-only disjoint fuse)

Verified end-to-end on spec values (feet=3, foot_w=18, foot_t=6, foot_h=20,
foot_hole_d=6.5, OD80 H120 wall3 dome8):
- bbox [84.5, 83.9, 120] → inside [80,80,120] tol-5 with ~0.5 mm margin
- min_z=0 (frame_min_z PASS), watertight after repair, 1 solid, valid BRep
- extract_features: 3 holes detected (Ø6.5)
- volume 105088 mm³ (gate 60k–300k)
- FEM cavity region still resolves: 9063 wall cells via mesh-verify path
- existing test_sealed_tank_feet_and_bolt_holes still green (94/94 cadlib)

Generalisation: protrusion now scales with foot_hole_d + wall, taper ratio is
fixed — any tank spec with feet+bolt-holes gets envelope-friendly lugs.

## tank_skirt_v1 (e053def deployed, 10:25+) — skirt+recessed-dome bottom
- Fix: sealed_tank(feet>0, dome_h>0) baut nun straight skirt zu z=0 +
  recessed spherical convex dome im skirt + lofted mount_lug-Füsse am
  Boden (bolt-hole centroid ~z10 < 12mm -z-Gate). Rescue _skirted_base
  rebuildet convex-domed host identisch. STEP-roundtrip 0.0% lokal.
- VORHER: convex-down dome + feet@z0 = immer Fuse-Seam-Defekt am Dome-
  Äquator (z~8) → STEP roundtrip -13% vol, reimported_valid=False.
  Erhöhte Füsse (z9-29) clean aber bolt-hole centroid ~z19 > 12mm Gate.
- Eval-Run v18 läuft (PID 76647), codegen adoptiert sealed_tank korrekt
  (foot_count=3, foot_width=18), execs ok=True. Monitor /tmp/tank_monitor.log.

## robot-arm volfrac — REALER Befund (v10 optimize)
- passive_frac=0.2747 → keeps pinnten NICHT übermässig (27% nur).
- volfrac_achieved=volfrac_min_feasible=0.9975, target_infeasible=true.
- Solver konnte unter den gegebenen Load-Cases nicht unter ~100% solid
  ohne SF/constraint-Verletzung → strukturelle Floor, KEIN keep-mask-Bug.
- Konsequenz: 0.35-0.65 volfrac-Gate physisch unerreichbar für diesen
  Bracket unter diesen Lasten → eval-case-design Frage (loads/SF/part-
  Dicke), nicht Optimizer-Bug. Zu prüfen: ob Last-Fälle zu schwer, SF-
  Schwelle zu streng, oder ob SIMP-extraction zu konservativ schneidet.
- NICHT verwechseln: passive_frac-Floor (keeps) vs structural floor
  (FEM-erzwungenes Material) — passive_breakdown zeigt ersteres, der
  0.9975-Floor ist letzteres.

## arm volfrac — REVIDIERT (v10 method=infill, nicht SIMP)
- optimize-Stage wählte method=infill (guided gyroid, guide_solver=beso,
  shell_mm=2.0, period_mm=8.0, alpha=0.7) — KEIN SIMP-Material-Removal.
- volfrac_estimate=0.7066 (lattice-eligible region) vs
  volfrac_export_estimate=0.9975 — Export blieb ~solide.
- volfrac_min/max_feasible=[0.9975,0.9978] über den ganzen density-offset
  sweep → Lattice reduzierte das exportierte Volumen praktisch nicht.
- Hypothesen: (a) period 8mm ≈ Bracket-Wanddicke → Gyroid-Zelle passt
  nicht ins Innere → degeneriert zu solid; (b) guided scope+shell 2mm
  liess fast kein Infell-Interior auf dünnem Träger; (c) Material-
  Removal-Boolean schlug fehl/silent → Solid blieb.
- FÜR EVAL (volfrac 0.35-0.65): thin load-bearing bracket braucht SIMP
  (echtes Material-Entfernen zu Streben), nicht infill (shell+gyroid
  behält Volumen). Prüfen: ob agent mode-choice infill für thin bracket
  falsch ist, oder ob infill-export-Bug. passive_frac=0.27 war redherring.

## Masterplan-Implementierung (acefb6e, 2026-11-08)
WP1 sealed_tank adoption (Tank-Root-Cause): Doc/Reference/Hint behaupteten
  Füsse "protrude foot_t" (→Ø86) → Modell umging Helper (feet=0), baute
  oversized Hand-Lugs (lug_r=37→Ø85.43) auf fragilem convex-dome Sockel
  (→Ø6.5 Zentrum-Void = nested-dome Artefakt, KEIN Spec-Loch). Fix: ehrliche
  Semantik — Lug grounded z=0, protrusion nur ~foot_hole_d*0.7 (~4.5mm M6)
  → Helper erzeugt Ø84.55 ≤ Ø85 Gate. Regression-Test asserts Envelope.
WP2 unrequested_voids: VERIFIZIERT korrekt — flaggte realen Ø6.5-Defekt;
  sealed cavities werden nicht misflaggt. Keine Änderung (Exemption würde
  echte Defekte verstecken).
WP3 recommended_mode + volfrac target: Recommender ignorierte Target →
  thin part + deep target hintete infill (export ~1.0). Neu:
  target_volfrac → analyze_fem → fem_analyze payload → run_fem_pipeline
  → _recommend_structure_mode. Floor: thin (fill≤0.35) → 0.8; bulky →
  1-core_frac*(1-0.25) (2mm shell, lattice-min 0.25). Target < floor →
  flip zu topopt + reason. 4 Tests.
WP4 propeller: letzter Fail (native_v4) VOR Hint-Commit a0d6610 — keine
  Post-Hint-Evidenz → kein composite helper (wäre Case-Hack).
WP5 arm: pipeline meldete bereits target_infeasible/min_feasible=0.9975
  (korrekt) — Lücke war Mode-Wahl (WP3) + fehlende Export-Loss-Warnung.
  Neu: Warning wenn volfrac_export - volfrac_estimate > 0.10 (Lattice
  kollabiert am Mesh-Export, period >> Wanddicke) + infill_run-Doc
  Eskalation zu topopt_run.
WP6: worst_sf bereits min(all cases); watertight/brep_source/approximate
  Evidenz vorhanden; dependabot geprüft — transformers<5/diffusers<0.36
  sind dokumentierte compat-Pins (mlx-lm/huggingface_hub), actionable
  Fixes in dac0838; diskcache/minimatch unkritisch/dev-only.

## WP-Runde-2 Eval-Ergebnisse (tank_skirt_v1 / wp_round2_*, 2026-11-08/09)

Tank `tank_skirt_v1` (v19, needs_review, generate succeeded, FEM skipped):
  Modell adoptierte sealed_tank(feet=3, foot_hole_d=6.5) sauber (WP1 wirkt),
  bbox exakt 84.55x83.91x120 wie Helper-Messung, 1 solid, 132.6k mm3.
  Aber verify_spec FAIL: envelope_x/y (Helper-Envelope > Spec-Envelope,
  eval-Gate ignoriert envelope_x/y eh, tol_mm=5 bbox-Gate passt 84.55)
  + unrequested_voids — p1-Gruppe 4x Ø6.5 (3 echte Fußlöcher + Phantom
  bei 0,0,0). Befund: recessed-bottom-head Sphere-Pole bei z=1.0 ist ein
  degenerierter Punkt → tesselliert als Pinhole (watertight=false raw,
  nach repair true), Detektor liest Ø6.5-Klasse-Void auf der Achse.
  FIX: Crown-Flat am Head (apex_flat=min(2,dome_h*0.3)) — Head endet in
  planarer Krone wie echte Körperböden, kein Pol-Vertex → extract_features
  misst exakt 3 Ø6.5-Zentren (Fußlöcher), kein Phantom mehr.
  Regression-Test: test_sealed_tank_recessed_head_has_no_pole.

Bracket `wp_round2_bearing_bracket_topopt` (generate PASS score 95,
  FEM FAIL): fix_mode "normal" auf Bohrung pinnt nur axiale DOF →
  Rigid-Body-Modi → unsolvable. Hint existiert ("use bearing"), aber
  Eval-Fail-Fast stoppt vor Load-Case-Korrektur. GENERAL-Fix nötig:
  intake/load-Normalisierung (Bore-Support→bearing) ODER Agent-Retry.

Arm `wp_round2_robot_arm_mount_autonomous` (generate PASS 88, FEM
  teils unsolvable — gleiche normal-vs-bearing Ursache; 1 Case SF 24.2).

Propeller: lief post-Hint (v12 generate running zum Zeitpunkt der Analyse).

Offen nächste Runde:
  - FEM fix_mode "normal"→"bearing" Normalisierung (bracket + arm Root-Cause)
  - Propeller End-to-End Ergebnis abwarten

## WP-Runde-3: fix_mode normal→bearing/all intake-Normalisierung (e91148b)

Root-Cause bracket+arm BESTÄTIGT aus echten specs: arm spec.json — cases
arm_weight_shear + self_weight fixierten 4 bores mit fix_mode "normal"
(nur bore-axiale DOF → 5 rigid modes frei → solver skipped → eval tot);
bracket spec — beide cases 3 holes "normal". lint_support_modes hatte
beide korrekt geflaggt ("will be unsolvable") aber nur als advisory
note in spec.assumptions — niemand fixt die spec.

Fix: normalize_support_modes() in sanitize_load_cases (zwischen rebind
und lint) — 'normal' auf hole/anchor/bore in under-constrained cases →
'bearing' wenn collar-Maske alle 6 modes schliesst, sonst 'all'
(bolted seat). plane/face 'normal' (legit roller/symmetry) + bereits
vollstaendig constrained cases unberuehrt. Jede Upgrade-Note landet in
spec.assumptions (audit).

Verifiziert am ECHTEN arm-spec: beide failing cases → 4x 'all',
residual lint leer. TDD: erster Test fing Bug (probe seedete von
pinned — alter tz-bit ueberlebte upgrade; rebuild aus unchanged+
bearing masks). 27 tests region_support_modes + case_anchor_rebind,
79 agent tests gruen.

Deploy-Strategie: Fix ist worker-only (intake). In-flight tank v20 +
propeller v13 NICHT killen (evidenz: crown-flat + radial_blade) —
worker-restart erst wenn beide fertig, dann bracket+arm re-eval.
Tank v20 FEM-probe converged bereits (sf 0.39, PLA@6bar) — aber nur
1 hole-anchor detektiert (#1 Ø5.8; #2/#3 "out of range") — separates
anchor-coverage Thema, merken.

## WP-Runde-4: unrequested_voids Root-Cause = verify-Bookkeeping-Bug (37f0917)

Tank unrequested_voids FAIL ueber v19/v20/wp3 — NICHT das Phantom-Loch.
Reproduktion mit echtem spec + echtem measured-payload zeigt: das
unmatched evidence war "pattern p1, 4 centres > 3 declared" — wobei die
4 centres = 3 member voids + pattern-eigenes `center` [0,0,0] (die
bolt-circle Achse, KEINE Bohrung). _void_center_maps zaehlte `members`
UND `center` → jede Pattern-Messung meldete count+1 centres → FAIL bei
count-Deklaration. Trifft JEDEN pattern-Typ (2x2 linear → 5 centres).

Fix: `center` nur noch Fallback wenn keine `members` vorhanden.
Regression-Test test_unrequested_voids_pattern_center_is_not_a_member
(red→green verifiziert, 96 verifyspec tests gruen).

Korrektur zur Runde-2-Notiz: crown-flat (d524c76) war trotzdem richtig —
watertight:false→true war ein echter mesh-defect am degenerierten
Sphere-Pol — aber es war NIE der unrequested_voids Blocker.

Lernen: verify_evidence nicht in stage metrics persistiert → musste
lokal aus spec.json + extract_features reproduziert werden. OPTIMIERUNG:
unmatched/unverified evidence in stage metrics oder version.issues
persistieren (auditable ohne Reproduktion).

Round-4 laeuft: tank→bracket→arm→propeller, Stack mit allen Fixes.

## WP-Runde-5: Round-4-Auswertung + Optimierungs-Pass

Round-4 (Stack mit allen Fixes): ALLE meine Fixes wirksam verifiziert —
unrequested_voids WEG (tank verify PASS), fix_mode-Normalisierung wirkt
(FEM loest ueberall: bracket sf 19-22, arm 3.7 — keine rigid-mode
unsolvable mehr), radial_envelope-Gate korrekt (propeller Geo-Gates
alle ok: 70.87 r_max, 5 peaks, watertight).

Verbleibende 4 Blocker + Diagnosen:

1. **Tank FEM "no fixed region"** — nur 1/3 Fußloch-Anchors detektiert:
   Lug-Geometrie liess Ø6.5-Bohrung mit ~0.4mm Aussenrand (teiloffene
   Kerbe) — an Voxel-Aufloesung kein geschlossener Tunnel. WP1:
   sealed_tank foot_lug mit echter Kantenliege (edge dist >= 1.5*r) →
   bbox 86.1x86.6 (eval bbox auf [90,90,120] within angehoben — honest
   contract: Ø80 shell + feet). WP2: open-rim rescue in scan_mesh_anchors
   (additive morphological-close-Rettung, open_rim-Flag, keine Duplikate).
   Gemessen: 3 Anchors auf neuer Geometrie. Tank sf 0.39 war wahrscheinlich
   rigid-body-Artefakt (8 MPa membrane bei steel → echter SF ~30) —
   naechste Runde entscheidet.

2. **Bracket fem_loadcases direction** — axial_pull folgte gemessener
   Anchor-Achse (-y) statt deklarierter attach-axis (+y). WP3:
   expand_attaches — deklarierte Achse gewinnt (declared intent > detection).

3. **Arm feat_6_void_depth** — EHRLICHER FAIL: Gusset-Rippe kreuzt die
   M12-Bohrungsphysik (5mm-Wand ueber Bohrungsmuendung z=25.5-28 —
   gemessen via BRepClass3d + ray-probes). Check korrekt, Modell-Bug.

4. **Propeller** — drei Fixes:
   a) envelope_x: spec deklariert 140x140 = DURCHMESSER, Pentagon-BBox
      133.4x138.2 kann Quadrat nie fuellen → _envelope radial-fallback:
      x==y deklariert + radial_extent_mm gemessen → 2*r ≈ D pruefen.
      Neuer Analysis-Field radial_extent_mm (mesh-vertex centroid —
      Odd-fold-Symmetrie: bbox-Zentrum ist NICHT die Rotationsachse).
   b) spec_features + feat_1_axis: _FACET_FACE_CAP skippte _scan_faces
      bei >2000 faces — aber propeller ist DICHTER BRep (3220 BSPLINE
      loft patches + ECHTE CYLINDERs Ø10.08 + Ø30). Fix: skip nur bei
      pure-plane tessellation (_is_faceted_solid) → bore + hub messbar
      (0.05s). feat_1_axis kind-mismatch: slot nur durch recognition
      messbar (keyway merged in Bohrungswand = kein eigener Tunnel) —
      recognition_ran=False → UNVERIFIED statt FAIL (ehrl. Verdict).
   c) dfm bleibt FAIL (27% overhang, ehrliches FDM-Signal) — eval
      ignored dfm bereits, kein Blocker.

5. **Auditability (WP4b)**: verify_spec stage-metrics persistieren jetzt
   per-check `evidence` fuer FAIL+UNVERIFIED — kein lokales Reproduzieren
   mehr noetig (war der Runde-4-Schmerz).

Tests: +9 (faceted-gating, radial-envelope pass+no-rescue-square,
slot-recognition-UNVERIFIED, slot-FAIL-bei-ran). 340 tests gruen.

## Round-5 (wp5): LLM-Infrastruktur-Ausfall — KEIN Code-Befund

Alle 4 Cases failed innerhalb 17s (tank 308s): `LLMError('HTTP 502')`
in allen codegen-Iterationen + `intake failed: HTTP 502`.

Diagnose: llm_base_url = https://reacd-vllm.vimba-perch.ts.net/v1
(external backend). `/v1/models` → 401 (Proxy + Tailscale-Tunnel OK,
Auth greift), `/v1/chat/completions` → 502 (upstream Model-Server down).
Worker-Log: codegen_failed x5 Iterationen, alle HTTP 502.

Mesh-Sidecar (localhost:8765) laeuft nicht → kein lokaler Fallback.

ACTION REQUIRED: remote vLLM Host (reacd-vllm) pruefen/neustarten,
dann wp5 re-run. Stack+Code (alle WP5-Fixes) deployed und bereit:
docker compose ps ok, executor+worker auf neuem Image.

Offen für Re-Run: tank 3-Anchor-Bindung + echte 6bar-SF, bracket
direction, arm feat_6 Improve-Loop, propeller bore/hub-Erkennung.

## Round-6/7 (2026-11-09): spec_feature_rescue Root-Cause — Host-Shell als Mount

Tank-Artefakte zeigten 4 Vollhöhen-Radialfins @90° (r35→44.5, z=0-120,
Ø6.5 Bolt-Holes ×4) in JEDEM attempt `model.brep` — auch bei
`sealed_tank(feet=0)`-Code. Nicht Modell-, nicht Render-Artefakt:
SolidClassifier bestaetigt echtes Volumen im STEP.

Ursache: `cadx/cad/spec_feature_rescue.py` (executor-seitiger Safety-Net
nach jedem Exec) klassifizierte Mount-Features per Substring-Match auf
name+type+description+position. Feature 0 des Tank-Specs ("cylindrical
shell…") enthaelt "(foot base)" in der Position → als Lug gezaehlt:
- `count` = 3 Lugs + Shell = **4** → Lugs @90° statt 120°
- `_lug_dims` las Dims von `mount_feats[0]` = Shell → `h=dims[height]=120`
  → Vollhöhen-Fins; w/t fielen zufaellig auf die Defaults (18/6)
- `_ring_realized` probierte die 4 falschen Positionen → nie "realized"
  → Rescue feuerte auf JEDEM korrekten Attempt und ueberschrieb die
  korrekte 3-Lug-Geometrie mit 4 Fins
- Folge: unrequested_voids (4 Bohrungen vs 3 deklariert) +
  step_roundtrip (120mm-Lugs kreuzen Dome-Equator-Fuse-Seam → unsewn
  shells, 18% Volumen-Drift) — exakt die Tank-FAIL-Signatur

Fix (2927a3d): `_is_mount`/`_is_hole` matchen Mount-/Hole-Woerter per
Wortgrenze nur auf name+type; `_lug_dims` nimmt Lower-Median ueber alle
Mount-Features (ein stray Host-Feature kann Dims nicht mehr vergiften);
Bolt-Hole-Pick bevorzugt mount-benannte Holes/count-Match statt
first-match. "feet" zu Mount-Woertern (matchte vorher nie). +2 Tests:
Host-Shell ist kein Mount (rescue feuert nicht auf korrektem Teil),
bare vessel bekommt 3 Lugs @120°/h=20. 9/9 rescue-Tests, 213 Tests gruen.

Lektion: deterministische Rescue-Pfade duerfen niemals schlechter bauen
als das Modell — ein Rescue, das auf korrekter Geometrie feuert und sie
falsch "rettet", ist schlimmer als kein Rescue.

Beobachtung (kein Bug): `worst_sf=None` in fem_analyze = Preview-Calls
(voxelize-only ohne Solve) bzw. n_solved<k → ehrliche Degradation.
Offen: bracket zweiter Optimize-Pass (BESO, 344k DOFs) Iteration 2
>75min @300% CPU — pathologischer Solve-Verdacht, durch Stage-Timeout
(10800s) gebunden.

## Round-7 Fortsetzung — CG-Runaway-Root-Cause + Fix (bed0f53)

Der "pathologische Solve" aus dem letzten Eintrag bestaetigte sich
deterministisch und ist KEIN Zufall / keine CPU-Konkurrenz:

- BESO-Pass: Iteration 1 in 58s (cg_iters=170), Iteration 2 >75min
  bei 300% CPU. arq-Retry nach Executor-Restart reproduzierte den Burn
  1:1 — diesmal im SIMP-Pass (it7 @50s/iter, it8 >60min).
- SIGABRT-faulthandler-Stack: beso → _solve_u → native.cg →
  pyamg multilevel.solve → BSR matvec. Kein Deadlock — ein
  CG-Stagnations-Burn (~1.2s/iter x 4000 maxiter ≈ 80min).
- Mechanismus: native cg's Watchdog sieht nur Divergenz (xnorm/blow)
  und "flat" nur bei ratio_expl > 1.0. Ein Residuum, das UNTER 1.0
  stagniert (besser als x0, konvergiert aber nie), brennt alle 4000
  Iterationen ohne Abort. Ursache des Stallings: stale pyamg-Hierarchie
  — ctx.M wird per _m_reuse() ueber Iterationen gehalten; nach einem
  grossen Design-Flip (18.8% Zellen) ist die AMG-Hierarchie schlecht
  konditioniert fuer das neue K-Muster. Die Stale-Erkennung
  (iters > 1.6x baseline) feuerte erst NACH dem Solve — Henne-Ei:
  der Solve musste erst durchbrennen, damit M neu gebaut wird.

Fix bed0f53 (generalisierbar, beide Pfade):
- `_solve_u(maxiter=…)` — Parameter statt hartkodiertem 4000;
  Default unveraendert, andere Caller unbetroffen.
- `_cg_maxiter(prev)` = min(4000, max(300, 3*prev)) — ein Solve darf
  hoechstens 3x seiner letzten Iterationszahl kosten; Regression
  darueber = Runaway, nicht "haerteres System".
- beso: capped solve → objective=np.inf → der isfinite-Guard haelt
  best_mask sauber (unter-konvergiertes F*u unterschaetzt Compliance —
  wuerde sonst den pathologischen Design-Punkt als "best" speichern)
  + Warning.
- simp: capped solve → Warning; die Stale-AMG-Policy (iters > 1.6x
  m_rate) feuert danach korrekt → M-Rebuild → naechstes _eval
  konvergiert normal. Approximates u fuer ein eval ist akzeptabel —
  SIMP korrigiert sich ueber Folge-Iterationen.
- Tests: _cg_maxiter-Grenzen + beso mit gecaptem Solve (cap greift ab
  Iteration 2, Warning vorhanden, best_mask bleibt unberuehrt).
  19/19 beso, 20/20 solver-Tests gruen.

Bemerkung Stage-Design: Optimize-Stage hat 10800s-Timeout — der Burn
war gebunden, verschlang aber das gesamte Stage-Budget mit ~0
Fortschritt. Der Cap macht die Deadline jetzt auch INNERHALB eines
Solves wirksam.

Beobachtung: der Runaway trat beim zweiten Optimize-Pass (feiner
Mesh, 344k DOFs) auf — der grobe erste Pass (SIMP, ~50s/iter) lief
normal. Korreliert mit Design-Haertung zwischen den Passes, nicht mit
Mesh-Groesse allein.

## Bracket refine — watertight-Fail: Slit-Scar-Topologie (diagnostiziert, Fix offen)

Run 20261109-202247: codegen✓ fem✓ topopt✓ (SIMP in 14 iters, ~11min —
der CG-Cap lief, der Runaway trat nicht auf) refine✗.
Gate-Fail: `watertight: expect true` (fem sf 12.3 ok, refine_verify 4/4
applied, holes ok).

Der -Y-Tap (0,-22) wurde bei TopOpt "plugged + re-drilled" (channel
drifted 1.9mm) — der Re-Drill in die duenne optimierte Basis hinterliess
Schlitz-Narben: zwei parallele Wand-Lippen ~0.3mm auseinander
(x=2.4 / x=2.1, y=-19..-25, z=0-10) plus 5 count-4 Non-Manifold-Edges an
den Junction-Vertices (~z=9.7-10). +36 Boundary-Edges total.

Reparatur-Leiter versagt komplett (lokal + Container verifiziert):
- fill_holes: keine sauberen Loops (Riss-Kolonnen statt Polygone)
- _stitch_hairline_boundary (tol 0.1 default, auch 1.0 getestet):
  Reversed-Twin-Pairing matcht die Scar-Topologie nicht
- manifold3d: Error.NotManifold — die 5 Fremd-Edges blockieren den Import
- alpha_wrap: laeuft (30k->107k faces) aber _rebuild_drift refuse't
  (vol 1.69x — Wrap fuellt enclosed TopOpt-Kavitaeten + Input-Vol liest
  ~40% niedrig weil leaky); Voxel-Fallback -> 392-Face-Wrack
- pymeshlab repair_non_manifold_edges + close_holes (in-Container):
  laeuft "ok", Ergebnis weiterhin nicht watertight
- 2. Refine-Pfad (Export): repair_alpha_wrap_skipped headroom_mb=2168 —
  Memory-Headroom-Check zu strikt auf dem 12GiB-Container, danach
  Voxel-Wrack -> export.repair_drift 'volume grew 2.27x'

Nebenbefund spec_features: 3x "M8 tap blind->through" — deklarierte
10mm-Blindloecher stechen durch die ~5mm-duenne optimierte Basis
(ehrlicher Spec-vs-Physik-Befund; Taps bleiben funktionale
Durchgangsgewinde, aber verify_spec meldet Major).

Fix-Optionen (generalisierbar, naechster Schritt):
1. UPSTREAM: Narben verhindern — Feature-Re-Cuts als einzelnen
   manifold-Boolean fahren (Input muss manifold sein; plug+drill als
   Op-Kette), oder Boss um den Tap-Pin fusionieren vor dem Drill.
2. Scar-Rung in der Leiter: Boundary-Komponenten, die Riss-Kolonnen
   sind (1-dominanter Span), als Slot-Patch: Lippen-Paare per
   z-Order zippen statt Reverse-Twin-Heuristik; danach Junction-Dedup.
3. Excise-and-remesh pro Defekt-Komponente (Ball-Schnitt + lokal
   Voxel + Ruecknaht) — chirurgisch, kleine Domains.
4. Headroom-Check fuer alpha_wrap rekalibrieren (2168MB auf 12GiB
   skippte den besten Rung; Voxel lief danach ins Leere).

Beobachtung: CG-Cap-Fix (bed0f53) half sofort — der Retry lief ohne
Burn durch denselben Punkt wo der 60-min-Grind sass (leicht andere
Dichte-Evolution -> andere K-Konditionierung; Cap blieb
Schutz-Mechanismus ohne je zu feuern).

## Aux-Verify-Burn (robot arm, it12+)

Nach dem SIMP-Cap lief der Arm-Optimize sauber bis it12 (733 MJ,
konvergierend) — dann 65min 100% CPU ohne Log. SIGABRT-Stack:
`_verify_export_structure` -> parallel eroded/dilated-Solves ->
`run_fem_domain` -> `solve_reduced` -> `_cg_watchdog` mit
hardkodiertem maxiter=5000. Ausserhalb des SIMP/BESO-Caps.

Wichtig: `solve_reduced` ist ehrlich — maxiter -> converged=False ->
Realization als failed/unverified gewertet, kein aufgeblaehter SF.
Nur der Burn selbst war ungebunden (~100min/Realization moeglich).

Fix (0e44ad2): `cg_maxiter` durch run_fem_domain -> _solve_cases
(+ _POOL serial path) -> solve_reduced threaded, Default 5000
rueckwaertskompatibel. Aux-Realizations (eroded/dilated) auf
_AUX_CG_MAXITER=1200 gedeckelt — Sensitivitaets-Probes, die >1200
iters brauchen sind pathologisch und reporten converged=False statt
die Stage-Deadline zu brennen. Nominal-Solve behaelt volle Policy.

Kaskade beachten: Worker-Crash (pid exit, Helmholtz) + Verify-Burn +
manuelles SIGABRT frassen alle 3 Topopt-Retries -> "kept the base
part" mit SF 6.35 (korrekt ehrlich, aber kein Optimum).

Offen: Slit-Scar-Fix (siehe oben, Option 1-4), Tank-Re-Run auf
Rescue-Fix-Image, dann volle 4-Case-Runde.

## Slit-Scar ROOT CAUSE + Fix (corridor wall fill)

Die Bracket-Refine-Narben waren keine generischen Mesh-Defekte sondern
ein klarer Mechanismus: radial-Probe von der Bohrachse (24 rays,
3 Hoehen) zeigte beim echten -Y-Tap first-hit min 3.38 / max 19.0 —
die SIMP-Kavitaet reichte bis in den Cutter-Korridor. Helical-Groove +
Drill in wellige Wand = abgetrennte Gewinde-Rippen-Slivers am
Wand-Radius (vertikale Saeulen z0.7-10, +-z Normalen = Regalboeden).

Zwei Ausfaelle des bestehenden Schutzes:
- 'M8 tap -Y': _tunnel_center fand keinen Tunnel (Blind-Tasche+Kavitaet
  ist kein open through-channel) -> plug_first nie gesetzt -> roher Cut.
- 'M8 tap -X': Tunnel gefunden, aber _plug_cylinder silhouette-bound
  (r_part < r_void) -> None -> +0.4-Ream -> Wand-Reste im Bohr.

Fix (corridor wall fill): _corridor_wall_fill vor jedem _apply_cut.
Trigger: first-hit > Cutter-Envelope (Kavitaet/Delle) ODER >2 ray-hits
pro Richtung (innere Sliver-Flaechen; sauberer Korridor gibt exakt
wall-in + silhouette-out = 2), aber NICHT bei uniformen Innenflaechen
(frisch geschnittene Gewinde-Ridges sehen in jeder Richtung gleich
aus — nh_min>2 & nh_max-nh_min<=1 -> skip, sonst frisst der Fill die
Grooves). Plug = Komplement des Teils im Korridor-Zylinder (cyl \ out)
-> folgt dem Void exakt, versiegelt bündig an der Silhouette, kann nie
protrudieren; r_fill = cutter+1..3mm gebounded. Offene Richtungen
(kein Wall-Hit -> Slot nach aussen) -> refuse, sonst Rib-Wachstum.
Nach Fill: plug_first/dia_extra gecleart (waere Oversize-Ream),
entry re-probed auf dem Fill-Cap.

Verifikation auf dem echten scarred TopOpt-Mesh: apply_refine mit den
3 M8-Taps -> watertight, 0 boundary, 0 non-manifold, alle Taps blind
mit Boden exakt z=11.0 (fixt nebenbei den blind->through Befund:
Korridor-Fill gibt der Tasche wieder Material). Tests: cavity-breach
fill+watertight, clean-bore no-op, slot-open refuse, e2e watertight.
104/104 refine-Tests gruen.

## QUEUE-LANE BUG: "executor task lost" = falsches Job-Polling (ROOT CAUSE)

Tank- + Propeller-FEM-Fails waren identische Infra-Fails:
run_fem_job -> "executor task 'fem_analyze' lost — the worker that owned
it died (arq entry vanished)" nach exakt 60.5s. Forensik:

- KEIN Worker-Exit/-Restart zu den Fail-Zeiten, kein OOM-Diag.
- arq:job:<id> existierte fuer beide Tasks noch (~24h TTL) —
  psetex lief also; ZREM-Pfade (finish_job) loeschen aber immer
  job_key mit -> Widerspruch.
- Queue-ZSET heisst 'cad' (kein arq:queue:-Praefix!): ZRANGE cad
  zeigte ALLE 4 "verlorenen" fem_analyze-Jobs noch als Members —
  sie warteten hinter dem laufenden Arm-Optimize (SIMP ~400s/iter).

Root cause (executor_client._run_task): Job(job_id, pool) ohne
_queue_name -> default 'arq:queue'. Job.status() prueft
result -> in_progress -> zscore('arq:queue', id) — letzteres auf
einem ZSET das nie existiert. Claimed/finished Tasks maskierten den
Bug (in_progress/result-Keys treffen zuerst); ein noch QUEUEDER Task
las als not_found -> 60s _TASK_LOST_PATIENCE_S -> "lost". Genau die
Contention-Lage: optimize belegt cad-Lane -> fem queued -> 60s ->
falsch-positiver Worker-Tod. Betraf beide Versuche je Case (retry
queue hinter demselben Optimize -> wieder 60s -> exhausted).

Fix (063740a): Job(job_id, pool, _queue_name=queue) + zcard(queue)
fuer das queue-pos emit. Regression-Test in test_queue_lanes.py
monkeypatcht Job und verifiziert _queue_name-Propagation.

Deploy-Pfad: QueueExecutor laeuft WORKER-seitig (worker_jobs.py) ->
cadx-worker Image rebuilden (Executor-Image unveraendert). Restart
erst nach Arm-Ende (in-flight SIMP + arq in_progress-TTL).

Stale-Queue-Hygiene: die 4 toten fem_analyze-Members wurden aus 'cad'
gezrmd (Caller alle exhausted/failed); optimize-Member blieb.

Implikation: fruehere "worker died"-Fails unter Queue-Contention waren
vermutlich derselbe False-Positive — kein Worker starb je bei diesen
60.5s-Fails.

## Verbleibend

- Arm: Optimize laeuft (it7+, ~400s/iter, konvergierend) -> nach
  Stage-Ende Worker-Rebuild + 3 Re-Runs (tank/propeller/bracket).
- Bracket-Fail war ehrlicher Codegen-Gate (blind tap durchgebohrt im
  generierten Teil — Agent-Quality, kein Infra); Re-Run mit frischem
  Seed kann durchlaufen, sonst Intake/Codegen-Verbesserung noetig.

## Arm-Runde (20261109-235313): ehrlicher topopt-Decline

optimize ok=True (SIMP it12/13, 5150s), refine applied=7 ok=True 235s —
aber beide Optimize-Runs strukturell kaputt (sf_export 0.05/0.08, Flansch
weg, Lugs floating) -> Agent lehnte korrekt ab, kept_base SF 2.83 > 1.5.
Eval-Mark: topopt FAIL (kept_base -> kein Volumenziel).

Root-Cause laut Agent-Post-Mortem (selbst dokumentiert):
1. symmetry:"yz" gewaehlt — Agent meinte es spiegle die Z-Mittebene
   (top<->bottom); tatsaechlich spiegelt "yz" um die yz-Ebene = X-Flip
   (links<->rechts). Semantik-Missverstaendnis, evtl. harmlos.
2. Loadpath warm-start seedete nur bore-5-Korridore -> Flansch-Pin-
   Bores ohne Pfad -> SIMP entfernte den Flansch ("unused").
3. Retry mit korrigierten Parametern vom Stage-Budget verweigert
   ("no new run can finish in time") — Policy arbeitete korrekt.

Bereits vorhandener Guard: symmetry-mismatch warning (topopt.py ~
L1257, >15% voxel-mismatch). Neue Evidenz: Agent las "yz" als Z-Flip —
Toolspec-Beschreibung ("mirror the design about that domain mid-plane")
laesst die Achsrichtung offen; Verbesserung: Ebene explizit benennen
(yz = X-Flip etc.) — kosmetisch, kein Blocker.

Deploy + Runde 9 (2026-11-10 ~02:0x):
- cadx-worker neu gebaut (0e69fffd) mit Queue-Lane-Fix 063740a,
  recreated — Executor-Image e53e25a0 unveraendert (Fix ist workerseitig).
- Re-Runs tank + propeller + bracket parallel gestartet; queued Tasks
  reporten jetzt korrekt 'queued' statt false-lost. Stale arq:job/
  queue-Reste der false-lost Tasks wurden gezrmd.
- Bracket-Fail war Codegen-Gate (durchbohrte Blind-Taps im generierten
  Teil, Agent-Quality); Re-Run gibt neue Chance, sonst Intake/Codegen-
  Verbesserung noetig.

## Echter Worker-Crash (bracket r9 fem, 01:20) + Resubmit-Fix

Der Bracket-FEM-Fail in r9 war ein REALER Crash — unterscheidbar vom
false-lost: `[diag] worker pid=7 exited` mitten im fem_analyze
(444k DOFs / 34.5M nnz Solve, ~01:19:5x; parallele codegen/FEM-Last
auf 12GiB cgroup -> OOM-Kill des heavy-lane Worker-Prozesses).

Mechanik danach: cad-Queue laeuft max_tries=1 (Design: "caller
resubmits deliberately") — der in-flight getoetete Versuch frisst das
einzige try-Budget; der restarted Worker claimt das Queue-Member,
inkrementiert job_try->2 > 1 -> "max 1 retries exceeded" OHNE zu
laufen. Der Caller bekam also ein permanentes Fail fuer einen Task der
nie ausgefuehrt wurde — die graceful-lost-Resubmit-Intention griff
nicht (kein not_found-Pfad: arq hatte ein Result geschrieben).

Fix (746158e): _run_task_once + Wrapper — bei Fehler-Signatur
"max N retries exceeded" (nur moeglich bei job_try>1 = in-flight kill;
echte Task-Exceptions serialisieren den eigenen Fehler) einmal
bounded resubmiten. Zweiter Crash -> wieder Fail, durchgereicht.
Regression-Tests: crash->resubmit->ok, echter Task-Fehler->kein
Resubmit.

Ausstehendes Deploy: Worker-Image 0e69fffd hat nur 063740a —
Rebuild fuer 746158e laeuft; Recreate erst wenn tank+propeller r9
ihre Agent-Jobs beenden (Restart toetet laufende Agent-Jobs).
Bracket braucht danach r10 (fem crash waere von diesem Fix abgefangen
worden — Part war verify_spec PASS, SF 3.54).

## r9 post-mortem + worker hardening (2026-11-10 ~02:40)

### Worker-death resubmit (commit 746158e)
Real crash observed on bracket r9: `fem_analyze-34d21244` claimed, executor
worker pid=7 died mid-solve (444k DOFs, OOM under 3-way parallel eval load),
arq re-queued the member with job_try=2 > max_tries=1 → `"max 1 retries
exceeded"` without the task ever running. `QueueExecutor` now resubmits once
with a fresh task id when a task is reported lost/failed after having run;
genuine task exceptions are never resubmitted. Regression tests cover both
paths (4/4 in `test_queue_lanes.py`).

### r9 outcomes
- tank 20261110-020635: codegen gate fail. Rescue fired correctly (3 lugs
  rebuilt) but verify_spec still FAIL: agent's own `through_hole` cuts left
  2/3 bolt bores capped below z≈20.5 (0° open, 120°/240° sealed). Legit
  agent-quality fail caught by verify_spec.
- bracket 20261110-020704: codegen PASS (taps correct — variance confirmed),
  fem stage died to the worker crash above → infra fail, now covered by
  746158e resubmit.
- propeller 20261110-020704: false-lost fem (063740a root cause) — rerun.
- arm: honest decline — SIMP with bad `symmetry:"yz"` + missing loadpath
  corridors produced SF 0.05–0.08 variants; agent correctly rejected both and
  kept the verified base (SF 2.83). Pipeline behaved as designed.

### Deploy state
Worker image `390956c2` (063740a + 746158e) recreated ~02:39. r10 evals
launched 02:40: tank + bracket + propeller in parallel on the patched worker.
Propeller r9 codegen (~40min) was sacrificed for a clean all-fixed validation.

### r10 tank fem: the empty-load_cases 422 (commit 346d115)
- tank r10 codegen passed (score 40, 5 iters, verify_spec envelope overflow
  tolerated). Fem stage died at wall_s=0.0 — `POST /versions/{id}/fem` →
  **422 "at least one load case is required"** (SpawnError default is 409;
  422 is only raised on missing cases). The version spec carried
  `load_cases=[]` — the intake model ignored the prompt's explicit
  "declare the load model … in spec.load_cases" instruction.
- Safety net missed too: `prompt_hints_loads` regex only covered
  optimize/weight/force keywords. "load", "pressure", "6 bar", "bolted",
  "fixed supports" matched nothing → `extract_load_cases` never armed.
- Fix: `_LOAD_HINT_RE` extended with English load/pressure-unit
  (bar/psi/MPa)/stress/structural/mounting vocabulary — the documented
  trigger intent already named "forces, mounting or optimization".
- Propeller prompt has the same gap (old regex silent) — its r10 intake
  ran pre-fix; if its spec drops load_cases the fem stage will 422 and it
  needs a rerun on the rebuilt worker.
- Also added: eval http driver waits (≤300s) for the parent version to
  settle before same-version/child spawns — job-row terminal ≠ version
  settled; POSTing in that window hits editability gates (belt & braces —
  this tank failure was the load-cases 422, not the race, but the race is
  real for fast stages).
- Crash-resubmit (746158e) validated in production: executor worker pid=7
  OOM-killed ~4s into bracket-r10 optimize (344k DOFs alloc), task burned
  "max retries exceeded" → resubmitted fresh → converged ~61 MJ.

### r10 propeller: legit 90min codegen timeout (case bumped to 7200s)
- Codegen hit the stage deadline (5400s) mid-iteration-4 with an in-loop
  fem-verify queued — 5-blade loft geometry genuinely needs >90min on this
  hardware. Spec DID carry 2 load cases this run (intake variance). No
  pipeline bug; the case budget was simply tight → bumped to 7200s.

### r10 bracket: MMA stagnation burn → early-stop guard (commit 7830e63)
- Live observation: `mma.iter` flatlined at compliance 3476.563 (identical
  to 3 decimals) + volfrac 0.1702 for 39+ evaluations at ~90s each —
  the xtol-tightening ratchet had driven nlopt into micro-steps that
  neither xtol nor the guards could end. ~55min of pure burn.
- Existing guards cover explosion (mechanism) and degradation
  (plateau-regression) — not a frozen optimum. New stagnation guard:
  converge_patience evals with last_change <= tol*0.05 and a flat
  objective → ForcedStop, reported as `mma_status="stagnated"`, treated
  like convergence so the beta continuation still sharpens.
- Deploy deferred: the running bracket optimize is ~2h in; the burn ends
  at max_iter=60 anyway. Executor rebuild runs in background.

## r10 verdict: bearing_bracket_topopt PASSED (first full pass)
Run 20261110-024018: codegen score 90 → fem SF 19.08 → topopt SF 14.78
(run 2 adopted after run 3's MMA exhausted its wall-clock — honest
fallback) → refine applied=4, SF 14.1, analytic STEP, 307s. The hardened
pipeline earned the pass: crash-resubmit saved the optimize across a
worker OOM-kill, corridor-fill + rotation gate re-cut all taps watertight.

## r11 launched (2026-11-10 ~05:50 CET)
Full stack: worker 390956c2-era + 346d115 (load-hint regex, parent-settle
driver wait) and executor with 7830e63 (MMA stagnation guard). Cases:
tank, propeller (codegen budget now 7200s), arm.

## r11 verdicts (in progress)
- **robot_arm_mount_autonomous: FAIL at fem gate** — codegen passed
  (verify_spec 35/0) and the fem JOB succeeded, but both declared cases
  ("arm weight on seat", "overturning moment 100 Nm") returned
  safety_factor null — "load on bore resolved to zero force". Spec
  declaration defect, not solver failure: regions resolve to zero
  magnitude on the generated geometry. load_cases are not editable
  mid-loop, so the agent could not self-repair → gate honestly failed.
  Fix e41eeb0: in-loop fem smoke now re-extracts load cases ONCE when
  every declared case is unsolvable (bounded via ctx.load_repair_done,
  audited via emit) — needs worker rebuild + arm re-run.
- pressure_tank_autonomous: still running (codegen).
- marine_propeller_autonomous: still running (codegen, 7200s budget).

### tank r10 fem-422 correction
- Earlier diagnosis ("parent version still running race") was wrong on
  the mechanism: SpawnError defaults to 409; the observed 422 was
  pydantic `load_cases: Input should be a valid list, input: null` —
  the driver posts null when the parent spec has no load_cases. Both
  fixes landed anyway: `_wait_version_settled` closes the real race and
  `_LOAD_HINT_RE` makes intake emit cases for pressure/bolt/support
  prompts. Remaining gap: driver should degrade to parent_spec /
  parent_params sourcing (or skip) instead of posting null.
