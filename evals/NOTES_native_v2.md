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
