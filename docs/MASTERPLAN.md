# KIARA — Technischer Masterplan (Repository Deep Audit)

**Stand:** 2026-11-19 · **Basis:** `feature/anleitungsbuilder` @ `d4d01c4` · **Scope:** gesamtes Repository (backend/kiara ~126k LOC Python, frontend ~27k LOC TS/TSX, mesh_server eigene Quellen, scripts/, evals/, docs/)

> Erstellt aus einem vollständigen Audit durch 3 parallele Analyse-Agents plus
> manuelle Verifikation der kritischen Befunde (Sandbox-Escape live bewiesen,
> N+1 / Pool-Leak / Duplikationen gegen den Code gegengeprüft).

---

## 1. Executive Summary

KIARA ist eine **überdurchschnittlich gereifte** Codebase: die Resilienz-Mechanik (Job-Recovery, Watchdog, Sandbox mit dualen Cold/Warm-Pfaden, `native.py` Rust-Adapter mit Kill-Switches und Parity-Toleranzen, atomic writes, honest-failure-Vertrag in allen Toolschichten) ist auf Produktionsniveau. Die Schwächen konzentrieren sich auf drei Cluster:

1. **Eine echte Sicherheitslücke** in der ersten Sandbox-Barriere (AST-Gate ist per Name-Aliasing umgehbar — live verifiziert). Die eigentliche Isolation (Container, rlimits, ImportGuard) hält trotzdem; die Lücke ist als Verstoß gegen das dokumentierte Verteidigungskonzept zu werten, nicht als Remote-Compromise.
2. **Zwei God-Modules**, die jede fachliche Änderung riskant machen: `cad/engine.py` (6,7k LOC, davon ~4k FEM-Orchestrierung in drei monolithischen Pipelines) und `agent/orchestrator.py` (6,6k LOC, 103 Top-Level-Defs, `_run_cad` = 1.235 LOC am Stück). Git-Churn bestätigt: das sind die zwei meist-editierten Dateien des Repos (118 / 112 Commits in 12 Monaten).
3. **Eine Schicht kleiner, sich summierender Effizienz-Lecks** (kein einzelner Blocker): N+1-Queries, per-Call-`httpx.Client`s, per-Call-GLB-Parsing, DB-Write pro Progress-Event, ein Redis-Pool-Leak im MCP-Session-Pool, ungeschützte Upload-Memory-Reads.

**Rust-Urteil vorweg:** Das Repo besitzt bereits ein ausgereiftes PyO3-Modul (`kiara_native`: BVH-Mesh-Queries, CSR-Assembly, SpMV, PCG, Feature-Edges, Collapse-Remap) mit sauberem Adapter-Pattern, Env-Killswitches und Parity-Tests — die naheliegenden Kernel sind **bereits migriert**. Verbleibend ist genau **ein** marginaler Kandidat (Boundary-Pair-Finding in `mesh_ops`). Kein pauschaler Rust-Rewrite ist zu empfehlen; die Strategie „Rust dort, wo ein messbarer Python-Hotspot existiert, hinter dem `native.py`-Adapter" bleibt die richtige.

---

## 2. Architektur-Überblick

```
┌─ frontend (React/Vite/Three, 132 files, lazy routes, TanStack Query)
│      │ REST /api/* + SSE
├─ api/routers (19 Router, dünn) ─ spawn.py ─ Job.payload {func,args}
│      │                                    │ arq queue "agent" (max_jobs=8)
├─ worker: worker_jobs.run_*_job (14 Funktionen, _guarded + make_emit)
│      │ → orchestrator.run_generation / chat_worker / guide_runner
│      │ → QueueExecutor ── arq lanes "cad"|"cad-light"|"cad-fem"
├─ executor: cad/tasks.py → cad/engine.py
│      │  ├─ sandbox.py/zygote.py (AST-Gate + rlimits + ImportGuard)
│      │  ├─ cad/features, verifyspec, exporters, reconstruct, mesh_ops, dfm
│      │  └─ cad/fem/* (voxelize → regions → solver → topopt/beso/infill/refine)
├─ guide/: PDF → doc/parts/steps-Missionen → preassembly (Port-Solver) →
│      │   assembly-Mission → scene/unity-Export   (eigener Tool-Session-Kosmos)
├─ mcpserver/ (strictly LLM-free Tool-Surface, Session-Pool pro Transport)
├─ mesh_server/meshserver (FastAPI-Nebendienst: GPU-Jobs, LLM-Proxy, Warm-Worker)
└─ kiara_native (Rust/PyO3-Wheel, in Docker-Images gebaut, Fallback-Adapter)
```

**Bewertete Schichtungsqualität:** `api` → `spawn` → `worker_jobs` → `cad/tasks` ist sauber entkoppelt; `guide/` hat eine klare Layering-Kette (`state`→`solver`→`session/tools`→`runner`). **Einzige echte Schichtverletzung:** `queue.py` lazy-importet `kiara.api.deps` innerhalb von Funktionen (queue.py:89,119,210,233,267) — die Queue-Schicht kennt die API-Schicht; funktional notwendig (Zirkularvermeidung), aber architektonisch invertiert. Zweitens zieht `executor_client.py:17-23` via Top-Level-Import `kiara.cad.tasks` den gesamten build123d/OCCT-Stack in jeden Prozess, der nur `QueueExecutor` braucht (inkl. API-Prozess via `mcpserver/session.py:56`).

**Churn-Heatmap (12 Monate, bestätigt die God-Module):** orchestrator.py 118 · engine.py 112 · guide/assembly_tools.py 76 · toolspecs.py 72 · worker_jobs.py 61 · guide/preassembly.py 55 · types.ts 51 · guide_runner.py 51.

---

## 3. Bestätigte Schwachstellen

### 3.1 Kritisch/Hoch

| # | Befund | Evidenz | Klasse |
|---|---|---|---|
| H1 | **AST-Gate per Name-Aliasing umgehbar** | `allowlist.py:527` prüft `node.func.id` nur bei direktem Call; `visit_Name` (allowlist.py:702) flaggt nur `__import__`. `o = open; o("/etc/hostname")` und `g = getattr; g(x, "__subclasses__")` passieren — **live verifiziert**: `check_code()` wirft keinen Fehler. Zusätzlich deckt `FORBIDDEN_ATTRS` keine String-Argumente ab (`getattr` umgeht Attribute-Knoten) und `np.ctypeslib.load_library`/`np.frombuffer`+`np.load(allow_pickle)`/`memmap`/`tofile` sind nicht gesperrt. | Security |
| H2 | **`engine.py` God-Module** | 6.732 LOC; `run_topopt_pipeline` 1.669 LOC (engine.py:2813, inkl. nested `_one_pass` ~860), `run_infill_pipeline` 1.400 (4485), `run_topopt_resample` 337 (5888), `run_refine` ~340 (6382); `# noqa: C901` ×5. Nested-Funktionen machen Stufen einzeln untestbar. | Wartbarkeit |
| H3 | **`orchestrator.py` God-Module** | 6.599 LOC, 103 Top-Level-Defs; `_run_cad` 1.232 LOC (3343), `_run_mesh` 744 (4578), `run_generation` ~300 (6295). 5× `noqa: C901`. Fachliche Änderungen verlangen Kontext über tausende Zeilen. | Wartbarkeit |
| H4 | **Mesh-Job-Upload: unsanitizierter `UploadFile.filename`** | `mesh_server/meshserver/app.py:~181` schreibt Staging-Dateien mit dem rohen Client-Dateinamen (`../../x.png` = Pfadtraversal). Referenz-Implementierung `openai_compat.py:212` sanitisiert bereits korrekt (`Path(name).name`). Gemildert durch Bearer-Auth auf Non-Loopback — auf Loopback-Binds (Default-Dev) unauthenticated by design. | Security |

### 3.2 Mittel

| # | Befund | Evidenz |
|---|---|---|
| M1 | MCP-Session-Pool: Redis-Pool-Leak — `mcpserver/session.py:61` erzeugt `QueueExecutor(None,…)`; `drop()` (168) poppt nur den Dict-Eintrag, `ArqRedis.aclose()` wird nie aufgerufen → ein Pool pro Connect/Disconnect-Zyklus. |
| M2 | `version_tree` N+1 — `versions.py:500` iteriert `_summary` pro Version; `_summary` (97-100) feuert `SELECT Artifact WHERE kind='thumbnail'` pro Zeile. |
| M3 | `_tool_search_library` Full-Table-Scan — `toolspecs.py:2737` `select(LibraryPart)` ohne `WHERE`, Python-Filter in-memory; läuft im Tool-Loop pro Suchaufruf. |
| M4 | `httpx.AsyncClient`-Churn — `llm/client.py:427,618` neu pro Request **und pro Retry**; `mesh_client.py:331` neu pro 2s-Poll über minutenlange Generierungen. Verliert Keep-Alive, erzeugt Socket-Druck. |
| M5 | 13× dupliziertes Worker-Job-Setup — `QueueExecutor(...)+make_emit+LLMClient+AgentCtx`-Blöcke (worker_jobs.py:574,765,875,1130,1262,1900,2166,2394,2617,2775,2905,2991,3372), ~50-100 Zeilen je Block; Drift-gefährdet (emit-Varianten divergieren bereits). |
| M6 | `emit()` schreibt DB pro Event — `worker_jobs.py:209-224`: `session_scope` + `Job.log`-Append + `live`-JSON-Rewrite pro Progress-Event (Mesh-Jobs alle ~2s über zehn+ Minuten); `Job.log` wächst unbegrenzt. |
| M7 | GLB-Reparse pro Kandidat — `Scene.collision_report` lädt `mesh_path` frisch (scene.py:134); `_annotate_candidates` ruft es pro Preview-Kandidat (~10× derselbe GLB-Parse pro Aufruf, assembly_tools.py:922). |
| M8 | Fork-Safety `_Zygote` — module-global `_ZYGOTE` (sandbox.py:194) überlebt `os.fork()` ohne Owner-PID-Guard; ein geforkter Pool-Worker würde in die Parent-Pipe schreiben (derzeit nur Konvention verhindert es). |
| M9 | FEM-Pool-Timeout lässt Kinder laufen — `analyze.py:795-810` `shutdown(wait=False)` ohne Kill der `_processes` → geleakte CPU bis Executor-Restart. |
| M10 | Upload-Endpoints lesen Body vor Cap-Check — `assets.py:85`, `library.py:57`, `guide.py:792`: `await f.read()` vollständig im RAM, dann `len(data) > cap`. |
| M11 | `resume_state["guide_dir"]` wird als FS-Root vertraut — `worker_jobs.py:796`: kein Re-Check unter `storage.data_root()`; korrupte/legacy Rows könnten außerhalb schreiben. |
| M12 | `stack/restart/{service}` reicht unvalidierten User-Input in Updater-URL — `system.py:146-150`; Service-Allowlist fehlt. |
| M13 | `run/mesh-server.env` enthält `MESH_API_KEY` im Klartext ohne `chmod 600` (`start.sh:278-287`); `--lan`-Modus schreibt den Key zusätzlich ins Start-Log (`start.sh:183-184` → `logs/start-*.log`). |
| M14 | i18n-Drift: ja + zh fehlen je ~27 Keys ggü. en/de (1932 vs. 1905) — stille en-Fallbacks. |
| M15 | Runtime-Signatur-Probing — `orchestrator.py:141-145` inspiziert `inspect.signature(critique_step.run)`; ein Step-Refactor flippt Orchestrator-Verhalten stillschweigend. |

### 3.3 Niedrig / verifiziert harmlos, aber bereinigenswert

- **`solve_multi_attachment` tot** (solver.py:1077-1120) — einziger Referent ist ein Test; durch `solve_bridge_attachment` ersetzt.
- **5× World-AABB-Duplikation** — `solver.world_bbox` (722), `solver.transform_bbox` (1287), `preassembly._rotated_bbox` (795), `preassembly._bbox_corners_world`+Wachstum (2199), `api/routers/guide.py._world_bbox` (225). Identische 8-Corner-Hull-Semantik.
- `rembg.new_session("u2net")` pro Bild (`vision/measure.py:58`, ~170MB Modell-Reload) — `assets.py:158` cacht korrekt global.
- Temp-Leaks: `images.py:89` `mkdtemp` ohne finally-Abdeckung des Generate-Pfads; `pdfdoc.py:121,200` `pdf/doc.close()` nicht in `finally`.
- `llm/client.py:434,652` Retry-Backoffs ohne Cancel-Check (bis ~64s Schlaf pro Versuch trotz Abbruch).
- `spawn.py`: ~15 `spawn_*`-Funktionen teilen ein nahezu identisches Version/Job/Enqueue-Gerüst.
- `evalkit` `_resolve` doppelt (http.py + local.py); `_phase_timings` doppelt (fem/beso.py + fem/topopt.py).
- `update.sh:412` referenziert nicht-existierendes `scripts/restore_db.sh`; `update_server.py:199` Tokenvergleich per `==` (nicht constant-time); `active_job_count` → 0 bei unerreichbarer API (fail-open).
- `llm_proxy` geteilter `httpx.AsyncClient` ohne Shutdown-`aclose` (mesh_server/meshserver/llm_proxy.py:~1066).
- `jobs_lifecycle_stream`: 4-Table-Join pro 15s-Keepalive pro Client (sse.py:147).
- `union_containment` reine Python-Triple-Loop (solver.py:771) — 4 Zeilen Broadcasting stattdessen.
- `mcpserver` `SESSIONS`-Pool unbounded (nur drop() bei Transport-Close); `service_slots._SLOTS` ohne Eviction.
- AGENTS.md `pytest`-Befehl ohne `-m "not verify"`; `docs/updating.md` erwähnt `restore_db.sh`.

### 3.4 Verifiziert KEINE Probleme (Explizit)

- **`_done_guard` in 5 Guide-Tool-Dateien** — Registry-Pattern (DONE_GUARDS), missionsspezifisch, keine Duplikation.
- **`guide/tools.py` `_fn/_s/_i` vs. `toolspecs.py`** — dokumentierte, absichtliche Isolation der Guide-Tool-Welt.
- **`InProcessExecutor`** — lebt in `cli.py:245`/`evalkit/local.py`, kein toter Code.
- **`_PATTERN_RE`/`_HOLE_TYPES` in verifyspec.py** — dokumentierte bewusste Lokalhaltung.
- Frontend: 0 `any`, 0 ts-ignore, keine ungenutzten Dependencies, Lazy-Routes + manuelles Three-Chunking bereits gesetzt.

---

## 4. Stabilitäts- & Sicherheitsbewertung

**Stark:** Sandbox Cold/Warm-Dualpfad (zygote `setsid`+`killpg`, Pipe-Drain vor waitpid, Protokoll-Resync), Job-Recovery (deterministische arq-IDs, AOF, 60s-Sweeper, Orphan-Reaper), Watchdog (Flap-Suppression, Log-Mtime-Stall-Detection statt Phase-Timing, ExitCode-0/143-Ausschluss, Supervision-Contract-File), Compose-Hardening (`read_only`, `cap_drop:ALL`, `pids_limit`, bounded logs), Secrets (`enc:v1:` Fernet at-rest, `public_values()` nur `_set`-Booleans), Update-Flow (Lock, Re-exec, PREV_SHA-Journal, Pre-Migration-pg_dump, Idempotenz-Marker), SSE subscribe-before-snapshot.

**Lücken (priorisiert):**
1. H1 AST-Alias-Gap — erste dokumentierte Barriere durchbrochen (Container bleibt echte Isolation).
2. H4 Upload-Filename-Traversal Mesh-Staging.
3. M10 Upload-RAM vor Cap — DoS-Vektor auf lokalen Instanzen.
4. M8/M9 Fork/Pool-Kind-Leaks — Latente Prozess-Hygiene.
5. M13 Secret-Hygiene in `run/mesh-server.env` + Start-Logs.
6. M12 Service-Name-Allowlist für Update-Proxy.
7. Suspected (unbewiesen, im Plan verankert als Verifikationstasks): `registry.get(strict=False)`-Aufrufstellen in `gen_worker.py` können die Memory-Admission umgehen (OOM-Pfad); Cancel/`wrong-model`-Cleanup-Pfade in `jobs.py`/`gen_worker.py` (Staging-Inputs); `chat_worker` While-Loop Boundedness aller Exit-Pfade.

**~967 `except Exception`/`suppress`** — durchgehend bewusster „Tools werfen nicht"-Vertrag (kein bare `except:` gefunden); akzeptabel, aber der Preis ist, dass einzelne echte Bugs als „fehlgeschlagene Operation" maskiert werden können — empfohlen: strukturierte Error-Klassifizierung in den kritischen Pfaden (fem solver, runner_entry mesh-boolean-Fallback).

---

## 5. Performance-Analyse

Gemessen/verfolgt (keine Theorie):

| Hotspot | Kostenmechanik | Befund |
|---|---|---|
| GLB-Reparse in Previews | `trimesh.load` ~50-300ms × Kandidaten × Preview-Calls | M7 — Cache auf Scene reicht |
| `version_tree` | N+1 Queries pro Baumaufruf | M2 — ein `IN_()`-Query |
| `search_library` Tool | ganze Tabelle in-memory pro Call | M3 — SQL `ILIKE`+`LIMIT` |
| `emit()` | DB-Write pro Progress-Event, `live` JSON-Rewrite | M6 — ≥1s-Flush + Log-Cap |
| `httpx`-Churn | TLS-Handshake + Pool-Aufbau pro Call/Poll | M4 — persistente Clients |
| `union_containment` | 512-Punkte-Python-Loop pro Instanz-Paar im Audit | numpy-Broadcasting |
| `rembg` Session | 170MB Modell-Reload pro Bild | globaler Cache wie assets.py |
| `estimate_prompt_tokens` | voller Message-Walk + `json.dumps` pro Turn | vermutlich akzeptabel; nur optimieren wenn Turn-Count-Hotspot gemessen (vermutetes Risiko) |

**Latenz-Dominanz ist LLM/Netz/DB — nicht Python-CPU.** In agent/llm/toolspecs existiert **kein** CPU-Hotspot, der Rust rechtfertigt (Token-Schätzung, Regex-Audits, Digest-Bildung sind ms-skalig ggü. Sekunden-bis-Minuten-Calls). Regex-Sammlungen sollten trotzdem modul-global `re.compile`d werden (Trivial-Fix).

---

## 6. Rust-Migrationspotenzial

**Ist-Zustand:** `backend/kiara_native` (PyO3 0.30, abi3-py312, maturin-Wheel im selben trixie-Base wie die Images gebaut) implementiert bereits: Mesh-BVH (`MeshIndex`: contains/ray_any/ray_locs/scanline_fill), FEM-CSR-Assembler, paralleles SpMV, PCG-Solver, `feature_edges`, `collapse_remap`. Adapter `cad/cad/native.py`: jede Funktion gibt `None` bei Fehlen/Fehler → Python-Fallback; `KIARA_NATIVE{,_GEOM,_FEM}=0` Killswitches; Determinismus-Dokumentation (rtol 1e-9 Assembly / 1e-5 Solve); 6 Parity-Testdateien in `tests/native/`. **In Produktion geladen** (worker + executor Container verifiziert, `kiara_native 0.1.0`).

| Kandidat | Stand | Urteil |
|---|---|---|
| FEM Assembly/SpMV/PCG | bereits native | ✅ done |
| Mesh contains/ray/scanline | bereits native | ✅ done |
| Boundary union-find | bereits native (`collapse_remap`) | ✅ done |
| Boundary-**Pair-Finding** (`mesh_ops.py:2288-2303`, kdtree `query_ball_point` pro Boundary-Edge in Python-Loop) | Boundary-Edges i.d.R. < wenige k | **potenziell sinnvoll** — erst `_qt`-Profilmessung; ggf. `scipy.spatial`-Batchquery statt Rust |
| `union_containment` Grid-Sampling | 512-Punkte-Loop | **nicht sinnvoll** — numpy fixt es |
| Guide Port-Matching/Transform-Scoring | Ports ≪ 100/Instanz, ms-skalig | **nicht sinnvoll** |
| FEM `_B_matrix`/`_ke_unit` | einmal pro (ν,pitch), 8 Iterationen | **nicht sinnvoll** |
| Sandbox-AST-Check | einmal pro Submission, ms | **nicht sinnvoll** — Korrektheit schlägt Speed |
| STEP/GLB-Parsing | OCC (C++) / trimesh | **nicht sinnvoll** — natives Backend; Callsite-Caching stattdessen |
| `patch.py` Fuzzy-Match | O(n²·k) Fenster, ms-skalig auf Modelldateien | **nicht sinnvoll** |
| `token estimation`/LLM-Stream | netz-gebunden | **nicht sinnvoll** |
| `cadmeta` Ray-Ringe | bereits vektorisiert + native rtree | **keep** |

**Empfehlung:** Rust-Budget in die **Verbreiterung des `kiara_native`-Vertrags** nur dort, wo ein `_qt`-Profil (features.py hat die Instrumentierung bereits) einen >20%-Anteil misst. Der einzige dokumentierte Kandidat ist das Boundary-Pair-Finding; vorher Numpy-Batch probieren.

---

## 7. Zielarchitektur (Delta, kein Rewrite)

Bestehende, bewährte Entscheidungen bleiben: Router-dünn + spawn/enqueue + arq-Lanes + QueueExecutor; Sandbox-Dualpfad; `native.py`-Adapter; Guide-Mission/Lint/Preassembly-Schichtung; `storage.py` als Path-Authority; Fernet-Secrets; Evalkit-Atomicity; Shell-Lib (`ui.sh`/`pkg.sh`) als Script-Basis.

**Änderungen:**

1. **`engine.py` → Package `cad/engine/`** — Aufteilung an den Pipeline-Nähten: `cad_exec.py` (Code-Exec/Export/Verify), `mesh.py` (run_mesh_pipeline), `fem_verify.py` (Cases/Solve/Audit), `topopt.py` (Pipeline + Nested-Stufen als Funktionen), `infill.py`, `resample.py`, `refine.py`, `shared.py` (die ~25 FEM-Helper 1587-2635). Öffentliche Entry-Points (`run_pipeline`, `run_topopt_pipeline`, `run_infill_pipeline`, `run_topopt_resample`, `run_refine`) bleiben über `cad/engine.py` als Fassade re-exportiert → **kein Breaking Change**.
2. **`orchestrator.py` → schlankere Orchestrierung** — Extraktion nach Fachlichkeit: `agent/orchestrator/` = `candidates.py` (149-258: Candidate/Outcome/_select/_regression), `artifacts.py` (969-1056: copy/promote/clean), `payloads.py` (_exec_payload/_mesh_payload), `feedback.py` (build_feedback/build_issues/issue-Helpers), `mode.py` (resolve_mode/_hybrid_prefers_cad), `topopt_phase.py` (1057-1308). `_run_cad`/`_run_mesh` werden Sequenz-Shells über Steps. Re-Export-Fassade für Worker-Imports.
3. **Gemeinsame Infrastruktur konsolidieren:** ein `world_bbox` (solver.world_bbox als einzige Implementierung); ein Worker-Job-Bootstrap (`worker_jobs._job_ctx()`); ein persistentes HTTP-Client-Pattern (shared `AsyncClient` pro Client-Klasse + `aclose` am Lebensende); modul-globale `re.compile` für alle per-Call-Regexe.
4. **Config-Konsolidierung (leichtgewichtig):** die ~45 `os.environ`-Stellen dokumentieren, welche davon Compose-Override (Dockerfile/.env) vs. Runtime-DB sind; `.env.example` als Vertragsdokument; kein Umbau nötig — nur Inventar + Kommentar-Referenz.
5. **Guide-Heuristik-Konstanten zentralisieren:** die numerischen Gates in `guide/solver.py`/`guide/preassembly.py` (GAP_GATE_MM, NORMAL_DOT_GATE, Sibling-Decay 150mm, Burial-Tiers 0.45/0.85, Exposure-Caps…) in benannte Konstanten mit physikalischer Begründung — das macht die „domain-general"-Behauptung pro Konstante auditierbar (Projektregel-konform) statt incident-getrieben.

---

## 8. Priorisierter Masterplan

Legende: **Aufwand** S<2h / M<1d / G>1d · **Prio** kritisch>hoch>mittel>niedrig

### Phase 0 — Sicherheit & sofortige Korrektheit (zuerst, risikolos)

| # | Maßnahme | Dateien | Prio | Aufw. | Abhäng. |
|---|---|---|---|---|---|
| 0.1 | AST-Alias-Gap schließen: `visit_Name` flaggt `Load`-Kontext `Name.id in FORBIDDEN_NAMES` (tötet `o=open`,`g=getattr`); numpy-scoped Attrs `ctypeslib/load/memmap/frombuffer/tofile/fromstring` in `FORBIDDEN_ATTRS`; String-Konstanten `^__\w+__$` in get()-ähnlichen Calls scannen sobald Aliasing zu ist | `cad/allowlist.py:527,702,708` + Escape-Regressionstests `tests/cad/` | kritisch | S | — |
| 0.2 | `UploadFile.filename` sanitizen (Pattern aus `openai_compat.py:212` übernehmen: `Path(name).name`) | `mesh_server/meshserver/app.py:~181` | kritisch | S | — |
| 0.3 | Upload-Cap vor Body-Read erzwingen: `Content-Length`-Header check + `f.read(cap+1)` Streaming-Grenze | `api/routers/assets.py:85`, `library.py:57`, `guide.py:792` | hoch | S | — |
| 0.4 | `chmod 600 run/mesh-server.env`; `MESH_API_KEY` nie ins Start-Log (redact oder `.env`-Hinweis) | `scripts/start.sh:183,278` | hoch | S | — |
| 0.5 | `stack/restart/{service}` → Allowlist valider Service-Namen | `api/routers/system.py:146` | mittel | S | — |
| 0.6 | `resume_state["guide_dir"]` unter `storage.data_root()` re-validieren, sonst re-derive | `worker_jobs.py:796` | mittel | S | — |
| 0.7 | `secrets.compare_digest` für Updater-Token; `active_job_count` fail-closed bei unerreichbarer API (oder explizit dokumentiert) | `scripts/update_server.py:199`, `update.sh:126` | mittel | S | — |
| 0.8 | Temp/Handle-Leaks: `images.py:89` mkdtemp→`TemporaryDirectory`+finally; `pdfdoc.py:121,200` close→finally; `rembg` Session global cachen | `api/routers/images.py`, `guide/pdfdoc.py`, `vision/measure.py:58` | mittel | S | — |
| 0.9 | Tote Referenz `restore_db.sh` entfernen oder Skript erstellen; AGENTS.md-`pytest`-Befehl um `-m "not verify"` ergänzen | `update.sh:412`, `docs/updating.md`, `AGENTS.md` | niedrig | S | — |

**Gate P0:** `pytest tests -x -q` grün; neuer AST-Escape-Test `g=getattr; g(...)` → reject; Upload mit `../`-Filename → 4xx; `make lint` sauber.

### Phase 1 — Effizienz-Lecks (klein, unabhängig, direkt messbar)

| # | Maßnahme | Dateien | Prio | Aufw. |
|---|---|---|---|---|
| 1.1 | MCP-Session-Executor: Pool teilen oder `drop()` → `await executor.aclose()` | `mcpserver/session.py:61,168` | hoch | S |
| 1.2 | `version_tree`: Thumbnails per `Artifact.version_id.in_(vids)` batchen | `api/routers/versions.py:97,500` | hoch | S |
| 1.3 | `search_library` → SQL `ILIKE`+`LIMIT` | `toolspecs.py:2737` | hoch | S |
| 1.4 | Persistente `AsyncClient`s (LLMClient, HttpMeshClient) + `aclose` auf Session-Ende | `llm/client.py:427,618`, `agent/mesh_client.py:331` | hoch | M |
| 1.5 | `emit()` drosseln: Flush ≥1s oder Statuswechsel; `Job.log` auf N Zeilen kappen | `worker_jobs.py:209` | mittel | S |
| 1.6 | Scene-Mesh-Cache (`path→trimesh`, transformierte Kopien für Candidate-Variants) | `guide/scene.py:26,134`, `assembly_tools.py:922` | mittel | S |
| 1.7 | `union_containment` auf numpy-Broadcasting | `guide/solver.py:771` | niedrig | S |
| 1.8 | Cancel-aware Sleeps im LLM-Retry (`ctx.check_cancel` in Backoff) | `llm/client.py:434,638` | niedrig | S |
| 1.9 | `zrange(0,-1)`→`zscan` im Chat-Sweeper; `SESSIONS`-TTL | `queue.py:213`, `mcpserver/session.py` | niedrig | S |
| 1.10 | `re.compile` zentral für per-Call-Regexe (autofix, toolspecs-Feature-Audit) | `autofix.py:36`, `toolspecs.py:475` | niedrig | S |

**Gate P1:** Benchmark `version_tree` (vorher/nachher Query-Count via assert), MCP-Session-Recycling ohne FD-Wachstum (`/proc`-FD-Count oder lsof über 50 Connect/Disconnects), kein neuer Client pro Poll (Code-Review), Suite grün.

### Phase 2 — Fork-/Prozess-Hygiene & Duplikationen

| # | Maßnahme | Dateien | Prio | Aufw. |
|---|---|---|---|---|
| 2.1 | `_Zygote` Owner-PID-Guard (`os.getpid()` bei Spawn; Mismatch→Cold-Path) | `cad/sandbox.py:194` | hoch | S |
| 2.2 | FEM-Pool-Timeout → Kindprozesse killen (`pool._processes` term) oder per-Case-Deadline im Child | `cad/fem/analyze.py:795` | mittel | M |
| 2.3 | World-AABB auf `solver.world_bbox` konsolidieren (4 weitere Implementierungen ersetzen) | solver.py:722,1287 · preassembly.py:795,2199 · routers/guide.py:225 | mittel | S |
| 2.4 | `solve_multi_attachment` löschen (oder explizit als API markieren) | `guide/solver.py:1077` | niedrig | S |
| 2.5 | Worker-Job-Bootstrap `worker_jobs._job_ctx(emit_variant, needs)` aus den 13 Blöcken extrahieren | `worker_jobs.py` (alle 13 Sites) | mittel | M |
| 2.6 | `spawn_*`-Familie auf einen parametrisierten Builder reduzieren (~15 fn) | `spawn.py` | niedrig | M |
| 2.7 | Signature-Probing → deklariertes Protokoll/Capability auf Steps | `orchestrator.py:141` | mittel | S |
| 2.8 | `InProcessExecutor` lazy-import `cad.tasks` in Methoden (entkoppelt API-Prozess von build123d) | `executor_client.py:17` | mittel | S |
| 2.9 | i18n-Parity-Check (Skript/CI: flatten Keys vs. `en`) + 27 fehlende ja/zh-Keys | `frontend/src/i18n/*` | niedrig | S |

**Gate P2:** Fork-Test (Worker in Pool ruft `run_user_code(warm=True)` → Cold-Fallback, keine Pipe-Korruption); AABB-Unit-Tests aller 5 Altpfade zeigen identische Hüllen vor Konsolidierung; Suite grün.

### Phase 3 — God-Module-Dekomposition (größtes Risiko, höchster langfristiger Wert)

| # | Maßnahme | Dateien | Prio | Aufw. |
|---|---|---|---|---|
| 3.1 | `engine.py`→Package (Zielbild §7.1); schrittweise: zuerst `fem/`→`engine/fem_*` Move+Reexport, dann Nested-Stufen herausheben | `cad/engine.py` 2813-6720 | hoch | G |
| 3.2 | `orchestrator.py`→`agent/orchestrator/`-Teilmodule (§7.2); Fassade behält Imports | `agent/orchestrator.py` | hoch | G |
| 3.3 | Nested Pipeline-Stufen (`_one_pass`, `_verify_one_pass_export`, `_resolve_cases_on`) zu modularen Funktionen heben → direkte Unit-Tests je Stufe | engine.py (innerhalb 3.1) | hoch | M |
| 3.4 | Guide-Heuristik-Konstanten zentralisieren + physikalische Begründung | `guide/solver.py`, `preassembly.py` | mittel | M |

**Risiken:** Orchestrator/Engine leben von geteiltem mutablem `AgentCtx`/Closure-State — Extraktion nur mit identischem Verhalten; Fassade-Reexports halten alle Import-Pfade stabil. **Strategie:** mechanische Moves zuerst (git-bewahrt Historie via `git mv`), dann Funktionshebung, jeweils mit vollem Suite-Lauf dazwischen. Kein Verhaltens-Refactoring im selben Commit wie Moves.

**Gate P3:** `pytest tests` + `make eval ARGS="--case l_bracket_rib"` unverändert; `ruff` ohne neue `noqa`; Public-API-Import-Test (`from kiara.cad.engine import run_pipeline` etc.).

### Phase 4 — Verifikation vermuteter Risiken (Audit→Code, kein Blindfix)

| # | Verifikation | Bei Bestätigung → Maßnahme | Aufw. |
|---|---|---|---|
| 4.1 | `gen_worker.py`: nutzt jeder Call `registry.get(strict=True)`? | strikte Admission erzwingen | S |
| 4.2 | `jobs.py`/`gen_worker.py` Staging-Cleanup auf cancel/`wrong-model`/kill-Pfaden | Staged-Input-Delete in allen Terminal-Pfaden | S |
| 4.3 | `chat_worker.run_chat_turn`: alle While-Loop-Exits enumerieren (Tool-Kill, Budget, Error-Streak, Infra) | fehlende Boundedness ergänzen | S |
| 4.4 | `worker_jobs.make_emit` `Job.log`-Größe auf Langläufern (DB-Zeilengröße messen) | Cap/Rollover (deckt sich mit 1.5) | S |
| 4.5 | `mcpserver/tools.py` Handler `_h_stats`/`_exec_payload`: N+1 + Payload-Bounds | analog 1.2/1.3 | S |
| 4.6 | `estimate_prompt_tokens`-Kosten bei >100-Turn-Sessions | ggf. inkrementelles Token-Budget-Caching | M |

### Phase 5 — Rust-Breite (nur evidenzbasiert)

| # | Maßnahme | Bedingung | Aufw. |
|---|---|---|---|
| 5.1 | `_qt`-Instrumentierung auf `mesh_ops` Boundary-Pair-Finding (`mesh_ops.py:2288-2303`) — Profil auf echten Mesh-Edit-Jobs | Wenn >20% der `process_mesh`-Zeit: zuerst `scipy.spatial.cKDTree.query_ball_point`-Batch (kein Rust nötig); erst dann `edges.rs` erweitern | S (Messung) |
| 5.2 | `kiara_native` Parity-Suite in `make test`-Pfad dokumentiert halten (bereits `tests/native/`) | — | — |
| 5.3 | Keine weiteren Rust-Kandidaten ohne Profil-Beweis | — | — |

### Phase 6 — Qualitätssicherung & Dokumentation

| # | Maßnahme | Aufw. |
|---|---|---|
| 6.1 | Test-Gap-Füllung: `agent/tripo_client`, `agent/concept_image`, `agent/model_pick`, `agent/mesh_cache`, `agent/trace`, `agent/webresearch`, `cad/contact_sheet`, `cad/analysis`, `mcpserver`-Handler, `vision` (derzeit dünn/keine dedizierten Dateien) | M |
| 6.2 | i18n-Parity als CI-Gate (`make lint` Schritt) | S |
| 6.3 | `evals/assets/kiara_heatpump` Binär-Fixtures → git-lfs oder `make eval-assets` Fetch | M |
| 6.4 | Docs-Nachzug: `restore_db.sh`, `pytest`-Befehl, Undocumented `scripts/sim_preassembly.py`/`evals/NOTES_*` | S |
| 6.5 | Error-Klassifikation in kritischen `except`-Pfaden (fem solver, runner_entry Boolean-Fallback) — Struktur statt generischem `suppress` | M |

---

## 9. Abhängigkeiten & Sequenz-Logik

```
P0 (Sicherheit) ── unabhängig, sofort
P1 (Effizienz)  ── unabhängig von P0; 1.4 vor 1.5 (Client-Persistenz erleichtert Emit-Drossel-Messung)
P2 (Hygiene)    ── 2.3/2.5/2.6 profitieren von P1's bereinigtem Call-Surface; 2.1/2.2 unabhängig
P3 (God-Module) ── zuletzt groß: saubere Basis (P0-P2) minimiert Konfliktfläche; 3.1 vor 3.3
P4 (Verifikation) ── parallel zu allem, liefert ggf. neue P0/P1-Items
P5 (Rust)       ── erst nach Messung; kann jederzeit laufen
P6 (QA/Docs)    ── kontinuierlich; i18n-CI nach 2.9
```

---

## 10. Abschlusskriterien (pro Phase)

- **P0:** Neue Security-Regressionstests grün (Alias-Escape reject, `../`-Upload reject, Traversal-`guide_dir` reject); volle Suite unverändert grün.
- **P1:** Query-Count-Assertions (`version_tree` konstant O(1)-Queries); FD-Stabilität MCP-Sessions über 50 Cycles; kein `AsyncClient()`-Neubau in Poll/Retry-Pfaden (grep-Assert im Test).
- **P2:** Fork-Safety-Test bestanden; AABB-Konsolidierung ohne Verhaltensdiff (Golden-Tests); 13 Setup-Blöcke → 1 Helper (grep-Count = 0).
- **P3:** Identische Eval-Ergebnisse (`make eval` Basis-Cases) vor/nach Split; keine API-Import-Breaks; neue Unit-Tests für gehobene Pipeline-Stufen.
- **P4:** Jede vermutete Gefahr dokumentiert als bestätigt+gefixt ODER widerlegt+geschlossen.
- **P5:** Benchmark-Protokoll (vorher/nachher `_qt`-Zahlen) für jede Native-Erweiterung, Parity-Tests grün.
- **P6:** `make lint` + i18n-Parity-Check + `pytest tests -x -q -m "not verify"` in CI-Pflicht; keine Module >3.5k LOC ohne dokumentierte Begründung in `AGENTS.md`.

**Globale Kriterien:** Ruff sauber (aktuell 11 minimale Befunde — mitziehen); TS strict ohne neue `any`; keine neuen `os.environ`-Stellen außerhalb des Settings-Moduls; `kiara_native` Kill-Switches bleiben für jede neue Kernel-Funktion verpflichtend.

---

## 11. Explizit NICHT empfohlen

- **Pauschaler Rust-Rewrite** — die messbaren Hotspots sind schon nativ; Rest ist I/O- oder Heuristik-skalig.
- **Framework-Wechsel** (FastAPI/arq/React bleiben) — kein Befund rechtfertigt Migration.
- **Auflösung der Tools-nie-werfen-Konvention** — konsistent durchgesetzt, bewusstes Design.
- **Frontend-Refactoring** — sauberste Schicht des Repos (0 `any`, Lazy-Routes, typed API); nur i18n-Drift + Einzelfixes.
- **`guide/`-Heuristik-Umbau** — gerade produktiv-härtende Phase; nur Konstanten-Dokumentation (3.4), keine Semantik-Änderung.
- **Worktree-Aufräumen (`3d_model_agent-fem`)** — separates aktives Feature-Branch-Checkout (`feature/fem-topopt-infill`), nicht dieses Audit's Aufgabe; Besitzer entscheidet.

---

## 12. Bewertung der Ausgangslage (ehrlich)

| Dimension | Note | Kommentar |
|---|---|---|
| Architektur | B+ | klare Schichten, 2 God-Module drücken die Note |
| Codequalität | B+ | konsistent, getypt; Hotspots konzentriert |
| Stabilität | A- | Recovery/Watchdog/Sandbox exzellent; Fork-/Pool-Ränder offen |
| Sicherheit | B | H1+H4 echte Lücken; ansonsten durchdacht (Fernet, Sandbox, Compose-Hardening) |
| Performance | B | Effizienz-Lecks summieren sich, keine einzelnen Blocker; native Kernel gut |
| Tests | A- | 3.371 Tests, breite Abdeckung, dedizierte Native-Parity; Gap-Liste §Phase 6.1 |
| Rust-Nutzung | A | Referenzimplementierung für „Rust wo es zählt" |
| Repo-Hygiene | A- | sauberes .gitignore, keine getrackten Artefakte; Binär-Fixtures + env-Sprawl kleinere Abzüge |

---

## 13. Ausführungs-Supplement (optimierte Umsetzungsreihenfolge)

Gegenüber §8 optimiert — gleiche Maßnahmen, bessere Parallelisierung und
geringeres Risiko:

1. **File-Ownership statt Maßnahmen-Ownership.** Parallele Workstreams werden
   nach Datei-Menge partitioniert (keine zwei Streams editieren dieselbe
   Datei). Dadurch entfallen Rebase-Konflikte; die ursprünglichen Phasen
   bleiben als Commit-Granularität erhalten.
   - **Stream A** (api/services): `api/routers/{versions,system,assets,library,images}.py`,
     `agent/{toolspecs,autofix,worker_jobs,mesh_client}.py`, `llm/client.py`,
     `mcpserver/session.py`, `queue.py`, `vision/measure.py`, `spawn.py`,
     `executor_client.py`, `agent/orchestrator.py` (nur :141), `chat_worker.py`
   - **Stream B** (guide/cad): `guide/{solver,preassembly,scene,assembly_tools,pdfdoc}.py`,
     `api/routers/guide.py`, `cad/sandbox.py`, `cad/fem/analyze.py`, `cad/allowlist.py`
   - **Stream C** (infra/frontend/docs): `meshserver/app.py`, `scripts/{start,update,update_server}.*`,
     `AGENTS.md`, `docs/updating.md`, `frontend/src/i18n/*`, `Makefile` (i18n-Gate)
2. **Blame-erhaltende God-Module-Splits (P3).** Statt Neuanlage:
   `git mv cad/engine.py cad/engine/__init__.py` — die Datei wird zum Package-
   `__init__`, alle `from kiara.cad.engine import X` bleiben funktional UND
   `git log --follow` behält die Historie. Danach Funktionsgruppen schrittweise
   in Geschwistermodule (`engine/topopt.py` etc.) heben; `__init__` re-exportiert.
   Analog `agent/orchestrator.py → agent/orchestrator/__init__.py`.
   Jeder Move-Schritt endet mit vollem Suite-Lauf vor dem nächsten.
3. **P4-Verifikationen in die Owner-Streams gefaltet** (4.1/4.2 → Stream C,
   4.3–4.5 → Stream A, 4.6 → dokumentiert, kein Code ohne Messbefund).
4. **P5** reduziert sich auf die `_qt`-Messung (keine Implementierung ohne
   Profil-Beweis).
5. **P6.1 (Test-Gaps)** auf die höchstwertigen ungetesteten Einheiten begrenzt
   (`agent/mesh_cache`, `agent/trace`, `agent/tripo_client`-Parsing) — der Rest
   folgt bei Bedarf. **P6.3 (git-lfs)** bleibt eine dokumentierte Entscheidung
   (History-Rewrite wäre destruktiv); Alternative `make eval-assets` nicht
   implementiert, solange keine Quelle existiert.
6. **Commit-Policy:** nur Audit-berührte Dateien pro Commit; die ~80
   uncommitted User-Dateien im Worktree bleiben unangetastet.
7. **Sandbox-Fix-Detail (0.1):** `visit_Name` flaggt `Load`-Kontext für
   `FORBIDDEN_NAMES \ {dir, input}` (beide bleiben als Variablennamen legal —
   `dir` ist ein legitimer Richtungsvektor-Name, ihre Aliase sind harmlos).
   `getattr` ist damit unreferenzierbar → die String-Arg-Lücke schließt sich
   mit. Numpy-scoped Forbids (`ctypeslib`, `load`, `loads`, `loadtxt`,
   `genfromtxt`, `fromfile`, `frombuffer`, `fromstring`, `memmap`, `save`,
   `savez*`, `savetxt`, `tofile`, `dump`, `dumps`) mit Root-Auflösung über
   `np`/`numpy` **plus** einstufige Alias-Präpass (`x = np`, `import numpy as x`).
