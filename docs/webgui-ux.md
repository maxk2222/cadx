# WebGUI Masterplan — UX/Design/Feature-Optimierung

> Status: implementiert und verifiziert (Oktober 2026) — siehe Commits
> `587c32f` (API), `555f8c7` (UI), `d034e28` (Scroll-Fix).

Ziel: Das Web-GUI soll die Fähigkeiten des Agentensystems vollständig sichtbar machen,
konsistenter aussehen und besser navigierbar sein. Scope: vorwiegend Frontend
(React/Vite/Three.js/Tailwind), plus kleine Backend-Ergänzungen, die die UI freischalten.

## Ist-Analyse

### Agentensystem (was die API bereits hergibt)

- Pipeline-Schritte mit Live-Events (SSE `step`-Events):
  `intake → plan → (meshgen → meshprocess/orient | codegen → execute → repair) → critique → finalize`,
  dazu `budget`, `done`, `question`. `Job` trägt `progress` (0→1), `step`, `iteration`,
  `log`, `questions`, `started_at`/`finished_at`.
- `VersionDetail` liefert u.a. `spec` (PartSpec: Maße mit Quelle, Features, Prozess,
  Material, Annahmen, offene Fragen), `plan` (ModelingPlan: Ansatz+Begründung, Steps,
  Parameter, Standardteile), `usage` (Token-Verbrauch), `critique` (Score, Discrepancies,
  Matches), `dfm`, `mesh` (Provenienz/Orientierung/Skalierung), `artifacts`.
- Versionsbaum: `generate | edit | params` mit `gen_mode`/`resolved_mode`.
- Endpunkte vorhanden aber im UI ungenutzt: `GET /catalog` (bd_warehouse-Standardteile),
  `jobs/{id}/answer`, Job-Historie pro Version, `usage`, `spec`, `plan`.
- Health: `GET /health` (api ok, llm_configured), `GET /mesh/health` (device, modelle).

### Schwächen des aktuellen GUI

1. **Versteckte Daten**: `spec`, `plan`, `usage`, `iterations`, Job-Laufzeiten und
   Critique-Matches werden abgerufen, aber nie angezeigt.
2. **Job-Observability**: JobLog ist ein roher Textstream — keine Pipeline-Ansicht,
   kein Fortschrittsbalken, keine Job-Auswahl (mehrere Jobs pro Version, z.B. resume),
   keine Laufzeit.
3. **Viewer3D**: nur OrbitControls — keine Ansichtspresets, kein Wireframe/Autorotate,
   kein Screenshot, kein Fullscreen, kein Loading-/Error-State.
4. **Projektliste**: einfache Zeilen ohne Thumbnails, Zeitstempel oder letzten
   Jobstatus; Anlegen nur mit Name (kein Beschreibungsfeld); kein Umbenennen/Löschen
   aus der Liste.
5. **Assets**: kein Löschen, keine Caption-Anzeige/-Bearbeitung, keine Lightbox;
   Auswahlzustand nur per Rahmen sichtbar.
6. **Navigation**: keine Breadcrumbs, keine Kontextinfo (Projektname auf
   Versionsseite nur als „← project"-Link), keine Statusindikatoren in der Top-Bar.
7. **Design**: kein gemeinsames UI-Kit — uneinheitliche Abstände/Buttons, native
   `confirm()`-Dialoge, keine Empty-States/Skeletons, keine Toasts.
8. **Server-State**: Ad-hoc `useEffect`-Polling; AGENTS.md schreibt TanStack Query
   vor (bislang gar nicht installiert).
9. Kleine UX-Lücken: kein Ctrl+Enter-Submit, kein Copy-Button für Code/Prompt,
   Code-Ansicht ohne Download, Params ohne Reset, Versionsliste ohne Filter,
   Artifact-Tabelle (alle Dateien inkl. `mesh_raw`) nicht sichtbar.

## Plan (Meilensteine)

### M0 — Backend-Support (klein, additiv)
- `ProjectOut` um `latest_thumbnail_url` + `latest_version_status` erweitern
  (Thumbnail des Versionszweigs mit höchster Nummer → Projekt-Karten).
- `PATCH /assets/{id}` zum Editieren der Caption.
- Frontend-Typen: `Job.started_at/finished_at` ergänzen (API liefert sie bereits).

### M1 — Foundation
- Deps: `@tanstack/react-query` (Server-State, Polling via `refetchInterval`,
  Invalidierung nach Mutationen), `lucide-react` (Icons).
- `components/ui.tsx`: Card, SectionTitle, Chip, Btn (Varianten), IconBtn, Spinner,
  EmptyState, Modal, ConfirmButton (ersetzt `window.confirm`), Lightbox (geteilt),
  ProgressBar, Tabs, CopyButton.
- `index.css`: Scrollbars, Focus-Rings, Selection, Card-Flächen.
- AppShell: Top-Bar mit Logo, Breadcrumb, Health-Dots (API/LLM/Mesh, 30s-Polling),
  Settings-Link.

### M2 — ProjectsPage
- Karten-Grid: Thumbnail der letzten Version, Name, Beschreibung,
  Versionsanzahl, Status des letzten Laufs, „updated … ago".
- Create-Dialog (Name + Beschreibung), Inline-Rename, Delete mit Confirm-Modal,
  Client-Search-Filter.

### M3 — ProjectPage
- Header: Titel/Beschreibung inline editierbar, Meta-Zeile.
- Asset-Manager: Grid mit Lightbox, Delete, Caption-Edit, „n Bilder ausgewählt",
  Select-all/none, Drag&Drop-Highlight.
- Prompt-Composer: größere Textarea, Ctrl+Enter, Draft im localStorage,
  Asset-Counter, ModeSelector unverändert.
- Versionsliste: Filter-Input, relatives Datum, bestehender Tree.

### M4 — VersionPage
- Header: Breadcrumb `projekt / vN`, Status, Mode-Badges, Score, Iterationen.
- Metrics-Strip: Dauer, Tokens (usage), Iterationen, DfM-Warnungen-Zähler.
- Tabs statt Endlos-Scrollen:
  `Overview` (Viewer + Mesh-/Analyse-Chips + Renders + Exports),
  `Spec & Plan` (PartSpec-Panel + ModelingPlan-Panel),
  `Code` (Viewer mit Copy/Download),
  `Quality` (DfM + Critique inkl. Matches),
  `Files` (Tabelle aller Artifacts mit Größe/Download).
- Sidebar: Agent-Aktivität (JobLog), Edit-Card, Params, Versionsbaum.

### M5 — JobLog / Observability
- Pipeline-Chips: gruppierte Schritte (Intake, Plan, Mesh, Code, Build, Critique,
  Export) mit pending/active/done-Zustand aus dem Event-Stream.
- Progress-Bar (lokal gespiegelt: +0.05/Step, done=1), Elapsed-Timer
  (`started_at`→jetzt/`finished_at`), Job-Auswahl über `listJobs`, Cancel im Log-Header,
  farbcodierte Logzeilen, Auto-Scroll nur wenn am unteren Rand.

### M6 — Viewer3D-Toolbar
- Overlay-Toolbar: Fit/Reset, Ansichtspresets (Iso/Front/Top/Right), Autorotate,
  Wireframe, Grid-Toggle, Screenshot-PNG (`preserveDrawingBuffer`), Fullscreen,
  Loading-Overlay, Fehlerzustand.

### M7 — SettingsPage & Extras
- Health-Cards oben (API/LLM-Key/Mesh inkl. Device + Modelle).
- Katalog-Browser (`GET /catalog`, lazy, collapsible pro Kategorie).
- `agent_clarification_mode` anzeigen; Kleinigkeiten: Copy-Buttons,
  Esc/Click-outside für Modals, Tastaturkürzel Ctrl+Enter.

## Verifikation
- `cd frontend && npm run build` (tsc strict + vite).
- Backend nach Schema-Änderung: `pytest tests -x -q`, `ruff check`, `mypy cadagent`.
- Smoke: `make infra` + `make dev-api` + `npm run dev`, Browser-Preview der drei
  Hauptseiten.
