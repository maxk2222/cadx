# TopOpt-Audit 2026-10 — Defektklassen-Check (Vergleichsmaßstab: webcae-Audit)

**Scope:** `backend/cadx/cad/fem/topopt.py`, `solver.py`, `extract.py`, `analyze.py`,
`regions.py`, `loads.py`, `backend/cadx/cad/engine.py` (`run_topopt_pipeline` Z.981–1465,
`run_topopt_resample` Z.1468–1648), `backend/cadx/llm/schemas.py` (`OptimizeSpec` Z.274),
plus `agent/steps/optimize.py`, `agent/toolspecs.py`, `cad/fem/voxelize.py` als Kontext.
Branch `dev`, READ-ONLY — nur verifizierte Code-Befunde, keine Vermutungen.

> **Nachtrag (Fix-Stand):** Die Befunde wurden seitdem adressiert — der Audit
> unten dokumentiert den damaligen Stand:
>
> - **BUG-1/BUG-2 gefixt:** `_one_pass` übergibt `void_cells | ~domain.solid` an
>   `density_to_mesh` (engine.py:~1615) und `density.npz` persistiert `voids`
>   (engine.py:~1788); der Resample nutzt den Key bzw. re-resolved
>   `void_regions` (engine.py:~1958–1977).
> - **RISK-1 gefixt:** `solver=oc` warnt jetzt explizit bei gesetzten
>   MMA-Constraints (topopt.py:~896).
> - **RISK-2 gefixt:** `OptimizeSpec` trägt `Field(ge/le)`-Bounds
>   (schemas.py:~300–355).
> - **RISK-3 gefixt:** `spn` ist 1/N-normiert (`/ n_sol`, topopt.py:~1425).
> - **RISK-4 gefixt:** `change_box` teilt `last_change` mit `_run_mma` — CG-rtol
>   liest den Live-Wert (topopt.py:~787, ~1267–1295).
> - **Item-10-Hinweis gefixt:** Pass-2-Renders bekommen Suffix `_p2`
>   (engine.py:~1587), kein Überschreiben mehr.
> - **Item-11 Telemetrie umgesetzt:** `res.topopt`/`verification` enthält jetzt
>   `compliance_export_j`, `max_disp_export_mm`, `max_vm_export_mpa`,
>   `gray_fraction`, `volfrac_mask`, `solver_used`, `beta_final`, `sf_gap`,
>   `case_sfs_export` u. a. (engine.py:~1716–1755); `_run_summary` nutzt
>   `sf_export` (optimize.py:~61–81).
> - **Offen:** Item 4c (heuristischer Objektiv-Gradient `oc_filter=True` im
>   MMA-Weighted-Pfad, topopt.py:~1353) und die übrigen Hinweise.

## Ergebnis-Tabelle

| # | Item | Status | file:line | Befund |
|---|------|--------|-----------|--------|
| 1 | Re-Analyse auf Export-Geometrie | **OK** (Hinweise) | engine.py:1327–1338, 1611–1621 | `verify_exp` läuft auf `domain_exp = voxelize(opt_mesh)` — NICHT auf `final_mask`. Es existieren beide Verifikationen (`sf_mask` auf exakter Maske Z.1312–1321, `sf_export` auf re-voxelisiertem Export-Mesh). Retry-Trigger nutzt `sf_export`. Der webcae-Defekt ist vermieden. |
| 2 | Signierte Displacement-Projektion | **OK** | topopt.py:857–871 | Constraint nutzt `max(|u|.max())` über alle Cases — Betragsnorm, beide Richtungen begrenzt. Adjoint-RHS nutzt `np.sign(u[dof_star])` — korrekter Subgradient von \|u\|. |
| 3 | P-norm ohne Volumen-/N-Gewichtung | **BUG** | topopt.py:889 | `spn = (Σ srel^P)^(1/P)` ohne 1/N-Normierung → Constraint ist N^(1/P)-skaliert, also mesh-abhängig (feineres Grid ⇒ implizit strengeres Limit; bei P=8 und Verdopplung der Auflösung ≈ +30 %). Gradient ist zum nicht-normierten spn konsistent, nur die Normierung fehlt. |
| 4a | Constraint-Sensitivitäten `oc_filter=False` | **OK** | topopt.py:845, 869, 904, 819 | Vol/Disp/Stress-Constraints und die minmax-Objective nutzen `oc_filter=False` (echte Chain-Rule). |
| 4b | Volumen-Constraint auf physical density | **OK** | topopt.py:558–561, 847 | OC-Bisektion wertet `_physical(cand)` aus sobald heaviside/AM aktiv; MMA `_g_vol` nutzt `p[solid_idx]` (post-AM physical). Konsistent. |
| 4c | Heuristischer Objektiv-Gradient in MMA | **RISK** | topopt.py:828 | `solver=mma` + `case_objective=weighted` speist den klassischen OC-Sensitivitätsfilter (`oc_filter=True`) als "Gradienten" an nlopt-MMA — heuristisch, nicht die echte Ableitung; Constraints sind exakt. Gemischte Konsistenz (dokumentiert, aber für MMA fragwürdig). minmax-Pfad (Z.819) nutzt korrekt `False`. |
| 5 | Degeneriertes Min-Mass-Objektiv | **N/A** + RISK | topopt.py:309–315, 373–379; engine.py:1147–1149; schemas.py:278 | Kein Massen-Minimierungspfad existiert — Objective ist immer Compliance (weighted/minmax-P-norm), Volumen ist Upper-Bound. Guards vorhanden (ndesign==0, free==0, "no solvable load case", passive_frac>volfrac-Warnung). **RISK:** `target_volume_fraction`, `density_threshold`, `penal`, `max_iterations` etc. haben KEINE `Field(ge/le)`-Bounds — Kommentar behauptet 0.1–0.9, Schema validiert nichts (volfrac=0.02 → leere Extraktion; volfrac>1 → volles Teil). |
| 6 | Parameter-Durchreichung / Boundary-Freeze | **BUG** (2×) + RISK | siehe Feld-Tabelle unten | `void_cells` wird in `_one_pass` NICHT an `density_to_mesh` übergeben (engine.py:1287–1295); `density.npz` speichert kein `voids`-Key (engine.py:1397–1407) → Resample-Guard engine.py:1562 ist für topopt-npz tot. `solver=oc` ignoriert mma-Felder still. |
| 7 | Stress-Pfad-Abdeckung | **OK** (Caveat) | solver.py:271–284 (`_B0`); topopt.py:656–694 | Von-Mises am Element-Zentroid ausgewertet (B am Zentrum) — konsistent in Constraint UND Verifikation. Keine Corner-Recovery → Peaks an Materialgrenzen/Lasteinleitung unterbewertet; durch "10–20 % accuracy"-Disclaimer abgedeckt (analyze.py:49–52). Caveat: `solve()` skaliert D mit einem `E_ref` für alle Zellen — Insert-Materialzonen bekommen Basis-E-Spannung (solver.py:271). |
| 8 | AM-Filter Chain-Rule | **OK** | topopt.py:152–197 | Forward `p=min(x, mean(5 supports))`; Backward verteilt `g/5` an valide Support-Slots, `own=(x<=m)` konsistenter Subgradient am Knick; top-down-Akkumulation über Layer rekursiv korrekt. Deposits auf Air-Zellen sind tot (werden nie gelesen) — korrekt, da deren p nicht von x abhängt. |
| 9 | Heaviside beta-Doubling | **OK** (Hinweis) | topopt.py:596–604 (OC), 930–939 (MMA) | OC: Stall → beta×2 bis beta_max, Warnung + `history[].beta`. MMA: Restart-Loop ≤4 Restarts, cap beta_max. Objektiv-Sprung erscheint implizit als History-Diskontinuität — kein dediziertes Metrik-Feld. Bei nicht-default beta0/beta_max-Spanne >4 Verdopplungen stoppt MMA vor beta_max. |
| 10 | sf_export-Retry wählt besseres Ergebnis | **OK** (Hinweis) | engine.py:1350–1368 | `if sf_exp2 is not None and sf_exp2 > sf_exp: best = p2` — strikt besserer Pass gewinnt; `best`-Bundle (result/opt_mesh/final_mask/verify/verify_export) konsistent. `retry`-Block protokolliert beide SFs. Hinweis: FEM-Renders beider Pässe schreiben gleiche Dateinamen in `out_dir/fem` — Pass-2-Renders überschreiben Pass-1, auch wenn p1 gewinnt (Render/Report-Mismatch). |
| 11 | Telemetrie-Gaps | s.u. | engine.py:1376–1395 | Siehe Abschnitt "Telemetrie-Empfehlung". |
| 12 | multires-Verifikation auf feiner Domain | **OK** | engine.py:1208–1255, 1312–1342 | Coarse-Pass liefert nur `x0`-Warmstart; `verify`/`verify_exp` laufen immer auf `domain` (fein) / `final_mask` / `voxelize(opt_mesh)`. Hinweis: coarse `pc` (Z.1227–1236) lässt solver/constraints/penal_init/beta weg — by design ("cheap"), feiner Pass wendet sie an. |

## Detail-Befunde

### BUG-1: `void_regions` geht in der Extraktion verloren (`_one_pass`)

- `void_cells` wird berechnet (`engine.py:1161`) und nur als `passive_void` an `simp`
  übergeben (`engine.py:1276`) — die Dichte dort ist RHO_MIN, das stimmt.
- `density_to_mesh` in `_one_pass` (`engine.py:1287–1295`) bekommt **kein** `void_cells=`.
  `density_to_mask` (extract.py:217–287) dokumentiert explizit, dass `void_cells`
  gegen Re-Wachstum durch `_enforce_min_member` (Dilation, extract.py:152–177) und
  `_regularize_mask` (Diagonal-Fill, extract.py:40–77) gewinnen muss.
- Folgen: (a) deklarierte void_regions können durch min-member-Dickung/Diagonal-Fill
  wieder aufgefüllt werden; (b) `mask |= passive_solid` (extract.py:241) lässt
  keep/load-Overlap über der Void-Region gewinnen; (c) Dilation/Regularisierung
  kann Material **außerhalb** `domain.solid` wachsen lassen — die resample-Seite
  OR't deshalb `~domain.solid` in die void-Maske (engine.py:1562, Kommentar:
  "or dilation grows material past the part surface").
- **Fix:** in `_one_pass`:
  `void_cells=void_cells.reshape(domain.dims) | ~domain.solid` an `density_to_mesh` übergeben.

### BUG-2: `density.npz` speichert keine `voids` → Resample-Guard tot

- `np.savez_compressed` (engine.py:1397–1407) schreibt keys: density, mask, solid,
  passive, origin, pitch, dims, flat_bottom — **kein `voids`**.
- `run_topopt_resample` liest `void_cells` nur wenn `"voids" in npz` (engine.py:1561–1563)
  → für topopt-erzeugte npz ist `void_cells=None` → kein Void-Schutz UND kein
  `~domain.solid`-Guard im Resample. `opt.void_regions` wird im Resample nicht
  re-resolved (nur `keep_list`/`passive`, engine.py:1539–1544).
- **Fix:** `voids=void_cells.reshape(domain.dims)` in den npz-Dump aufnehmen
  (oder `void_regions` im Resample via `keep_element_mask` re-resolven und mit
  `~domain.solid` OR'en).

### RISK-1: `solver=oc` schluckt MMA-Constraints still

- `max_displacement_mm`, `max_stress_mpa`, `case_objective=minmax` werden zu
  `TopOptParams` durchgereicht (engine.py:1180–1183), aber der OC-Loop
  (topopt.py:527–633) referenziert sie nie. Kein Warning-Pfad wie bei
  `minmax`-mit-1-Case (topopt.py:744) oder MMA-Fallback (topopt.py:505).
- Agent kann per Tool `max_stress_mpa` setzen ohne `solver="mma"` → Ergebnis
  verletzt die gewünschte Constraint ohne jeden Hinweis.
- **Fix:** in `simp` (oder engine): Warning `"solver=oc ignoriert
  max_displacement_mm/max_stress_mpa/case_objective=minmax"` (oder Auto-Upgrade
  zu mma wenn Constraints gesetzt).

### RISK-2: Fehlende Schema-Bounds auf OptimizeSpec

- schemas.py:278–311: `target_volume_fraction` (Kommentar "0.1..0.9"),
  `density_threshold` ("0.2..0.9"), `penal`, `penal_init`, `max_iterations`,
  `min_member_mm`, `min_gap_mm`, `max_displacement_mm`, `max_stress_mpa` — keine
  `Field(ge=…, le=…)` Constraints. LLM-Output erreicht den Solver ungeprüft.
- **Fix:** Bounds ergänzen (z. B. `volfrac ∈ [0.05, 0.95]`, `threshold ∈ [0.2, 0.9]`).

### RISK-3: P-norm-Stress-Constraint mesh-abhängig (siehe Item 3)

- `spn = (Σ_{solid} srel^P)^(1/P)` (topopt.py:889) — bei uniformen Voxeln ist
  Volumen-Gewichtung ≡ Mittelwert; die fehlende 1/N-Normierung macht das Limit
  von der Zellzahl abhängig (P=8, Auflösung ×2 ⇒ spn ×~1.3).
- **Fix:** `spn = (np.mean(srel[solid_idx]**P))**(1.0/P)`; im Gradienten
  `coef` bzw. den direkten dgdp-Term entsprechend durch `solid_idx.size` teilen.

### RISK-4: CG-Genauigkeit im MMA-Pfad fix bei 1e-3

- `_eval` wählt `cg_rtol` nach `change` (topopt.py:424). `change` ist eine
  `simp`-Scope-Variable, die im MMA-Pfad nie aktualisiert wird — `_run_mma`
  trackt `last_change` separat (topopt.py:770, 786–788) und schreibt erst beim
  Return zurück. Alle MMA-FEM-Solves (Primär + Constraints auf CG-Domänen
  > `spsolve_dofs`) laufen mit `cg_rtol=1e-3` → rauschbehaftete
  Objektiv-/Constraint-Werte und Gradienten für MMA.
- **Fix:** `change` für MMA-Evals aus `last_change` speisen, oder für MMA
  fest `1e-5` verwenden.

### Hinweise (kein Bug)

- **Item 1:** Agent-Run-Tabelle (`_run_summary`, optimize.py:60–62) nutzt
  `res.fem` (mask!) als `worst_sf` — Runs werden nach der optimistischeren
  Masken-Verifikation verglichen, nicht `sf_export`. Einzel-Cases können in
  `verify_exp` still ohne SF enden, wenn Fix-/Load-Regionen auf der
  Export-Domain nicht mehr matchen (analyze.py:262–264 — Case wird mit
  `safety_factor=None` appended, `worst_safety_factor` ignoriert None).
  Export-Voxelize kann den Pitch adaptiv vergröbern (voxelize.py:155–168).
- **Item 8 (Beobachtung):** AM-Filter kann passive Zellen unter 1 drücken;
  `density_to_mask` re-addiert `passive_solid` (extract.py:241) — Dichte und
  exportierte Maske divergieren auf gepinnten Zellen (Verifikation nutzt die
  Maske — konsistent zum gedruckten Teil, nur zur Kenntnis).
- **Item 10:** `volfrac_target` in `res.topopt` zeigt bei Retry-Gewinn das
  erhöhte volfrac (Original nur in `retry.volfrac_first`); Pass-2-Renders
  überschreiben Pass-1-Renders gleichen Namens.
- **minmax ignoriert `case.weight`** (smooth P-norm über Cases ungewichtet,
  topopt.py:817) — semantisch vertretbar, erwähnenswert.

## OptimizeSpec → Konsument (Item 6 Tabelle)

| Feld (schemas.py:274–323) | Konsumiert wo | Status |
|---|---|---|
| `enabled` | orchestrator.py:4794+ (Pipeline-Dispatch) | OK |
| `target_volume_fraction` | engine.py:1170 → `params.volfrac` | OK (kein Bound) |
| `max_iterations` | engine.py:1173 → `params.max_iter` | OK |
| `safety_factor_min` | engine.py:1353 (Retry-Gate) | OK |
| `keep_regions` | engine.py:1152 → keep_list → passive | OK |
| `void_regions` | engine.py:1161 → `passive_void` | **BUG:** Extraktion (BUG-1) + Resample (BUG-2) |
| `keep_mode` / `auto_keep_margin_mm` | engine.py:1153–1155 | OK |
| `min_member_mm` | engine.py:1167 → Extraktion + rmin | OK |
| `min_gap_mm` | engine.py:1168 → `_min_gap_warning` | OK (nur Warnung, by design) |
| `density_threshold` | engine.py:1174 → `params.threshold` | OK (kein Bound) |
| `warm_density_relpath` | engine.py:1193–1203 | OK |
| `multires` | engine.py:1208 | OK |
| `self_supporting` / `print_direction` | engine.py:1176–1177 | OK |
| `filter` | engine.py:1175 | OK |
| `penal` / `penal_init` | engine.py:1178–1179 | OK |
| `solver` | engine.py:1180 | OK (RISK-1: oc schluckt Constraints still) |
| `case_objective` | engine.py:1181 | OK (oc/minmax-Warnung nur in mma) |
| `max_displacement_mm` / `max_stress_mpa` | engine.py:1182–1183 | OK-Wert, RISK-1 |
| `auto_refine_features` | worker_jobs.py:939–940 → refine-Enqueue | OK |
| `restore_planes` / `auto_plane_restore` / `helical_threads` | worker_jobs.py:1458–1469 → engine.py:1786–1848 (refine-Pfad) | OK |
| `note` | kein Code-Konsument (LLM-Kontext) | by design (semi-tot) |

Nicht über `OptimizeSpec` exponierte `TopOptParams` (Defaults only):
`tol`, `converge_patience`, `move`, `beta0`, `beta_max`, `stress_pnorm`,
`stress_relax_q`, `penal_ramp_iters`, `rmin_mm` (aus runtime+min_member abgeleitet).

**"Boundary-Layer-Freeze":** fix/load-Zellen werden über `passive_solid_mask`
(topopt.py:953–971) dauerhaft auf ρ=1 gepinnt — bewusstes Design ("functional
features preserved by construction"), quantifiziert via `passive_frac` +
Warning bei >volfrac (topopt.py:509–517, 617–625). OK.

## Telemetrie-Empfehlung (Item 11)

Bereits berechnet, aber NICHT in `res.topopt` (nur in `res.fem`/`res.fem_export`
model_dumps oder gar nicht):

| Metrik | Quelle | Status |
|---|---|---|
| `compliance_export_j` (pro Case) | `verify_export.cases[].compliance_j` | berechnet, nicht propagiert |
| `displacement_export_mm` | `verify_export.cases[].max_disp_mm` | berechnet, nicht propagiert |
| `max_vm_export_mpa` | `verify_export.cases[].max_vm_mpa` | berechnet, nicht propagiert |
| `compliance_pred_mj` | `history[-1].compliance_mj` (SIMP-Objektiv; Achtung Einheit mJ vs. J) | in history vergraben |
| `volfrac_mask` | `final_mask.sum()/nsolid` — Extraktions-Volfrac vs. `volfrac_achieved` (Dichte-Integral) zeigt Thickening-/Erosions-Effekte | nicht berechnet |
| `gray_fraction` | Anteil Design-Zellen mit 0.05 < ρ < 0.95 (Filter-/Extraktions-Qualität) | **nirgends berechnet** |
| `beta_final`, `penal_final` | history (MMA-Recs Z.790–797 tragen `beta` nicht einmal) | partiell |
| `solver_used` / `mma_fallback` | nur in warnings erkennbar | nicht propagiert |
| `sf_gap` | `sf_mask − sf_export` (Erosions-Indikator der Export-Kette) | beide Werte vorhanden, Differenz nicht |
| `dofs` | `report.grid["dofs"]` | nur in res.fem |
| `n_cases_effective` | Anzahl resolved Cases inkl. Cone-Expansion | nicht propagiert |
| `simp_s` / `verify_s` | `res.timings` | nicht in topopt |
| `case_sfs_mask` / `case_sfs_export` | pro-Case SF-Arrays beider Verifikationen | nur implizit in res.fem/fem_export |

Empfehlung: Flat-Summary-Felder in `res.topopt.update` (engine.py:1376)
ergänzen — der Agent vergleicht Runs sonst anhand von `res.fem` (Masken-SF)
statt `sf_export`.

## Item 12 — multires

`verify` (Z.1312) und `verify_exp` (Z.1327) laufen beide auf der feinen Domain;
der Coarse-Pass erzeugt nur `x0` (`_upsample_density`, engine.py:1917–1935).
Hinweis: `pc` (engine.py:1227–1236) setzt `solver`/`case_objective`/
`max_displacement_mm`/`max_stress_mpa`/`penal_init`/`beta`-Kontrolle nicht —
OC+cone im Coarse-Lauf ist die dokumentierte "cheap"-Heuristik; Nutzer-Constraints
greifen erst im feinen Pass (warm-gestartet).
