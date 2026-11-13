# Ground truth — kiara_heatpump (vom Auftraggeber bestätigt)

## Teile
Gehäuse (Anker, auf Ebene), Adapter DN22 IG/Muffe, Rohr DN22 l=70, Rohr DN22 l=500, Bogen Muffe→Spitzende, Bogen Muffe→Muffe. Wärmepumpe selbst ist KEIN CAD-Teil — Gehäuse allein plausibel/stabil auf Ebene positionieren.

## Ausgangszustand
Zwei vormontierte Adapter UNTERHALB des Gehäuses (links+rechts) gehören zum Ausgangszustand — nicht mit oberen Adaptern (Schritt 1/2) verwechseln. Falls ohne Wärmepumpen-CAD nicht real platzierbar: virtuelle Anschlussreferenzen/Ankerpunkte erlaubt — Entscheidung dokumentieren, Anschlusspunkte konsistent abbilden. Keine fehlende Geometrie erfinden.

## Verbindliche Schritte 1–10 (pro Seite L/R gespiegelt)
1/2. Adapter auf oberen linken/rechten Gehäuse-Anschluss schrauben.
3/4. Rohr l=70 in oberen linken/rechten Adapter einsetzen.
5/6. Bogen Muffe→Spitzende auf vormontierten UNTEREN linken/rechten Adapter — freie Öffnung zeigt NACH OBEN.
7/9. Rohr l=500 in unteren Bogen einsetzen (vertikales Rohr nach oben).
8/10. Bogen Muffe→Muffe auf langes Rohr UND freie Seite auf oberes kurzes Rohr — BRÜCKE untere↔obere Baugruppe, beide Anschlüsse geometrisch konsistent (Multi-Mate).

## Erwartete Joint-Struktur
- adapter#lower_L/R → casing untere Stutzen (vormontiert), adapter#upper_L/R → casing obere Stutzen
- pipe_l70_L/R → adapter_upper_L/R
- elbow_spigot_L/R → adapter_lower_L/R, Öffnung +Z
- pipe_l500_L/R → elbow_spigot_L/R (vertikal)
- elbow_socket_L/R → pipe_l500_L/R:top UND pipe_l70_L/R:top (2 Joints je Bogen)

## Klärungen vom Auftraggeber (2026-11-13)
- Das Gehäuse sitzt AUF der Wärmepumpe (Pumpen-CAD existiert nicht, wird NICHT gebaut) — das Gehäuse steht aufrecht (Render-Perspektive täuscht, es ist NICHT gekippt).
- Die vormontierten Adapter sitzen an der HINTEREN Seite der Wärmepumpe, ca. 600 mm UNTERHALB des Gehäuses — praktisch auf Boden-/Pumpenniveau. Darum: die 500-mm-Rohre zeigen VOM GEWICHTPUNKT DES GEWICHTPUNKT GEHÄUSES NACH UNTEN (Kette: unterer Adapter → Bogen Muffe→Spitzende, freie Öffnung oben → L500 vertikal hoch → Muffe-Muffe-Bogen als Brücke zum L70 oben).
- Aus den PDF-Figuren: die „oberen" Gehäuse-Anschlüsse sind die waagerechten Stutzen an der FRONT-Seite (Adapters + L70 ragen horizontal heraus); die unteren Gehäuse-Stutzen (rim_19/20, hinten unten) sind Referenzpunkte für die vormontierten Adapter — die Adapter selbst gehören aber zur Pumpe: Referenz-Platzierung, keine erfundene Pumpen-Geometrie.
- Höhenlogik: Adapter unten (~Bodenniveau) + Bogen + L500 → L500-Top landet in der Höhe der Gehäuse-Front-Anschlüsse; dort verbindet der Muffe-Muffe-Bogen vertikal↔horizontal.
- Adapter-Seiten (Auftraggeber): eine Seite hat ein INNENGEWINDE → schraubt auf die Gewinde-Stutzen am Gehäuse; die andere Seite ist die FEMALE-AUFNAHME für das L70-Rohr. Der Adapter ist also richtungsgebunden: Gewinde→Stutzen, Aufnahme→Rohr.

## Port-Modell (solver/cadmeta, 2026-11-13)
- Jeder Rim-Port misst: `radius` (Lippe), `inner_radius` (lichter Radius bis zum Seat), `outer_radius` (Hüllkurve), `depth` (Seat-Tiefe; 500=Durchgangsbohrung/offenes Rohrende; <4=feste Fläche).
- Verbindungsregel: ein Joint braucht einen SOCKET (Seat ≥4mm ODER durchgehende klare Bohrung für Push-Fit ODER eine female Port mit gemessener Bohrung — Überwurfmutter-Lippe misst ~0.3mm und bleibt trotzdem Receiver) + einen PLUG dessen `outer_radius ≤ socket.inner_radius·1.15+0.5`. Die Seite OHNE Messdaten behält ihren gezeichneten Radius als Hüllkurve — ein unvermessener Plug, der nicht in die gemessene Bohrung passt, wird hart abgelehnt.
- Gehäuse-Geometrie (vermessen): die echten Anschluss-Stutzen sind die zwei hohlen **Ø27-Rohrenden** (rim_0/37: bore r10.7, Wand r13.1, ~17mm über die Fläche ragend) — im GLB zeigen sie lokal +Z; die Trichteröffnungen (rim_33/35, r30→r27) sind die ovalen Löcher der gegenüberliegenden Fläche. `_anchor_orientation` dreht den Anker über alle 24 Würfelrotationen so, dass die fittbaren Interfaces lateral erreichbar zur Arbeitsseite zeigen — hier R1: Stutzen→+Y-Front, Trichter→oben.
- Adapter-Montage (verifiziert): die Adapter-Mutter (rim_2, bore 11.06) umschließt das Rohrende — **socket-over-tube** (bore ≥ OD via 15%-Mess-Slack), Mutter-Mündung liegt ~19mm über dem Stub-Ende, Körper ragt ~42mm heraus. Gewindeflächen verschachteln sich nominal — kein Clearance-Fit nötig.
- Thread-Seat (solver): eine FEMALE-Mündung sitzt auf einem boss-nested MALE-Stub auch wenn `plug > bore` (grobe Assets: das Rohr steht für den Gewinde-Anschluss) — Score ~0.5 schlägt die übergroße Socket-Tasche (~0.16), verliert gegen echte snug-Fits (~0.9); ~3mm Tuck an der Mündung. Tiefe Mover-Sockets seaten schwächer (gemessene Seat-Tiefe <4mm = Gewindemund → 0.5).
- Engagement: nur die EMPFANGENDE Socket-Tiefe zählt — lose Kragen-Mates (Bore ≫ Plug) sitzen am Mund (~≤6mm), enge Fits fahren bis zum Seat (~85%); Thread-Seats = ~3mm.
- Adapter ist richtungsgebunden: Innengewinde-Ende (rim_2, flache Mündung) → Gehäuse-Rohrstutzen, Muffen-Ende (rim_5, Tiefe 500) ← L70.
- Kandidaten-Auswahl (preassembly): Lagewörter ("oberer/linker Anschluss") normalisieren die Port-Weltposition gegen die Host-Spanne (~±0.5/Achse) — Roh-mm addierten früher +376mm für einen höheren Port und ließen einen Trichter-Stub die echte Verbindung schlagen. Solver-Fit zählt mit ×4 in denselben Hint-Term: snug schlägt "nur höher". Ports >60mm unter der Host-Silhouette längs ihrer Achse sind Innen-Junctions — in Orientierung UND Attach-Pfad geskippt.

## Physikalisches Audit (deterministisch, 2026-11-13)
- `joint_geometry_audit`: pro Joint `gap_mm`/`normal_dot` mit Socket-Tiefe als Eingriffs-Toleranz (Überwurfmutter sitzt nie plan — Rest ≤ Tiefe = noch im Eingriff); `buried_frac` = Union-Containment der Welt-AABB gegen Nicht-Partner-Hosts (Boolean-Kollision ist blind für Teile in Hohlräumen — Rohr in der Mutterbohrung schneidet nichts).
- Approximierte Interfaces: nur wenn der Seat-Tier tatsächlich griff — female-Mündung auf boss-nested MALE-Stub dessen OD NICHT in die Bohrung passt (Stub steht für unmodelliertes Gewinde). Schlanker Nested-Stub, der physisch passt (Rohrende in Muffe), ist ein normaler Spigot-Joint — kein Finding.
- Derselbe Audit speist: `validate_assembly_geometry` (Tool), `write_review_report` (verweigert score ≥8 bei offenen Findings), `unity_guide.json` Joint-Metriken, Evalkit `guide_mate_quality`-Gate.
