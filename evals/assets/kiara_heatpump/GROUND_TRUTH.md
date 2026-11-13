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
