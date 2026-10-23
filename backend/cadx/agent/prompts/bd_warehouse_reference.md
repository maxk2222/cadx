## bd_warehouse 0.4 reference — standard parts (verified against installed version)

Imports:
```python
from bd_warehouse.fastener import (
    SocketHeadCapScrew,
    CounterSunkScrew,
    HexHeadScrew,
    ButtonHeadScrew,
    SetScrew,
    HexNut,
    HexNutWithFlange,
    SquareNut,
    DomedCapNut,
    HeatSetNut,
    PlainWasher,
    ChamferedWasher,
    ClearanceHole,
    TapHole,
    ThreadedHole,
    InsertHole,
)
from bd_warehouse.thread import IsoThread, MetricTrapezoidalThread, PlasticBottleThread
from bd_warehouse.bearing import SingleRowDeepGrooveBallBearing, PressFitHole
from bd_warehouse.gear import SpurGear, HelicalGear, RackGear
```

### Fasteners (norm tables, metric + imperial)
```python
screw = CounterSunkScrew(
    size="M4-0.7", fastener_type="iso10642", length=12
)  # DIN 965 ~ ISO countersunk
cap = SocketHeadCapScrew(size="M3-0.5", fastener_type="iso4762", length=10)  # DIN 912
hexs = HexHeadScrew(size="M6-1", fastener_type="iso4017", length=20)
nut = HexNut(size="M6-1", fastener_type="iso4032")
insert = HeatSetNut(size="M3-0.5-6", fastener_type="Hilitchi")  # sizes like 'M3-0.5-6'/'M3-0.5-8'
bearing = SingleRowDeepGrooveBallBearing(size="M8-22-7")  # 608
# Query data:  screw.clearance_hole_diameters -> {'Close':4.3,'Normal':4.5,'Loose':4.8}
#              screw.tap_hole_diameters -> {'Soft':..., 'Hard':...}
#              <Cls>.types(), <Cls>.sizes(type), <Cls>.select_by_size("M6-1")
# simple=True (default) = no real thread geometry; hand="right"|"left".
```
Common fastener_type values: screws iso10642/iso7046/iso14581 (countersunk), iso4762 (socket head),
iso4017/iso4014/din931 (hex head), iso7380_1/iso7380_2 (button), iso1207/iso7048/iso14580 (cheese),
asme_b18.3 (socket, imperial), iso4026 (set screw). Nuts: iso4032/iso4033/iso4035, din557, din1587,
din1665, iso4036. Washers: iso7093/iso7094/iso7089/iso7091, iso7090, iso7092.
Metric size strings: M2-0.4 M2.5-0.45 M3-0.5 M4-0.7 M5-0.8 M6-1 M8-1.25 M10-1.5 M12-1.75.

### Norm holes — ALWAYS use these for named fasteners
```python
with BuildPart() as p:
    Box(100, 60, 8, align=(Align.CENTER, Align.CENTER, Align.MIN))
    top = p.faces().sort_by(Axis.Z)[-1]
    with Locations(top):  # workplane on the face -> holes cut into the part
        with GridLocations(80, 40, 2, 2):
            ClearanceHole(fastener=screw, fit="Normal", counter_sunk=True)
            # counter_sunk=True + countersunk screw -> conical countersink; with a cap screw -> counterbore
        with Locations((0, 0)):
            TapHole(
                fastener=cap, material="Soft"
            )  # tap drill dia; 'Soft' plastics/alu, 'Hard' steel
        # ThreadedHole(fastener=cap, depth=8, simple=True)     # cosmetic thread, depth REQUIRED
        # InsertHole(fastener=insert, depth=6, manufacturing_compensation=0.1)  # heat-set pocket
        # ClearanceHole(fastener=nut, captive_nut=True, depth=5)               # hex nut pocket
        # PressFitHole(bearing=bearing, interference=0.025)                    # bearing seat
result = p.part
# hole positions are recorded on fastener.hole_locations -> build assemblies if include_fasteners
```

### Thread detail contract
```python
core = Cylinder(d / 2 - 0.3, L, align=(Align.CENTER, Align.CENTER, Align.MIN))
thread = IsoThread(
    major_diameter=d,
    pitch=1.0,
    length=L,
    external=True,
    end_finishes=("fade", "square"),
    manufacturing_compensation=PARAMS.get("thread_comp", 0.0),
    simple=(CADX_DETAIL_MODE == "preview"),
)
result = core + thread  # fuse -> one solid
# internal: IsoThread(..., external=False) inside a bore of major_diameter,
#           or ThreadedHole(fastener=..., depth=L, simple=False)
# bottle cap: PlasticBottleThread(size="M38SP444", external=False)
# metric trapezoidal (lead screw): MetricTrapezoidalThread(size="Tr16x4", length=L)
# Include CADX_FEATURES metadata for the thread so preview renders can draw a
# lightweight visible helix when simple=True. The executor rebuilds final exports
# with CADX_DETAIL_MODE="export"; there the real helix must be present.
# end_finishes: ("fade","square") is a good fast default; "chamfer" is ~20x slower.
```

### Gears / other
```python
gear = SpurGear(module=1.0, tooth_count=20, pressure_angle=20, thickness=6)  # module system, mm
# bore/keyway: model the gear, then cut a Hole / keyway slot on top of it.
# also available: HelicalGear, RackGear, WormGear, WormWheel; ORing (iso3601 sizes);
# ExternalSnapRing/InternalSnapRing (+ RetainingRingGroove); ShaftKey (+ Keyway);
# HexFlangedEccentricBushing; Sprocket; Pipe (nps/material/identifier);
# Flange family (BlindFlange, LappedFlange, SlipOnFlange, SocketWeldFlange,
# WeldNeckFlange — nps + flange_class); open_builds (V-Slot/C-Beam profiles,
# spacers, gantry plates — see catalog for exact class names).
```

### Catalog excerpt
When the request names standard parts, the user message contains a
"STANDARD PARTS — valid catalog values" section listing the exact `type` and
`size` strings each class accepts. ALWAYS take type/size strings from that list —
anything else fails the static pre-check. If the size you need is not listed,
pick the nearest valid one and note it in `reasoning`.

### Rules
- size string must exist in the norm table — when in doubt use the common pitch table above;
  `Class.sizes(fastener_type)` lists valid ones.
- screw `length` matters only for `include_fasteners` assemblies and blind-hole depth checks.
- `depth=None` on a hole = through the whole part; give `depth` for blind holes.
- `HeatSetNut` pairs ONLY with `InsertHole` (other holes raise ValueError).
- Agent preview uses simple fasteners plus a lightweight visible thread cue. If a
  thread is part of the requested standard fastener/feature, the final export is
  rebuilt with real helix geometry; never ship the preview core as the final thread.
- after fusing thread+core or gear+bore verify the part is still ONE solid.
