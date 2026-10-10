from bd_warehouse.fastener import ClearanceHole, CounterSunkScrew
from build123d import *

PARAMS = {
    "length": 100.0,
    "width": 60.0,
    "thickness": 8.0,
    "screw": "M4-0.7",
}

screw = CounterSunkScrew(size=PARAMS["screw"], fastener_type="iso10642", length=12)

with BuildPart() as p:
    Box(
        PARAMS["length"],
        PARAMS["width"],
        PARAMS["thickness"],
        align=(Align.CENTER, Align.CENTER, Align.MIN),
    )
    with Locations(p.faces().sort_by(Axis.Z)[-1]):
        with GridLocations(PARAMS["length"] - 20, PARAMS["width"] - 20, 2, 2):
            ClearanceHole(fastener=screw, fit="Normal", counter_sunk=True)
result = p.part
