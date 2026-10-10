from build123d import *

PARAMS = {
    "length": 40.0,
    "width": 20.0,
    "height": 10.0,
    "fillet_r": 1.5,
}

with BuildPart() as p:
    Box(
        PARAMS["length"],
        PARAMS["width"],
        PARAMS["height"],
        align=(Align.CENTER, Align.CENTER, Align.MIN),
    )
    fillet(p.edges().filter_by(Axis.Z), radius=PARAMS["fillet_r"])
result = p.part
