"""Physical tolerance constants for the deterministic guide solver.

Every value here is a DOMAIN-LEVEL property of mechanical assembly —
socket geometry, connector reach, burial fractions — never a number tuned
to a particular manual or fixture. Millimeters and unitless fractions only.
"""

# A mated port pair lands its sealing faces within this gap. Sub-print-
# accuracy tight on purpose: a joint that still shows 0.05 mm of air is
# not bearing on its seat.
GAP_GATE_MM = 0.05

# Port normals must be antiparallel to within ~2.6° — looser gates admit
# edge-on "mates" that a manual never calls a connection.
NORMAL_DOT_GATE = -0.999

# Interface score weights for cylindrical mating. A male↔female pair is
# the only unambiguous mechanical joint (+0.5); socket-over-socket stays
# physically plausible but ambiguous at the rim (0.2 malus keeps the
# ordering stable without banning it).
IFACE_SCORE_BASE = 0.1
IFACE_SCORE_MF = 0.5
IFACE_SCORE_FF = 0.2

# Radial margin bands that distinguish a true socket fit from a loose
# collar/nut mate: ≤2 mm clearance grips the plug, ≥10 mm is a pass-
# through collar. Between them the fit depth blends linearly.
SOCKET_FIT_MARGIN_MM = 2.0
COLLAR_MARGIN_SPAN_MM = 8.0

# A bore whose seat probe ran out at this depth never grips — the probe
# range (~500 mm) exceeds any real fitting's travel, so the value marks
# "through hole", not a measured dimension.
THROUGH_BORE_DEPTH_MM = 400.0

# A HOST-side bore this much wider than the plug is a pass-through
# opening (wall aperture, housing cutout), not a connection — nothing
# grips the plug. Factor covers loose clearances, the flat term covers
# thin-walled cutouts near thin plugs.
OPENING_RADIUS_FACTOR = 1.85
OPENING_RADIUS_MM = 12.0

# Ray-measured clear radius reads ~10 % low near rims/chamfers — accept a
# plug within that slack plus a half-millimeter floor, still rejecting
# anything that physically cannot enter the bore.
CLEAR_RADIUS_SLACK = 1.15
CLEAR_RADIUS_FLOOR_MM = 0.5

# Fraction of pose candidates landing far from every host port before a
# part pair is declared unrelated. Empirically a real interface keeps
# most candidates within reach; 0.45 sits well between "sparse but real"
# and "coincidental geometry".
UNRELATED_FAR_FRAC = 0.45

# Alignment gates for axis features: |dot| > 0.9 means within ~26° of
# parallel (line/mouth axes), > 0.998 effectively collinear (≈3.6°).
LINE_AXIS_DOT = 0.9
COLLINEAR_DOT = 0.998

# A bridge mate whose two port axes land further than ~37° from (anti-)
# parallel is no mate in either sign convention.
BRIDGE_AXIS_DOT = 0.8

# Recessed ports deeper than any fitting's reach are interior junctions,
# not mount interfaces. 60 mm exceeds the insertion depth of typical
# consumer hardware while still admitting counter-sunk connectors.
PORT_EXPOSURE_MAX_MM = 60.0

# Lift-correction sanity bounds: a residual larger than this is a
# pairing error, not a pose error; per-axis corrections disagreeing by
# more than the spread are contradictory evidence and get dropped.
MISS_RESIDUAL_MAX_MM = 80.0
SHIFT_SPREAD_MM = 10.0
SHIFT_MEAN_MIN_MM = 0.5

# Single lift shifts are clamped to the same scale as the exposure
# bound — a 60 mm vertical correction is already at the edge of what a
# pose error should ever request.
LIFT_SHIFT_CLAMP_MM = 60.0

# Predicted-chain-closure slack for premount pose ranking: the estimate
# (chain height along the leg axis) is coarse, so the flag only marks
# predictions beyond the same scale as a real pairing error — the
# candidate ordering itself is driven by the raw residual, not this gate.
CLOSURE_SLACK_MM = 80.0

# Bridge-retry rejection gates: a candidate buried >45 % inside another
# part — or >60 % swallowed by the host itself — is intersecting, not
# mated.
BURIAL_FRAC_GATE = 0.45
SWALLOW_FRAC_GATE = 0.6

# A "premounted" instance only counts as attached to the host tree when
# it sits within this bbox gap of a host instance — close enough that a
# human would call it mounted, far enough to catch a satellite subtree
# spawned a meter away.
PREMOUNT_HOST_GAP_MM = 100.0

# Post-hoc joint verification tolerances are deliberately wider than the
# solve-time gate: GAP_GATE * this factor (or the socket depth, capped)
# still counts as seated after the fact.
POSTHOC_GAP_FACTOR = 20.0
POSTHOC_GAP_CAP_MM = 30.0
POSTHOC_NDOT_GATE = -0.95
