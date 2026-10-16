"""Subprocess entry for process_mesh payloads.

Usage: python -m cadx.cad.mesh_payload_entry <workdir>
Reads:  <workdir>/input.pkl  (the arq payload dict)
Writes: <workdir>/result.pkl (the run_mesh_payload result dict)

Isolation rationale: mesh post-processing allocates heavily in trimesh/OCC/VTK
and a long-lived arq worker never returns that RSS to the OS — peaks stacked
across jobs used to trip the container cgroup (rc=137), killing the worker and
orphaning the in-flight job. A child process hands all of it back on exit and
turns an OOM into a clean missing-result error instead of a dead worker.

Always exits 0 — the parent detects failures via result.pkl.
"""

from __future__ import annotations

import pickle
import sys
import traceback
from pathlib import Path


def main() -> None:
    workdir = Path(sys.argv[1])
    try:
        payload = pickle.loads((workdir / "input.pkl").read_bytes())
        from cadx.cad.tasks import run_mesh_payload

        d = run_mesh_payload(payload)
    except Exception:
        d = {
            "ok": False,
            "error": "mesh payload crashed",
            "traceback": traceback.format_exc()[-4000:],
        }
    try:
        (workdir / "result.pkl").write_bytes(pickle.dumps(d))
    except Exception:
        (workdir / "result.pkl").write_bytes(
            pickle.dumps({"ok": False, "error": "mesh result not serializable"})
        )


if __name__ == "__main__":
    main()
