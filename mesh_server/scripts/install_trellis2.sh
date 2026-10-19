#!/usr/bin/env bash
# Install Microsoft TRELLIS.2-4B on a CUDA box (>=24 GB VRAM recommended).
# Usage: mesh_server/scripts/install_trellis2.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source scripts/lib/pkg.sh

TRELLIS_REPO="https://github.com/microsoft/TRELLIS.2.git"
VENV="${VENV:-mesh_server/.venv}"
SRC="${TRELLIS_SRC:-mesh_server/.src/TRELLIS.2}"
need_venv
pkg_detect

say "cloning $TRELLIS_REPO (recursive submodules!)"
clone_or_update "$TRELLIS_REPO" "$SRC" main 1

say "TRELLIS.2 setup (basic + flash-attn + cumesh + o-voxel + flexgemm)"
# sourced in the current shell (as upstream intends) so exported build env
# survives for the `INSTALL -e` build below
cd "$SRC"
. ./setup.sh --new-env --basic --flash-attn --cumesh --o-voxel --flexgemm
cd "$ROOT"
INSTALL -e "$SRC"

say "downloading weights"
hf_snapshot "microsoft/TRELLIS.2-4B"

ok "done — 'meshserver models' should list trellis2 on a CUDA device"
