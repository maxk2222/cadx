"""Voxel-FEM + SIMP topology optimization.

Pure engine code — no DB, no queue, no LLM. Runs inside the cad-executor.
Units: mm, N, MPa. Linear isotropic elasticity on a uniform Hex8 grid
(8-node brick elements, 2x2x2 Gauss integration).
"""
