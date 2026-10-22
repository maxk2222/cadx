"""evalkit — standardized evaluation for the CADX pipeline.

Two drivers over one shared core (suite definitions, gates, metrics):

- ``http``  — drives the public REST API against the running stack; every
  stage produces real Version/Job rows under fixed ``eval:*`` projects.
  Results land in ``evals/results/`` and are baseline-capable.
- ``local`` — in-process driver for quick dev iteration: a scratch sqlite
  DB + scratch data dir, no API server, no Redis, no Docker build. Results
  land in ``evals/scratch/`` and are never baselined.
"""
