"""`cadx eval` — Typer sub-app wiring the evalkit commands.

Imports stay function-local (cli.py convention): the ``--local`` bootstrap
must be able to set DATABASE_URL/DATA_DIR before cadx.db.session binds its
engine.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Any

import typer

eval_app = typer.Typer(
    add_completion=False,
    help="Standardized eval: suite cases x stages, http (DB-backed) or --local scratch runs.",
    no_args_is_help=True,
)


def _parse_variants(specs: list[str] | None) -> list[tuple[str | None, dict[str, Any] | None]]:
    from cadx.evalkit.stats import parse_variant

    out: list[tuple[str | None, dict[str, Any] | None]] = []
    try:
        for spec in specs or []:
            out.append(parse_variant(spec))
    except ValueError as e:
        print(f"error: {e}")
        raise typer.Exit(2) from None
    return out or [(None, None)]


def _load_cases(names: list[str] | None) -> list:
    from cadx.evalkit.suite import load_suite

    cases = load_suite()
    if not names:
        return cases
    wanted = set(names)
    out = [c for c in cases if c.name in wanted]
    missing = wanted - {c.name for c in out}
    if missing:
        print(f"error: unknown case(s): {', '.join(sorted(missing))}")
        raise typer.Exit(1)
    return out


@eval_app.command(name="run")
def run(
    local: bool = typer.Option(
        False,
        "--local",
        help="scratch run: no API/Redis/real-DB writes, results under evals/scratch",
    ),
    case: Annotated[
        list[str] | None, typer.Option("--case", "-c", help="suite case name(s)")
    ] = None,
    stages: str | None = typer.Option(
        None, "--stages", "-s", help="comma-separated stage ids/types to run"
    ),
    from_version: str | None = typer.Option(
        None, "--from-version", help="http: parent version id for the first stage"
    ),
    from_dir: str | None = typer.Option(
        None, "--from", help="local: directory with an STL/BREP as parent for the first stage"
    ),
    repeat: int = typer.Option(1, "--repeat", "-n", min=1),
    variant: Annotated[
        list[str] | None, typer.Option("--variant", help="NAME=<json> RuntimeSettings")
    ] = None,
    settings: str | None = typer.Option(
        None, "--settings", help="extra RuntimeSettings JSON applied to all variants"
    ),
    label: str = typer.Option("", "--label", "-l"),
    timeout: float | None = typer.Option(None, "--timeout", help="default stage timeout override"),
    api_url: str = typer.Option("http://localhost:8080", "--api-url"),
    no_artifacts: bool = typer.Option(False, "--no-artifacts"),
    no_fail_fast: bool = typer.Option(False, "--no-fail-fast"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    json_out: str | None = typer.Option(None, "--json", help="write combined results JSON"),
) -> None:
    """Run suite cases — http driver by default, `--local` for scratch dev runs."""

    async def _run() -> int:
        from cadx.evalkit.runner import run_case

        cases = _load_cases(case)
        variants = _parse_variants(variant)
        stage_filter = {s.strip() for s in stages.split(",") if s.strip()} if stages else None
        settings_overrides: dict[str, Any] = {}
        if settings:
            try:
                settings_overrides = json.loads(settings)
            except json.JSONDecodeError as e:
                print(f"error: --settings malformed JSON: {e}")
                return 2
        results = []
        for vname, voverrides in variants:
            if vname:
                print(f"=== variant {vname}: {json.dumps(voverrides)} ===")
            for c in cases:
                for rep in range(repeat):
                    # stage timeout override applies to every stage
                    if timeout:
                        for st in c.stages:
                            st.timeout_s = timeout
                    try:
                        r = await run_case(
                            c,
                            driver="local" if local else "http",
                            label=label,
                            variant=vname,
                            variant_overrides=voverrides,
                            settings_overrides=settings_overrides,
                            stages_filter=stage_filter,
                            from_version=from_version,
                            from_dir=Path(from_dir) if from_dir else None,
                            api_url=api_url,
                            repeat_index=rep if repeat > 1 else 0,
                            keep_artifacts=not no_artifacts,
                            fail_fast=not no_fail_fast,
                            verbose=verbose,
                        )
                    except Exception as e:
                        print(f"{c.name}: crashed ({type(e).__name__}: {e})")
                        continue
                    results.append(r)
                    st = ", ".join(
                        f"{s.stage_id}:{s.status}{'✓' if s.passed else '✗'}" for s in r.stages
                    )
                    err = f" error={r.error!r}" if r.error else ""
                    print(
                        f"{r.case} [{r.driver}{('/' + r.variant) if r.variant != 'default' else ''}] "
                        f"passed={r.passed} ({r.wall_s:.0f}s) {st}{err} -> evals/"
                        f"{'scratch' if local else 'results'}/{r.case}/{r.run_id}"
                    )
        if json_out:
            payload = json.dumps([r.to_dict() for r in results], default=str, indent=1)
            await asyncio.to_thread(Path(json_out).write_text, payload, "utf-8")
            print(f"results written to {json_out}")
        if not results:
            return 1
        return 0 if all(r.passed for r in results) else 1

    raise typer.Exit(asyncio.run(_run()))


@eval_app.command(name="cases")
def list_cases() -> None:
    """List suite cases, their stages and gate counts."""
    cases = _load_cases(None)
    for c in cases:
        print(f"{c.name}  (difficulty {c.difficulty}, project {c.project})")
        if c.description:
            print(f"  {c.description}")
        for s in c.stages:
            flag = " " if s.enabled else "(disabled) "
            print(
                f"  {flag}{s.id:<14} {s.type:<12} gates={len(s.gates)} timeout={s.timeout_s:.0f}s"
            )


@eval_app.command(name="report")
def report(
    baseline: str | None = typer.Option(
        None, "--baseline", help="diff the aggregated view against a saved baseline label"
    ),
    variant: str | None = typer.Option(None, "--variant"),
    include_scratch: bool = typer.Option(
        False, "--include-scratch", help="also fold evals/scratch runs into the report"
    ),
    json_out: str | None = typer.Option(None, "--json"),
    save: str | None = typer.Option(None, "--save", help="write markdown report to evals/reports/"),
) -> None:
    """Aggregate result.json trees into a variant x case table."""
    from cadx.evalkit import baseline as bl
    from cadx.evalkit.paths import BASELINES_DIR, REPORTS_DIR, RESULTS_DIR, SCRATCH_DIR
    from cadx.evalkit.report import aggregate, load_results, render_markdown, rows_to_dicts

    roots = [RESULTS_DIR] + ([SCRATCH_DIR] if include_scratch else [])
    results = load_results(*roots)
    if variant:
        results = [r for r in results if r.variant == variant]
    rows = aggregate(results)
    md = render_markdown(
        rows, title="eval report" + (" (incl. scratch)" if include_scratch else "")
    )
    print(md)
    if json_out:
        Path(json_out).write_text(json.dumps(rows_to_dicts(rows), indent=1), encoding="utf-8")
        print(f"written to {json_out}")
    if save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORTS_DIR / save
        path.write_text(md, encoding="utf-8")
        print(f"written to {path}")
    if baseline:
        try:
            cur = bl.save("report_tmp", out_dir=REPORTS_DIR)
        except ValueError as e:
            print(f"error: {e}")
            raise typer.Exit(1) from None
        diffs = bl.diff(bl.load_baseline(baseline), json.loads(cur.read_text()))
        print()
        print(bl.render_diff(diffs, baseline, "current"))
        cur.unlink(missing_ok=True)
        BASELINES_DIR.mkdir(parents=True, exist_ok=True)
        raise typer.Exit(1 if diffs else 0)


@eval_app.command(name="baseline")
def baseline_cmd(
    action: str = typer.Argument(..., help="save <label> | diff <a> <b> | list"),
    a: str | None = typer.Argument(None),
    b: str | None = typer.Argument(None),
) -> None:
    """Save/diff/list committed baselines under evals/baselines/."""
    from cadx.evalkit import baseline as bl
    from cadx.evalkit.paths import BASELINES_DIR

    if action == "save":
        if not a:
            print("usage: cadx eval baseline save <label>")
            raise typer.Exit(2)
        try:
            path = bl.save(a)
        except ValueError as e:
            print(f"error: {e}")
            raise typer.Exit(1) from None
        print(f"baseline saved: {path}")
        return
    if action == "list":
        for p in sorted(BASELINES_DIR.glob("*.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            print(f"{p.stem:<24} {d.get('created_at', '')}  cells={len(d.get('cells') or {})}")
        return
    if action == "diff":
        if not a or not b:
            print("usage: cadx eval baseline diff <label_a> <label_b>")
            raise typer.Exit(2)
        diffs = bl.diff(bl.load_baseline(a), bl.load_baseline(b))
        print(bl.render_diff(diffs, a, b))
        raise typer.Exit(1 if diffs else 0)
    print(f"unknown action {action!r} — save|diff|list")
    raise typer.Exit(2)


@eval_app.command(name="clean")
def clean(
    wipe_db: bool = typer.Option(False, "--wipe-db", help="delete all project-scoped DB rows"),
    purge_files: bool = typer.Option(False, "--purge-files", help="remove eval run file trees"),
    with_settings: bool = typer.Option(
        False, "--with-settings", help="also delete AppSetting rows (settings+secrets)"
    ),
    force: bool = typer.Option(False, "--force", help="wipe even with active jobs"),
    yes: bool = typer.Option(False, "--yes", "-y", help="execute (default is dry-run)"),
) -> None:
    """Guarded cleanup — dry-run unless --yes."""

    async def _run() -> int:
        from cadx.evalkit.clean import purge_files as pf
        from cadx.evalkit.clean import wipe_db as wd

        dry = not yes
        if dry:
            print("DRY-RUN (pass --yes to execute)")
        if wipe_db:
            try:
                rep = await wd(dry_run=dry, force=force, with_settings=with_settings)
            except Exception as e:
                print(
                    f"error: database unreachable ({type(e).__name__}) — is the stack up? "
                    f"({str(e).splitlines()[0][:160]})"
                )
                return 1
            for name, n in rep.deleted.items():
                print(f"  {'would delete' if dry else 'deleted'} {n:>6} rows from {name}")
            for s in rep.skipped:
                print(f"  skipped: {s}")
        if purge_files:
            rep = pf(dry_run=dry)
            for f in rep.files_removed:
                print(f"  {'would remove' if dry else 'removed'} {f}")
            if not rep.files_removed:
                print("  nothing to remove")
        if not (wipe_db or purge_files):
            print("nothing selected — pass --wipe-db and/or --purge-files")
        return 0

    raise typer.Exit(asyncio.run(_run()))
