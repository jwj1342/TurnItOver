"""Command-line entry points. Thin: parse args, load config, call the engine."""
from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path

from turnitover.engine.sharding import parse_shard
from turnitover.telemetry import configure_logging

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="turnitover")
    parser.add_argument("--log-json", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="run the data engine for one shard")
    g.add_argument("--config", type=Path, required=True)
    g.add_argument("--shard", default="0/1", help="i/n")
    g.add_argument("--out", type=Path)
    g.add_argument("--n-samples", type=int)

    d = sub.add_parser("detectability", help="compute a detectability matrix on the configured asset")
    d.add_argument("--config", type=Path, required=True)
    d.add_argument("--out", type=Path, required=True)

    i = sub.add_parser("inspect", help="print a summary of samples in a shard tar")
    i.add_argument("tar", type=Path)
    i.add_argument("--n", type=int, default=5)

    sub.add_parser("render-docs", help="regenerate docs/taxonomy.md")

    args = parser.parse_args(argv)
    configure_logging(json_lines=args.log_json, level=args.log_level)
    return {"generate": _generate, "detectability": _detectability, "inspect": _inspect, "render-docs": _render_docs}[args.cmd](args)


def _generate(args) -> int:
    from turnitover.config import load_generate_config
    from turnitover.engine.generate import run_generate

    cfg = load_generate_config(args.config, REPO_ROOT)
    changes = {}
    if args.out:
        changes["output_dir"] = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if args.n_samples:
        changes["n_samples"] = args.n_samples
    cfg = dataclasses.replace(cfg, **changes)
    shard, n_shards = parse_shard(args.shard)
    path = run_generate(cfg, shard, n_shards)
    print(path)
    return 0


def _detectability(args) -> int:
    import yaml

    from turnitover.assets.source import make_source
    from turnitover.config import parse_action
    from turnitover.core.program import ObjectProgram
    from turnitover.corruptions import get_corruption
    from turnitover.detectability.matrix import Thresholds, compute_on_asset
    from turnitover.render.session import ObservationSession, RenderConfig
    from turnitover.render.views import load_views

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    r = raw.get("render", {})
    render = RenderConfig(harness_dir=REPO_ROOT / r.get("harness_dir", "web/dist"),
                          esbuild_bin=REPO_ROOT / r.get("esbuild_bin", "web/node_modules/.bin/esbuild"),
                          width=int(r.get("width", 256)), height=int(r.get("height", 256)))
    views = load_views(REPO_ROOT / raw.get("views", "configs/views.yaml"))
    source = make_source(**raw.get("asset_source", {"kind": "toy"}))
    reference = ObjectProgram.from_spec(next(iter(source)))
    corruptions = [get_corruption(c) for c in raw["corruptions"]]
    actions = [parse_action(a) for a in raw["actions"]]
    thr = Thresholds(**raw.get("thresholds", {}))
    with ObservationSession(render, views) as session:
        m = compute_on_asset(session, reference, corruptions, actions, int(raw.get("n_seeds", 4)), int(raw.get("seed", 0)), thr)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    m.to_json(args.out)
    print(m.to_markdown())
    return 0


def _inspect(args) -> int:
    from turnitover.storage.reader import iter_samples

    for k, (sample, members) in enumerate(iter_samples(args.tar)):
        if k >= args.n:
            break
        print(f"== {sample.sample_id}  compile_ok={sample.compile_ok}  images={sum(m.startswith('obs/') for m in members)}")
        for c in sample.corruptions:
            print(f"   corruption {c.defect_id} parts={list(c.parts)} severity={c.severity:.2f}")
        for ch in sample.checks:
            print(f"   check {ch.checker_id}: {'PASS' if ch.passed else 'FAIL'} score={ch.score}")
            for inv in ch.invariants:
                if not inv.passed:
                    print(f"      x {inv.name} state={inv.state} parts={list(inv.parts)} value={inv.value:.4g} tol={inv.tolerance:.4g}")
        print(f"   timing_ms={{{', '.join(f'{k}={v:.0f}' for k, v in sample.timing_ms.items())}}}")
    return 0


def _render_docs(args) -> int:
    from turnitover.taxonomy.render_docs import write_docs

    out = REPO_ROOT / "docs" / "taxonomy.md"
    write_docs(out)
    logging.getLogger("turnitover").info("docs.written", extra={"path": str(out)})
    return 0
