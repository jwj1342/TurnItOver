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

    p = sub.add_parser("preview", help="export PNG screenshots, MP4 videos, and an offline gallery")
    p.add_argument("--out", type=Path, help="new/empty output directory (default: output/preview-TIMESTAMP)")
    p.add_argument("--program", type=Path, help="ABI-compatible TypeScript; defaults to the toy cabinet")
    p.add_argument("--size", type=int, default=640, help="square image size in pixels")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--seconds", type=float, default=4, help="duration of orbit and of each joint sweep")
    p.add_argument("--no-video", action="store_true", help="screenshots only; does not require FFmpeg")
    p.add_argument("--ffmpeg", default="ffmpeg", help="FFmpeg executable with libx264 support")

    mc = sub.add_parser("models-check", help="check role configuration locally; never calls APIs")
    mc.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    for command, help_text in (("model-call", "call a configured text/vision model once"),
                               ("reconstruct", "single-pass photo-to-program baseline (not an upstream pipeline)")):
        m = sub.add_parser(command, help=help_text)
        m.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
        m.add_argument("--role", choices=("generator", "judge", "diagnosis"), default="generator")
        m.add_argument("--prompt-file", type=Path, required=command == "model-call",
                       default=REPO_ROOT / "docs/prompts/reconstruct.txt")
        m.add_argument("--image", type=Path, action="append", help="repeat for multiple images")
        m.add_argument("--out", type=Path, required=True)
        m.add_argument("--dry-run", action="store_true", help="prepare inputs without API calls or keys")

    sub.add_parser("render-docs", help="regenerate docs/taxonomy.md")

    v = sub.add_parser("verify", help="budgeted verifier with evidence-linked verdict and offline report")
    v.add_argument("--program", type=Path, required=True)
    v.add_argument("--out", type=Path, required=True)
    v.add_argument("--reference-image", type=Path, action="append", default=[])
    v.add_argument("--reference-program", type=Path, help="optional privileged gold audit; never sent to judge")
    v.add_argument("--task", default="Verify the articulated object against the reference inputs.")
    v.add_argument("--policy", choices=("active", "fixed", "random", "runtime"), default="active")
    v.add_argument("--budget", type=int, default=8)
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--size", type=int, default=384)
    v.add_argument("--views", type=Path, default=REPO_ROOT / "configs/views.yaml")
    v.add_argument("--actions", nargs="+", choices=("request_view", "actuate_joint", "query_runtime"),
                   default=["request_view", "actuate_joint", "query_runtime"])
    v.add_argument("--max-triangles", type=int, default=5000)
    v.add_argument("--max-draw-calls", type=int, default=32)
    v.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    v.add_argument("--local-model", type=Path, help="Qwen3-VL weights directory (CUDA); bypasses API configuration")

    args = parser.parse_args(argv)
    configure_logging(json_lines=args.log_json, level=args.log_level)
    return {"generate": _generate, "detectability": _detectability, "inspect": _inspect,
            "preview": _preview, "models-check": _models_check, "model-call": _model_call,
            "reconstruct": _model_call, "verify": _verify, "render-docs": _render_docs}[args.cmd](args)


def _verify(args) -> int:
    from turnitover.core.program import ObjectProgram
    from turnitover.models.config import model_config, read_environment
    from turnitover.render.session import RenderConfig
    from turnitover.render.views import load_views
    from turnitover.verifier.runner import VerifyConfig, verify

    model = None if args.local_model or args.policy == "runtime" else model_config("judge", read_environment(args.env_file))
    result = verify(ObjectProgram(args.program.read_text()), args.out,
                    RenderConfig(REPO_ROOT / "web/dist", REPO_ROOT / "web/node_modules/.bin/esbuild", args.size, args.size),
                    load_views(args.views), VerifyConfig(budget=args.budget, mode=args.policy, seed=args.seed,
                        action_types=tuple(args.actions), max_triangles=args.max_triangles, max_draw_calls=args.max_draw_calls),
                    task=args.task, references=tuple(args.reference_image), model=model, local_model=args.local_model,
                    reference_program=ObjectProgram(args.reference_program.read_text()) if args.reference_program else None)
    print(f"{result['verdict']['status']} ({result['termination']}): {args.out.resolve() / 'index.html'}")
    return 0 if result["status"] == "complete" else 1


def _models_check(args) -> int:
    from turnitover.models.commands import check_models

    return check_models(args)


def _model_call(args) -> int:
    from turnitover.models.commands import run_call

    return run_call(args, REPO_ROOT, reconstruct=args.cmd == "reconstruct")


def _preview(args) -> int:
    from datetime import datetime, timezone

    from turnitover.assets.toy import toy_cabinet
    from turnitover.core.program import ObjectProgram
    from turnitover.preview import export_preview
    from turnitover.render.session import RenderConfig
    from turnitover.render.views import load_views

    output = args.out or Path("output") / datetime.now(timezone.utc).strftime("preview-%Y%m%dT%H%M%S%fZ")
    if not output.is_absolute():
        output = REPO_ROOT / output
    program = ObjectProgram(args.program.read_text(encoding="utf-8")) if args.program else ObjectProgram.from_spec(toy_cabinet())
    render = RenderConfig(REPO_ROOT / "web/dist", REPO_ROOT / "web/node_modules/.bin/esbuild",
                          width=args.size, height=args.size)
    path = export_preview(program, render, load_views(REPO_ROOT / "configs/views.yaml"), output,
                          fps=args.fps, seconds=args.seconds, videos=not args.no_video, ffmpeg=args.ffmpeg)
    print(path)
    return 0


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
