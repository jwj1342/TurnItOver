"""The vertical slice: reference -> corrupt -> observe -> evidence -> check -> write.

Stateless per shard: everything derives from (config, shard, n_shards).
"""
from __future__ import annotations

import dataclasses
import json
import logging
import socket
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from turnitover import __version__
from turnitover.assets.source import make_source
from turnitover.checkers import make_checkers
from turnitover.checkers.base import Checker, Evidence
from turnitover.checkers.evidence import collect_evidence, failed_evidence, resolve_states
from turnitover.config import GenerateConfig
from turnitover.core.program import ObjectProgram
from turnitover.core.sample import SCHEMA_VERSION, Framing, Sample
from turnitover.corruptions import Corruption, get_corruption
from turnitover.engine.sharding import sample_seed, shard_indices
from turnitover.policy.base import PolicyContext
from turnitover.policy.loop import run_observation_loop
from turnitover.policy.scripted import ScriptedPolicy
from turnitover.render.session import CompileError, HarnessError, ObservationSession
from turnitover.render.views import load_views
from turnitover.storage.writer import ShardWriter, shard_paths
from turnitover.telemetry import environment_info, git_state, now_iso, summarize_timings, timed

log = logging.getLogger("turnitover.engine")


@dataclass(frozen=True)
class ReferenceBundle:
    program: ObjectProgram
    framing: Framing
    evidence: Evidence
    states: tuple[str, ...]
    source_id: str


def build_reference(session: ObservationSession, program: ObjectProgram, evidence_states, source_id: str) -> ReferenceBundle:
    info = session.load(program, framing=None)
    states = resolve_states(evidence_states, tuple(info.joints))
    evidence = collect_evidence(session, program.sha, states)
    session.dispose()
    return ReferenceBundle(program=program, framing=info.framing, evidence=evidence, states=states, source_id=source_id)


def generate_sample(
    idx: int,
    cfg: GenerateConfig,
    session: ObservationSession,
    corruptions: tuple[Corruption, ...],
    checkers: tuple[Checker, ...],
    reference: ReferenceBundle,
) -> tuple[Sample, dict[str, bytes], dict]:
    seed = sample_seed(cfg.seed, idx)
    rng = np.random.default_rng(seed)
    timing: dict[str, float] = {}
    spec = reference.program.spec
    sample_id = f"{spec.asset_id}-{idx:08d}"
    log.info("sample.start", extra={"sample_id": sample_id, "seed": seed})

    with timed(timing, "corrupt"):
        corruption = corruptions[rng.integers(len(corruptions))]
        program, cspec = corruption.apply(reference.program, rng)

    images: dict[str, bytes] = {}
    trajectory = ()
    evidence: Evidence
    error = None
    try:
        with timed(timing, "load"):
            info = session.load(program, framing=reference.framing)
        with timed(timing, "observe"):
            ctx = PolicyContext(joints=tuple(info.joints), views=tuple(v.id for v in session.views), budget=cfg.budget)

            def sink(step: int, png: bytes) -> str:
                ref = f"obs/{step:02d}.png"
                images[ref] = png
                return ref

            trajectory = run_observation_loop(session, ScriptedPolicy(cfg.observation_script), ctx, sink)
        with timed(timing, "evidence"):
            evidence = collect_evidence(session, program.sha, reference.states)
    except (CompileError, HarnessError) as e:
        error = str(e)
        evidence = failed_evidence(program.sha, error)
        log.warning("sample.compile_error", extra={"sample_id": sample_id, "error": error})
    finally:
        session.dispose()

    with timed(timing, "checks"):
        checks = tuple(c.check(evidence, reference.evidence if c.needs_reference else None) for c in checkers)

    sample = Sample(
        schema_version=SCHEMA_VERSION,
        sample_id=sample_id,
        run_id=cfg.run_id,
        idx=idx,
        seed=seed,
        asset_id=spec.asset_id,
        source_id=reference.source_id,
        program_sha=program.sha,
        program_source=program.source,
        reference_sha=reference.program.sha,
        corruptions=(cspec,),
        framing=reference.framing,
        trajectory=trajectory,
        checks=checks,
        compile_ok=evidence.compile_ok,
        error=error,
        timing_ms=timing,
        host=socket.gethostname(),
        created_at=now_iso(),
    )
    log.info("sample.done", extra={"sample_id": sample_id, "defect": cspec.defect_id, "compile_ok": sample.compile_ok,
                                   "checks": {c.checker_id: c.passed for c in checks}, "timing_ms": timing})
    return sample, images, dict(evidence.geometry)


def run_generate(cfg: GenerateConfig, shard: int = 0, n_shards: int = 1) -> Path:
    paths = shard_paths(cfg.output_dir, shard)
    if paths["manifest"].exists() and json.loads(paths["manifest"].read_text()).get("status") == "complete":
        log.info("shard.skip_complete", extra={"shard": shard})
        return paths["tar"]

    views = load_views(cfg.views_path)
    source = make_source(**cfg.asset_source)
    spec = next(iter(source))
    corruptions = tuple(get_corruption(cid, **cfg.corruption_params.get(cid, {})) for cid in cfg.corruptions)
    checkers = make_checkers(cfg.checkers)
    indices = shard_indices(cfg.n_samples, shard, n_shards)
    counters = {"n_ok": 0, "n_compile_error": 0, "n_failed": 0}
    timings: list[dict[str, float]] = []
    started = now_iso()

    with ObservationSession(cfg.render, views) as session, ShardWriter(cfg.output_dir, shard) as writer:
        gl = session.gl_info()
        log.info("session.start", extra={"shard": shard, "gl": gl, "browser": session.browser_version()})
        reference = build_reference(session, ObjectProgram.from_spec(spec), cfg.evidence_states, source.source_id)
        for idx in indices:
            try:
                sample, images, geometry = generate_sample(idx, cfg, session, corruptions, checkers, reference)
            except Exception:
                counters["n_failed"] += 1
                log.exception("sample.failed", extra={"idx": idx})
                continue
            with timed(sample.timing_ms, "write"):
                writer.add(sample, images, geometry)
            counters["n_ok" if sample.compile_ok else "n_compile_error"] += 1
            timings.append(sample.timing_ms)
        manifest = {
            "status": "complete",
            "version": __version__,
            "run_id": cfg.run_id,
            "shard": shard,
            "n_shards": n_shards,
            "indices": [indices.start, indices.stop, indices.step],
            "config_path": str(cfg.config_path),
            "config": _config_dict(cfg),
            "git": git_state(Path(__file__).resolve().parents[2]),
            "browser": session.browser_version(),
            "gl": gl,
            "started_at": started,
            "finished_at": now_iso(),
            **environment_info(),
            **counters,
            "timings": summarize_timings(timings),
        }
        writer.write_manifest(manifest)
    log.info("shard.done", extra={"shard": shard, **counters})
    return paths["tar"]


def _config_dict(cfg: GenerateConfig) -> dict:
    from turnitover.core.serde import to_dict

    d = to_dict(cfg)
    return json.loads(json.dumps(d, default=str))
