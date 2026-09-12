"""The vertical slice: reference -> corrupt -> observe -> evidence -> check -> write.

Stateless per shard: everything derives from (config, shard, n_shards).
"""
from __future__ import annotations

import dataclasses
import hashlib
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
from turnitover.core.actions import ActuateJoint
from turnitover.core.sample import SCHEMA_VERSION, Framing, Sample
from turnitover.corruptions import Corruption, get_corruption
from turnitover.engine.sharding import sample_seed, shard_indices
from turnitover.engine.quality import qualify
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
    images: dict[str, bytes] = dataclasses.field(default_factory=dict)
    metadata: dict = dataclasses.field(default_factory=dict)


def build_reference(session: ObservationSession, program: ObjectProgram, evidence_states, source_id: str) -> ReferenceBundle:
    info = session.load(program, framing=None)
    states = resolve_states(evidence_states, tuple(info.joints))
    evidence = collect_evidence(session, program.sha, states)
    images = {"reference/00.png": session.request_view(session.views[0].id).png}
    session.dispose()
    return ReferenceBundle(program=program, framing=info.framing, evidence=evidence, states=states, source_id=source_id, images=images)


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
        program = reference.program
        labels, isolated_programs = [], []
        # Preserve the legacy RNG sequence for the default single-corruption run.
        clean = cfg.clean_fraction > 0 and rng.random() < cfg.clean_fraction
        if not clean:
            available = [c for c in corruptions if c.applicable(program)]
            if len(available) < cfg.max_corruptions:
                raise ValueError("Not enough applicable corruption types for this asset")
            count = 1 if cfg.max_corruptions == 1 else int(rng.integers(1, cfg.max_corruptions + 1))
            for _ in range(count):
                corruption = available.pop(int(rng.integers(len(available))))
                state = rng.bit_generator.state
                isolated_rng = np.random.default_rng(0)
                isolated_rng.bit_generator.state = state
                isolated, _ = corruption.apply(reference.program, isolated_rng)
                program, label = corruption.apply(program, rng)
                labels.append(label)
                isolated_programs.append(isolated)

    images: dict[str, bytes] = dict(reference.images)
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

            script = tuple(expanded for action in cfg.observation_script for expanded in (
                tuple(ActuateJoint(j, action.detent) for j in info.joints)
                if isinstance(action, ActuateJoint) and action.joint_id == "*" else (action,)))
            trajectory = run_observation_loop(session, ScriptedPolicy(script), ctx, sink)
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

    isolated_checks = []
    with timed(timing, "label_validation"):
        for isolated in isolated_programs:
            if len(labels) == 1:
                isolated_checks.append(checks)
                continue
            try:
                session.load(isolated, framing=reference.framing)
                isolated_evidence = collect_evidence(session, isolated.sha, reference.states)
            except (CompileError, HarnessError) as exc:
                isolated_evidence = failed_evidence(isolated.sha, str(exc))
            finally:
                session.dispose()
            isolated_checks.append(tuple(c.check(isolated_evidence, reference.evidence if c.needs_reference else None) for c in checkers))
    quality = qualify(labels, checks, isolated_checks)
    if not evidence.compile_ok:
        quality["status"] = "quarantined"

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
        corruptions=tuple(labels),
        framing=reference.framing,
        trajectory=trajectory,
        checks=checks,
        compile_ok=evidence.compile_ok,
        error=error,
        timing_ms=timing,
        host=socket.gethostname(),
        created_at=now_iso(),
        dataset={"category": spec.category, **reference.metadata, "quality": quality,
                 "reference_images": list(reference.images), "reference_view": str(session.views[0].id),
                 "reference_program_source": reference.program.source,
                 "parts": list(program.spec.part_ids), "joints": list(program.spec.joint_ids),
                 "views": [str(v.id) for v in session.views],
                 "constraints": cfg.checkers, "trajectory_policy": "scripted_not_expert"},
    )
    log.info("sample.done", extra={"sample_id": sample_id, "defects": [c.defect_id for c in labels], "compile_ok": sample.compile_ok,
                                   "checks": {c.checker_id: c.passed for c in checks}, "timing_ms": timing})
    return sample, images, dict(evidence.geometry)


def run_generate(cfg: GenerateConfig, shard: int = 0, n_shards: int = 1) -> Path:
    from turnitover.assets.catalog import validate_spec
    from turnitover.core.serde import to_dict

    paths = shard_paths(cfg.output_dir, shard)
    source = make_source(**cfg.asset_source)
    specs = sorted(source, key=lambda s: s.asset_id)
    if not specs or len({s.asset_id for s in specs}) != len(specs):
        raise ValueError("Asset source must contain unique assets")
    metadata = {}
    for spec in specs:
        validate_spec(spec)
        metadata[spec.asset_id] = source.metadata(spec.asset_id) if hasattr(source, "metadata") else {
            "split_group": f"{source.source_id}:{spec.asset_id}", "origin": "builtin procedural toy fixture",
            "license": "repository terms; not an external asset license"}
    fingerprint_data = {"schema_version": SCHEMA_VERSION, "config": _config_dict(cfg), "shard": shard,
                        "n_shards": n_shards, "assets": to_dict(specs), "metadata": metadata}
    package_root = Path(__file__).resolve().parents[1]
    code_digest = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        code_digest.update(str(path.relative_to(package_root)).encode())
        code_digest.update(path.read_bytes())
    fingerprint_data["code_sha256"] = code_digest.hexdigest()
    fingerprint_data["views_sha256"] = hashlib.sha256(cfg.views_path.read_bytes()).hexdigest()
    fingerprint_data["harness_sha256"] = hashlib.sha256((cfg.render.harness_dir / "harness.js").read_bytes()).hexdigest()
    fingerprint = hashlib.sha256(json.dumps(fingerprint_data, sort_keys=True).encode()).hexdigest()
    if paths["manifest"].exists():
        previous = json.loads(paths["manifest"].read_text())
        if previous.get("status") == "complete" and previous.get("fingerprint") == fingerprint:
            if not all(p.is_file() for p in paths.values()):
                raise ValueError("Completed shard is missing files; choose a new output directory")
            log.info("shard.skip_complete", extra={"shard": shard})
            return paths["tar"]
        raise ValueError("Existing shard has different inputs or is incomplete; choose a new output directory")
    if any(p.exists() for p in paths.values()):
        raise ValueError("Partial shard exists; choose a new output directory")

    views = load_views(cfg.views_path)
    corruptions = tuple(get_corruption(cid, **cfg.corruption_params.get(cid, {})) for cid in cfg.corruptions)
    checkers = make_checkers(cfg.checkers)
    if not checkers:
        raise ValueError("Dataset generation requires reference quality checkers")
    indices = shard_indices(cfg.n_samples, shard, n_shards)
    counters = {"n_ok": 0, "n_compile_error": 0, "n_failed": 0, "n_accepted": 0, "n_quarantined": 0}
    timings: list[dict[str, float]] = []
    failures = []
    started = now_iso()

    with ObservationSession(cfg.render, views) as session, ShardWriter(cfg.output_dir, shard) as writer:
        gl = session.gl_info()
        # Group execution by asset to retain only one reference geometry bundle in memory.
        # Selection depends on the global index, never the worker/shard count.
        for asset_index, spec in enumerate(specs):
            asset_indices = (i for i in indices if i % len(specs) == asset_index)
            reference = None
            for idx in asset_indices:
                try:
                    if reference is None:
                        reference = build_reference(session, ObjectProgram.from_spec(spec), cfg.evidence_states, source.source_id)
                        reference_checks = tuple(c.check(reference.evidence, reference.evidence if c.needs_reference else None) for c in checkers)
                        if not reference.evidence.compile_ok or not reference.evidence.geometry or not all(
                                c.passed and c.invariants for c in reference_checks):
                            reference = None
                            raise ValueError("Reference fails configured baseline constraints")
                        reference = dataclasses.replace(reference, metadata=metadata[spec.asset_id])
                    sample, images, geometry = generate_sample(idx, cfg, session, corruptions, checkers, reference)
                except Exception as exc:
                    counters["n_failed"] += 1
                    failures.append({"idx": idx, "asset_id": spec.asset_id, "error": str(exc), "error_type": type(exc).__name__})
                    log.exception("sample.failed", extra={"idx": idx})
                    continue
                with timed(sample.timing_ms, "write"):
                    writer.add(sample, images, geometry)
                counters["n_ok" if sample.compile_ok else "n_compile_error"] += 1
                counters["n_accepted" if sample.dataset["quality"]["status"] == "accepted" else "n_quarantined"] += 1
                timings.append(sample.timing_ms)
        manifest = {
            "status": "complete" if not counters["n_failed"] else "incomplete",
            "fingerprint": fingerprint,
            "schema_version": SCHEMA_VERSION,
            "version": __version__,
            "run_id": cfg.run_id,
            "shard": shard,
            "n_shards": n_shards,
            "indices": [indices.start, indices.stop, indices.step],
            "assets": [{"asset_id": s.asset_id, "category": s.category, **metadata[s.asset_id]} for s in specs],
            "config_path": str(cfg.config_path),
            "config": _config_dict(cfg),
            "git": git_state(Path(__file__).resolve().parents[2]),
            "browser": session.browser_version(),
            "gl": gl,
            "started_at": started,
            "finished_at": now_iso(),
            "failures": failures,
            **environment_info(),
            **counters,
            "timings": summarize_timings(timings),
        }
        writer.close()
        writer.write_manifest(manifest)
    log.info("shard.done", extra={"shard": shard, **counters})
    if counters["n_failed"]:
        raise RuntimeError(f"Shard incomplete: {counters['n_failed']} failed samples; see manifest")
    return paths["tar"]


def _config_dict(cfg: GenerateConfig) -> dict:
    from turnitover.core.serde import to_dict

    d = to_dict(cfg)
    return json.loads(json.dumps(d, default=str))
