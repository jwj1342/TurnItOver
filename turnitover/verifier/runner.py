"""File-backed verification entry point; privileged audit runs only after the verdict."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from turnitover.core.program import ObjectProgram
from turnitover.core.serde import to_dict
from turnitover.models.client import complete, image_part
from turnitover.models.config import ModelConfig
from turnitover.render.session import CompileError, HarnessError, ObservationSession, RenderConfig
from turnitover.render.views import ViewDef
from turnitover.telemetry import git_state, now_iso
from turnitover.verifier.contracts import Context, VERIFIER_SCHEMA_VERSION, uncertain
from turnitover.verifier.loop import Episode, run_verification
from turnitover.verifier.policies import RuntimePolicy, VisionPolicy
from turnitover.verifier.report import write_report


@dataclass(frozen=True)
class VerifyConfig:
    budget: int = 8
    mode: str = "active"
    seed: int = 0
    action_types: tuple[str, ...] = ("request_view", "actuate_joint", "query_runtime")
    runtime_properties: tuple[str, ...] = ("stats", "hierarchy", "joint_state", "state_delta")
    max_triangles: int = 5000
    max_draw_calls: int = 32

    def validate(self):
        if type(self.budget) is not int or not 0 <= self.budget <= 64:
            raise ValueError("Verifier budget must be an integer in [0,64]")
        if self.mode not in {"active", "fixed", "random", "runtime"}:
            raise ValueError("Unknown verifier policy")
        if not set(self.action_types) <= {"request_view", "actuate_joint", "query_runtime"}:
            raise ValueError("Invalid action allowlist")
        if not set(self.runtime_properties) <= {"stats", "hierarchy", "joint_state", "state_delta"}:
            raise ValueError("Invalid runtime property allowlist")
        if self.max_triangles <= 0 or self.max_draw_calls <= 0:
            raise ValueError("Runtime thresholds must be positive")


def verify(program: ObjectProgram, output: Path, render: RenderConfig, views: tuple[ViewDef, ...],
           cfg: VerifyConfig, *, task: str = "Verify this articulated object.", references: tuple[Path, ...] = (),
           model: ModelConfig | None = None, local_model: Path | None = None,
           reference_program: ObjectProgram | None = None, model_call=None) -> dict:
    cfg.validate()
    if not views or render.width <= 0 or render.height <= 0:
        raise ValueError("Views and positive render dimensions are required")
    if cfg.mode != "runtime" and not (model_call or local_model):
        if model is None:
            raise ValueError("Model configuration or local weights are required")
        model.validate()
    for reference in references:
        image_part(reference)
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Verifier output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    (output / "observations").mkdir()
    (output / "candidate.ts").write_text(program.source)
    copied = []
    image_metadata = []
    for i, path in enumerate(references):
        dest = output / f"reference-{i:02d}{path.suffix.lower()}"
        data = path.read_bytes()
        dest.write_bytes(data)
        copied.append(dest)
        image_metadata.append({"path": dest.name, "sha256": hashlib.sha256(data).hexdigest()})
    metadata = {"schema_version": VERIFIER_SCHEMA_VERSION, "created_at": now_iso(), "status": "running",
                "task": task, "program_sha": program.sha, "config": dataclasses.asdict(cfg),
                "render": {"width": render.width, "height": render.height, "views": [v.to_json() for v in views]},
                "references": image_metadata, "git": git_state(Path(__file__).resolve().parents[2]),
                "model": model.public() if model else {"kind": "local_qwen" if local_model else "offline",
                                                       "name": local_model.name if local_model else None}}
    if local_model and (local_model / "download-manifest.json").exists():
        metadata["local_weights"] = json.loads((local_model / "download-manifest.json").read_text())
    _json(output / "result.json", metadata)
    if cfg.mode == "runtime":
        policy = RuntimePolicy()
    else:
        if local_model:
            from turnitover.models.local_qwen import LocalQwen

            call = LocalQwen(local_model)
        else:
            call = model_call or (lambda prompt, images: complete(model, prompt, images))
        policy = VisionPolicy(call, output, copied, cfg.mode, cfg.seed)
    trajectory_path = output / "trajectory.jsonl"
    with trajectory_path.open("w") as trace:
        def save_observation(obs):
            trace.write(json.dumps(to_dict(obs)) + "\n")
            trace.flush()

        def image_sink(step, png):
            relative = f"observations/{step:03d}.png"
            (output / relative).write_bytes(png)
            return relative

        stage = "environment"
        try:
            with ObservationSession(render, views) as session:
                metadata.update(browser=session.browser_version(), gl=session.gl_info())
                stage = "program_load"
                info = session.load(program, framing=None)
                metadata.update(program_loaded=True, framing=to_dict(info.framing))
                context = Context(task, info.parts, tuple(info.joints), tuple(v.id for v in views),
                                  cfg.runtime_properties, cfg.action_types, cfg.max_triangles, cfg.max_draw_calls)
                metadata["context"] = dataclasses.asdict(context)
                stage = "verification"
                episode = run_verification(session, policy, context, cfg.budget, image_sink, save_observation)
        except (CompileError, HarnessError) as exc:
            episode = Episode(uncertain("Candidate could not be loaded or observed."), "program_error", (), 0, type(exc).__name__)
            metadata["program_loaded"] = False
        except Exception as exc:
            # Preserve prior evidence if an infrastructure failure interrupts the loop.
            metadata.update(status="error", termination="environment_error" if stage == "environment" else "internal_error",
                            error_type=type(exc).__name__, verdict=dataclasses.asdict(uncertain("Verifier execution failed.")),
                            model_calls=list(policy.calls), finished_at=now_iso())
            _json(output / "result.json", metadata)
            write_report(output, metadata, [])
            raise
    metadata.update(status="complete" if episode.termination in {"judge_finished", "budget_exhausted"} else "error",
                    termination=episode.termination, verdict=dataclasses.asdict(episode.verdict),
                    spent=episode.spent, error_type=episode.error_type,
                    attempted_action=episode.attempted_action,
                    error_message=episode.error_message,
                    trajectory=[to_dict(o) for o in episode.trajectory], model_calls=list(policy.calls), finished_at=now_iso())
    # Persist the judge's result before any privileged evaluation happens.
    _json(output / "result.json", metadata)
    _json(output / "feedback.json", {"verdict": metadata["verdict"], "termination": episode.termination,
                                     "program_sha": program.sha, "evidence_file": "trajectory.jsonl"})
    if reference_program:
        from turnitover.verifier.audit import audit

        (output / "reference-program.ts").write_text(reference_program.source)
        metadata["audit"] = audit(program, reference_program, render, views, cfg.max_triangles, cfg.max_draw_calls)
        _json(output / "audit.json", metadata["audit"])
        _json(output / "result.json", metadata)
    write_report(output, metadata, episode.trajectory)
    return metadata


def _json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)
