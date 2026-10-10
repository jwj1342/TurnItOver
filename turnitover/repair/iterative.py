"""Photo-to-program generation, render gating and fresh verification loop."""
from __future__ import annotations

import dataclasses
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from turnitover.core.program import ObjectProgram
from turnitover.core.jsonio import write_json_atomic
from turnitover.models.client import ModelResult, image_part
from turnitover.models.generation import generate_program
from turnitover.render.gate import RenderGateResult, run_render_gate
from turnitover.render.session import RenderConfig
from turnitover.render.views import ViewDef
from turnitover.repair.loop import Feedback, RepairProposal, propose_repair, usage_tokens
from turnitover.telemetry import git_state, now_iso
from turnitover.verifier.runner import VerifyConfig, verify


@dataclass(frozen=True)
class IterativeConfig:
    max_runtime_repairs: int = 3
    max_visual_revisions: int = 2
    blank_stddev_threshold: float = 1.0
    verifier: VerifyConfig = VerifyConfig()
    task: str = "Verify the articulated object against the reference inputs."

    def validate(self) -> None:
        if type(self.max_runtime_repairs) is not int or self.max_runtime_repairs < 0:
            raise ValueError("Runtime-repair budget must be a nonnegative integer")
        if type(self.max_visual_revisions) is not int or self.max_visual_revisions < 0:
            raise ValueError("Visual-revision budget must be a nonnegative integer")
        if self.blank_stddev_threshold < 0:
            raise ValueError("Blank-image threshold must be nonnegative")
        if not self.task.strip():
            raise ValueError("Task must be nonempty")
        self.verifier.validate()


def run_iterative(
    references: tuple[Path, ...],
    output: Path,
    render: RenderConfig,
    views: tuple[ViewDef, ...],
    cfg: IterativeConfig,
    generator_call: Callable[[str, list[Path]], ModelResult],
    *,
    generation_prompt: str | None = None,
    initial_program: ObjectProgram | None = None,
    judge_call: Callable[[str, list[Path]], ModelResult] | None = None,
    judge_model=None,
    gate_call=run_render_gate,
    verifier_call=verify,
    repair_call=propose_repair,
) -> dict:
    """Run the bounded loop. Only an evidence-linked verifier pass is accepted."""
    cfg.validate()
    if not references:
        raise ValueError("At least one reference image is required")
    if not views:
        raise ValueError("At least one view is required")
    for path in references:
        image_part(path)
    if initial_program is None and not generation_prompt:
        raise ValueError("A generation prompt is required without an initial program")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Iterative output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    copied_references = _copy_references(references, output)
    report = {
        "schema_version": 1,
        "kind": "iterative_synthesis",
        "status": "running",
        "accepted": False,
        "created_at": now_iso(),
        "task": cfg.task,
        "config": dataclasses.asdict(cfg),
        "render": {"width": render.width, "height": render.height,
                   "views": [view.to_json() for view in views]},
        "references": [_file_record(path, output) for path in copied_references],
        "git": git_state(Path(__file__).resolve().parents[2]),
        "rounds": [],
        "model_calls": {"generation": 0, "runtime_repair": 0, "visual_revision": 0, "judge": 0},
        "total_tokens": 0,
    }
    write_json_atomic(output / "result.json", report)

    def tracked_generator(kind: str):
        def call(prompt: str, images: list[Path]) -> ModelResult:
            report["model_calls"][kind] += 1
            response = generator_call(prompt, images)
            report["total_tokens"] += usage_tokens(response.usage)
            return response
        return call

    def record_proposal_if_untracked(kind: str, calls_before: int, proposal: RepairProposal) -> None:
        if report["model_calls"][kind] == calls_before:
            report["model_calls"][kind] += 1
            report["total_tokens"] += usage_tokens(proposal.usage)

    try:
        if initial_program is None:
            generated = generate_program(generation_prompt or "", copied_references, output / "generation",
                                         tracked_generator("generation"))
            program = generated.program
        else:
            program = initial_program
            (output / "initial.ts").write_text(program.source, encoding="utf-8")
    except Exception as exc:
        return _finish(output, report, "generation_failed", error_type=type(exc).__name__)

    visual_memory: list[dict] = []
    visual_calls = 0
    round_index = 0
    while True:
        round_dir = output / f"round-{round_index:03d}"
        round_dir.mkdir()
        (round_dir / "candidate.ts").write_text(program.source, encoding="utf-8")
        round_record = {"round": round_index, "program_sha": program.sha, "render_attempts": [],
                        "verifier": None, "visual_revisions": []}
        report["rounds"].append(round_record)
        write_json_atomic(output / "result.json", report)

        runtime_memory: list[dict] = []
        gate_result: RenderGateResult | None = None
        runtime_calls = 0
        gate_index = 0
        render_required = True
        while True:
            gate_dir = round_dir / "render-gate" / f"attempt-{gate_index:03d}"
            if render_required:
                gate_result = gate_call(program, gate_dir, render, views, cfg.blank_stddev_threshold)
                gate_record = dataclasses.asdict(gate_result)
                gate_record.update(attempt=gate_index, program_sha=program.sha,
                                   artifact=str(gate_dir.relative_to(output) / "result.json"))
                round_record["render_attempts"].append(gate_record)
                write_json_atomic(output / "result.json", report)
                if gate_result.success:
                    round_record["program_sha"] = program.sha
                    break
                if gate_result.failure_kind == "environment":
                    return _finish(output, report, "render_failed", final_program=program,
                                   termination="environment_error", error_type=gate_result.error_type,
                                   message=gate_result.message, stage=gate_result.stage,
                                   failure_kind=gate_result.failure_kind)
                render_required = False
            if runtime_calls >= cfg.max_runtime_repairs:
                return _finish(output, report, "render_failed", final_program=program)
            assert gate_result is not None
            packet = _render_feedback(cfg.task, copied_references, gate_result, gate_dir)
            repair_dir = round_dir / "runtime-repairs" / f"repair-{runtime_calls:03d}"
            calls_before = report["model_calls"]["runtime_repair"]
            try:
                proposal = repair_call(program.source, tracked_generator("runtime_repair"), packet, repair_dir,
                                       previous_attempts=runtime_memory,
                                       remaining_calls=cfg.max_runtime_repairs-runtime_calls)
            except Exception as exc:
                return _finish(output, report, "generation_failed", final_program=program,
                               phase="runtime_repair", error_type=type(exc).__name__)
            runtime_calls += 1
            record_proposal_if_untracked("runtime_repair", calls_before, proposal)
            runtime_memory.append(_proposal_memory(proposal, runtime_calls - 1))
            if proposal.outcome == "stop":
                return _finish(output, report, "render_failed", final_program=program,
                               termination="generator_stop")
            if proposal.source is not None:
                candidate = ObjectProgram(proposal.source)
                if candidate.sha != program.sha:
                    program = candidate
                    (round_dir / "candidate.ts").write_text(program.source, encoding="utf-8")
                    gate_index += 1
                    render_required = True

        verifier_dir = round_dir / "verifier"
        try:
            verification = verifier_call(program, verifier_dir, render, views, cfg.verifier,
                                         task=cfg.task, references=copied_references,
                                         model=judge_model, model_call=judge_call)
        except Exception as exc:
            return _finish(output, report, "verification_error", final_program=program,
                           error_type=type(exc).__name__)
        judge_calls = verification.get("model_calls", [])
        report["model_calls"]["judge"] += len(judge_calls)
        report["total_tokens"] += sum(usage_tokens(call.get("usage", {})) for call in judge_calls)
        round_record["verifier"] = {
            "status": verification.get("status"),
            "termination": verification.get("termination"),
            "verdict": verification.get("verdict"),
            "spent": verification.get("spent"),
            "artifact": str(verifier_dir.relative_to(output) / "result.json"),
        }
        write_json_atomic(output / "result.json", report)
        if verification.get("status") != "complete":
            return _finish(output, report, "verification_error", final_program=program,
                           termination=verification.get("termination"))
        verdict = verification.get("verdict", {})
        if verdict.get("status") == "pass":
            return _finish(output, report, "accepted", accepted=True, final_program=program)
        if verdict.get("status") == "uncertain":
            return _finish(output, report, "verification_uncertain", final_program=program)
        if verdict.get("status") != "fail":
            return _finish(output, report, "verification_error", final_program=program,
                           termination="invalid_verdict")
        packet = _verification_feedback(cfg.task, copied_references, verification, verifier_dir)
        revised = None
        while visual_calls < cfg.max_visual_revisions:
            repair_dir = round_dir / "visual-revisions" / f"revision-{visual_calls:03d}"
            calls_before = report["model_calls"]["visual_revision"]
            try:
                proposal = repair_call(program.source, tracked_generator("visual_revision"), packet, repair_dir,
                                       previous_attempts=visual_memory,
                                       remaining_calls=cfg.max_visual_revisions-visual_calls)
            except Exception as exc:
                return _finish(output, report, "generation_failed", final_program=program,
                               phase="visual_revision", error_type=type(exc).__name__)
            attempt_index = visual_calls
            visual_calls += 1
            record_proposal_if_untracked("visual_revision", calls_before, proposal)
            round_record["visual_revisions"].append({
                "outcome": proposal.outcome,
                "error": proposal.error,
                "artifact": str(repair_dir.relative_to(output) / "result.json"),
            })
            visual_memory.append(_proposal_memory(proposal, attempt_index))
            write_json_atomic(output / "result.json", report)
            if proposal.outcome == "stop":
                return _finish(output, report, "max_visual_revisions", final_program=program,
                               termination="generator_stop")
            if proposal.source is not None:
                revised = ObjectProgram(proposal.source)
                break
        if revised is None:
            return _finish(output, report, "max_visual_revisions", final_program=program,
                           termination="revision_limit")
        program = revised
        round_index += 1


def _render_feedback(task: str, references: tuple[Path, ...], result: RenderGateResult,
                     gate_dir: Path) -> Feedback:
    images = list(references)
    observations = []
    if result.image_ref:
        rendered = gate_dir / result.image_ref
        images.append(rendered)
        observations.append({"step": 0, "kind": "render_gate", "image_ref": rendered.name,
                             "stage": result.stage, "image_stddev": result.image_stddev})
    public = {"task": task, "phase": "runtime_repair", "render_gate": dataclasses.asdict(result),
              "observations": observations}
    return Feedback(public, tuple(images), len(observations))


def _verification_feedback(task: str, references: tuple[Path, ...], result: dict,
                           verifier_dir: Path) -> Feedback:
    verdict = result["verdict"]
    cited = {step for finding in verdict.get("findings", []) for step in finding.get("evidence_steps", [])}
    observations = []
    images = list(references)
    for observation in result.get("trajectory", []):
        if observation.get("step") not in cited:
            continue
        item = dict(observation)
        image_ref = item.get("image_ref")
        if image_ref:
            image = verifier_dir / image_ref
            images.append(image)
            item["image_ref"] = image.name
        observations.append(item)
    public = {"task": task, "phase": "visual_revision", "verdict": verdict,
              "termination": result.get("termination"), "observations": observations}
    return Feedback(public, tuple(images), len(observations))


def _proposal_memory(proposal: RepairProposal, index: int) -> dict:
    return {"attempt": index, "outcome": proposal.outcome, "error": proposal.error,
            "response": proposal.response}


def _copy_references(references: tuple[Path, ...], output: Path) -> tuple[Path, ...]:
    copied = []
    for index, source in enumerate(references):
        target = output / f"reference-{index:02d}{source.suffix.lower()}"
        shutil.copyfile(source, target)
        copied.append(target)
    return tuple(copied)


def _file_record(path: Path, root: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path.relative_to(root)), "sha256": hashlib.sha256(data).hexdigest()}


def _finish(output: Path, report: dict, status: str, *, accepted: bool = False,
            final_program: ObjectProgram | None = None, **fields) -> dict:
    report.update(status=status, accepted=accepted, finished_at=now_iso(), **fields)
    if final_program is not None:
        (output / "final.ts").write_text(final_program.source, encoding="utf-8")
        report["final_program"] = {"path": "final.ts", "sha": final_program.sha}
    write_json_atomic(output / "result.json", report)
    return report
