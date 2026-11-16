"""Judge conditions for the repair table: each one turns the current program into Feedback for the generator.

A judge never sees the private corruption record except the gold judge, which is the upper bound.
The sweep renders the reference and the candidate at the same viewpoints and joint states with the case's fixed
framing, in the order and naming of the three-tier-900 training export, so that fine-tuned verifiers are queried
in-distribution. Every judge hands the generator the same images, the case's observation group (per sweep cell the
reference then the candidate); judges differ only in their own section of the feedback (verdict or pixel report).
Renders are cached by (sweep, framing, program hash), verdicts by (model identity, prompt, image bytes).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from PIL import Image, ImageChops, ImageStat

from turnitover.core.actions import Detent
from turnitover.core.program import ObjectProgram
from turnitover.core.sample import Framing
from turnitover.render.session import ObservationSession
from turnitover.repair.loop import Feedback
from turnitover.taxonomy import load_taxonomy

PROMPT_900 = (Path(__file__).parent / "prompt_900.txt").read_text()   # her export's prompt text (code900), adds joint_id
PROMPT_PREFIX = PROMPT_900.rstrip("\n") + "\n\n"                         # one blank line before the JSON, as in the export
FRAMING_900 = Path(__file__).parent / "framing_900.json"               # per-case export framing, scripts/exp900/derive_pack_framing.py
# Train-fitted status threshold of scripts/exp900/pixel_status_baseline.py (same score: mean absolute RGB difference in
# [0,1] of one reference/candidate pair), from tio-runs/out/level900-base-2B4B-26612686/pixel-status/metrics.json.
PIXEL_THRESHOLD = 0.0010002983617596328
STATES = ("rest", "mid", "limit")
TIER_STATES = {"easy": ("rest", "limit"), "medium": STATES, "hard": STATES}
TASK = "Repair deviations from the reference while preserving task joint identities."
FIXES = {
    "kinematics.joint_axis": ("The part rotates about the wrong axis.", "Set the axis of joint {joint} to the reference direction."),
    "kinematics.direction_flip": ("The part moves in the opposite direction from the reference.", "Negate the axis or swap the limits of joint {joint}."),
    "kinematics.joint_anchor": ("The part rotates about an offset pivot.", "Move the anchor of joint {joint} back onto the hinge line."),
    "kinematics.joint_type": ("The joint has the wrong type.", "Change joint {joint} to the reference joint type."),
    "kinematics.joint_range": ("The joint range differs from the reference.", "Set the limits of joint {joint} to the reference range."),
    "structure.part_offset": ("The part is displaced from its reference position.", "Translate {parts} back to the reference position."),
    "structure.part_rotation": ("The part is rotated relative to the reference.", "Reset the rotation of {parts} to the reference."),
}


@dataclass(frozen=True)
class Sweep:
    views: tuple[str, ...]
    joints: tuple[str, ...]
    states: tuple[str, ...]
    values: tuple[tuple[str, float], ...] = ()   # explicit joint values per state (her exports); else detents

    @property
    def key(self) -> str:
        return hashlib.sha256(json.dumps(dataclasses.asdict(self), sort_keys=True).encode()).hexdigest()[:12]


@dataclass(frozen=True)
class Cell:
    view: str
    state: str
    kind: str            # reference | candidate
    name: str            # image file name in training convention
    path: Path


@dataclass
class Case:
    uid: str
    tier: str
    initial: str
    reference: ObjectProgram
    framing: Framing | None
    labels: list[dict]
    sweep: Sweep
    grid: list[dict] = field(default_factory=list)
    exposures: list[list[int]] = field(default_factory=list)
    parts: tuple[str, ...] = ()      # part and joint order of the training export's prompt context; empty: harness order
    joints: tuple[str, ...] = ()

    @staticmethod
    def load(folder: Path, default_views: tuple[str, ...] = ("front",)) -> "Case":
        """One case directory as prepare_repair_experiment.py writes it, plus optional tier and sweep fields."""
        data = json.loads((folder / "case.private.json").read_text())
        annotation = data["annotation"]
        framing = Framing(**data["framing"]) if data.get("framing") else None
        tier = data.get("tier") or annotation.get("tier") or "medium"
        spec = data.get("sweep") or annotation.get("sweep") or {}
        joints = tuple(spec.get("joints") or sorted({l["params"]["joint"] for l in annotation["labels"]
                                                     if isinstance(l.get("params"), dict) and l["params"].get("joint")}))
        sweep = Sweep(tuple(spec.get("views") or default_views), joints, tuple(spec.get("states") or TIER_STATES[tier]))
        return Case(annotation["id"], tier, (folder / "initial.ts").read_text(),
                    ObjectProgram((folder / "reference.private.ts").read_text()), framing, annotation["labels"], sweep,
                    annotation.get("grid", []), annotation.get("exposures", []))


PACK_FORMAT = "three-tier-900-validation-repair-programs-v1"


def exposing_cells_by_defect(episode: dict) -> dict[str, set[str]]:
    """defect_id -> candidate image names (basename) cited by the evidence-constrained target of a 900 episode."""
    content = episode["messages"][0]["content"]
    payload = json.loads(content[content.index('{"context"'):])
    step_image = {h["step"]: h["image_ref"].split("/")[-1] for h in payload["history"]}
    out: dict[str, set[str]] = {}
    for finding in json.loads(episode["messages"][-1]["content"])["verdict"]["findings"]:
        for step in finding["evidence_steps"]:
            out.setdefault(finding["defect_id"], set()).add(step_image[step])
    return out


def load_pack(root: Path, labels_path: Path | None = None, episodes_path: Path | None = None,
              framing_path: Path = FRAMING_900) -> list[Case]:
    """Her 58-case repair pack (manifest format three-tier-900-validation-repair-programs-v1).
    Labels come from the 900 export's annotations.private.jsonl, per-cell exposure from validation.episodes.jsonl,
    the fixed framing of her images from framing_path, the prompt's part/joint order from the reference AssetSpec."""
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("format") != PACK_FORMAT:
        raise ValueError(f"unexpected pack format: {manifest.get('format')}")
    framings = json.loads(Path(framing_path).read_text())["cases"]
    labels: dict[str, list[dict]] = {}
    if labels_path:
        for line in Path(labels_path).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                labels[row["id"]] = row["labels"]
    exposing: dict[str, dict[str, set[str]]] = {}
    if episodes_path:
        for line in Path(episodes_path).read_text().splitlines():
            if line.strip():
                episode = json.loads(line)
                exposing[episode["id"].split(":")[0]] = exposing_cells_by_defect(episode)
    cases = []
    for entry in json.loads((root / "cases.json").read_text()):
        uid = entry["case_id"]
        texts = {}
        for kind in ("candidate", "reference"):
            program = entry["programs"][kind]
            text = (root / program["typescript"]).read_text()
            if hashlib.sha256(text.encode()).hexdigest() != program["typescript_sha256"]:
                raise ValueError(f"program hash mismatch: {uid} {kind}")
            texts[kind] = text
        states = tuple(s for s in STATES if s in {c["state"] for c in entry["observation_cells"]})
        values = tuple((s, float(entry["state_values"][s])) for s in states if s != "rest")
        sweep = Sweep(tuple(entry["views"]), (entry["target_joint_id"],), states, values)
        case_labels = labels.get(uid, [])
        grid, exposures = [], []
        cited = exposing.get(uid, {})
        for cell in entry["observation_cells"]:
            grid.append({"view": cell["view_id"], "joint": entry["target_joint_id"] if cell["state"] != "rest" else None,
                         "detent": None if cell["state"] == "rest" else cell["state"]})
            name = cell["candidate_image"].split("/")[-1]
            exposures.append([i for i, l in enumerate(case_labels) if name in cited.get(l["defect_id"], ())])
        spec = json.loads((root / entry["programs"]["reference"]["spec"]).read_text())
        framing = Framing(tuple(framings[uid]["center"]), framings[uid]["radius"])
        cases.append(Case(uid, entry["difficulty"], texts["candidate"], ObjectProgram(texts["reference"]), framing,
                          case_labels, sweep, grid, exposures,
                          tuple(p["id"] for p in spec["parts"]), tuple(j["id"] for j in spec["joints"])))
    return cases


def cell_name(index: int, view: str, kind: str, state: str, single_view: bool) -> str:
    return f"{index:02d}-{kind}_{state}.png" if single_view else f"{index:02d}-{view}_{kind}_{state}.png"


def render_program(session: ObservationSession, program: ObjectProgram, framing: Framing | None,
                   sweep: Sweep) -> tuple[dict, list[bytes]]:
    """Render every sweep cell of one program, view-major then state order; returns the load info and the PNGs."""
    info = session.load(program, framing=framing)
    joints = sweep.joints or tuple(info.joints)
    pngs = []
    try:
        for view in sweep.views:
            for state in sweep.states:
                session.reset_joints()
                png = session.request_view(view).png
                if state != "rest":
                    value = dict(sweep.values).get(state)
                    for joint in joints:
                        png = (session.actuate_joint(joint, value) if value is not None else session.actuate(joint, Detent(state))).png
                pngs.append(png)
    finally:
        session.dispose()
    return dict(parts=list(info.parts), joints=list(info.joints), framing=dataclasses.asdict(info.framing)), pngs


def sweep_layout(case: Case, folders: dict[str, Path]) -> list[Cell]:
    """Sweep cells in the training order (per view, references then candidates), named as in the training export."""
    single = len(case.sweep.views) == 1
    cells, index = [], 0
    for view in case.sweep.views:
        for kind in ("reference", "candidate"):
            for state in case.sweep.states:
                name = cell_name(index, view, kind, state, single)
                cells.append(Cell(view, state, kind, name, folders[kind] / name))
                index += 1
    return cells


def sweep_cells(session: ObservationSession, case: Case, candidate: str, cache: Path) -> tuple[list[Cell], dict]:
    """Reference and candidate renders of the sweep with the case's framing; names are unique across both programs.
    Cached per (sweep, framing, program, kind)."""
    program = ObjectProgram(candidate)
    framing = hashlib.sha256(json.dumps(dataclasses.asdict(case.framing)).encode()).hexdigest()[:12] if case.framing else "auto"
    root = cache / "renders" / f"{case.sweep.key}-{framing}"
    folders = {"reference": root / case.reference.sha / "reference", "candidate": root / program.sha / "candidate"}
    cells = sweep_layout(case, folders)
    for kind, source in (("reference", case.reference), ("candidate", program)):
        folder = folders[kind]
        if (folder / "info.json").exists():
            continue
        folder.mkdir(parents=True, exist_ok=True)
        info, pngs = render_program(session, source, case.framing, case.sweep)
        for cell, png in zip([c for c in cells if c.kind == kind], pngs, strict=True):
            cell.path.write_bytes(png)
        (folder / "info.json").write_text(json.dumps(info))
    info = json.loads((folders["candidate"] / "info.json").read_text())
    return cells, info


def image_ref(case: Case, cell: Cell) -> str:
    return f"images/{case.uid}/{cell.name}"


def judge_prompt(case: Case, info: dict, cells: list[Cell]) -> tuple[str, list[Path]]:
    """The three-tier-900 training prompt: references first, every later image is a request_view step."""
    context = {"task": "compare reference and candidate articulation", "parts": list(case.parts or info["parts"]),
               "joints": list(case.joints or info["joints"]),
               "views": list(case.sweep.views), "runtime_properties": [], "action_types": ["request_view"],
               "max_triangles": 1, "max_draw_calls": 1}
    history = [{"step": step, "image_ref": image_ref(case, cell), "action": {"type": "request_view", "view_id": cell.view}}
               for step, cell in enumerate(cells[1:])]
    payload = {"context": context, "remaining": 0, "history": history, "image_order": [image_ref(case, c) for c in cells],
               "taxonomy_ids": list(load_taxonomy().ids), "training_objective": "comprehensive_within_budget"}
    return PROMPT_PREFIX + json.dumps(payload), [c.path for c in cells]


def validate_verdict(text: str, parts: list[str], steps: set[int]) -> tuple[dict, str | None]:
    """Method-section rules. Findings may carry joint_id (her export format). Invalid -> uncertain with the reason."""
    def uncertain(reason: str) -> dict:
        return {"status": "uncertain", "confidence": 0.0, "findings": [], "summary": "The judge returned an invalid verdict.",
                "limitations": [reason]}
    raw = text.strip()
    if raw.startswith("```json\n") and raw.endswith("```"):
        raw = raw[8:-3].strip()
    try:
        value = json.loads(raw)
    except ValueError:
        return uncertain("invalid JSON"), "invalid JSON"
    if not isinstance(value, dict) or set(value) != {"verdict"} or not isinstance(value["verdict"], dict):
        return uncertain("expected one verdict object"), "expected one verdict object"
    v = value["verdict"]
    status = v.get("status")
    findings = v.get("findings")
    if status not in ("pass", "fail", "uncertain") or not isinstance(findings, list):
        return uncertain("invalid status or findings"), "invalid status or findings"
    ids = set(load_taxonomy().ids)
    clean = []
    for f in findings:
        if not isinstance(f, dict) or f.get("defect_id") not in ids:
            return uncertain("unknown defect id"), "unknown defect id"
        fparts = f.get("parts")
        if not isinstance(fparts, list) or not set(fparts) <= set(parts):
            return uncertain("unknown part id"), "unknown part id"
        ev = f.get("evidence_steps")
        if not isinstance(ev, list) or not ev or any(type(s) is not int for s in ev) or not set(ev) <= steps:
            return uncertain("finding cites a step that does not exist"), "finding cites a step that does not exist"
        clean.append({"defect_id": f["defect_id"], "parts": list(fparts), "joint_id": f.get("joint_id"),
                      "severity": f.get("severity"), "confidence": f.get("confidence"), "evidence_steps": sorted(set(ev)),
                      "description": str(f.get("description", "")), "suggested_fix": str(f.get("suggested_fix", ""))})
    if status == "pass" and (clean or not steps):
        return uncertain("pass with findings or without observations"), "pass with findings or without observations"
    if status == "fail" and not clean:
        return uncertain("fail without findings"), "fail without findings"
    verdict = {"status": status, "confidence": v.get("confidence"), "findings": clean, "summary": str(v.get("summary", "")),
               "limitations": [str(x) for x in v.get("limitations", [])] if isinstance(v.get("limitations"), list) else []}
    return verdict, None


class JudgeModel:
    """Qwen3-VL base with an optional LoRA adapter, greedy decoding, loaded on first use."""

    def __init__(self, base: Path, adapter: Path | None = None, max_new_tokens: int = 512):
        self.base, self.adapter, self.max_new_tokens = Path(base), Path(adapter) if adapter else None, max_new_tokens
        self.model = self.processor = None
        self.identity = {"base": self.base.name, "adapter": str(self.adapter) if self.adapter else None,
                         "max_new_tokens": max_new_tokens}
        if self.adapter:   # content, not path: jobs copy the checkpoint to node-local disk
            digest = hashlib.sha256()
            for name in ("adapter_config.json", "adapter_model.safetensors"):
                digest.update((self.adapter / name).read_bytes())
            self.identity["adapter_sha256"] = digest.hexdigest()

    def _load(self):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        source = self.adapter if self.adapter and (self.adapter / "preprocessor_config.json").exists() else self.base
        self.processor = AutoProcessor.from_pretrained(source, local_files_only=True, trust_remote_code=False)
        model = Qwen3VLForConditionalGeneration.from_pretrained(self.base, local_files_only=True, trust_remote_code=False,
                                                                torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")   # CPU only for reruns without a free GPU
        if self.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, self.adapter, local_files_only=True)
        self.model = model.eval()

    def __call__(self, prompt: str, images: list[Path]) -> tuple[str, dict]:
        import torch
        if self.model is None:
            self._load()
        content = [{"type": "image", "image": str(p.resolve())} for p in images] + [{"type": "text", "text": prompt}]
        inputs = self.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=True,
                                                    add_generation_prompt=True, return_dict=True, return_tensors="pt").to(self.model.device)
        start = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        new = output[:, inputs["input_ids"].shape[1]:]
        text = self.processor.batch_decode(new, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return text, {"input_tokens": int(inputs["input_ids"].shape[1]), "output_tokens": int(new.shape[1]),
                      "elapsed_ms": (time.perf_counter() - start) * 1000}


class ApiJudgeModel:
    """A judge model behind an API (the TIO_JUDGE_* settings, e.g. a large open VLM served by vLLM), called with the same
    prompt and images as the local judges. `call(config, prompt, images) -> ModelResult` is the caller's retrying
    completion; a ModelError it raises ends the episode like a generator failure (status generator_error, rerun on resume)."""

    def __init__(self, config, call: Callable):
        self.config, self.call = config, call
        self.identity = {"api_model": config.model, "provider": config.provider, "max_tokens": config.max_tokens,
                         "chat_template_kwargs": config.chat_template_kwargs}

    def __call__(self, prompt: str, images: list[Path]) -> tuple[str, dict]:
        result = self.call(self.config, prompt, images)
        return result.text, {**result.usage, "finish_reason": result.finish_reason, "elapsed_ms": result.elapsed_ms}


def pixel_difference(reference: Path, candidate: Path) -> float:
    with Image.open(reference) as a, Image.open(candidate) as b:
        a, b = a.convert("RGB"), b.convert("RGB")
        if a.size != b.size:
            raise ValueError("paired renders differ in size")
        return sum(ImageStat.Stat(ImageChops.difference(a, b)).mean) / (3 * 255.0)


def observation_feedback(case: Case, cells: list[Cell], section: dict) -> Feedback:
    """What every judge gives the generator: the case's observation group, per sweep cell (view-major, then state)
    the reference image then the candidate image, plus the judge's own section. Observation cost is one per cell."""
    by_key = {(c.view, c.state, c.kind): c for c in cells}
    group = [by_key[(view, state, kind)] for view in case.sweep.views for state in case.sweep.states
             for kind in ("reference", "candidate")]
    public = {"task": TASK, "observations": [{"view": c.view, "state": c.state, "kind": c.kind, "image_ref": c.name} for c in group],
              "reference_condition": "reference_then_candidate_per_cell", **section}
    return Feedback(public, tuple(c.path for c in group), len(group) // 2)


def verdict_section(verdict: dict, shown: list[Cell], folder: Path) -> dict:
    """The verdict as validated, plus the observation cell behind every step it cites (step k is the judge's image k+1)."""
    (folder / "verdict.json").write_text(json.dumps(verdict, indent=2))
    cited = sorted({s for f in verdict["findings"] for s in f["evidence_steps"]})
    return {"verdict": verdict, "evidence_step_cells": [{"step": s, "view": shown[s + 1].view, "state": shown[s + 1].state,
                                                         "kind": shown[s + 1].kind} for s in cited]}


class Judge:
    """Base: bind(session, case, cache) -> observe(source, gold, folder)."""
    judge_id = "base"

    def describe(self) -> dict:
        return {"judge": self.judge_id}

    def bind(self, session: ObservationSession, case: Case, cache: Path) -> Callable[[str, dict, Path], Feedback]:
        raise NotImplementedError


class NoneJudge(Judge):
    judge_id = "none"

    def bind(self, session, case, cache):
        def observe(source: str, gold: dict, folder: Path) -> Feedback:
            cells, _ = sweep_cells(session, case, source, cache)
            return observation_feedback(case, cells, {})
        return observe


class PixelJudge(Judge):
    judge_id = "pixel"

    def __init__(self, threshold: float = PIXEL_THRESHOLD):
        self.threshold = threshold

    def describe(self):
        return {"judge": self.judge_id, "threshold": self.threshold}

    def bind(self, session, case, cache):
        def observe(source: str, gold: dict, folder: Path) -> Feedback:
            cells, _ = sweep_cells(session, case, source, cache)
            by_key = {(c.view, c.state, c.kind): c for c in cells}
            report = []
            for view in case.sweep.views:
                for state in case.sweep.states:
                    diff = pixel_difference(by_key[(view, state, "reference")].path, by_key[(view, state, "candidate")].path)
                    report.append({"view": view, "state": state, "differs": diff > self.threshold, "difference": round(diff, 6)})
            (folder / "pixel.json").write_text(json.dumps(report, indent=2))
            return observation_feedback(case, cells, {"pixel_difference": {
                "threshold": self.threshold, "cells": report,
                "summary": "Cells (view, state) whose candidate rendering differs from the reference. No defect type or part is identified."}})
        return observe


class VlmJudge(Judge):
    """Zero-shot or fine-tuned Qwen3-VL on the single rest screenshot or the full sweep; verdicts cached by
    (model identity, prompt, image bytes)."""

    def __init__(self, judge_id: str, model: JudgeModel, mode: str = "sweep"):
        if mode not in ("screenshot", "sweep"):
            raise ValueError("mode must be screenshot or sweep")
        self.judge_id, self.model, self.mode = judge_id, model, mode

    def describe(self):
        return {"judge": self.judge_id, "mode": self.mode, **self.model.identity}

    def bind(self, session, case, cache):
        def observe(source: str, gold: dict, folder: Path) -> Feedback:
            cells, info = sweep_cells(session, case, source, cache)
            if self.mode == "screenshot":
                first = cells[0]
                rest = next(c for c in cells if c.kind == "candidate" and c.state == "rest" and c.view == first.view)
                name = cell_name(1, rest.view, "candidate", "rest", len(case.sweep.views) == 1)
                (folder / name).write_bytes(rest.path.read_bytes())     # step 0 of a two-image prompt
                shown = [first, dataclasses.replace(rest, name=name, path=folder / name)]
            else:
                shown = cells
            prompt, images = judge_prompt(case, info, shown)
            digest = hashlib.sha256(json.dumps(self.model.identity, sort_keys=True).encode())
            digest.update(b"\0"); digest.update(prompt.encode())
            for path in images:
                digest.update(b"\0"); digest.update(hashlib.sha256(path.read_bytes()).digest())
            key = cache / "verdicts" / self.judge_id / f"{digest.hexdigest()}.json"
            if key.exists():
                record = json.loads(key.read_text())
                record["cached"] = True
            else:
                text, usage = self.model(prompt, images)
                record = {"prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "response": text, "usage": usage,
                          "model": self.model.identity, "cached": False}
                key.parent.mkdir(parents=True, exist_ok=True)
                key.write_text(json.dumps(record))
            steps = set(range(len(shown) - 1))
            verdict, error = validate_verdict(record["response"], info["parts"], steps)
            (folder / "judge_call.json").write_text(json.dumps({**record, "validation_error": error, "images": [c.name for c in shown]}, indent=2))
            (folder / "judge_prompt.txt").write_text(prompt)
            return observation_feedback(case, cells, verdict_section(verdict, shown, folder))
        return observe


class GoldJudge(Judge):
    """Findings built from the private corruption record: the upper bound on what feedback can contribute."""
    judge_id = "gold"

    def bind(self, session, case, cache):
        if not case.labels:
            raise ValueError(f"the gold judge needs the case's defect labels (--labels): {case.uid}")

        def observe(source: str, gold: dict, folder: Path) -> Feedback:
            cells, info = sweep_cells(session, case, source, cache)
            # steps whose candidate cell exposes the label, from the annotation grid when it covers the sweep
            exposed_by_label: dict[int, set[int]] = {}
            cell_index = {(g["view"], g["detent"] or "rest"): i for i, g in enumerate(case.grid)} if case.grid else {}
            for step, cell in enumerate(cells[1:]):
                if cell.kind != "candidate":
                    continue
                g = cell_index.get((cell.view, cell.state))
                if g is not None and g < len(case.exposures):
                    for label_index in case.exposures[g]:
                        exposed_by_label.setdefault(label_index, set()).add(step)
            candidate_steps = [i for i, c in enumerate(cells[1:]) if c.kind == "candidate"]
            findings = []
            failing_parts = {p for c in gold.get("checks", []) for i in c.get("invariants", []) if not i.get("passed")
                             for p in i.get("parts", ())}
            labels = [] if gold.get("passed") else [l for l in case.labels if not failing_parts or set(l.get("parts", ())) & failing_parts]
            if not gold.get("passed") and not labels:
                labels = list(case.labels)
            for index, label in enumerate(case.labels):
                if label not in labels:
                    continue
                defect = label["defect_id"]
                joint = (label.get("params") or {}).get("joint") if isinstance(label.get("params"), dict) else None
                parts = [p for p in label.get("parts", []) if p in info["parts"]]
                description, fix = FIXES.get(defect, ("The part deviates from the reference.", "Edit {parts} to match the reference."))
                steps = sorted(exposed_by_label.get(index) or candidate_steps)
                findings.append({"defect_id": defect, "parts": parts, "joint_id": joint if defect.startswith("kinematics") else None,
                                 "severity": float(label.get("severity", 0.5)), "confidence": 1.0, "evidence_steps": steps,
                                 "description": description, "suggested_fix": fix.format(joint=joint or "the joint", parts=", ".join(parts) or "the part")})
            verdict = {"status": "fail" if findings else "pass", "confidence": 1.0, "findings": findings,
                       "summary": "Findings from the corruption record and the privileged audit." if findings else "The audit passes.",
                       "limitations": ["Gold findings are an upper bound and never available to a deployed verifier."]}
            return observation_feedback(case, cells, verdict_section(verdict, cells, folder))
        return observe


def judge_from_id(judge_id: str, model_root: Path | None, adapters: dict[str, Path],
                  api_model: ApiJudgeModel | None = None) -> Judge:
    if judge_id == "none":
        return NoneJudge()
    if judge_id == "pixel":
        return PixelJudge()
    if judge_id == "gold":
        return GoldJudge()
    if judge_id.startswith(("zs2b_", "zs4b_", "zs8b_")):
        if model_root is None:
            raise ValueError("--model-root is required for VLM judges")
        size, mode = judge_id[2:4].upper(), judge_id[5:]
        return VlmJudge(judge_id, JudgeModel(model_root / f"Qwen3-VL-{size}-Instruct"), mode)
    if judge_id.startswith("zsapi_"):
        if api_model is None:
            raise ValueError("zsapi_ judges need the TIO_JUDGE_* settings")
        return VlmJudge(judge_id, api_model, judge_id.removeprefix("zsapi_"))
    if judge_id.startswith("ours_"):
        size = judge_id.split("_")[1].upper()
        if judge_id not in adapters or model_root is None:
            raise ValueError(f"--adapter {judge_id}=<checkpoint> and --model-root are required")
        return VlmJudge(judge_id, JudgeModel(model_root / f"Qwen3-VL-{size}-Instruct", adapters[judge_id]), "sweep")
    raise ValueError(f"Unknown judge: {judge_id}")
