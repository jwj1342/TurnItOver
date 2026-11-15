"""Judge conditions for the repair table: each one turns the current program into Feedback for the generator.

A judge never sees the private corruption record except the gold judge, which is the upper bound.
The sweep renders the reference and the candidate at the same viewpoints and joint states, in the
order and naming of the three-tier-900 training export, so that fine-tuned verifiers are queried
in-distribution. Renders and verdicts are cached by program hash so repeated rounds and judges share work.
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


def load_pack(root: Path, labels_path: Path | None = None, episodes_path: Path | None = None) -> list[Case]:
    """Her 58-case repair pack (manifest format three-tier-900-validation-repair-programs-v1).
    Labels come from the 900 export's annotations.private.jsonl, per-cell exposure from validation.episodes.jsonl."""
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("format") != PACK_FORMAT:
        raise ValueError(f"unexpected pack format: {manifest.get('format')}")
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
        cases.append(Case(uid, entry["difficulty"], texts["candidate"], ObjectProgram(texts["reference"]), None,
                          case_labels, sweep, grid, exposures))
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


def sweep_cells(session: ObservationSession, case: Case, candidate: str, cache: Path) -> tuple[list[Cell], dict]:
    """Reference and candidate renders in the training order (per view, references then candidates), named as in the
    training export so that feedback image names are unique. Cached per (sweep, framing, program, kind)."""
    program = ObjectProgram(candidate)
    root = cache / "renders" / f"{case.sweep.key}-{case.framing.radius if case.framing else 'auto'}"
    folders = {"reference": root / case.reference.sha / "reference", "candidate": root / program.sha / "candidate"}
    single = len(case.sweep.views) == 1
    cells, index = [], 0
    for view in case.sweep.views:
        for kind in ("reference", "candidate"):
            for state in case.sweep.states:
                name = cell_name(index, view, kind, state, single)
                cells.append(Cell(view, state, kind, name, folders[kind] / name))
                index += 1
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
    context = {"task": "compare reference and candidate articulation", "parts": info["parts"], "joints": info["joints"],
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

    def ident(self) -> dict:
        out = {"base": self.base.name, "adapter": str(self.adapter) if self.adapter else None}
        if self.adapter and (self.adapter / "adapter_model.safetensors").exists():
            out["adapter_sha256"] = hashlib.sha256((self.adapter / "adapter_model.safetensors").read_bytes()).hexdigest()
        return out

    def _load(self):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        source = self.adapter if self.adapter and (self.adapter / "preprocessor_config.json").exists() else self.base
        self.processor = AutoProcessor.from_pretrained(source, local_files_only=True, trust_remote_code=False)
        model = Qwen3VLForConditionalGeneration.from_pretrained(self.base, local_files_only=True, trust_remote_code=False,
                                                                torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
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


def pixel_difference(reference: Path, candidate: Path) -> float:
    with Image.open(reference) as a, Image.open(candidate) as b:
        a, b = a.convert("RGB"), b.convert("RGB")
        if a.size != b.size:
            raise ValueError("paired renders differ in size")
        return sum(ImageStat.Stat(ImageChops.difference(a, b)).mean) / (3 * 255.0)


def observation_entries(case: Case, cells: list[Cell], steps: list[int]) -> list[dict]:
    out = []
    for step in steps:
        cell = cells[step + 1] if step >= 0 else cells[0]
        out.append({"step": step, "view": cell.view, "state": cell.state, "kind": cell.kind, "image_ref": cell.name})
    return out


def verdict_feedback(case: Case, cells: list[Cell], verdict: dict, folder: Path, extra: dict | None = None) -> Feedback:
    """The verdict plus the images it cites, reference first; a verdict without findings shows the rest pair."""
    cited = sorted({s for f in verdict["findings"] for s in f["evidence_steps"]})
    if not cited:
        cited = [i for i, c in enumerate(cells[1:]) if c.kind == "candidate" and c.state == "rest" and c.view == cells[0].view][:1]
    images = [cells[0].path] + [cells[s + 1].path for s in cited]
    public = {"task": TASK, "verdict": verdict, "observations": observation_entries(case, cells, [-1] + cited),
              "reference_condition": "first_reference_image_then_cited_observations"}
    if extra:
        public.update(extra)
    (folder / "verdict.json").write_text(json.dumps(verdict, indent=2))
    return Feedback(public, tuple(images), len(cells) - 1)


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
            rest = next(i for i, c in enumerate(cells[1:]) if c.kind == "candidate" and c.state == "rest")
            public = {"task": TASK, "observations": observation_entries(case, cells, [-1, rest]),
                      "reference_condition": "reference_rest_then_candidate_rest"}
            return Feedback(public, (cells[0].path, cells[rest + 1].path), 1)
        return observe


class PixelJudge(Judge):
    judge_id = "pixel"

    def __init__(self, threshold: float = 0.002):
        self.threshold = threshold

    def describe(self):
        return {"judge": self.judge_id, "threshold": self.threshold}

    def bind(self, session, case, cache):
        def observe(source: str, gold: dict, folder: Path) -> Feedback:
            cells, _ = sweep_cells(session, case, source, cache)
            by_key = {(c.view, c.state, c.kind): (i, c) for i, c in enumerate(cells)}
            report, images, cited = [], [cells[0].path], []
            for view in case.sweep.views:
                for state in case.sweep.states:
                    ri, rc = by_key[(view, state, "reference")]
                    ci, cc = by_key[(view, state, "candidate")]
                    diff = pixel_difference(rc.path, cc.path)
                    differs = diff > self.threshold
                    report.append({"view": view, "state": state, "differs": differs, "difference": round(diff, 5),
                                   "image_ref": cc.name if differs else None})
                    if differs:
                        images.append(cc.path)
                        cited.append(ci - 1)
            public = {"task": TASK, "pixel_difference": {"threshold": self.threshold, "cells": report,
                                                         "summary": "Cells whose candidate rendering differs from the reference. No defect type or part is identified."},
                      "observations": observation_entries(case, cells, [-1] + cited),
                      "reference_condition": "first_reference_image_then_differing_candidate_cells"}
            (folder / "pixel.json").write_text(json.dumps(report, indent=2))
            return Feedback(public, tuple(images), len(cells) - 1)
        return observe


class VlmJudge(Judge):
    """Zero-shot or fine-tuned Qwen3-VL on the single rest screenshot or the full sweep; verdicts cached by program hash."""

    def __init__(self, judge_id: str, model: JudgeModel, mode: str = "sweep"):
        if mode not in ("screenshot", "sweep"):
            raise ValueError("mode must be screenshot or sweep")
        self.judge_id, self.model, self.mode = judge_id, model, mode

    def describe(self):
        return {"judge": self.judge_id, "mode": self.mode, **self.model.ident()}

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
            key = cache / "verdicts" / self.judge_id / f"{case.sweep.key}-{self.mode}" / f"{ObjectProgram(source).sha}.json"
            if key.exists():
                record = json.loads(key.read_text())
                record["cached"] = True
            else:
                text, usage = self.model(prompt, images)
                record = {"prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "response": text, "usage": usage,
                          "model": self.model.ident(), "cached": False}
                key.parent.mkdir(parents=True, exist_ok=True)
                key.write_text(json.dumps(record))
            steps = set(range(len(shown) - 1))
            verdict, error = validate_verdict(record["response"], info["parts"], steps)
            (folder / "judge_call.json").write_text(json.dumps({**record, "validation_error": error, "images": [c.name for c in shown]}, indent=2))
            (folder / "judge_prompt.txt").write_text(prompt)
            return verdict_feedback(case, shown, verdict, folder)
        return observe


class GoldJudge(Judge):
    """Findings built from the private corruption record: the upper bound on what feedback can contribute."""
    judge_id = "gold"

    def bind(self, session, case, cache):
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
            return verdict_feedback(case, cells, verdict, folder)
        return observe


def judge_from_id(judge_id: str, model_root: Path | None, adapters: dict[str, Path]) -> Judge:
    if judge_id == "none":
        return NoneJudge()
    if judge_id == "pixel":
        return PixelJudge()
    if judge_id == "gold":
        return GoldJudge()
    if judge_id.startswith("zs8b_"):
        if model_root is None:
            raise ValueError("--model-root is required for VLM judges")
        return VlmJudge(judge_id, JudgeModel(model_root / "Qwen3-VL-8B-Instruct"), judge_id.removeprefix("zs8b_"))
    if judge_id.startswith("ours_"):
        size = judge_id.split("_")[1].upper()
        if judge_id not in adapters or model_root is None:
            raise ValueError(f"--adapter {judge_id}=<checkpoint> and --model-root are required")
        return VlmJudge(judge_id, JudgeModel(model_root / f"Qwen3-VL-{size}-Instruct", adapters[judge_id]), "sweep")
    raise ValueError(f"Unknown judge: {judge_id}")
