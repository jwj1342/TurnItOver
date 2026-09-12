"""YAML -> frozen config dataclasses. No framework; explicit fields only."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from turnitover.core.actions import Action, ActuateJoint, Detent, QueryRuntime, RequestView, RuntimeProperty, Stop, ViewId
from turnitover.render.session import RenderConfig


@dataclass(frozen=True)
class GenerateConfig:
    run_id: str
    seed: int
    n_samples: int
    asset_source: dict[str, Any]
    corruptions: tuple[str, ...]
    observation_script: tuple[Action, ...]
    evidence_states: tuple[str, ...]
    checkers: dict[str, dict]
    render: RenderConfig
    views_path: Path
    output_dir: Path
    budget: int | None = None
    config_path: Path | None = None
    corruption_params: dict[str, dict] = field(default_factory=dict)
    clean_fraction: float = 0.0
    max_corruptions: int = 1

    def __post_init__(self):
        if self.n_samples <= 0 or not 0 <= self.clean_fraction <= 1:
            raise ValueError("n_samples must be positive and clean_fraction must be in [0, 1]")
        if self.max_corruptions < 1 or len(set(self.corruptions)) != len(self.corruptions):
            raise ValueError("max_corruptions must be positive and corruption IDs unique")
        if self.clean_fraction < 1 and self.max_corruptions > len(self.corruptions):
            raise ValueError("max_corruptions exceeds the number of configured corruption types")


def parse_action(d: dict) -> Action:
    t = d["type"]
    if t == "request_view":
        return RequestView(ViewId(d["view_id"]))
    if t == "actuate_joint":
        return ActuateJoint(d["joint_id"], Detent(d["detent"]))
    if t == "query_runtime":
        return QueryRuntime(RuntimeProperty(d["property"]))
    if t == "stop":
        return Stop()
    raise ValueError(f"unknown action type {t!r}")


def _resolve(root: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else root / path


def load_generate_config(path: Path, repo_root: Path | None = None) -> GenerateConfig:
    root = repo_root or Path(__file__).resolve().parents[1]
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    r = raw.get("render", {})
    render = RenderConfig(
        harness_dir=_resolve(root, r.get("harness_dir", "web/dist")),
        esbuild_bin=_resolve(root, r.get("esbuild_bin", "web/node_modules/.bin/esbuild")),
        width=int(r.get("width", 512)),
        height=int(r.get("height", 512)),
        browser_restart_every=int(r.get("browser_restart_every", 200)),
    )
    run_id = str(raw["run_id"])
    output_dir = _resolve(root, str(raw.get("output_dir", "data/runs/${run_id}")).replace("${run_id}", run_id))
    asset_source = dict(raw.get("asset_source", {"kind": "toy"}))
    if "manifest" in asset_source:
        asset_source["manifest"] = str(_resolve(root, asset_source["manifest"]))
    return GenerateConfig(
        run_id=run_id,
        seed=int(raw["seed"]),
        n_samples=int(raw["n_samples"]),
        asset_source=asset_source,
        corruptions=tuple(raw["corruptions"]),
        observation_script=tuple(parse_action(a) for a in raw["observation_script"]),
        evidence_states=tuple(raw.get("evidence_states", ["rest"])),
        checkers={k: (v or {}) for k, v in raw.get("checkers", {}).items()},
        render=render,
        views_path=_resolve(root, raw.get("views", "configs/views.yaml")),
        output_dir=output_dir,
        budget=raw.get("budget"),
        config_path=path,
        corruption_params={k: (v or {}) for k, v in raw.get("corruption_params", {}).items()},
        clean_fraction=float(raw.get("clean_fraction", 0.0)),
        max_corruptions=int(raw.get("max_corruptions", 1)),
    )
