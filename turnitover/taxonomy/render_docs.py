"""Render ``defects.yaml`` to Markdown (docs/taxonomy.md). Pure function + writer."""
from __future__ import annotations

from pathlib import Path

from turnitover.taxonomy import Taxonomy, load_taxonomy

HEADER = (
    "# Defect taxonomy\n\n"
    "GENERATED from `turnitover/taxonomy/defects.yaml` by `python -m turnitover render-docs`. "
    "Do not edit by hand. Source table: RP.md, section 方法/缺陷分类与数据构造.\n\n"
)


def render_markdown(tax: Taxonomy) -> str:
    lines = [HEADER]
    lines.append("| id | 层级 | 缺陷 | 注入方式 | 可见性类别 | 说明 | 标签形式 | 已实现 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for d in tax.defects:
        lines.append(
            f"| `{d.id}` | {d.layer.value} | {d.name_zh} | {d.injection} | {d.visibility.value} | "
            f"{d.visibility_note} | {d.label_form} | {'yes' if d.implemented else 'no'} |"
        )
    lines.append("")
    counts = {}
    for d in tax.defects:
        counts.setdefault(d.layer.value, {}).setdefault(d.visibility.value, 0)
        counts[d.layer.value][d.visibility.value] += 1
    lines.append("## Visibility by layer\n")
    lines.append("| layer | " + " | ".join(v for v in _visibility_order()) + " |")
    lines.append("|---|" + "---|" * len(_visibility_order()))
    for layer, per in counts.items():
        lines.append(f"| {layer} | " + " | ".join(str(per.get(v, 0)) for v in _visibility_order()) + " |")
    lines.append("")
    return "\n".join(lines)


def _visibility_order() -> tuple[str, ...]:
    from turnitover.taxonomy import Visibility

    return tuple(v.value for v in Visibility)


def write_docs(path: Path) -> None:
    path.write_text(render_markdown(load_taxonomy()), encoding="utf-8")
