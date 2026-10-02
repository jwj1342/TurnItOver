from __future__ import annotations

import base64
import hashlib
import html as html_lib
import mimetypes
import re
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

try:
    from src.bundle import materialize_session_bundle
except ModuleNotFoundError:
    materialize_session_bundle = None

from src.live_preview import build_live_preview
from src.parser import discover_sessions, display_session_name, parse_session
from src.pipeline_parser import discover_pipeline_runs, display_pipeline_name, parse_pipeline_run


ROOT = Path(__file__).resolve().parent
SESSIONS_DIR = ROOT / "data" / "sessions"
HUMAN_SESSIONS_DIR = SESSIONS_DIR / "human_demo"
AGENT_SESSIONS_DIR = SESSIONS_DIR / "agent_demo"
REPO_RUNS_DIR = ROOT / "data" / "runs"
BUNDLE_DIR = ROOT / "data" / "bundles"

st.set_page_config(page_title="img2threejs 轨迹查看器", page_icon="🧭", layout="wide")

st.markdown(
    """
    <style>
      .block-container {
        padding-top: 1.35rem;
        padding-bottom: 1.5rem;
      }
      div[data-testid="stVerticalBlock"] {
        gap: 0.65rem;
      }
      div[data-testid="stImage"] img {
        width: 100% !important;
        aspect-ratio: 16 / 9 !important;
        object-fit: contain !important;
        background: #f6f8fa;
        border: 1px solid #d0d7de;
        border-radius: 8px;
      }
      div[data-testid="stMetric"] {
        padding: 0 !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_session(source_type: str, path_str: str, mtime_ns: int):
    if source_type == "pipeline":
        return parse_pipeline_run(path_str)
    return parse_session(path_str)


def github_diff_html(diff_text: str) -> tuple[str, int]:
    """Render a full unified diff as one GitHub-like, line-numbered region."""
    hunk_re = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")
    rows: list[str] = []
    old_line: int | None = None
    new_line: int | None = None
    additions = 0
    deletions = 0

    for raw_line in diff_text.splitlines():
        if raw_line.startswith("--- ") or raw_line.startswith("+++ "):
            continue

        hunk = hunk_re.match(raw_line)
        if hunk:
            old_line = int(hunk.group(1))
            new_line = int(hunk.group(3))
            rows.append(
                '<tr class="hunk">'
                '<td class="ln"></td><td class="ln"></td>'
                f'<td class="code">{html_lib.escape(raw_line)}</td>'
                '</tr>'
            )
            continue

        kind = "context"
        old_num = ""
        new_num = ""
        prefix = raw_line[:1]
        code = raw_line[1:] if prefix in {"+", "-", " "} else raw_line

        if prefix == "+":
            kind = "add"
            new_num = str(new_line) if new_line is not None else ""
            if new_line is not None:
                new_line += 1
            additions += 1
        elif prefix == "-":
            kind = "del"
            old_num = str(old_line) if old_line is not None else ""
            if old_line is not None:
                old_line += 1
            deletions += 1
        else:
            old_num = str(old_line) if old_line is not None else ""
            new_num = str(new_line) if new_line is not None else ""
            if old_line is not None:
                old_line += 1
            if new_line is not None:
                new_line += 1

        escaped_code = html_lib.escape(code).replace("\t", "    ")
        marker = "+" if kind == "add" else "-" if kind == "del" else " "
        rows.append(
            f'<tr class="{kind}">'
            f'<td class="ln">{old_num}</td>'
            f'<td class="ln">{new_num}</td>'
            f'<td class="code"><span class="marker">{marker}</span>{escaped_code}</td>'
            '</tr>'
        )

    summary = f'<span class="minus">−{deletions}</span> <span class="plus">+{additions}</span>'
    body = "".join(rows)
    doc = f"""
<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<style>
  :root {{ color-scheme: light; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: transparent; color: #1f2328; font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
  .diff-box {{ border: 1px solid #d0d7de; border-radius: 8px; overflow: hidden; background: #fff; }}
  .diff-head {{ height: 42px; padding: 0 12px; display: flex; align-items: center; justify-content: space-between; background: #f6f8fa; border-bottom: 1px solid #d0d7de; font-size: 12px; }}
  .file {{ font-weight: 600; font-family: ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}
  .meta {{ color: #656d76; display: flex; gap: 10px; align-items: center; }}
  .minus {{ color: #cf222e; font-weight: 600; }}
  .plus {{ color: #1a7f37; font-weight: 600; }}
  .scroll {{ overflow-x: auto; }}
  table {{ width: 100%; border-collapse: collapse; table-layout: auto; font: 12px/20px ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace; }}
  td {{ padding: 0; vertical-align: top; }}
  .ln {{ width: 44px; min-width: 44px; padding: 0 8px; text-align: right; color: #656d76; background: #f6f8fa; border-right: 1px solid #d8dee4; user-select: none; }}
  .code {{ padding: 0 8px; white-space: pre; min-width: 0; }}
  .marker {{ display: inline-block; width: 14px; user-select: none; }}
  .add .ln {{ background: #ccffd8; }}
  .add .code {{ background: #e6ffec; }}
  .del .ln {{ background: #ffd7d5; }}
  .del .code {{ background: #ffebe9; }}
  .hunk .ln, .hunk .code {{ background: #ddf4ff; color: #0969da; }}
  .hunk .code {{ padding: 4px 8px; }}
  .context:hover .code, .context:hover .ln {{ background: #f6f8fa; }}
</style>
</head>
<body>
  <div class="diff-box">
    <div class="diff-head">
      <div><span class="file">index.html</span></div>
      <div class="meta"><span>完整文件 · 格式化比较</span>{summary}</div>
    </div>
    <div class="scroll"><table><tbody>{body}</tbody></table></div>
  </div>
</body>
</html>
"""
    return doc, len(rows)


def _view_sort_key(view: dict) -> tuple[int, str]:
    view_id = str(view.get("view_id") or "")
    match = re.search(r"(\d+)$", view_id)
    return (int(match.group(1)) if match else 10**9, view_id)


def _image_data_url(path: str | Path) -> str:
    image_path = Path(path)
    mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def render_verifier_evidence_gallery(ckpt: dict) -> None:
    """Render every observation as an ordered thumbnail with hover preview."""
    views = sorted(ckpt.get("verifier_views") or [], key=_view_sort_key)
    if not views:
        st.info("该 round 没有保存 Verifier observation。")
        return

    cards: list[str] = []
    for view in views:
        path = view.get("path")
        view_id = html_lib.escape(str(view.get("view_id") or "view"))
        action = html_lib.escape(str(view.get("action") or "capture"))
        selected_class = " selected" if view.get("selected") else ""

        if path and Path(path).is_file():
            data_url = _image_data_url(path)
            image = (
                f'<img class="evidence-thumb" src="{data_url}" '
                f'alt="{view_id} · {action}">'
            )
            preview = (
                '<div class="evidence-preview">'
                f'<img src="{data_url}" alt="{view_id} · {action}">'
                f'<div class="evidence-preview-label">{view_id} · {action}</div>'
                '</div>'
            )
        else:
            image = '<div class="evidence-missing">截图缺失</div>'
            preview = ""

        cards.append(
            f'<div class="evidence-card{selected_class}">'
            f'{image}'
            f'<div class="evidence-caption">{view_id}<br>{action}</div>'
            f'{preview}'
            '</div>'
        )

    gallery = "".join(cards)
    st.markdown(
        f"""
        <style>
          .evidence-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(76px, 1fr));
            gap: 10px;
            align-items: start;
            overflow: visible;
          }}
          .evidence-card {{
            position: relative;
            min-width: 0;
            padding: 5px;
            border: 1px solid #d0d7de;
            border-radius: 9px;
            background: #fff;
            overflow: visible;
          }}
          .evidence-card.selected {{
            border: 3px solid #39ff88;
            padding: 3px;
            box-shadow:
              0 0 7px rgba(57,255,136,.95),
              0 0 18px rgba(57,255,136,.55);
          }}
          .evidence-thumb {{
            display: block;
            width: 100%;
            aspect-ratio: 16 / 9;
            object-fit: contain;
            background: #f6f8fa;
            border-radius: 5px;
          }}
          .evidence-caption {{
            margin-top: 5px;
            color: #656d76;
            font-size: 11px;
            line-height: 1.35;
            text-align: center;
            overflow-wrap: anywhere;
          }}
          .evidence-missing {{
            display: flex;
            width: 100%;
            aspect-ratio: 16 / 9;
            align-items: center;
            justify-content: center;
            background: #f6f8fa;
            color: #656d76;
            font-size: 11px;
            border-radius: 5px;
          }}
          .evidence-preview {{
            display: none;
            position: fixed;
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
            width: min(720px, 68vw);
            max-height: 78vh;
            padding: 10px;
            background: rgba(255,255,255,.98);
            border: 1px solid #d0d7de;
            border-radius: 12px;
            box-shadow: 0 18px 60px rgba(31,35,40,.30);
            z-index: 999999;
            pointer-events: none;
          }}
          .evidence-preview img {{
            display: block;
            width: 100%;
            max-height: 70vh;
            object-fit: contain;
            background: #f6f8fa;
            border-radius: 8px;
          }}
          .evidence-preview-label {{
            padding: 7px 2px 0;
            color: #24292f;
            font-size: 12px;
            text-align: center;
          }}
          .evidence-card:hover .evidence-preview {{
            display: block;
          }}
        </style>
        <div class="evidence-grid">{gallery}</div>
        """,
        unsafe_allow_html=True,
    )

def render_pipeline_gate(ckpt: dict) -> None:
    render_check = ckpt.get("render_check") or {}
    st.markdown("#### Render gate")
    gate_cols = st.columns(4)
    gate_cols[0].metric("Success", "✓" if render_check.get("success") else "✗")
    gate_cols[1].metric("Canvas", "✓" if render_check.get("canvas_nonempty") else "✗")
    stddev = render_check.get("image_stddev")
    gate_cols[2].metric("Image stddev", f"{stddev:.2f}" if isinstance(stddev, (int, float)) else "—")
    error_count = sum(
        len(render_check.get(key) or [])
        for key in ("load_errors", "page_errors", "console_errors")
    )
    gate_cols[3].metric("Browser errors", error_count)


st.markdown("## img2threejs · 轨迹查看器")

sources: list[dict] = []
for path in discover_sessions(HUMAN_SESSIONS_DIR):
    sources.append(
        {"type": "chatbox", "path": path, "label": f"[Human] {display_session_name(path)}"}
    )

bundled_root = None
if materialize_session_bundle is not None:
    try:
        bundled_root = materialize_session_bundle(BUNDLE_DIR)
        if bundled_root:
            for path in discover_sessions(bundled_root):
                sources.append(
                    {"type": "chatbox", "path": path, "label": f"[Human] {display_session_name(path)}"}
                )
    except Exception as exc:
        st.error(f"内置 Chatbox 数据包解码失败：{exc}")

for pipeline_root in (AGENT_SESSIONS_DIR, REPO_RUNS_DIR):
    for path in discover_pipeline_runs(pipeline_root):
        sources.append(
            {"type": "pipeline", "path": path, "label": f"[Agent] {display_pipeline_name(path)}"}
        )

unique_sources: dict[tuple[str, str], dict] = {}
for source in sources:
    key = (source["type"], str(source["path"].resolve()))
    unique_sources[key] = source
sources = sorted(unique_sources.values(), key=lambda item: item["label"].lower())

if not sources:
    st.info(
        "还没有可展示的会话。Human session 放入 `data/sessions/human_demo/<session-id>/`；"
        "Agent session 放入 `data/sessions/agent_demo/<run-id>/`。本地 Pipeline run 仍可从 `data/runs/<run-id>/` 直接查看。"
    )
    st.stop()

labels: dict[str, dict] = {}
for source in sources:
    label = source["label"]
    if label in labels:
        label = f"{label} · {source['path'].name[:8]}"
    labels[label] = source

selected_name = st.sidebar.selectbox("Session", list(labels.keys()))
selected_source = labels[selected_name]
selected_dir = selected_source["path"]
source_type = selected_source["type"]
source_manifest = selected_dir / ("trajectory.json" if source_type == "pipeline" else "session.json")
session_data = load_session(source_type, str(selected_dir), source_manifest.stat().st_mtime_ns)
meta = session_data["session"]
checkpoints = session_data["checkpoints"]

if not checkpoints:
    st.error("该会话中没有可展示的 checkpoint。")
    st.stop()

checkpoint_ids = [item["checkpoint"] for item in checkpoints]
selected_ckpt_id = st.sidebar.select_slider("Checkpoint", options=checkpoint_ids, value=checkpoint_ids[0])
ckpt = next(item for item in checkpoints if item["checkpoint"] == selected_ckpt_id)

if source_type == "pipeline":
    accepted_text = "accepted" if meta.get("accepted") else "not accepted"
    st.sidebar.caption(
        f"{meta.get('status') or 'unknown'} · {accepted_text} · "
        f"{meta['checkpoint_count']} rounds · {meta['render_count']} renders"
    )
else:
    st.sidebar.caption(
        f"{meta.get('model') or '未知模型'} · "
        f"{meta['checkpoint_count']} ckpt · {meta['render_count']} render"
    )

with st.sidebar.expander("Session details", expanded=False):
    st.caption(f"ID：`{meta['session_id']}`")
    st.caption(f"Source：`{source_type}`")
    if source_type == "pipeline":
        st.caption(f"Status：`{meta.get('status') or 'unknown'}`")
        st.caption(f"Accepted：`{meta.get('accepted')}`")
        st.caption(f"Visual revisions：`{meta.get('visual_revisions')}`")
    else:
        st.caption(f"Provider：`{meta.get('provider') or '未知'}`")
        if meta.get("missing_mapped_images"):
            st.warning(f"{meta['missing_mapped_images']} 个用户图像缺少本地 resource")

renders = session_data.get("render_images") or []
with st.expander(f"Render trajectory · {len(renders)}", expanded=False):
    if not renders:
        st.info("该会话没有导出的 render 图片。")
    else:
        output_owner = {
            item.get("final_render_path"): item["checkpoint"]
            for item in checkpoints
            if item.get("final_render_path")
        }
        for row_start in range(0, len(renders), 6):
            row_renders = renders[row_start : row_start + 6]
            gallery_cols = st.columns(len(row_renders), gap="small")
            for col, render_path in zip(gallery_cols, row_renders):
                owner = output_owner.get(render_path)
                label = f"ckpt {owner}" if owner is not None else "unmatched"
                with col:
                    st.image(render_path, use_container_width=True)
                    st.caption(label)

if source_type == "pipeline":
    phase_label = "初始生成" if selected_ckpt_id == 0 else f"第 {selected_ckpt_id} 次视觉修改"
else:
    phase_label = "初始生成" if selected_ckpt_id == 0 else f"第 {selected_ckpt_id} 次修改"
duration = ckpt.get("generation_duration_ms")
duration_text = f"{duration / 1000:.1f}s" if duration else "—"
reasoning_tokens = ckpt.get("reasoning_tokens") or "—"
model = ckpt.get("model") or meta.get("model") or "未知模型"

st.markdown(f"### ckpt {selected_ckpt_id} · {phase_label}")
if source_type == "pipeline":
    st.caption(
        f"source：{ckpt.get('source') or 'unknown'} · "
        f"render：{'success' if ckpt.get('render_success') else 'failed'} · "
        f"verifier：{ckpt.get('verifier_verdict') or '—'}"
    )
else:
    st.caption(f"模型：{model} · 推理 tokens：{reasoning_tokens} · 生成耗时：{duration_text}")

reference = meta.get("reference_path")
before_render = ckpt.get("input_render_path")
prompt_col, ref_col, before_col = st.columns([1.25, 1, 1], gap="small")
with prompt_col:
    if source_type == "pipeline":
        st.caption("GENERATION INPUT" if selected_ckpt_id == 0 else "REVISION INPUT")
    else:
        st.caption("用户输入")
    with st.container(height=230, border=True):
        st.write(ckpt.get("prompt") or "这一轮没有提取到输入文本。")

with ref_col:
    st.caption("REFERENCE RGB")
    if reference:
        st.image(reference, use_container_width=True)
    else:
        st.info("无 reference")

with before_col:
    st.caption("BEFORE")
    if before_render:
        st.image(before_render, use_container_width=True)
    elif selected_ckpt_id == 0:
        st.info("初始生成")
    else:
        st.info("无 before render")

if source_type == "pipeline":
    compare_tab, verifier_tab, code_tab = st.tabs(["Coder × Verifier", "Render / Evidence", "代码 Diff"])
else:
    compare_tab, code_tab = st.tabs(["推理 × 实时渲染", "代码 Diff"])
    verifier_tab = None

with compare_tab:
    if source_type == "pipeline":
        coder_col, verifier_col = st.columns(2, gap="medium")

        with coder_col:
            st.markdown("#### Coder")
            st.caption(f"阶段：{ckpt.get('source') or 'unknown'} · 负责生成/修改 Three.js")
            coder_reasoning = ckpt.get("coder_reasoning") or ckpt.get("reasoning_trace")
            if coder_reasoning:
                with st.container(height=500, border=True):
                    st.markdown(coder_reasoning)
            else:
                st.info("该 round 没有保存 Coder reasoning。")

            coder_final = ckpt.get("coder_final_answer")
            if coder_final:
                with st.expander("Coder final answer", expanded=False):
                    st.markdown(coder_final)
            coder_events = ckpt.get("agent_events") or []
            if coder_events:
                with st.expander(f"Coder events · {len(coder_events)}", expanded=False):
                    st.json(coder_events)

        with verifier_col:
            st.markdown("#### Verifier")
            st.caption(
                f"verdict：{ckpt.get('verifier_verdict') or '—'} · "
                f"观察视角：{len(ckpt.get('verifier_views') or [])} · "
                f"选中证据：{len(ckpt.get('selected_view_ids') or [])}"
            )
            verifier_reasoning = ckpt.get("verifier_reasoning")
            if verifier_reasoning:
                with st.container(height=500, border=True):
                    st.markdown(verifier_reasoning)
            else:
                st.info("该 round 没有保存 Verifier reasoning。")

            feedback = ckpt.get("verifier_feedback")
            if feedback:
                st.caption("Verifier feedback → next Coder")
                with st.container(border=True):
                    st.write(feedback)
            verifier_final = ckpt.get("verifier_final_answer")
            if verifier_final:
                with st.expander("Verifier final answer", expanded=False):
                    st.markdown(verifier_final)
            verifier_events = ckpt.get("verifier_events") or []
            if verifier_events:
                with st.expander(f"Verifier events · {len(verifier_events)}", expanded=False):
                    st.json(verifier_events)
    else:
        reasoning_col, live_col = st.columns(2, gap="medium")

        with reasoning_col:
            st.markdown("#### Reasoning")
            stats = (
                f"推理 tokens：{ckpt.get('reasoning_tokens') or '—'} · "
                f"输入 tokens：{ckpt.get('input_tokens') or '—'} · "
                f"输出 tokens：{ckpt.get('output_tokens') or '—'} · "
                f"生成耗时：{duration_text}"
            )
            st.caption(stats)
            if ckpt.get("reasoning_trace"):
                with st.container(height=520, border=True):
                    st.markdown(ckpt["reasoning_trace"])
            else:
                st.info("无可见 reasoning")

        with live_col:
            raw_html = ckpt.get("code_after") or ""
            st.markdown("#### Live render")
            if not raw_html:
                st.info("当前 checkpoint 没有可执行 HTML。")
            else:
                html_hash = hashlib.sha256(raw_html.encode("utf-8")).hexdigest()[:16]
                nonce_key = f"live-preview-nonce:{meta['session_id']}:{selected_ckpt_id}:{html_hash}"
                if nonce_key not in st.session_state:
                    st.session_state[nonce_key] = 0
                if st.button(
                    "重新加载实时预览",
                    key=f"reload:{nonce_key}",
                    type="primary",
                    use_container_width=True,
                ):
                    st.session_state[nonce_key] += 1
                components.html(
                    build_live_preview(raw_html, nonce=st.session_state[nonce_key]),
                    height=560,
                    scrolling=False,
                )

if verifier_tab is not None:
    with verifier_tab:
        render_col, evidence_col = st.columns([1, 1.2], gap="medium")

        with render_col:
            st.markdown("#### Live render")
            st.caption("当前 round authoritative HTML 的实时渲染。")
            raw_html = ckpt.get("code_after") or ""
            if raw_html:
                html_hash = hashlib.sha256(raw_html.encode("utf-8")).hexdigest()[:16]
                nonce_key = (
                    f"pipeline-live-preview-nonce:"
                    f"{meta['session_id']}:{selected_ckpt_id}:{html_hash}"
                )
                if nonce_key not in st.session_state:
                    st.session_state[nonce_key] = 0
                if st.button(
                    "重新加载当前 round",
                    key=f"reload:{nonce_key}",
                    use_container_width=True,
                ):
                    st.session_state[nonce_key] += 1
                components.html(
                    build_live_preview(raw_html, nonce=st.session_state[nonce_key]),
                    height=500,
                    scrolling=False,
                )
            else:
                st.info("当前 round 没有可执行 HTML。")

        with evidence_col:
            views = ckpt.get("verifier_views") or []
            selected_count = sum(1 for view in views if view.get("selected"))
            st.markdown("#### Verifier evidence")
            st.caption(
                f"主动观察 {len(views)} 个视角 · 选中 {selected_count} 个作为反馈证据"
            )
            render_verifier_evidence_gallery(ckpt)

        render_pipeline_gate(ckpt)

with code_tab:
    if ckpt.get("code_diff"):
        diff_doc, diff_rows = github_diff_html(ckpt["code_diff"])
        diff_height = min(820, max(170, 42 + diff_rows * 20))
        components.html(diff_doc, height=diff_height, scrolling=diff_rows > 38)
    else:
        st.info("这个 checkpoint 没有 HTML 变化。")

if source_type == "pipeline":
    prev_text = "无" if selected_ckpt_id == 0 else f"ckpt {selected_ckpt_id - 1}"
    st.caption(f"{prev_text} → ckpt {selected_ckpt_id} · Pipeline round artifacts")
else:
    prev_text = "无" if selected_ckpt_id == 0 else f"ckpt {selected_ckpt_id - 1}"
    st.caption(f"{prev_text} → ckpt {selected_ckpt_id} · Chatbox 顶层 messages")
