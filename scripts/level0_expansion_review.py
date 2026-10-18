"""Write an offline, label-free native-pixel review page for the current bundle."""
from __future__ import annotations

import html
from pathlib import Path


def write_review_page(out: Path, public: list[dict], reviews: dict[str, dict]) -> None:
    pending = [p for p in public if reviews[p['case_id']]['status'] == 'pending_human_review']
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Level 0 v2 新增审核</title>',
        '<style>body{font:16px/1.6 system-ui;background:#fafaf8;color:#222;margin:24px}'
        'main{max-width:1080px;margin:auto}article{border-top:2px solid #777;padding:24px 0}'
        '.grid{display:grid;grid-template-columns:256px 256px;gap:10px;width:522px}'
        'img{width:256px;height:256px;max-width:none}figure{margin:0}a{color:#18559e}</style><main>',
        f'<h1>Level 0 v2：新增待审核 {len(pending)} 个样本</h1>',
        '<p>每组四张原生 256×256 图：第一行参考 rest / limit，第二行候选 rest / limit。'
        '点击六图板可额外检查 mid。页面不包含 Gold、注入参数或答案。'
        '请确认正常与候选的运动是否清晰、rest 是否匹配、是否有出框/遮挡干扰；'
        '逐例记录 pass/fail、缺陷类型和一句运动方向描述，再更新 human_review.jsonl。'
        '不能把默认批准或自动通过替代实际逐图审核。</p>']
    for p in sorted(pending, key=lambda p: p['case_id']):
        parts += [f'<article id="{p["case_id"]}"><h2>{p["case_id"]} · {html.escape(p["category"])}</h2>',
            f'<p>视角 {html.escape(p["view"])} · <a href="{p["board"]}">rest/mid/limit 六图板</a> · '
            f'<a href="{p["input_preview"]}">原始四宫格</a></p><div class="grid">']
        for path, title in zip(p['images'], ('参考 rest', '参考 limit', '候选 rest', '候选 limit')):
            parts.append(f'<figure><figcaption>{title}</figcaption><img loading="lazy" src="{path}" alt="{title}"></figure>')
        parts.append('</div></article>')
    parts.append('</main></html>')
    (out / 'review-expansion.html').write_text('\n'.join(parts), encoding='utf-8')
