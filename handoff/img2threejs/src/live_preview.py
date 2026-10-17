from __future__ import annotations

import re


_BOOTSTRAP = r"""
<script>
(() => {
  const errors = [];

  function ensurePanel() {
    let panel = document.getElementById('__trajectory_preview_errors__');
    if (panel) return panel;

    panel = document.createElement('div');
    panel.id = '__trajectory_preview_errors__';
    panel.style.cssText = [
      'position:fixed',
      'left:12px',
      'right:12px',
      'bottom:12px',
      'z-index:2147483647',
      'display:none',
      'max-height:36vh',
      'overflow:auto',
      'padding:10px 12px',
      'border:1px solid #ff8182',
      'border-radius:8px',
      'background:rgba(255,235,233,.97)',
      'color:#82071e',
      'font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace',
      'white-space:pre-wrap',
      'box-shadow:0 4px 18px rgba(0,0,0,.18)'
    ].join(';');

    document.documentElement.appendChild(panel);
    return panel;
  }

  function renderErrors() {
    const panel = ensurePanel();
    if (!errors.length) {
      panel.style.display = 'none';
      panel.textContent = '';
      return;
    }
    panel.style.display = 'block';
    panel.textContent = '实时渲染错误\n\n' + errors.slice(-8).join('\n\n');
  }

  function record(message) {
    const text = String(message || 'Unknown error');
    if (!errors.includes(text)) errors.push(text);
    renderErrors();
  }

  window.addEventListener('error', (event) => {
    const location = event.filename
      ? `\n${event.filename}:${event.lineno || 0}:${event.colno || 0}`
      : '';
    record(`${event.message || 'JavaScript error'}${location}`);
  });

  window.addEventListener('unhandledrejection', (event) => {
    const reason = event.reason && event.reason.stack
      ? event.reason.stack
      : event.reason;
    record(`Unhandled promise rejection: ${reason}`);
  });

  window.addEventListener('load', () => {
    // A second resize after the iframe has its final dimensions fixes many
    // Three.js scenes that size the renderer from window.innerWidth/innerHeight.
    setTimeout(() => {
      try { window.dispatchEvent(new Event('resize')); } catch (_) {}
    }, 80);
  });
})();
</script>
""".strip()


_VIEWPORT_STYLE = r"""
<style id="__trajectory_preview_style__">
html, body { width: 100%; height: 100%; min-height: 100%; }
canvas { max-width: 100%; }
</style>
""".strip()


def build_live_preview(source: str, *, nonce: int = 0) -> str:
    """Prepare raw checkpoint HTML for execution inside Streamlit's browser iframe.

    The trajectory HTML itself is preserved. We only inject lightweight preview
    instrumentation before the page's own scripts so runtime failures are visible
    in the iframe and a post-load resize event helps full-window Three.js scenes
    adapt to the component dimensions.

    ``nonce`` is emitted as an inert comment so Streamlit receives a different
    component payload when the user explicitly requests a reload.
    """
    if not source or not source.strip():
        return ""

    injection = f"<!-- trajectory-preview-nonce:{int(nonce)} -->\n{_VIEWPORT_STYLE}\n{_BOOTSTRAP}\n"

    head_match = re.search(r"<head\b[^>]*>", source, flags=re.I)
    if head_match:
        pos = head_match.end()
        return source[:pos] + "\n" + injection + source[pos:]

    html_match = re.search(r"<html\b[^>]*>", source, flags=re.I)
    if html_match:
        pos = html_match.end()
        return source[:pos] + "\n<head>\n" + injection + "</head>\n" + source[pos:]

    # Fragments are still valid inside components.html; wrap them so module
    # scripts and viewport sizing behave consistently.
    return (
        "<!doctype html>\n<html><head>\n"
        + injection
        + "</head><body>\n"
        + source
        + "\n</body></html>"
    )
