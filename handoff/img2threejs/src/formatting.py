from __future__ import annotations

import difflib

from bs4 import BeautifulSoup
import jsbeautifier


def _js_options():
    options = jsbeautifier.default_options()
    options.indent_size = 2
    options.indent_char = " "
    options.preserve_newlines = True
    options.max_preserve_newlines = 2
    options.wrap_line_length = 120
    options.end_with_newline = False
    return options


def _format_css(css: str) -> str:
    """Small display-only CSS formatter.

    This deliberately avoids changing strings or comments aggressively. It is
    only a readability fallback for the usually short inline style blocks in
    generated Three.js pages.
    """
    text = css.strip()
    if not text:
        return ""

    out: list[str] = []
    buffer: list[str] = []
    indent = 0
    quote: str | None = None
    escape = False
    in_comment = False
    i = 0

    def flush() -> None:
        value = "".join(buffer).strip()
        buffer.clear()
        if value:
            out.append("  " * indent + value)

    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""

        if in_comment:
            buffer.append(ch)
            if ch == "*" and nxt == "/":
                buffer.append(nxt)
                i += 1
                in_comment = False
            i += 1
            continue

        if quote:
            buffer.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                quote = None
            i += 1
            continue

        if ch in {"'", '"'}:
            quote = ch
            buffer.append(ch)
        elif ch == "/" and nxt == "*":
            in_comment = True
            buffer.extend([ch, nxt])
            i += 1
        elif ch == "{":
            buffer.append(" {") if buffer and not "".join(buffer).rstrip().endswith(" ") else buffer.append("{")
            flush()
            indent += 1
        elif ch == "}":
            flush()
            indent = max(0, indent - 1)
            out.append("  " * indent + "}")
        elif ch == ";":
            buffer.append(";")
            flush()
        else:
            buffer.append(ch)
        i += 1

    flush()
    return "\n".join(out)


def format_html_for_display(source: str) -> str:
    """Return a human-readable copy of HTML without touching the raw source.

    HTML is normalized/indented with BeautifulSoup, JavaScript inside <script>
    tags is formatted with jsbeautifier, and inline CSS blocks get lightweight
    indentation. The returned string is for UI display and diffing only; the
    original trajectory code remains unchanged and is still used for rendering.
    """
    if not source or not source.strip():
        return ""

    soup = BeautifulSoup(source, "html.parser")
    options = _js_options()

    for script in soup.find_all("script"):
        raw = script.string
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        try:
            formatted = jsbeautifier.beautify(text, options)
        except Exception:
            formatted = text
        script.clear()
        script.append("\n" + formatted.rstrip() + "\n")

    for style in soup.find_all("style"):
        raw = style.string
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        try:
            formatted = _format_css(text)
        except Exception:
            formatted = text
        style.clear()
        style.append("\n" + formatted.rstrip() + "\n")

    return soup.prettify().rstrip() + "\n"


def formatted_html_diff(before: str, after: str) -> str:
    """Return a full-file diff of two human-readable HTML states.

    Unlike a conventional compact unified diff, this deliberately keeps every
    unchanged line in the output. The UI can therefore show the complete
    formatted ``index.html`` in one GitHub-like region while still highlighting
    deleted lines in red and added lines in green. This is display-only: the
    original trajectory HTML remains untouched.
    """
    before_pretty = format_html_for_display(before)
    after_pretty = format_html_for_display(after)
    if before_pretty == after_pretty:
        return ""

    before_lines = before_pretty.splitlines(keepends=True)
    after_lines = after_pretty.splitlines(keepends=True)

    # A context radius at least as large as the whole file forces difflib to
    # emit one complete-file hunk instead of collapsing unchanged regions.
    full_context = max(len(before_lines), len(after_lines), 1)

    return "".join(
        difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile="修改前/index.html",
            tofile="修改后/index.html",
            n=full_context,
        )
    )
