"""Tests for static frontend asset security and offline self-containment."""

from __future__ import annotations

import re
from pathlib import Path

STATIC_DIR = Path(__file__).parent.parent / "src" / "ag" / "web" / "static"


def test_static_assets_exist():
    """Verify index.html, app.js, and style.css exist in src/ag/web/static."""
    assert (STATIC_DIR / "index.html").is_file(), "index.html missing"
    assert (STATIC_DIR / "app.js").is_file(), "app.js missing"
    assert (STATIC_DIR / "style.css").is_file(), "style.css missing"


def test_no_external_urls_or_cdns():
    """Verify static assets contain no external URLs (http://, https://, or //)."""
    # Regex matching external resources, excluding local loopback URLs in comments if any
    external_url_pattern = re.compile(r"""(?i)(?:https?:)?//(?!127\.0\.0\.1|localhost)[\w.-]+""")

    for path in (STATIC_DIR / "index.html", STATIC_DIR / "app.js", STATIC_DIR / "style.css"):
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8")
        matches = external_url_pattern.findall(content)
        assert not matches, f"External URLs found in {path.name}: {matches}"


def test_index_html_has_no_inline_scripts_or_styles():
    """Verify index.html has no inline <script>...</script> or <style>...</style> blocks, and no inline style attributes."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.is_file():
        assert False, "index.html does not exist"

    content = index_path.read_text(encoding="utf-8")

    # Match <script> tags that have inline content (not src=...)
    inline_script = re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", content, re.DOTALL | re.IGNORECASE)
    # Strip whitespace to check if there is actual code
    inline_script_code = [s.strip() for s in inline_script if s.strip()]
    assert not inline_script_code, f"Inline script found in index.html: {inline_script_code}"

    # Match <style> tags
    inline_styles = re.findall(r"<style[^>]*>(.*?)</style>", content, re.DOTALL | re.IGNORECASE)
    assert not inline_styles, f"Inline <style> tags found in index.html: {inline_styles}"

    # Match inline style="..." attributes
    style_attrs = re.findall(r"""\bstyle\s*=\s*['"][^'"]*['"]""", content, re.IGNORECASE)
    assert not style_attrs, f"Inline style attributes found in index.html: {style_attrs}"


def test_app_js_uses_no_inner_html():
    """Verify app.js does not use innerHTML or outerHTML for DOM manipulation."""
    app_js_path = STATIC_DIR / "app.js"
    if not app_js_path.is_file():
        assert False, "app.js does not exist"

    content = app_js_path.read_text(encoding="utf-8")
    assert "innerHTML" not in content, "innerHTML used in app.js - must use textContent or createTextNode"
    assert "outerHTML" not in content, "outerHTML used in app.js"
    assert "document.write" not in content, "document.write used in app.js"
