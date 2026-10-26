"""Accessibility and interactive DOM snapshot engine with server-side ref mapping and TOON serialization."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ag.wire.wire import to_llm

if TYPE_CHECKING:
    from playwright.async_api import Page

CONTROL_AND_BIDI_REGEX = re.compile(r"[\x00-\x1F\x7F\u200E\u200F\u202A-\u202E]")


def sanitize_text(text: str, max_chars: int = 80) -> str:
    """Strip control and BiDi characters, collapse whitespace, and truncate."""
    if not text or not isinstance(text, str):
        return ""
    clean = CONTROL_AND_BIDI_REGEX.sub("", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    if len(clean) > max_chars:
        return clean[: max_chars - 1] + "…"
    return clean


class DOMElement(BaseModel):
    """Metadata for an interactive element tracked during snapshot epoch."""

    ref: str
    role: str
    name: str
    tag: str
    selector: str
    element_type: str = ""
    autocomplete: str = ""
    epoch: int = 0
    is_editable: bool = False
    is_password: bool = False
    is_card: bool = False
    is_otp: bool = False


class SnapshotResult(BaseModel):
    """Result of taking a browser snapshot."""

    url: str
    title: str
    toon_table: str
    elements: dict[str, DOMElement] = Field(default_factory=dict)
    epoch: int = 0


# JavaScript expression executed in page context to extract visible interactive elements
EXTRACT_ELEMENTS_JS = """
(() => {
    const isVisible = (elem) => {
        if (!elem) return false;
        if (elem.closest('[aria-hidden="true"]')) return false;
        if (typeof elem.checkVisibility === 'function' && !elem.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true })) {
            return false;
        }
        let curr = elem;
        while (curr && curr !== document.documentElement) {
            const pStyle = window.getComputedStyle(curr);
            if (pStyle.display === 'none' || pStyle.visibility === 'hidden' || pStyle.opacity === '0') return false;
            const r = curr.getBoundingClientRect();
            if ((pStyle.overflow === 'hidden' || pStyle.overflowX === 'hidden' || pStyle.overflowY === 'hidden') && (r.width < 1 || r.height < 1)) {
                return false;
            }
            if (r.right < 0 || r.bottom < 0) return false;
            curr = curr.parentElement;
        }
        const rect = elem.getBoundingClientRect();
        if (rect.width < 1 || rect.height < 1) return false;
        if (rect.right < 0 || rect.bottom < 0) return false;
        return true;
    };

    const getRole = (elem) => {
        const explicitRole = elem.getAttribute('role');
        if (explicitRole) return explicitRole.toLowerCase();
        const tag = elem.tagName.toLowerCase();
        if (tag === 'button') return 'button';
        if (tag === 'a' && elem.hasAttribute('href')) return 'link';
        if (tag === 'input') {
            const type = (elem.getAttribute('type') || 'text').toLowerCase();
            if (['button', 'submit', 'reset'].includes(type)) return 'button';
            if (['checkbox', 'radio'].includes(type)) return type;
            return 'textbox';
        }
        if (tag === 'textarea') return 'textbox';
        if (tag === 'select') return 'combobox';
        return '';
    };

    const getName = (elem) => {
        const ariaLabel = elem.getAttribute('aria-label');
        if (ariaLabel && ariaLabel.trim()) return ariaLabel.trim();
        const placeholder = elem.getAttribute('placeholder');
        if (placeholder && placeholder.trim()) return placeholder.trim();
        const title = elem.getAttribute('title');
        if (title && title.trim()) return title.trim();
        const alt = elem.getAttribute('alt');
        if (alt && alt.trim()) return alt.trim();
        
        // For inputs, check value or associated label
        if (elem.tagName.toLowerCase() === 'input') {
            const val = elem.value || '';
            const type = (elem.getAttribute('type') || '').toLowerCase();
            if (['button', 'submit'].includes(type) && val.trim()) return val.trim();
            if (elem.id) {
                const label = document.querySelector(`label[for="${elem.id}"]`);
                if (label && label.innerText.trim()) return label.innerText.trim();
            }
        }
        return (elem.innerText || elem.textContent || '').trim();
    };

    // Build unique CSS selector for element without injecting DOM attributes
    const getSelector = (elem) => {
        if (elem.id) return `#${CSS.escape(elem.id)}`;
        const tag = elem.tagName.toLowerCase();
        const parent = elem.parentElement;
        if (!parent) return tag;
        const siblings = Array.from(parent.children).filter(c => c.tagName.toLowerCase() === tag);
        if (siblings.length === 1) return `${getSelector(parent)} > ${tag}`;
        const index = siblings.indexOf(elem) + 1;
        return `${getSelector(parent)} > ${tag}:nth-of-type(${index})`;
    };

    const candidates = document.querySelectorAll(
        'button, a[href], input, textarea, select, [role="button"], [role="link"], [role="textbox"], [role="checkbox"], [tabindex]:not([tabindex="-1"])'
    );

    const items = [];
    for (const elem of candidates) {
        if (!isVisible(elem)) continue;
        const role = getRole(elem);
        if (!role) continue;
        const name = getName(elem);
        const tag = elem.tagName.toLowerCase();
        const type = (elem.getAttribute('type') || '').toLowerCase();
        const auto = (elem.getAttribute('autocomplete') || '').toLowerCase();
        const nameAttr = (elem.getAttribute('name') || '').toLowerCase();
        const idAttr = (elem.id || '').toLowerCase();

        items.push({
            role: role,
            name: name,
            tag: tag,
            type: type,
            autocomplete: auto,
            name_attr: nameAttr,
            id_attr: idAttr,
            selector: getSelector(elem),
            is_editable: tag === 'textarea' || (tag === 'input' && !['button', 'submit', 'checkbox', 'radio'].includes(type)),
            is_password: type === 'password' || auto.includes('password') || idAttr.includes('password') || nameAttr.includes('password'),
            is_card: auto.includes('cc-') || idAttr.includes('card') || nameAttr.includes('card'),
            is_otp: auto === 'one-time-code' || idAttr.includes('otp') || nameAttr.includes('otp')
        });
    }
    return items;
})()
"""


class SnapshotEngine:
    """Manages snapshot generation, element caching, and TOON formatting."""

    def __init__(self) -> None:
        self.current_epoch: int = 0
        self.elements_by_ref: dict[str, DOMElement] = {}

    async def take_snapshot(self, page: Page, max_elements: int = 80) -> SnapshotResult:
        """Capture visible interactive elements from page and format as TOON table."""
        self.current_epoch += 1
        url = page.url
        try:
            title = await page.title()
        except Exception:  # noqa: BLE001
            title = ""

        # Extract items via JS execution
        items: list[dict[str, Any]] = await page.evaluate(EXTRACT_ELEMENTS_JS)

        self.elements_by_ref.clear()
        rows: list[dict[str, str]] = []
        truncated_items = items[:max_elements]

        for idx, item in enumerate(truncated_items, start=1):
            ref = f"@e{idx}"
            clean_name = sanitize_text(item.get("name", ""), max_chars=80)
            role = sanitize_text(item.get("role", "button"), max_chars=30)
            tag = item.get("tag", "")
            selector = item.get("selector", "")

            elem = DOMElement(
                ref=ref,
                role=role,
                name=clean_name,
                tag=tag,
                selector=selector,
                element_type=item.get("type", ""),
                autocomplete=item.get("autocomplete", ""),
                epoch=self.current_epoch,
                is_editable=item.get("is_editable", False),
                is_password=item.get("is_password", False),
                is_card=item.get("is_card", False),
                is_otp=item.get("is_otp", False),
            )
            self.elements_by_ref[ref] = elem
            rows.append({"ref": ref, "role": role, "name": clean_name})

        # Build declared length TOON table
        toon_table = to_llm({"elements": rows}, fmt="toon_in")

        return SnapshotResult(
            url=url,
            title=sanitize_text(title, max_chars=120),
            toon_table=toon_table,
            elements=dict(self.elements_by_ref),
            epoch=self.current_epoch,
        )

    def get_element(self, ref: str) -> DOMElement | None:
        """Retrieve cached element by ref."""
        return self.elements_by_ref.get(ref)
