"""Gmail compose tool generating prefilled draft URLs without sending."""

from __future__ import annotations

import urllib.parse
import webbrowser
from typing import Any

MAX_BODY_CHARS = 2000


def compose_email(to: str, subject: str, body: str) -> dict[str, Any]:
    """Build prefilled Gmail compose URL with strict URL-encoding and open in browser."""
    to_str = str(to or "").strip()
    subject_str = str(subject or "").strip()
    body_str = str(body or "")

    truncated = len(body_str) > MAX_BODY_CHARS
    if truncated:
        body_str = body_str[:MAX_BODY_CHARS]

    # Query params encoded properly with quote
    params = {
        "view": "cm",
        "fs": "1",
        "to": to_str,
        "su": subject_str,
        "body": body_str,
    }
    query_str = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    compose_url = f"https://mail.google.com/mail/?{query_str}"

    try:
        webbrowser.open(compose_url)
        return {
            "ok": True,
            "data": {
                "url": compose_url,
                "to": to_str,
                "subject": subject_str,
                "truncated": truncated,
                "note": "Draft opened in browser. Will never send automatically.",
            },
            "error": None,
        }
    except Exception as err:  # noqa: BLE001
        return {"ok": False, "data": None, "error": f"Failed to open compose URL: {err}"}
