"""Gmail compose tool generating prefilled draft URLs without sending."""

from __future__ import annotations

import urllib.parse
from typing import Any

from ag.skills.launcher import get_launcher

MAX_BODY_CHARS = 2000


def compose_email(to: str, subject: str = "", body: str = "") -> dict[str, Any]:
    """Build prefilled Gmail compose URL with strict URL-encoding and open in browser."""
    to_str = str(to or "").strip()
    subject_str = str(subject or "").strip()
    body_str = str(body or "")

    truncated = len(body_str) > MAX_BODY_CHARS
    if truncated:
        body_str = body_str[:MAX_BODY_CHARS]

    # Query params encoded properly with quote. Omit 'su' if subject is empty.
    params = {
        "view": "cm",
        "fs": "1",
        "to": to_str,
        "body": body_str,
    }
    if subject_str:
        params["su"] = subject_str

    query_str = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    compose_url = f"https://mail.google.com/mail/?{query_str}"

    try:
        get_launcher().open(compose_url)
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
