"""Post-check validation verifying tool results against manifest post_conditions."""

from __future__ import annotations

import urllib.parse
from pathlib import Path
from typing import Any

from ag.skills.manifest import ToolManifest


def verify_post_conditions(
    tool_name: str,
    manifest: ToolManifest,
    args: dict[str, Any],
    result_data: Any,
    workspace_root: Path,
) -> tuple[bool, str]:
    """Verify tool execution output against declared post_conditions.

    Returns:
        (success, error_message_if_failed)
    """
    conditions = manifest.post_conditions or []
    if not conditions:
        return True, ""

    data = result_data if isinstance(result_data, dict) else {}

    for cond in conditions:
        cond_clean = cond.strip()

        # 1. file_exists check
        if cond_clean == "file_exists":
            path_arg = args.get("path") or args.get("target") or data.get("path")
            if not path_arg:
                return False, "Post-check failure: 'file_exists' condition but no path argument was provided"
            target_path = Path(path_arg)
            if not target_path.is_absolute():
                target_path = (workspace_root / target_path).resolve()
            if not target_path.is_file():
                return False, f"Post-check failure: expected file does not exist at {path_arg}"

        # 2. content_returned check
        elif cond_clean == "content_returned":
            if "content" not in data or data["content"] is None:
                return False, "Post-check failure: 'content_returned' condition failed, no content in result"

        # 3. entries_returned check
        elif cond_clean == "entries_returned":
            if "entries" not in data or not isinstance(data["entries"], list):
                return False, "Post-check failure: 'entries_returned' condition failed, no directory entries list"

        # 4. url_opened check
        elif cond_clean in ("url_opened", "draft_opened"):
            url_val = data.get("url")
            if not url_val or not isinstance(url_val, str):
                return False, f"Post-check failure: '{cond_clean}' condition failed, no URL found in tool result"

        # 5. draft_url_has(...) check
        if "draft_url_has" in cond_clean:
            url_str = str(data.get("url") or "")
            if not url_str:
                return False, "Post-check failure: draft_url_has requires 'url' in tool output"
            try:
                parsed = urllib.parse.urlparse(url_str)
                query_params = urllib.parse.parse_qs(parsed.query)
            except Exception as err:  # noqa: BLE001
                return False, f"Post-check failure parsing draft URL query: {err}"

            # Check 'to' argument match if specified
            if args.get("to"):
                expected_to = str(args["to"]).strip().lower()
                actual_to = [t.lower() for t in query_params.get("to", [])]
                if expected_to not in actual_to and not any(expected_to in t for t in actual_to):
                    return False, f"Post-check failure: draft URL missing recipient 'to={args['to']}'"

            # Check 'subject' argument match if specified
            if args.get("subject"):
                expected_su = str(args["subject"]).strip().lower()
                actual_su = [s.lower() for s in query_params.get("su", []) + query_params.get("subject", [])]
                if expected_su not in actual_su and not any(expected_su in s for s in actual_su):
                    return False, f"Post-check failure: draft URL missing subject 'su={args['subject']}'"

    return True, ""
