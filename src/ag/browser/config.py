"""Browser configuration, escalation matching, and engagement safety rules."""

from __future__ import annotations

import os
import re
import tomllib
import urllib.parse
from pathlib import Path

from pydantic import BaseModel, Field

# Normalized word-boundary escalation pattern
ESCALATION_PATTERN = re.compile(
    r"\b(send|post|submit|delete|buy|pay|confirm|publish|place\s+order|checkout|remove|transfer|withdraw|accept|agree|donate)\b",
    re.IGNORECASE,
)

# Engagement-related actions/labels restricted on social platforms
ENGAGEMENT_ACTIONS = {
    "like",
    "follow",
    "comment",
    "reply",
    "share",
    "repost",
    "post",
    "message",
    "subscribe",
}

DEFAULT_SOCIAL_DOMAINS = {
    "instagram.com",
    "facebook.com",
    "twitter.com",
    "x.com",
    "threads.net",
    "tiktok.com",
    "reddit.com",
    "linkedin.com",
}


def normalize_label(label: str) -> str:
    """Normalize label string by lowercasing and collapsing whitespace."""
    if not label or not isinstance(label, str):
        return ""
    # Collapse multiple whitespaces and trim
    return re.sub(r"\s+", " ", label.strip().lower())


def is_escalation_label(label: str) -> bool:
    """Check if label matches R2 escalation keywords using word boundaries."""
    norm = normalize_label(label)
    if not norm:
        return False
    return bool(ESCALATION_PATTERN.search(norm))


def is_engagement_action(action_or_label: str, domain_or_url: str) -> bool:
    """Return True if action/label represents an engagement action on a social domain."""
    norm_action = normalize_label(action_or_label)
    if not norm_action:
        return False

    domain = extract_domain(domain_or_url)
    if not any(social in domain for social in DEFAULT_SOCIAL_DOMAINS):
        return False

    words = set(re.findall(r"\b\w+\b", norm_action))
    return bool(words.intersection(ENGAGEMENT_ACTIONS))


def extract_domain(url_or_domain: str) -> str:
    """Extract domain host from URL or host string."""
    if not url_or_domain:
        return ""
    if "://" in url_or_domain:
        try:
            parsed = urllib.parse.urlparse(url_or_domain)
            return (parsed.hostname or "").lower()
        except Exception:  # noqa: BLE001
            return ""
    return url_or_domain.split(":")[0].strip().lower()


class BrowserConfig(BaseModel):
    """Configuration loaded from ~/.ag/browser.toml or defaults."""

    min_delay_ms: int = 250
    wait_cap_s: float = 10.0
    social_domains: list[str] = Field(default_factory=lambda: sorted(DEFAULT_SOCIAL_DOMAINS))
    domain_delays_ms: dict[str, int] = Field(
        default_factory=lambda: {"instagram.com": 2000, "threads.net": 1000}
    )
    executable: str | None = None
    cdp_port: int | None = None
    allow_non_loopback: bool = False
    max_tabs: int = 5
    max_actions: int = 30
    max_scrolls: int = 20
    idle_timeout_s: int = 900  # 15 minutes auto-stop

    def get_delay_for_url(self, url: str) -> float:
        """Return inter-action delay in seconds for given URL."""
        domain = extract_domain(url)
        for dom, delay_ms in self.domain_delays_ms.items():
            if dom in domain:
                return delay_ms / 1000.0
        return self.min_delay_ms / 1000.0


def get_default_browser_config_path() -> Path:
    """Return ~/.ag/browser.toml location."""
    env_dir = os.environ.get("AG_WORKSPACE_DIR")
    if env_dir:
        base = Path(env_dir).expanduser()
    else:
        base = Path("~/.ag").expanduser()
    return base / "browser.toml"


def load_browser_config(path: Path | None = None) -> BrowserConfig:
    """Load browser configuration from TOML file if present, else defaults."""
    cfg_path = path or get_default_browser_config_path()
    if cfg_path.exists():
        try:
            data = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
            return BrowserConfig(**data)
        except Exception:  # noqa: BLE001, S110
            pass
    return BrowserConfig()
