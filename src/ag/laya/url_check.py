"""URL safety checks: scheme enforcement, userinfo detection, punycode, IP literals, and local port warnings."""

from __future__ import annotations

import ipaddress
import urllib.parse


def check_url(url: str, web_port: int = 3180) -> tuple[bool, list[str], str]:
    """Inspect URL for safety violations and risk factors.

    Returns:
        (is_blocked, warning_reasons, parsed_hostname)
    """
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception as err:  # noqa: BLE001
        return True, [f"Invalid URL structure: {err}"], ""

    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        return True, [f"Disallowed URL scheme '{scheme}': only http and https are permitted"], ""

    hostname = (parsed.hostname or "").lower()
    netloc = parsed.netloc or ""
    warnings: list[str] = []

    # 1. Userinfo trick (e.g. https://google.com@evil.com)
    if "@" in netloc:
        userinfo = netloc.split("@", 1)[0]
        warnings.append(
            f"Detected deceptive userinfo '{userinfo}' in URL; true destination host is '{hostname}'"
        )

    # 2. Punycode check (IDN homograph attack prevention)
    if "xn--" in hostname:
        warnings.append(f"Punycode encoded internationalized hostname detected: '{hostname}'")

    # 3. IP address literal checks (private, loopback, link-local)
    is_ip = False
    try:
        ip = ipaddress.ip_address(hostname)
        is_ip = True
        if ip.is_loopback:
            warnings.append(f"URL targets loopback IP address '{hostname}'")
        elif ip.is_private:
            warnings.append(f"URL targets private network IP address '{hostname}'")
        elif ip.is_link_local:
            warnings.append(f"URL targets link-local IP address '{hostname}'")
        else:
            warnings.append(f"URL targets direct IP address literal '{hostname}'")
    except ValueError:
        pass

    if not is_ip and (hostname == "localhost" or hostname.endswith(".localhost")):
        warnings.append(f"URL targets local loopback host '{hostname}'")

    # 4. Check web port target
    try:
        port = parsed.port
        if port == web_port or (
            port is not None and port == 3180 and hostname in ("127.0.0.1", "localhost", "::1")
        ):
            warnings.append(f"URL targets Sowl web harness control port {web_port}")
    except ValueError:
        pass

    return False, warnings, hostname
