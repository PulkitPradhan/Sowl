"""Dedicated Chrome process management with loopback CDP binding and PID tracking."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
import psutil

from ag.browser.config import BrowserConfig, load_browser_config


def find_chrome_executable(config: BrowserConfig | None = None) -> str | None:
    """Find installed Chrome or Chromium binary, honoring configuration and env."""
    cfg = config or load_browser_config()
    if cfg.executable and Path(cfg.executable).exists() and os.access(cfg.executable, os.X_OK):
        return cfg.executable

    env_bin = os.environ.get("AG_CHROME_BIN")
    if env_bin and Path(env_bin).exists() and os.access(env_bin, os.X_OK):
        return env_bin

    candidates = [
        "chromium",
        "google-chrome",
        "google-chrome-stable",
        "chromium-browser",
    ]
    for name in candidates:
        path = shutil.which(name)
        if path:
            return path
    return None


def get_default_profile_dir() -> Path:
    """Return ~/.ag/chrome-profile directory."""
    env_dir = os.environ.get("AG_WORKSPACE_DIR")
    if env_dir:
        base = Path(env_dir).expanduser()
    else:
        base = Path("~/.ag").expanduser()
    return base / "chrome-profile"


def get_default_pid_file() -> Path:
    """Return ~/.ag/browser.json PID tracking file."""
    env_dir = os.environ.get("AG_WORKSPACE_DIR")
    if env_dir:
        base = Path(env_dir).expanduser()
    else:
        base = Path("~/.ag").expanduser()
    return base / "browser.json"


def find_free_port() -> int:
    """Find a free high port on 127.0.0.1."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def is_browser_attached(pid_file: Path | None = None) -> bool:
    """Synchronously check whether the browser process is alive and attached."""
    pf = pid_file or get_default_pid_file()
    if not pf.exists():
        return False
    try:
        data = json.loads(pf.read_text(encoding="utf-8"))
        pid = int(data.get("pid", 0))
        return bool(pid and psutil.pid_exists(pid))
    except Exception:  # noqa: BLE001
        return False


async def get_browser_status(pid_file: Path | None = None) -> dict[str, Any]:
    """Check whether dedicated browser process is running and responding on CDP."""
    pf = pid_file or get_default_pid_file()
    if not pf.exists():
        return {"running": False, "pid": None, "port": None, "cdp_url": None}

    try:
        data = json.loads(pf.read_text(encoding="utf-8"))
        pid = int(data.get("pid", 0))
        port = int(data.get("port", 0))
    except Exception:  # noqa: BLE001
        return {"running": False, "pid": None, "port": None, "cdp_url": None}

    if not pid or not psutil.pid_exists(pid):
        try:
            pf.unlink(missing_ok=True)
        except OSError:
            pass
        return {"running": False, "pid": None, "port": None, "cdp_url": None}

    # Verify CDP responsiveness
    cdp_url = f"http://127.0.0.1:{port}"
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            resp = await client.get(f"{cdp_url}/json/version")
            if resp.status_code == 200:
                return {
                    "running": True,
                    "pid": pid,
                    "port": port,
                    "cdp_url": cdp_url,
                    "browser": resp.json().get("Browser", "Chrome"),
                }
    except Exception:  # noqa: BLE001, S110
        pass

    return {"running": False, "pid": pid, "port": port, "cdp_url": cdp_url}


async def start_browser(
    profile_dir: Path | None = None,
    pid_file: Path | None = None,
    port: int | None = None,
    config: BrowserConfig | None = None,
    idle_timeout_s: int = 900,
) -> dict[str, Any]:
    """Launch dedicated Chrome process bound to 127.0.0.1 with dedicated profile."""
    cfg = config or load_browser_config()
    executable = find_chrome_executable(cfg)
    if not executable:
        raise RuntimeError(
            "No Chrome or Chromium browser executable found. Please install Chromium or set 'executable' in ~/.ag/browser.toml."
        )

    pf = pid_file or get_default_pid_file()
    status = await get_browser_status(pf)
    if status["running"]:
        return status

    p_dir = profile_dir or get_default_profile_dir()
    p_dir.mkdir(parents=True, exist_ok=True)
    try:
        p_dir.chmod(0o700)
    except OSError:
        pass

    cdp_port = port or cfg.cdp_port or find_free_port()

    # Launch Chrome detached with loopback CDP
    cmd = [
        executable,
        f"--user-data-dir={p_dir}",
        f"--remote-debugging-port={cdp_port}",
        "--remote-debugging-address=127.0.0.1",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-background-networking",
        "--disable-sync",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--disable-gpu",
        "--disable-crash-reporter",
        "--headless=new" if os.environ.get("AG_BROWSER_HEADLESS") == "1" else "",
    ]
    # Filter empty args
    cmd = [arg for arg in cmd if arg]

    proc = subprocess.Popen(  # noqa: ASYNC220
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    # Save PID file with 0600 permissions
    data = {
        "pid": proc.pid,
        "port": cdp_port,
        "cdp_url": f"http://127.0.0.1:{cdp_port}",
        "profile_dir": str(p_dir),
        "started_at": time.time(),
        "idle_timeout_s": idle_timeout_s,
    }
    pf.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        pf.chmod(0o600)
    except OSError:
        pass

    # Poll CDP until ready
    cdp_url = f"http://127.0.0.1:{cdp_port}"
    deadline = time.time() + 10.0
    ready = False
    async with httpx.AsyncClient(timeout=1.0) as client:
        while time.time() < deadline:
            try:
                resp = await client.get(f"{cdp_url}/json/version")
                if resp.status_code == 200:
                    ready = True
                    break
            except Exception:  # noqa: BLE001
                await asyncio.sleep(0.15)

    if not ready:
        await stop_browser(pf)
        raise RuntimeError(f"Timed out waiting for Chrome CDP to respond on {cdp_url}")

    return {
        "running": True,
        "pid": proc.pid,
        "port": cdp_port,
        "cdp_url": cdp_url,
    }


async def stop_browser(pid_file: Path | None = None) -> bool:
    """Stop the dedicated Chrome process and clean up PID tracking."""
    pf = pid_file or get_default_pid_file()
    if not pf.exists():
        return False

    try:
        data = json.loads(pf.read_text(encoding="utf-8"))
        pid = int(data.get("pid", 0))
    except Exception:  # noqa: BLE001
        pid = 0

    if pid and psutil.pid_exists(pid):
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            for child in children:
                child.send_signal(signal.SIGTERM)
            parent.send_signal(signal.SIGTERM)

            # Wait up to 3 seconds for termination
            _, alive = psutil.wait_procs(children + [parent], timeout=3.0)
            for p in alive:
                p.kill()
        except Exception:  # noqa: BLE001, S110
            pass

    try:
        pf.unlink(missing_ok=True)
    except OSError:
        pass

    return True
