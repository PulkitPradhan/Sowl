"""Background sampler tracking target process memory (Ollama) and GPU VRAM deltas."""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass

import psutil


@dataclass
class ResourceStats:
    peak_process_ram_mb: float = 0.0
    peak_gpu_delta_mb: float = 0.0


def get_gpu_vram_used_mb() -> float:
    """Query current GPU VRAM used in MB via nvidia-smi."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if res.returncode == 0:
            lines = res.stdout.strip().splitlines()
            if lines:
                return float(lines[0].strip())
    except Exception:  # noqa: BLE001, S110
        pass
    return 0.0


def find_monitored_process(name_pattern: str = "ollama") -> psutil.Process | None:
    """Find process matching the monitor pattern."""
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            pname = (proc.info.get("name") or "").lower()
            cmdline = " ".join(proc.info.get("cmdline") or []).lower()
            if name_pattern.lower() in pname or name_pattern.lower() in cmdline:
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


class ProcessGpuSampler:
    """Background sampling thread monitoring target process RAM and GPU memory delta every 200 ms."""

    def __init__(self, process_name: str = "ollama", interval_s: float = 0.2) -> None:
        self.process_name = process_name
        self.interval_s = interval_s
        self._target_proc: psutil.Process | None = None
        self._baseline_gpu_mb: float = 0.0
        self._peak_proc_ram_mb: float = 0.0
        self._peak_gpu_delta_mb: float = 0.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._target_proc = find_monitored_process(self.process_name)
        self._baseline_gpu_mb = get_gpu_vram_used_mb()
        self._peak_proc_ram_mb = 0.0
        self._peak_gpu_delta_mb = 0.0
        self._stop_event.clear()

        self._thread = threading.Thread(target=self._run_sampling, daemon=True)
        self._thread.start()

    def _run_sampling(self) -> None:
        while not self._stop_event.is_set():
            # Sample process RAM
            if self._target_proc is not None:
                try:
                    ram_mb = self._target_proc.memory_info().rss / (1024 * 1024)
                    self._peak_proc_ram_mb = max(self._peak_proc_ram_mb, ram_mb)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    self._target_proc = find_monitored_process(self.process_name)

            # Sample GPU memory delta
            current_gpu = get_gpu_vram_used_mb()
            delta = max(0.0, current_gpu - self._baseline_gpu_mb)
            self._peak_gpu_delta_mb = max(self._peak_gpu_delta_mb, delta)

            time.sleep(self.interval_s)

    def stop(self) -> ResourceStats:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

        return ResourceStats(
            peak_process_ram_mb=round(self._peak_proc_ram_mb, 1),
            peak_gpu_delta_mb=round(self._peak_gpu_delta_mb, 1),
        )
