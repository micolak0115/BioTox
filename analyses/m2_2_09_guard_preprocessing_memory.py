# memory_guard.py
# ============================================================
# MemoryGuard — monitors RAM usage, warns, and kills if critical
#
# Usage:
#   guard = MemoryGuard(min_free_gb=10, warn_pct=0.75, kill_pct=0.80)
#
#   guard.check_available(needed_gb=50, label="load X")
#   guard.log_info("after X mmap | ")
#   guard.collect()
#   guard.kill_if_critical()
#   guard.array_gb((1_000_000, 12328))
#   guard.estimate_safe_chunk(n_cols=12328, target_pct=0.05)
#
#   with guard:          # starts background log + kill monitors
#       do_heavy_work()  # SIGTERM → 10s grace → SIGKILL if RAM critical
# ============================================================

import gc
import os
import sys
import signal
import logging
import threading
import time
import numpy as np

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def _get_logger(name=__name__):
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger

log = _get_logger()


def _mem_info() -> dict:
    """
    Returns memory info dict with keys:
        total_gb, used_gb, available_gb, used_pct
    Falls back to /proc/meminfo if psutil not available.
    """
    if _HAS_PSUTIL:
        vm           = psutil.virtual_memory()
        return {
            "total_gb":     vm.total     / 1e9,
            "used_gb":      vm.used      / 1e9,
            "available_gb": vm.available / 1e9,
            "used_pct":     vm.percent   / 100.0,
        }
    else:
        # fallback: parse /proc/meminfo (Linux only)
        info = {}
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        info[parts[0].rstrip(":")] = int(parts[1]) * 1024
            total     = info.get("MemTotal",     0)
            available = info.get("MemAvailable", 0)
            used      = total - available
            return {
                "total_gb":     total     / 1e9,
                "used_gb":      used      / 1e9,
                "available_gb": available / 1e9,
                "used_pct":     used / total if total > 0 else 0.0,
            }
        except Exception:
            return {
                "total_gb":     0.0,
                "used_gb":      0.0,
                "available_gb": 0.0,
                "used_pct":     0.0,
            }


class MemoryGuard:
    """
    Monitors system RAM and kills the process if usage exceeds kill_pct.

    Parameters
    ----------
    min_free_gb      : minimum free RAM required for check_available()
    warn_pct         : log warning when used RAM exceeds this fraction
    kill_pct         : send SIGTERM when used RAM exceeds this fraction
    monitor_interval : background log thread interval (seconds)
    kill_interval    : background kill thread interval (seconds)
    grace_period     : seconds between SIGTERM and SIGKILL
    """

    def __init__(
        self,
        min_free_gb:      float = 10.0,
        warn_pct:         float = 0.75,
        kill_pct:         float = 0.90,
        monitor_interval: float = 30.0,
        kill_interval:    float = 5.0,
        grace_period:     float = 3.0,
    ):
        if not _HAS_PSUTIL:
            log.warning(
                "psutil not found — falling back to /proc/meminfo. "
                "Install psutil for accurate memory monitoring: pip install psutil"
            )

        self.min_free_gb      = min_free_gb
        self.warn_pct         = warn_pct
        self.kill_pct         = kill_pct
        self.monitor_interval = monitor_interval
        self.kill_interval    = kill_interval
        self.grace_period     = grace_period

        self._stop_event      = threading.Event()
        self._log_thread:  threading.Thread | None = None
        self._kill_thread: threading.Thread | None = None
        self._killing       = False   # prevent double kill

    # ----------------------------------------------------------
    # public API
    # ----------------------------------------------------------

    def check_available(self, needed_gb: float, label: str = "") -> None:
        """
        Raise MemoryError if available RAM < needed_gb or < min_free_gb.
        Logs current state regardless.
        """
        m   = _mem_info()
        avail = m["available_gb"]
        tag   = f"[{label}] " if label else ""

        log.info(
            f"{tag}RAM check: "
            f"available={avail:.1f} GB | "
            f"needed={needed_gb:.1f} GB | "
            f"used={m['used_pct']*100:.1f}%"
        )

        if avail < self.min_free_gb:
            raise MemoryError(
                f"{tag}Insufficient RAM: "
                f"available={avail:.1f} GB < min_free={self.min_free_gb:.1f} GB"
            )

        if avail < needed_gb:
            raise MemoryError(
                f"{tag}Insufficient RAM: "
                f"available={avail:.1f} GB < needed={needed_gb:.1f} GB"
            )

        if m["used_pct"] >= self.warn_pct:
            log.warning(
                f"{tag}RAM warning: "
                f"used={m['used_pct']*100:.1f}% ≥ warn_pct={self.warn_pct*100:.0f}%"
            )

    def log_info(self, label: str = "") -> None:
        """Log current RAM state. label is prepended."""
        m = _mem_info()
        log.info(
            f"{label}"
            f"RAM: {m['used_gb']:.1f}/{m['total_gb']:.1f} GB used "
            f"({m['used_pct']*100:.1f}%) | "
            f"available={m['available_gb']:.1f} GB"
        )

    def collect(self) -> None:
        """Run gc.collect() and log RAM state."""
        gc.collect()
        self.log_info("after gc.collect() | ")

    def kill_if_critical(self) -> bool:
        """
        Manually check RAM and trigger kill sequence if critical.
        Returns True if kill was triggered, False otherwise.
        """
        m = _mem_info()
        if m["used_pct"] >= self.kill_pct and not self._killing:
            log.error(
                f"CRITICAL RAM: used={m['used_pct']*100:.1f}% ≥ "
                f"kill_pct={self.kill_pct*100:.0f}% — triggering kill"
            )
            self._trigger_kill()
            return True
        return False

    @staticmethod
    def array_gb(shape: tuple, dtype=np.float32) -> float:
        """Estimate RAM needed for a dense numpy array in GB."""
        n_bytes = np.prod(shape) * np.dtype(dtype).itemsize
        return float(n_bytes) / 1e9

    def estimate_safe_chunk(
        self,
        n_cols:     int,
        target_pct: float = 0.05,
        dtype                = np.float32,
    ) -> int:
        """
        Compute safe number of rows such that one dense chunk of
        shape (chunk_rows, n_cols) uses at most target_pct of total RAM.

        Returns an integer number of rows (minimum 1).

        Example:
            128 GB machine, target_pct=0.05, n_cols=12328:
            budget = 0.05 × 128 GB = 6.4 GB
            chunk_rows = 6.4 GB / (12328 × 4) = ~129,000 rows
        """
        m          = _mem_info()
        total_gb   = m["total_gb"] if m["total_gb"] > 0 else 128.0
        budget_gb  = total_gb * target_pct
        bytes_per_row = n_cols * np.dtype(dtype).itemsize
        chunk_rows = max(1, int(budget_gb * 1e9 / bytes_per_row))
        return chunk_rows

    def estimate_max_loaded_elems(
        self,
        target_pct: float = 0.05,
        dtype             = np.float32,
    ) -> int:
        """
        Compute maximum total elements that fit in target_pct of total RAM.
        Returns plain int (never a tuple).

        Example:
            128 GB machine, target_pct=0.05:
            budget = 6.4 GB → 1,600,000,000 float32 elements
        """
        m         = _mem_info()
        total_gb  = m["total_gb"] if m["total_gb"] > 0 else 128.0
        budget_gb = total_gb * target_pct
        n_elems   = int(budget_gb * 1e9 / np.dtype(dtype).itemsize)
        return n_elems

    # ----------------------------------------------------------
    # context manager — starts background monitors
    # ----------------------------------------------------------

    def __enter__(self):
        self._stop_event.clear()
        self._killing = False
        self._start_log_monitor()
        self._start_kill_monitor()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_event.set()
        if self._log_thread and self._log_thread.is_alive():
            self._log_thread.join(timeout=2.0)
        if self._kill_thread and self._kill_thread.is_alive():
            self._kill_thread.join(timeout=2.0)
        return False   # do not suppress exceptions

    # ----------------------------------------------------------
    # background threads
    # ----------------------------------------------------------

    def _start_log_monitor(self) -> None:
        """Background thread: log RAM every monitor_interval seconds."""
        def _run():
            while not self._stop_event.wait(timeout=self.monitor_interval):
                m = _mem_info()
                msg = (
                    f"[monitor] RAM: {m['used_gb']:.1f}/{m['total_gb']:.1f} GB "
                    f"({m['used_pct']*100:.1f}%) | "
                    f"available={m['available_gb']:.1f} GB"
                )
                if m["used_pct"] >= self.warn_pct:
                    log.warning(msg)
                else:
                    log.info(msg)

        self._log_thread = threading.Thread(target=_run, daemon=True,
                                            name="MemGuard-log")
        self._log_thread.start()

    def _start_kill_monitor(self) -> None:
        """Background thread: send SIGTERM→SIGKILL if RAM exceeds kill_pct."""
        def _run():
            while not self._stop_event.wait(timeout=self.kill_interval):
                m = _mem_info()
                if m["used_pct"] >= self.kill_pct and not self._killing:
                    log.error(
                        f"[kill-monitor] CRITICAL RAM: "
                        f"{m['used_pct']*100:.1f}% ≥ {self.kill_pct*100:.0f}% — "
                        f"sending SIGTERM (SIGKILL in {self.grace_period:.0f}s)"
                    )
                    self._trigger_kill()
                    break

        self._kill_thread = threading.Thread(target=_run, daemon=True,
                                             name="MemGuard-kill")
        self._kill_thread.start()

    def _trigger_kill(self) -> None:
        """Send SIGTERM now, SIGKILL after grace_period."""
        if self._killing:
            return
        self._killing = True
        pid = os.getpid()

        def _kill_sequence():
            try:
                log.error(f"[MemoryGuard] SIGTERM → pid {pid}")
                os.kill(pid, signal.SIGTERM)
                time.sleep(self.grace_period)
                log.error(f"[MemoryGuard] SIGKILL → pid {pid}")
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass   # process already gone

        t = threading.Thread(target=_kill_sequence, daemon=True,
                             name="MemGuard-killer")
        t.start()