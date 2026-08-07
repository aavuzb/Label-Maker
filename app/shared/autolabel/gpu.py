"""Hardware GPU detection and live status.

Deliberately independent of what torch/onnxruntime themselves report:
`torch.cuda.is_available()` and `"CUDAExecutionProvider" in
onnxruntime.get_available_providers()` both return False whenever the
*installed package* happens to be a CPU-only build — which says nothing
about whether the machine actually has a GPU. Querying `nvidia-smi`
directly (it ships with the NVIDIA driver itself, independent of any
Python package) tells the two apart, so the UI can say "you have a GPU but
the installed packages can't use it" instead of a flat, misleading "no
GPU found".
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

_NVIDIA_SMI = "nvidia-smi"


def apply_gpu_library_path() -> None:
    """Must run before any other import in this process ever touches
    onnxruntime/torch — see main.py's call site.

    pip-installed NVIDIA packages (nvidia-cublas-cu12, nvidia-cudnn-cu13,
    ...) ship real .so files under site-packages/nvidia/*/lib, but nothing
    adds that directory to the dynamic linker's search path on its own.
    PyTorch works around this internally (it dlopen()s its own dependencies
    by explicit path), but onnxruntime's CUDA execution provider does a
    plain dlopen("libcublasLt.so.13") and silently falls back to CPU if the
    linker can't find it — exactly the "GPU is right there but the app
    can't find it" symptom this whole module exists to avoid. Fixed here by
    adding every such lib directory to LD_LIBRARY_PATH.

    Only Linux needs this (the platform LD_LIBRARY_PATH applies to; Windows
    resolves DLLs via PATH instead, and onnxruntime-gpu isn't published for
    macOS anyway). A no-op if no such packages are installed, or the path
    is already correct.

    LD_LIBRARY_PATH is read once by the dynamic linker at process start —
    changing os.environ after that point doesn't reliably reach every
    dlopen() a library performs later — so when the path is wrong, this
    re-execs the current process instead of just mutating the environment
    in place, ensuring the fix actually takes effect for this run.
    """
    import os
    import site
    import sys
    from pathlib import Path

    if not sys.platform.startswith("linux"):
        return

    lib_dirs: list[str] = []
    for site_dir in site.getsitepackages():
        nvidia_root = Path(site_dir) / "nvidia"
        if not nvidia_root.is_dir():
            continue
        for lib_dir in sorted(nvidia_root.glob("*/lib")):
            if lib_dir.is_dir():
                lib_dirs.append(str(lib_dir))
    if not lib_dirs:
        return

    existing_parts = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    missing = [d for d in lib_dirs if d not in existing_parts]
    if not missing:
        return

    os.environ["LD_LIBRARY_PATH"] = ":".join(missing + [p for p in existing_parts if p])
    os.execv(sys.executable, [sys.executable, *sys.argv])

# Below this much free memory, a run is likely to hit an out-of-memory
# error partway through rather than actually complete — there's no
# generic way to know a given model's real footprint ahead of time, so
# this is just a "GPU is essentially full" floor, not a precise estimate.
MIN_FREE_MEMORY_MB = 750
# Above this utilization, the GPU is already doing meaningful other work
# (another app, a game, a training run) and queuing more onto it risks
# contending for the same compute instead of running promptly.
BUSY_UTILIZATION_PERCENT = 85


@dataclass
class GpuStatus:
    hardware_present: bool  # an NVIDIA GPU physically exists on this machine
    name: str | None = None
    total_mb: int | None = None
    used_mb: int | None = None
    free_mb: int | None = None
    utilization_percent: int | None = None
    torch_cuda_ready: bool = False  # the installed torch build can actually use it
    onnxruntime_cuda_ready: bool = False  # the installed onnxruntime build can actually use it

    @property
    def library_ready(self) -> bool:
        """Whether *any* installed inference backend can actually put work
        on the GPU — hardware alone isn't enough if every installed
        package happens to be a CPU-only build."""
        return self.torch_cuda_ready or self.onnxruntime_cuda_ready

    @property
    def usable_now(self) -> bool:
        """Whether starting a GPU run right now is expected to go well —
        hardware present, at least one backend can use it, not already
        busy, and not nearly out of memory."""
        return self.hardware_present and self.library_ready and not self.busy_or_low_memory

    @property
    def busy_or_low_memory(self) -> bool:
        if self.utilization_percent is not None and self.utilization_percent >= BUSY_UTILIZATION_PERCENT:
            return True
        if self.free_mb is not None and self.free_mb < MIN_FREE_MEMORY_MB:
            return True
        return False

    def unusable_reason(self) -> str:
        """Human-readable reason `usable_now` is False. Only meaningful
        when it actually is False."""
        if not self.hardware_present:
            return "No GPU was detected on this machine."
        if not self.library_ready:
            return (
                "A GPU is present, but the installed onnxruntime/torch packages are CPU-only builds. "
                "Install onnxruntime-gpu and/or a CUDA build of torch to use it."
            )
        if self.utilization_percent is not None and self.utilization_percent >= BUSY_UTILIZATION_PERCENT:
            return f"The GPU is currently busy ({self.utilization_percent}% utilization)."
        if self.free_mb is not None and self.free_mb < MIN_FREE_MEMORY_MB:
            return f"The GPU doesn't have enough free memory right now ({self.free_mb:,} MB free)."
        return "The GPU isn't usable right now."

    def describe(self) -> str:
        if not self.hardware_present:
            return "No GPU detected on this machine."
        parts = [self.name or "GPU"]
        if self.free_mb is not None and self.total_mb is not None:
            parts.append(f"{self.free_mb:,} / {self.total_mb:,} MB free")
        if self.utilization_percent is not None:
            parts.append(f"{self.utilization_percent}% busy")
        detail = " — ".join(parts)
        if not self.library_ready:
            detail += " (installed onnxruntime/torch packages are CPU-only — see above)"
        return detail


def _query_nvidia_smi() -> dict | None:
    """Best-effort hardware query. Returns None if there's no NVIDIA
    driver installed (covers "no NVIDIA GPU" and "GPU present but driver
    missing" alike — either way, nothing here can use it)."""
    if shutil.which(_NVIDIA_SMI) is None:
        return None
    try:
        completed = subprocess.run(
            [
                _NVIDIA_SMI,
                "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    output = completed.stdout.strip()
    if not output:
        return None
    # First GPU only — multi-GPU device selection isn't something this
    # app's single-model auto-labeling run needs to handle.
    fields = [f.strip() for f in output.splitlines()[0].split(",")]
    if len(fields) != 5:
        return None
    name, total, used, free, util = fields
    try:
        return {
            "name": name,
            "total_mb": int(total),
            "used_mb": int(used),
            "free_mb": int(free),
            "utilization_percent": int(util),
        }
    except ValueError:
        return None


def _torch_cuda_ready() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 — torch not installed, or a broken install
        return False


def _onnxruntime_cuda_ready() -> bool:
    try:
        import onnxruntime as ort

        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:  # noqa: BLE001
        return False


def detect_gpu() -> GpuStatus:
    """A fresh, live snapshot — call this each time current status is
    needed (e.g. right before starting a run), not once and cached, since
    utilization/free memory change constantly as other processes use the
    GPU."""
    hw = _query_nvidia_smi()
    return GpuStatus(
        hardware_present=hw is not None,
        name=hw["name"] if hw else None,
        total_mb=hw["total_mb"] if hw else None,
        used_mb=hw["used_mb"] if hw else None,
        free_mb=hw["free_mb"] if hw else None,
        utilization_percent=hw["utilization_percent"] if hw else None,
        torch_cuda_ready=_torch_cuda_ready(),
        onnxruntime_cuda_ready=_onnxruntime_cuda_ready(),
    )
