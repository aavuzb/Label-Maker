"""How many CPU threads the ML inference backends are allowed to use —
kept dependency-free (stdlib only) so it can run at the very top of
main.py, before numpy/scipy/torch/cv2 are imported anywhere in the app.

That ordering isn't cosmetic: numpy's BLAS backend (OpenBLAS here) reads
OPENBLAS_NUM_THREADS once, the moment numpy is first imported, and locks in
that many threads for the rest of the process — setting the env var any
later (e.g. from inside model.py, right before actually loading a model)
has no effect at all, because numpy/cv2 are already imported transitively
by the time the app's main window even finishes constructing. Left
unset, OpenBLAS defaults to one thread per logical core, which is exactly
what pins every core during a long auto-labeling run and starves the
desktop compositor of CPU time — that starvation is what shows up as the
whole screen freezing, not just this app being slow.
"""

from __future__ import annotations

import os

# Every env var an ML library in this stack might read for its own
# BLAS/OpenMP-style thread pool. Harmless to set the ones a given machine's
# libraries don't use.
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def cpu_thread_budget() -> int:
    """Leaves at least one core free for the OS/desktop, however many the
    machine has."""
    total = os.cpu_count() or 4
    return max(1, total - 1)


def apply_cpu_thread_budget() -> None:
    """Must be called before numpy/scipy/torch/cv2 are imported anywhere
    in the process — see module docstring. `setdefault` so a value the user
    already set themselves (e.g. power users tuning this by hand) wins."""
    budget = str(cpu_thread_budget())
    for name in _THREAD_ENV_VARS:
        os.environ.setdefault(name, budget)
