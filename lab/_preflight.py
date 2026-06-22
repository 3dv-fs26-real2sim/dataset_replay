"""Startup guards for the training/eval entry scripts.

Two recurring failure modes on a laptop-class (8 GB) GPU, both surfacing as the
cryptic ``'NoneType' object has no attribute 'create_rigid_body_view'`` (PhysX
could not allocate VRAM → the physics sim view is None):

1. A *previous* Isaac Sim run crashed/was killed and left a zombie process
   holding GPU memory — the next run then OOMs at sim-view creation.
2. The desktop / browser is using enough VRAM that little is left.

:func:`preflight` makes the EULA non-interactive and warns early (with the exact
remedy) when free VRAM looks too low, instead of letting it fail deep inside the
sim with an unreadable traceback. Pure stdlib — safe to call before the app boots.
"""

from __future__ import annotations

import os
import subprocess


def _gpu_free_and_apps() -> tuple[int | None, list[str]]:
    """Return (free_MiB, [process lines]) via nvidia-smi, or (None, []) if absent."""
    try:
        free = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        free_mib = int(free.stdout.strip().splitlines()[0])
        app_lines = [ln for ln in apps.stdout.strip().splitlines() if ln.strip()]
        return free_mib, app_lines
    except Exception:
        return None, []


def preflight(min_free_mib: int = 3500) -> None:
    """Accept the EULA non-interactively and warn on low free VRAM.

    ``min_free_mib`` is a soft threshold (default 3.5 GB) — below it the sim is
    likely to OOM at sim-view creation. We warn (not abort) so the user can still
    try, but point at the usual cause (a leftover Isaac process holding VRAM).
    """
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

    free_mib, apps = _gpu_free_and_apps()
    if free_mib is None:
        return
    print(f"[preflight] GPU free VRAM: {free_mib} MiB", flush=True)
    if free_mib >= min_free_mib:
        return
    print(f"[preflight] WARNING: only {free_mib} MiB free (< {min_free_mib} MiB). PhysX may "
          f"fail to allocate VRAM and crash with 'create_rigid_body_view' / 'Failed to create "
          f"simulation view backend'.", flush=True)
    leftover = [a for a in apps if "python" in a.lower()]
    if leftover:
        print("[preflight] Leftover GPU processes (a previous crashed run may be holding VRAM):", flush=True)
        for a in leftover:
            print(f"             {a}", flush=True)
        print("[preflight] Free it with:  kill -9 <pid>   (or: pkill -9 -f isaac-sim)", flush=True)
    print("[preflight] Also: always run with --headless, and lower --num_envs if needed.", flush=True)
