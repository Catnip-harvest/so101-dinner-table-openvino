"""Pick the MuJoCo OpenGL backend for whatever machine we happen to be on.

This module MUST be imported (and `select_backend()` called) before `mujoco`
is imported for the first time, because MuJoCo reads MUJOCO_GL at import time.

Kaggle / Colab / any headless Linux box -> EGL.
Windows / macOS dev boxes -> leave MuJoCo's default (WGL / CGL), which works
with a hidden window and needs no extra setup.
"""

from __future__ import annotations

import os
import sys


def select_backend(verbose: bool = False) -> str:
    """Set MUJOCO_GL if the user has not already pinned it. Returns the choice."""
    explicit = os.environ.get("MUJOCO_GL")
    if explicit:
        choice = explicit
    elif sys.platform.startswith("linux"):
        # Headless Linux (Kaggle, Colab, CI). EGL needs no X server.
        os.environ["MUJOCO_GL"] = "egl"
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        choice = "egl"
    else:
        # Windows (WGL) and macOS (CGL) are fine on defaults.
        choice = "default"
    if verbose:
        print(f"[mj_backend] platform={sys.platform} MUJOCO_GL={os.environ.get('MUJOCO_GL', '<default>')}")
    return choice


def probe() -> str:
    """Report which backend MuJoCo actually ended up using (call after import)."""
    import mujoco  # noqa: F401

    return os.environ.get("MUJOCO_GL", "default")
