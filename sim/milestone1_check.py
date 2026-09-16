"""MILESTONE 1 gate: load the bimanual scene, render `top`, `left_wrist` and
`right_wrist` offscreen at 640x480, write PNGs, print the state/action dims."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mj_backend import select_backend  # noqa: E402

BACKEND = select_backend(verbose=True)

import mujoco  # noqa: E402

from bimanual_scene import (  # noqa: E402
    ARMS,
    JOINT_NAMES,
    OBJECTS,
    build_model,
    reset_home,
    state_vector,
)

W, H = 640, 480
OUT = Path(__file__).resolve().parent.parent / "out" / "milestone1"


def save_png(path: Path, rgb: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(rgb).save(path)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model, idx = build_model()
    data = mujoco.MjData(model)
    reset_home(model, data, idx)

    # Let the position servos settle so the arms are actually holding the pose.
    for _ in range(500):
        mujoco.mj_step(model, data)
    settle_t = time.time() - t0

    print(f"[build+settle] {settle_t:.2f}s  backend={BACKEND}")
    print(f"model: nq={model.nq} nv={model.nv} nu={model.nu} "
          f"nbody={model.nbody} ngeom={model.ngeom} ncam={model.ncam} neq={model.neq}")

    renderer = mujoco.Renderer(model, height=H, width=W)
    paths = []
    for cam in ("top", "left_wrist", "right_wrist"):
        renderer.update_scene(data, camera=cam)
        rgb = renderer.render()
        p = OUT / f"{cam}.png"
        save_png(p, rgb)
        paths.append(p)
        print(f"  rendered {cam:12s} -> {p}  shape={rgb.shape} "
              f"mean={rgb.mean():.1f} std={rgb.std():.1f}")
    renderer.close()

    state = state_vector(data, idx)
    action = np.concatenate(
        [[data.ctrl[a] for a in idx.arm_act_ids[arm]] for arm in ARMS]
    )
    print()
    print("=== dimensionality ===")
    print(f"observation.state : {state.shape[0]}-D "
          f"({len(ARMS)} arms x {len(JOINT_NAMES)} joints)")
    print(f"action            : {action.shape[0]}-D (same joint-space layout)")
    print(f"joint order       : " + ", ".join(
        f"{arm}_{j}" for arm in ARMS for j in JOINT_NAMES))
    print(f"state  = {np.round(state, 4)}")
    print(f"action = {np.round(action, 4)}")
    print(f"images            : 3 x ({H}, {W}, 3) uint8  "
          f"keys top / left_wrist / right_wrist")
    print(f"free objects      : {', '.join(OBJECTS)}")
    for o in OBJECTS:
        print(f"    {o:6s} pos = {np.round(data.xpos[idx.obj_body[o]], 4)}")

    ok = all(p.exists() and p.stat().st_size > 5000 for p in paths)
    print()
    print("MILESTONE 1:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
