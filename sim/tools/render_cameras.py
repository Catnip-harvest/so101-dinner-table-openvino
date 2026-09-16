"""Render the three fixed camera checks at the neutral scene pose."""

from pathlib import Path
import sys

import mujoco
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bimanual_scene import build_model, reset_home  # noqa: E402

CAMERAS = ("top", "right_wrist", "left_wrist")

model, idx = build_model()
data = mujoco.MjData(model)
reset_home(model, data, idx)
renderer = mujoco.Renderer(model, height=240, width=320)
out = Path(__file__).resolve().parents[2] / "out" / "menagerie_cams"
out.mkdir(parents=True, exist_ok=True)
for camera in CAMERAS:
    renderer.update_scene(data, camera=camera)
    path = out / f"{camera}.png"
    Image.fromarray(renderer.render()).save(path)
    print(path)
renderer.close()
