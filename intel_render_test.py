"""Minimal MuJoCo render gate used before any Intel workload."""

from __future__ import annotations

import argparse
from pathlib import Path

from sim.mj_backend import select_backend

select_backend(verbose=True)

import imageio.v3 as iio  # noqa: E402

from sim.episode import BimanualEnv  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=r"C:\pai\out\render_test.png")
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    env = BimanualEnv(img_w=256, img_h=256, record_images=True, seed=0)
    try:
        env.reset()
        env.randomize_objects()
        frame = env.capture()["top"]
        iio.imwrite(out, frame)
        print(f"RENDER OK {out} shape={frame.shape} dtype={frame.dtype}")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
