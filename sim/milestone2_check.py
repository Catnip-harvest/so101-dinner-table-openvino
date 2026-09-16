"""MILESTONE 2 gate: run one scripted episode of each task and report timing,
frame counts, IK health and whether the success predicate fires."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from episode import (  # noqa: E402
    CAMERAS,
    FPS,
    TASKS,
    BimanualEnv,
)

OUT = Path(__file__).resolve().parent.parent / "out" / "milestone2"


def save_strip(env, tag: str, every: int = 12) -> None:
    from PIL import Image

    OUT.mkdir(parents=True, exist_ok=True)
    for cam in CAMERAS:
        frames = env.rec.images[cam][::every]
        if not frames:
            return
        h, w, _ = frames[0].shape
        strip = np.concatenate(frames[:10], axis=1)
        Image.fromarray(strip).save(OUT / f"{tag}_{cam}_strip.png")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="cup,plate")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--strip", action="store_true")
    args = ap.parse_args()

    env = BimanualEnv(
        img_w=args.width,
        img_h=args.height,
        record_images=not args.no_images,
        seed=args.seed,
    )
    print(f"fps={FPS} control_dt={1/FPS:.3f}s  images={'off' if args.no_images else f'{args.width}x{args.height}'}")

    allok = True
    for name in args.tasks.split(","):
        lang, script, predicate = TASKS[name]
        for rep in range(args.reps):
            env.reset()
            env.randomize_objects()
            env.randomize_visuals()
            t0 = time.perf_counter()
            info = script(env)
            wall = time.perf_counter() - t0
            ok, why = predicate(env, info)
            n = len(env.rec)
            print(
                f"[{name}] rep{rep} frames={n:4d} ({n/FPS:5.2f}s sim)  "
                f"wall={wall:6.2f}s  {n/wall:6.1f} fps  "
                f"ik_maxerr={env.rec.max_ik_pos_err*1000:5.1f}mm "
                f"ik_fail={env.rec.n_ik_fail:3d}  "
                f"{'SUCCESS' if ok else 'FAIL   '}  {why}"
            )
            print(f'        task string: "{lang}"')
            if args.strip and rep == 0:
                save_strip(env, name)
            allok &= ok
    env.close()
    print("MILESTONE 2:", "PASS" if allok else "FAIL")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
