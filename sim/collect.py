"""Parallel scripted demonstration collector for the dinner-table simulator."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from mj_backend import select_backend

select_backend()

from bimanual_scene import ARMS, CARRY_MODE, CUP_HALF_H, HOME_QPOS  # noqa: E402
from episode import (  # noqa: E402
    BimanualEnv,
    CAMERAS,
    FPS,
    TASK_STRINGS,
    go_home,
    object_pos,
    pick_cup,
    place_cup,
    success_cup_handoff,
    task_cup_relay,
)

IMAGE_SIZE = 256
DART_RATE = 0.30
DART_SIGMA = 0.02
INSTRUCTION_B = "slide the cup to the middle of the table"
RECORD_ARMS = ("right", "left")
STATE_ACTION_ORDER = (
    "right.shoulder_pan", "right.shoulder_lift", "right.elbow_flex",
    "right.wrist_flex", "right.wrist_roll", "right.gripper",
    "left.shoulder_pan", "left.shoulder_lift", "left.elbow_flex",
    "left.wrist_flex", "left.wrist_roll", "left.gripper",
)

_ENV: "RecordingEnv | None" = None
_OUT: Path | None = None


class _OrderedSamples(list):
    """Adapt the parent's arm order only at the recording boundary."""

    _indices = [6 * ARMS.index(arm) + joint
                for arm in RECORD_ARMS for joint in range(6)]

    def append(self, value: np.ndarray) -> None:
        super().append(np.asarray(value)[self._indices].copy())


class _ExecutionRNG:
    """Keep seeded DART draws right-first while the parent executes in ARMS order."""

    _indices = [6 * RECORD_ARMS.index(arm) + joint
                for arm in ARMS for joint in range(6)]

    def __init__(self, generator) -> None:
        self.generator = generator

    def __getattr__(self, name):
        return getattr(self.generator, name)

    def normal(self, loc=0.0, scale=1.0, size=None):
        values = self.generator.normal(loc, scale, size)
        return values[self._indices] if size == 12 else values


class _VideoFrames:
    """Stream the parent's pre-step images without retaining them in RAM."""

    def __init__(self, env: "RecordingEnv", camera: str) -> None:
        self.env, self.camera = env, camera

    def append(self, frame: np.ndarray) -> None:
        writer = self.env._writers.get(self.camera)
        if writer is not None:
            writer.append_data(frame)


class RecordingEnv(BimanualEnv):
    """Parent physics/DART/phase execution with streamed right-then-left records."""

    def __init__(self, seed: int) -> None:
        super().__init__(img_w=IMAGE_SIZE, img_h=IMAGE_SIZE, record_images=True, seed=seed)
        self._writers: dict[str, object] = {}

    def reset(self, noise_sigma: float = 0.0) -> None:
        super().reset(noise_sigma=noise_sigma)
        self.rec.states = _OrderedSamples()
        self.rec.actions = _OrderedSamples()
        self.rec.images = {cam: _VideoFrames(self, cam) for cam in CAMERAS}

    def _parent_step(self, step, record: bool) -> None:
        # Adapt only the assignment of the existing 12 Gaussian draws. The
        # parent owns IK, clean labels, ctrl writes, substeps and carry hooks.
        rng = self.rng
        if self.noise_sigma > 0.0:
            self.rng = _ExecutionRNG(rng)
        try:
            step(record=record)
        finally:
            self.rng = rng

    def joint_step(self, record: bool = True) -> None:
        self._parent_step(super().joint_step, record)

    def control_step(self, record: bool = True) -> None:
        self._parent_step(super().control_step, record)

    def begin_videos(self, paths: dict[str, Path]) -> str:
        try:
            import imageio.v2 as imageio
            import imageio_ffmpeg

            imageio_ffmpeg.get_ffmpeg_exe()
            opened = {}
            for cam, path in paths.items():
                opened[cam] = imageio.get_writer(
                    str(path), fps=FPS, codec="libx264", quality=8,
                    macro_block_size=None, pixelformat="yuv420p",
                )
            self._writers = opened
            return "mp4"
        except Exception:
            for writer in locals().get("opened", {}).values():
                try:
                    writer.close()
                except Exception:
                    pass
            self.close_videos()
            self._writers = {cam: _PngWriter(path.with_suffix("")) for cam, path in paths.items()}
            return "png"

    def close_videos(self) -> None:
        for writer in self._writers.values():
            try:
                writer.close()
            except Exception:
                pass
        self._writers = {}


class _PngWriter:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index = 0

    def append_data(self, image: np.ndarray) -> None:
        from PIL import Image
        Image.fromarray(image).save(self.directory / f"{self.index:06d}.png")
        self.index += 1

    def close(self) -> None:
        return None


def _task_b(env: RecordingEnv) -> dict:
    relay = env.relay
    if relay is None:
        env.fault("no relay spot for this layout")
        return {}
    pick_cup(env, "right")
    place_cup(env, "right", relay["xy"], rest_z=CUP_HALF_H + 0.001, dz=relay["dz_put"])
    go_home(env, "right")
    env.hold(0.6)
    return {"relay_xy": np.asarray(relay["xy"], dtype=float)}


def _success_b(env: RecordingEnv, info: dict) -> tuple[bool, str]:
    if env.rec.faults:
        return False, f"faulted: {env.rec.faults[0]}"
    cup = object_pos(env.data, env.idx, "cup")
    rotation = env.data.xmat[env.idx.obj_body["cup"]].reshape(3, 3)
    distance = float(np.linalg.norm(cup[:2] - info["relay_xy"]))
    upright = float(rotation[:, 2] @ np.array([0.0, 0.0, 1.0]))
    right_q = np.asarray(env.data.qpos[env.idx.arm_qpos_ids["right"]])
    home_error = float(np.max(np.abs(right_q - HOME_QPOS)))
    if distance > 0.020:
        return False, f"cup {distance * 1000:.0f}mm from relay point"
    if upright < 0.85:
        return False, f"cup tipped over (up={upright:.2f})"
    if home_error > 0.08:
        return False, f"right arm {home_error * 1000:.0f}mrad from home"
    return True, "success"


def _worker_init(out: str, base_seed: int) -> None:
    global _ENV, _OUT
    _OUT = Path(out)
    _ENV = RecordingEnv(base_seed + os.getpid())
    print(f"worker pid={os.getpid()} BIMANUAL_CARRY={os.environ.get('BIMANUAL_CARRY', '<unset>')} "
          f"CARRY_MODE={CARRY_MODE}", flush=True)


def _remove_artifact(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _attempt(spec: tuple[int, int, str, bool]) -> dict:
    attempt, seed, instruction, perturbed = spec
    assert _ENV is not None and _OUT is not None
    env = _ENV
    env.rng = np.random.default_rng(seed)
    env.reset(noise_sigma=DART_SIGMA if perturbed else 0.0)
    env.randomize_objects()
    env.randomize_visuals()
    stem = f"episode_{attempt:06d}"
    temp_paths = {cam: _OUT / f".{stem}.{os.getpid()}.{cam}.mp4" for cam in CAMERAS}
    started = time.perf_counter()
    storage = env.begin_videos(temp_paths)
    try:
        if instruction == "A":
            info = task_cup_relay(env)
            success, reason = success_cup_handoff(env, info)
            task = TASK_STRINGS["cup"]
        else:
            info = _task_b(env)
            success, reason = _success_b(env, info)
            task = INSTRUCTION_B
    except Exception as exc:
        success, reason, task = False, f"exception: {type(exc).__name__}: {exc}", (
            TASK_STRINGS["cup"] if instruction == "A" else INSTRUCTION_B
        )
    finally:
        env.close_videos()
    elapsed = time.perf_counter() - started
    faults = list(env.rec.faults)
    if not success and not faults:
        faults = [reason]
    files: dict[str, str] = {}
    if success:
        states = np.asarray(env.rec.states, dtype=np.float32)
        actions = np.asarray(env.rec.actions, dtype=np.float32)
        timestamps = np.arange(len(states), dtype=np.float64) / FPS
        flags = np.zeros(len(states), dtype=np.uint8)
        if perturbed:
            flags[:] = 1
        npz_path = _OUT / f"{stem}.npz"
        np.savez_compressed(
            npz_path, state=states, action=actions, timestamps=timestamps, flags=flags,
            task=np.asarray(task), seed=np.asarray(seed, dtype=np.int64),
            success=np.asarray(True), faults=np.asarray(faults, dtype=np.str_),
            is_perturbed=np.asarray(perturbed), state_action_order=np.asarray(STATE_ACTION_ORDER),
        )
        files["npz"] = npz_path.name
        for cam, temp in temp_paths.items():
            source = temp if storage == "mp4" else temp.with_suffix("")
            target = _OUT / (f"{stem}_{cam}.mp4" if storage == "mp4" else f"{stem}_{cam}")
            source.replace(target)
            files[cam] = target.name
    else:
        for temp in temp_paths.values():
            _remove_artifact(temp)
            _remove_artifact(temp.with_suffix(""))
    return {
        "attempt": attempt, "seed": seed, "instruction": instruction, "task": task,
        "is_perturbed": perturbed, "success": bool(success), "reason": reason,
        "faults": faults, "frames": len(env.rec.states), "wall_time_s": round(elapsed, 3),
        "storage": storage, "files": files,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    count = parser.add_mutually_exclusive_group(required=True)
    count.add_argument("--episodes", type=int, help="Number of attempts (legacy mode)")
    count.add_argument("--target-kept", type=int, help="Collect until this many successes; stop below 30%% after 200 attempts")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--instruction", choices=("A", "B", "both"), default="both")
    return parser.parse_args()


def _write_manifest(out: Path, results: list[dict], started: float, workers_used: int) -> dict:
    wall = time.perf_counter() - started
    fault_counts = Counter(fault for result in results for fault in result["faults"] if not result["success"])
    per_instruction = defaultdict(lambda: {"attempts": 0, "kept": 0})
    for result in results:
        per_instruction[result["instruction"]]["attempts"] += 1
        per_instruction[result["instruction"]]["kept"] += int(result["success"])
    manifest = {
        "format_version": 1, "fps": FPS, "carry_mode": CARRY_MODE,
        "state_action_order": list(STATE_ACTION_ORDER),
        "attempts": len(results), "kept": sum(r["success"] for r in results),
        "per_fault_counts": dict(sorted(fault_counts.items())),
        "per_instruction_counts": dict(per_instruction), "wall_time_s_total": round(wall, 3),
        "wall_time_s_per_attempted_episode": round(wall / max(1, len(results)), 3),
        "seconds_per_attempted_episode_per_worker": round(wall * workers_used / max(1, len(results)), 3),
        "workers": workers_used, "episodes": sorted(results, key=lambda item: item["attempt"]),
    }
    temporary = out / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(out / "manifest.json")
    return manifest


def _batch_size(attempts: int, kept: int, workers: int, episodes: int | None,
                target_kept: int | None) -> int:
    if target_kept is None:
        return min(workers, episodes - attempts)
    if attempts >= 200 and kept / attempts < 0.30:
        return 0
    size = min(workers, target_kept - kept)
    # Evaluate the gate at exactly 200, without queued surplus work.
    return min(size, 200 - attempts) if attempts < 200 else size


def main() -> int:
    args = parse_args()
    count = args.episodes if args.episodes is not None else args.target_kept
    if count < 1 or args.workers < 1:
        raise SystemExit("episode count and --workers must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise SystemExit(f"output directory is not empty: {args.out}")
    rng = np.random.default_rng(args.seed)
    started = time.perf_counter()
    results = []
    kept = 0
    workers_used = min(args.workers, count)
    manifest = _write_manifest(args.out, results, started, workers_used)
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers_used, initializer=_worker_init,
                  initargs=(str(args.out), args.seed)) as pool:
        while True:
            size = _batch_size(len(results), kept, workers_used, args.episodes, args.target_kept)
            if size <= 0:
                break
            specs = []
            for i in range(len(results), len(results) + size):
                instruction = args.instruction if args.instruction != "both" else ("A" if i % 2 == 0 else "B")
                specs.append((i, args.seed + i, instruction, bool(rng.random() < DART_RATE)))
            for result in pool.imap_unordered(_attempt, specs, chunksize=1):
                results.append(result)
                kept += int(result["success"])
                manifest = _write_manifest(args.out, results, started, workers_used)
    manifest = _write_manifest(args.out, results, started, workers_used)
    if args.target_kept is not None and kept < args.target_kept:
        print("STOP: kept/attempt below 30% at/after 200 attempts", flush=True)
    print(json.dumps({k: v for k, v in manifest.items() if k != "episodes"}, indent=2))
    print(
        f"TIMING seconds_per_attempted_episode_per_worker="
        f"{manifest['seconds_per_attempted_episode_per_worker']:.3f}"
    )
    return 0 if (kept >= args.target_kept if args.target_kept is not None else kept > 0) else 2


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
