"""Torch-free closed-loop SmolVLA OpenVINO evaluation in MuJoCo.

The exported ``smolvla.xml`` is a full action-chunk graph.  Its traced forward
contains image/language prefix encoding, KV-cache construction and the ten-step
Euler flow-matching loop.  Only the manifest-declared preprocessing and
postprocessing live outside the graph and are reproduced here with NumPy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from sim.mj_backend import select_backend

select_backend()

import mujoco  # noqa: E402
import openvino as ov  # noqa: E402
import openvino_tokenizers  # noqa: F401, E402  (register tokenizer ops)

# The protected simulator intentionally supports direct ``python sim/episode.py``
# execution and therefore uses top-level sibling imports.  Put its directory on
# sys.path here instead of requiring edits to the concurrently maintained files.
SIM_DIR = Path(__file__).resolve().parent / "sim"
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

from sim.episode import BimanualEnv, TASK_STRINGS, success_cup_handoff  # noqa: E402
from sim.bimanual_scene import CUP_HALF_H, HOME_QPOS, PLATE_RADIUS, PLATE_TOP_Z, object_pos  # noqa: E402


CAMERAS = ("top", "left_wrist", "right_wrist")
STATE_ACTION_ORDER = ("right", "left")
TASK_B = "slide the cup to the middle of the table"
CONTROL_HZ = 50
EPS = 1e-8


def _cpu_name() -> str:
    if os.name == "nt":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor() or platform.uname().processor or "unknown"


def _nearest_rank(values: list[float], percentile: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = -(-percentile * len(ordered) // 100) - 1
    return ordered[idx]


def _timing(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean_ms": None, "p50_ms": None, "p95_ms": None, "fps": None, "runs": 0}
    mean = statistics.mean(values)
    p95 = _nearest_rank(values, 95)
    assert p95 is not None
    return {
        "mean_ms": round(mean, 2),
        "p50_ms": round(sorted(values)[len(values) // 2], 2),
        "p95_ms": round(p95, 2),
        "fps": round(1000.0 / mean, 1),
        "runs": len(values),
    }


def _port_name(port: Any) -> str:
    try:
        return port.get_any_name()
    except Exception:
        return port.any_name


def _port_dtype(port: Any) -> np.dtype:
    return np.dtype(port.get_element_type().to_dtype())


def _resize_square_chw(image: np.ndarray, size: int) -> np.ndarray:
    """Dependency-free bilinear resize matching the export's square 256->512 path."""
    src = np.asarray(image, dtype=np.float32)
    if src.ndim == 3 and src.shape[-1] == 3:
        src = np.transpose(src, (2, 0, 1))
    _, height, width = src.shape
    if height == size and width == size:
        return src
    ys = (np.arange(size, dtype=np.float32) + 0.5) * height / size - 0.5
    xs = (np.arange(size, dtype=np.float32) + 0.5) * width / size - 0.5
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    wy = ys - y0
    wx = xs - x0
    y0 = np.clip(y0, 0, height - 1)
    x0 = np.clip(x0, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    x1 = np.clip(x0 + 1, 0, width - 1)
    top = src[:, y0[:, None], x0[None, :]] * (1.0 - wx)[None, None, :]
    top += src[:, y0[:, None], x1[None, :]] * wx[None, None, :]
    bottom = src[:, y1[:, None], x0[None, :]] * (1.0 - wx)[None, None, :]
    bottom += src[:, y1[:, None], x1[None, :]] * wx[None, None, :]
    return top * (1.0 - wy)[None, :, None] + bottom * wy[None, :, None]


def _state_right_left(env: BimanualEnv) -> np.ndarray:
    chunks = []
    for arm in STATE_ACTION_ORDER:
        chunks.append(np.asarray([env.data.qpos[q] for q in env.idx.arm_qpos_ids[arm]], dtype=np.float32))
    return np.concatenate(chunks)


def _stats_vector(stats: dict[str, Any], key: str, field: str, width: int) -> np.ndarray:
    row = stats.get(key, {})
    value = np.asarray(row.get(field, [0.0 if field == "mean" else 1.0] * width), dtype=np.float32)
    if value.size != width:
        raise ValueError(f"{key}.{field} has {value.size} values, expected {width}")
    return value


class OpenVINOPolicy:
    def __init__(self, bundle: Path, device: str, stats: dict[str, Any]) -> None:
        self.bundle = bundle
        self.device = device
        self.stats = stats
        core = ov.Core()
        model_path = bundle / "smolvla.xml"
        tokenizer_path = bundle / "tokenizer.xml"
        if not model_path.exists() or not tokenizer_path.exists():
            raise FileNotFoundError(f"bundle needs smolvla.xml and tokenizer.xml: {bundle}")

        t0 = time.perf_counter()
        self.compiled = core.compile_model(core.read_model(str(model_path)), device)
        self.compile_s = time.perf_counter() - t0
        self.request = self.compiled.create_infer_request()
        self.inputs = {_port_name(port): port for port in self.compiled.inputs}
        self.outputs = [_port_name(port) for port in self.compiled.outputs]

        tok_model = core.read_model(str(tokenizer_path))
        self.tokenizer = core.compile_model(tok_model, "CPU")
        self.tokenizer_input = _port_name(self.tokenizer.inputs[0])
        self.tokenizer_outputs = [_port_name(port) for port in self.tokenizer.outputs]
        self.image_size = self._discover_image_size()
        self.latencies_ms: list[float] = []
        self.output_finite = True
        print("IR inputs:")
        for name, port in self.inputs.items():
            print(f"  {name}: {port.get_partial_shape()} {port.get_element_type()}")
        print("IR outputs:")
        for port in self.compiled.outputs:
            print(f"  {_port_name(port)}: {port.get_partial_shape()} {port.get_element_type()}")

    def _discover_image_size(self) -> int:
        for name, port in self.inputs.items():
            if "image" in name.lower() and "mask" not in name.lower():
                shape = list(port.get_partial_shape())
                dims = [d.get_length() for d in shape if not d.is_dynamic]
                if len(dims) >= 2:
                    return int(dims[-1])
        return 512

    def _tokenize(self, instruction: str) -> tuple[np.ndarray, np.ndarray]:
        raw = self.tokenizer({self.tokenizer_input: [instruction.rstrip() + "\n"]})
        by_name = {_port_name(port): np.asarray(raw[port]) for port in self.tokenizer.outputs}
        ids = next((v for k, v in by_name.items() if "input_ids" in k), None)
        mask = next((v for k, v in by_name.items() if "attention_mask" in k), None)
        if ids is None or mask is None:
            ordered = [by_name[n] for n in self.tokenizer_outputs]
            ids, mask = ordered[0], ordered[1]
        return ids, mask.astype(np.bool_)

    def _feed(self, frames: dict[str, np.ndarray], state: np.ndarray, instruction: str) -> dict[str, np.ndarray]:
        mean = _stats_vector(self.stats, "observation.state", "mean", state.size)
        std = _stats_vector(self.stats, "observation.state", "std", state.size)
        state_norm = ((state - mean) / (std + EPS))[None, :].astype(np.float32)
        images = []
        for camera in CAMERAS:
            image = _resize_square_chw(frames[camera], self.image_size) / 255.0
            images.append((image * 2.0 - 1.0)[None, ...])
        image_tensor = np.stack(images, axis=0).astype(np.float32)
        image_masks = np.ones((len(CAMERAS), 1), dtype=np.bool_)
        token_ids, token_mask = self._tokenize(instruction)
        canonical = {
            "images": image_tensor,
            "image_masks": image_masks,
            "state": state_norm,
            "tokenized_prompt": token_ids,
            "tokenized_prompt_mask": token_mask,
        }
        feed: dict[str, np.ndarray] = {}
        for name, port in self.inputs.items():
            key = name.lower().replace(".", "_").replace("/", "_")
            matches = [k for k in canonical if key == k or key.endswith(k)]
            if not matches:
                if "mask" in key and "image" in key:
                    matches = ["image_masks"]
                elif "mask" in key:
                    matches = ["tokenized_prompt_mask"]
                elif "token" in key or "prompt" in key:
                    matches = ["tokenized_prompt"]
                elif "image" in key:
                    matches = ["images"]
                elif "state" in key:
                    matches = ["state"]
            if not matches:
                # torch/OpenVINO occasionally gives an otherwise unnamed input a
                # numeric tensor id (observed as ``1762`` for state).  Bind that
                # port from its unique manifest-declared static shape.
                shape = tuple(
                    1 if dim.is_dynamic else dim.get_length()
                    for dim in port.get_partial_shape()
                )
                shape_matches = [k for k, value in canonical.items() if np.asarray(value).shape == shape]
                if len(shape_matches) == 1:
                    matches = shape_matches
            if not matches:
                raise KeyError(f"cannot map IR input {name!r}; ports={list(self.inputs)}")
            feed[name] = np.asarray(canonical[matches[0]], dtype=_port_dtype(port))
        return feed

    def predict(self, frames: dict[str, np.ndarray], state: np.ndarray, instruction: str) -> np.ndarray:
        feed = self._feed(frames, state, instruction)
        started = time.perf_counter()
        result = self.request.infer(feed)
        self.latencies_ms.append((time.perf_counter() - started) * 1000.0)
        value = np.asarray(next(iter(result.values())), dtype=np.float32)
        finite = bool(np.isfinite(value).all())
        self.output_finite = self.output_finite and finite
        if not finite:
            raise FloatingPointError("OpenVINO output is not finite")
        if value.ndim == 3:
            value = value[0]
        mean = _stats_vector(self.stats, "action", "mean", value.shape[-1])
        std = _stats_vector(self.stats, "action", "std", value.shape[-1])
        return value * (std + EPS) + mean


class VideoSink:
    def __init__(self, path: Path, fps: int) -> None:
        self.path = path
        self.frames_dir = path.with_suffix("")
        self.writer = None
        self.index = 0
        try:
            import imageio.v2 as imageio

            path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=1)
        except Exception as exc:
            print(f"VIDEO ENCODER FAILED {path}: {exc}; writing PNG frames")
            self.frames_dir.mkdir(parents=True, exist_ok=True)

    def add(self, frame: np.ndarray) -> None:
        if self.writer is not None:
            try:
                self.writer.append_data(frame)
                self.index += 1
                return
            except Exception as exc:
                print(f"VIDEO WRITE FAILED {self.path}: {exc}; subsequent frames are PNG")
                try:
                    self.writer.close()
                finally:
                    self.writer = None
                    self.frames_dir.mkdir(parents=True, exist_ok=True)
        import imageio.v3 as iio

        iio.imwrite(self.frames_dir / f"frame_{self.index:06d}.png", frame)
        self.index += 1

    def close(self) -> str:
        if self.writer is not None:
            self.writer.close()
            return str(self.path)
        return str(self.frames_dir)


def _render_top(env: BimanualEnv) -> np.ndarray:
    if env.renderer is None:
        raise RuntimeError("environment renderer is unavailable")
    env.renderer.update_scene(env.data, camera="top")
    return env.renderer.render().copy()


def _apply_action(env: BimanualEnv, action: np.ndarray) -> None:
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    if action.size != 12:
        raise ValueError(f"policy action has {action.size} values, expected 12")
    for i, arm in enumerate(STATE_ACTION_ORDER):
        q = action[i * 6 : (i + 1) * 6].copy()
        for k, actuator in enumerate(env.idx.arm_act_ids[arm]):
            if bool(env.model.actuator_ctrllimited[actuator]):
                lo, hi = env.model.actuator_ctrlrange[actuator]
                q[k] = np.clip(q[k], lo, hi)
            env.data.ctrl[actuator] = q[k]
        env.q_cmd[arm] = q
    substeps = max(1, int(round((1.0 / CONTROL_HZ) / float(env.model.opt.timestep))))
    for _ in range(substeps):
        mujoco.mj_step(env.model, env.data)
    if not bool(np.isfinite(env.data.qpos).all()):
        env.fault("MuJoCo qpos became non-finite")


def _done_b(env: BimanualEnv) -> tuple[bool, str, float | None]:
    if env.rec.faults:
        return False, f"faulted: {env.rec.faults[0]}", None
    if env.relay is None:
        return False, "no relay target", None
    cup = object_pos(env.data, env.idx, "cup")
    err = float(np.linalg.norm(cup[:2] - np.asarray(env.relay["xy"])))
    rotation = env.data.xmat[env.idx.obj_body["cup"]].reshape(3, 3)
    upright = float(rotation[:, 2] @ np.array([0.0, 0.0, 1.0]))
    right = np.asarray([env.data.qpos[q] for q in env.idx.arm_qpos_ids["right"]])
    home_err = float(np.max(np.abs(right - HOME_QPOS)))
    ok = err <= 0.020 and upright >= 0.85 and home_err <= 0.08
    return ok, f"relay_error_mm={err * 1000:.1f}, upright={upright:.3f}, right_home_max_rad={home_err:.3f}", err


def _outcomes(env: BimanualEnv, plate_xy: np.ndarray) -> tuple[dict[str, bool], dict[str, str], float]:
    a_ok, a_reason = success_cup_handoff(env, {"plate_xy": plate_xy})
    b_ok, b_reason, _ = _done_b(env)
    cup = object_pos(env.data, env.idx, "cup")
    plate = object_pos(env.data, env.idx, "plate")
    return {"A": bool(a_ok), "B": bool(b_ok)}, {"A": a_reason, "B": b_reason}, float(np.linalg.norm(cup[:2] - plate[:2]))


def run_episode(policy: OpenVINOPolicy, seed: int, given: str, out_dir: Path, video: bool, max_steps: int) -> dict[str, Any]:
    env = BimanualEnv(img_w=256, img_h=256, record_images=True, seed=seed)
    sink = VideoSink(out_dir / f"seed_{seed}_{given}.mp4", CONTROL_HZ) if video else None
    started = time.perf_counter()
    step = 0
    instruction = TASK_STRINGS["cup"] if given == "A" else TASK_B
    try:
        env.reset()
        env.randomize_objects()
        plate_xy = object_pos(env.data, env.idx, "plate")[:2].copy()
        outcomes = {"A": False, "B": False}
        while step < max_steps and not env.rec.faults:
            frames = env.capture()
            chunk = policy.predict(frames, _state_right_left(env), instruction)
            for action in chunk:
                _apply_action(env, action)
                step += 1
                if sink is not None:
                    sink.add(_render_top(env))
                outcomes, _, _ = _outcomes(env, plate_xy)
                if outcomes[given] or step >= max_steps:
                    break
            if outcomes[given]:
                break
        outcomes, reasons, cup_error = _outcomes(env, plate_xy)
        cup_xy = object_pos(env.data, env.idx, "cup")[:2]
        target_xy = plate_xy if given == "A" else np.asarray(env.relay["xy"])
        target_error = float(np.linalg.norm(cup_xy - target_xy))
        elapsed = time.perf_counter() - started
        return {
            "seed": seed,
            "instruction_given": given,
            "instruction_text": instruction,
            "steps": step,
            "success": outcomes[given],
            "outcomes": outcomes,
            "outcome_reasons": reasons,
            "final_cup_position_error_to_target_m": round(target_error, 6),
            "final_cup_position_error_to_plate_m": round(cup_error, 6),
            "time_to_success_s": round(step / CONTROL_HZ, 3) if outcomes[given] else None,
            "episode_wall_s": round(elapsed, 3),
            "faults": list(env.rec.faults),
            "video_or_frames": sink.close() if sink is not None else None,
        }
    except Exception as exc:
        if sink is not None:
            media = sink.close()
        else:
            media = None
        return {
            "seed": seed,
            "instruction_given": given,
            "instruction_text": instruction,
            "steps": step,
            "success": False,
            "outcomes": {"A": False, "B": False},
            "outcome_reasons": {"A": "evaluation exception", "B": "evaluation exception"},
            "final_cup_position_error_to_target_m": None,
            "final_cup_position_error_to_plate_m": None,
            "time_to_success_s": None,
            "episode_wall_s": round(time.perf_counter() - started, 3),
            "faults": [f"{type(exc).__name__}: {exc}"],
            "evaluation_error": True,
            "video_or_frames": media,
        }
    finally:
        env.close()


def make_tiled_video(episodes: list[dict[str, Any]], given: str, out_dir: Path) -> str | None:
    paths = [Path(row["video_or_frames"]) for row in episodes if row["instruction_given"] == given and row.get("video_or_frames")]
    paths = [p for p in paths if p.is_file() and p.suffix.lower() == ".mp4"]
    if not paths:
        return None
    import imageio.v2 as imageio

    readers = [imageio.get_reader(path) for path in paths]
    target = out_dir / f"tiled_{given}.mp4"
    writer = imageio.get_writer(target, fps=CONTROL_HZ, codec="libx264", quality=7, macro_block_size=1)
    active = [True] * len(readers)
    last = [None] * len(readers)
    frame_no = 0
    try:
        while any(active):
            for i, reader in enumerate(readers):
                if not active[i]:
                    continue
                try:
                    last[i] = reader.get_data(frame_no)
                except (IndexError, RuntimeError):
                    active[i] = False
            if not any(frame is not None for frame in last):
                break
            cells = []
            blank = np.zeros_like(next(frame for frame in last if frame is not None))
            for i in range(12):
                cells.append(last[i] if i < len(last) and last[i] is not None else blank)
            rows = [np.concatenate(cells[i : i + 4], axis=1) for i in range(0, 12, 4)]
            writer.append_data(np.concatenate(rows, axis=0))
            frame_no += 1
    except Exception as exc:
        print(f"TILED VIDEO FAILED: {exc}")
        return None
    finally:
        writer.close()
        for reader in readers:
            reader.close()
    return str(target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir", required=True, help="precision bundle directory or smolvla.xml")
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed0", type=int, default=1000)
    parser.add_argument("--instruction", choices=("A", "B"), default="A")
    parser.add_argument("--out", required=True)
    parser.add_argument("--swap", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--max-steps", type=int, default=1500)
    args = parser.parse_args()

    bundle = Path(args.ir).resolve()
    if bundle.is_file():
        bundle = bundle.parent
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = json.loads((bundle / "stats.json").read_text(encoding="utf-8"))
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    cpu = _cpu_name()
    precision = bundle.name.lower()
    policy = OpenVINOPolicy(bundle, args.device, stats)
    givens = [args.instruction]
    if args.swap:
        givens.append("B" if args.instruction == "A" else "A")
    episodes = []
    for seed in range(args.seed0, args.seed0 + args.seeds):
        for given in givens:
            row = run_episode(policy, seed, given, out_dir, args.video, args.max_steps)
            episodes.append(row)
            print(f"seed={seed} given={given} success={row['success']} steps={row['steps']} wall={row['episode_wall_s']:.2f}s faults={row['faults']}")

    table = {f"{given}-given/{done}-done": 0 for given in ("A", "B") for done in ("A", "B")}
    for row in episodes:
        for done in ("A", "B"):
            table[f"{row['instruction_given']}-given/{done}-done"] += int(row["outcomes"][done])
    tiled = make_tiled_video(episodes, args.instruction, out_dir) if args.video else None
    ir_bytes = sum((bundle / name).stat().st_size for name in ("smolvla.xml", "smolvla.bin") if (bundle / name).exists())
    inference = _timing(policy.latencies_ms)
    inference.update({"compile_s": round(policy.compile_s, 2), "output_finite": policy.output_finite})
    result = {
        "device": args.device,
        "precision": precision,
        "ir_path": str(bundle / "smolvla.xml"),
        "ir_size_mb": round(ir_bytes / 1e6, 1),
        "openvino_version": ov.__version__,
        "cpu_name": cpu,
        "NOT_THE_REQUIRED_SILICON": "intel" not in cpu.lower() or "core" not in cpu.lower() or "ultra" not in cpu.lower(),
        "seeds": args.seeds,
        "seed0": args.seed0,
        "max_steps": args.max_steps,
        "control_hz": CONTROL_HZ,
        "state_action_order": list(STATE_ACTION_ORDER),
        "graph_contract": {"full_policy": True, "flow_matching_steps_inside_ir": 10, "schedule": "Euler dt=-0.1; t=1.0,0.9,...,0.1"},
        "manifest": manifest,
        "inference": inference,
        "episodes": episodes,
        "successes": sum(int(row["success"]) for row in episodes),
        "instruction_swap_table": table,
        "tiled_video": tiled,
    }
    result_path = out_dir / "eval_results.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"inference p50={inference['p50_ms']} ms p95={inference['p95_ms']} ms calls={inference['runs']} output_finite={inference['output_finite']}")
    print("instruction swap table:", json.dumps(table, sort_keys=True))
    print("NOT_THE_REQUIRED_SILICON:", str(result["NOT_THE_REQUIRED_SILICON"]).lower())
    print("wrote", result_path)
    return 0 if policy.output_finite and not any(row.get("evaluation_error") for row in episodes) else 2


if __name__ == "__main__":
    raise SystemExit(main())
