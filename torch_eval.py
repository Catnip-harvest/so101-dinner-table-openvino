"""Closed-loop eval of a SmolVLA checkpoint with lerobot's own torch inference.

No OpenVINO, no physicalai: loads the checkpoint with lerobot 0.6.1, drives the same
BimanualEnv and success checks the OpenVINO evaluator uses, and reports the identical
JSON. Purpose: answer "does run 3 complete the task?" on the box GPU.

    python torch_eval.py --checkpoint <dir> --seeds 5 --seed0 3000 --instruction A --out out/eval
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch

from sim.mj_backend import select_backend
select_backend()
import mujoco  # noqa: E402
from sim.episode import BimanualEnv, TASK_STRINGS, success_cup_handoff  # noqa: E402
from sim.bimanual_scene import HOME_QPOS, object_pos  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402

CAMERAS = ("top", "left_wrist", "right_wrist")
STATE_ACTION_ORDER = ("right", "left")
TASK_B = "slide the cup to the middle of the table"
CONTROL_HZ = 50


def _state_right_left(env):
    chunks = []
    for arm in STATE_ACTION_ORDER:
        chunks.append(np.asarray([env.data.qpos[q] for q in env.idx.arm_qpos_ids[arm]], dtype=np.float32))
    return np.concatenate(chunks)


def _apply_action(env, action):
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    if action.size != 12:
        raise ValueError(f"action has {action.size} values, expected 12")
    for i, arm in enumerate(STATE_ACTION_ORDER):
        q = action[i * 6:(i + 1) * 6].copy()
        for k, act in enumerate(env.idx.arm_act_ids[arm]):
            if bool(env.model.actuator_ctrllimited[act]):
                lo, hi = env.model.actuator_ctrlrange[act]
                q[k] = np.clip(q[k], lo, hi)
            env.data.ctrl[act] = q[k]
        env.q_cmd[arm] = q
    substeps = max(1, int(round((1.0 / CONTROL_HZ) / float(env.model.opt.timestep))))
    for _ in range(substeps):
        mujoco.mj_step(env.model, env.data)


def _done_b(env):
    if env.rec.faults:
        return False, f"faulted: {env.rec.faults[0]}"
    if env.relay is None:
        return False, "no relay target"
    cup = object_pos(env.data, env.idx, "cup")
    err = float(np.linalg.norm(cup[:2] - np.asarray(env.relay["xy"])))
    rot = env.data.xmat[env.idx.obj_body["cup"]].reshape(3, 3)
    upright = float(rot[:, 2] @ np.array([0.0, 0.0, 1.0]))
    right = np.asarray([env.data.qpos[q] for q in env.idx.arm_qpos_ids["right"]])
    home_err = float(np.max(np.abs(right - HOME_QPOS)))
    ok = err <= 0.020 and upright >= 0.85 and home_err <= 0.08
    return ok, f"relay_error_mm={err*1000:.1f}, upright={upright:.3f}"


def _outcomes(env, plate_xy):
    a_ok, a_reason = success_cup_handoff(env, {"plate_xy": plate_xy})
    b_ok, b_reason = _done_b(env)
    cup = object_pos(env.data, env.idx, "cup")
    plate = object_pos(env.data, env.idx, "plate")
    return {"A": bool(a_ok), "B": bool(b_ok)}, {"A": a_reason, "B": b_reason}, float(np.linalg.norm(cup[:2] - plate[:2]))


class TorchPolicy:
    def __init__(self, checkpoint, device):
        from lerobot.policies.factory import make_pre_post_processors
        self.device = device
        self.policy = SmolVLAPolicy.from_pretrained(checkpoint).to(device).eval()
        self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=checkpoint)
        self.policy.reset()

    def new_episode(self):
        self.policy.reset()

    @torch.no_grad()
    def act(self, frames, state, instruction):
        obs = {"task": instruction,
               "observation.state": torch.from_numpy(state).float().unsqueeze(0).to(self.device)}
        for cam in CAMERAS:
            img = frames[cam]  # HWC uint8
            obs[f"observation.images.{cam}"] = (
                torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(self.device))
        obs = self.pre(obs)                       # normalize inputs + tokenize the task
        action = self.policy.select_action(obs)   # lerobot manages the 50-step chunk queue
        action = self.post(action)                # unnormalize back to joint space
        return action.squeeze(0).float().cpu().numpy()


def run_episode(policy, seed, given, max_steps):
    env = BimanualEnv(img_w=256, img_h=256, record_images=True, seed=seed)
    policy.new_episode()
    started = time.perf_counter()
    step = 0
    instruction = TASK_STRINGS["cup"] if given == "A" else TASK_B
    try:
        env.reset(); env.randomize_objects()
        plate_xy = object_pos(env.data, env.idx, "plate")[:2].copy()
        outcomes = {"A": False, "B": False}
        while step < max_steps and not env.rec.faults:
            frames = env.capture()
            action = policy.act(frames, _state_right_left(env), instruction)
            _apply_action(env, action)
            step += 1
            outcomes, _, _ = _outcomes(env, plate_xy)
            if outcomes[given]:
                break
        outcomes, reasons, cup_err = _outcomes(env, plate_xy)
        return {"seed": seed, "instruction_given": given, "steps": step,
                "success": bool(outcomes[given]), "outcomes": outcomes, "outcome_reasons": reasons,
                "final_cup_position_error_to_plate_m": round(cup_err, 6),
                "episode_wall_s": round(time.perf_counter() - started, 3),
                "faults": list(env.rec.faults)}
    except Exception as exc:
        return {"seed": seed, "instruction_given": given, "steps": step, "success": False,
                "faults": [f"{type(exc).__name__}: {exc}"], "evaluation_error": True}
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=3000)
    ap.add_argument("--instruction", choices=("A", "B", "both"), default="A")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-steps", type=int, default=1500)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = TorchPolicy(args.checkpoint, device)
    givens = ["A", "B"] if args.instruction == "both" else [args.instruction]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    eps = []
    for s in range(args.seed0, args.seed0 + args.seeds):
        for g in givens:
            r = run_episode(policy, s, g, args.max_steps)
            eps.append(r)
            print(f"seed={s} given={g} success={r['success']} steps={r.get('steps')} "
                  f"err_mm={int(1000*r['final_cup_position_error_to_plate_m']) if r.get('final_cup_position_error_to_plate_m') is not None else 'NA'} "
                  f"faults={r.get('faults')}", flush=True)
    succ = sum(e["success"] for e in eps)
    result = {"device": device, "checkpoint": args.checkpoint, "backend": "lerobot-torch",
              "seeds": args.seeds, "episodes": eps, "successes": succ}
    (out / "eval_results.json").write_text(json.dumps(result, indent=2))
    print(f"\nRESULT: {succ}/{len(eps)} success")


if __name__ == "__main__":
    main()
