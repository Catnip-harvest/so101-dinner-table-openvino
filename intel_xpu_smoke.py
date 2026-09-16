"""Optional 200-step SmolVLA XPU smoke wrapper (separate non-deploy venv)."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
import time
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="local LeRobotDataset root")
    parser.add_argument("--repo", default="local/tiber-smoke")
    parser.add_argument("--out", default=r"C:\pai\out\xpu_smoke.json")
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    available = bool(torch.xpu.is_available())
    print("torch", torch.__version__)
    print("torch.xpu.is_available()", available)
    if not available:
        out.write_text(json.dumps({"torch": torch.__version__, "xpu_available": False}, indent=2))
        return 0
    if not Path(args.dataset).exists():
        print("XPU SMOKE SKIPPED: local dataset slice is missing:", args.dataset)
        out.write_text(json.dumps({"torch": torch.__version__, "xpu_available": True, "error": "dataset slice missing"}, indent=2))
        return 0

    torch.xpu.reset_peak_memory_stats()
    started = time.perf_counter()
    argv = [
        "lerobot-train",
        "--policy.path=lerobot/smolvla_base",
        f"--dataset.repo_id={args.repo}",
        f"--dataset.root={Path(args.dataset).resolve()}",
        "--policy.device=xpu",
        "--policy.push_to_hub=false",
        "--batch_size=1",
        "--steps=200",
        "--save_freq=999999",
        r"--output_dir=C:\pai\xpu_smoke_run",
        "--job_name=tiber_xpu_smoke",
        "--wandb.enable=false",
        "--log_freq=10",
    ]
    old_argv = sys.argv
    exit_code = 0
    try:
        sys.argv = argv
        runpy.run_module("lerobot.scripts.lerobot_train", run_name="__main__")
    except SystemExit as exc:
        exit_code = int(exc.code or 0)
    finally:
        sys.argv = old_argv
    elapsed = time.perf_counter() - started
    peak = int(torch.xpu.max_memory_allocated())
    result = {
        "torch": torch.__version__,
        "xpu_available": True,
        "device": torch.xpu.get_device_name(0),
        "steps": 200,
        "wall_s": round(elapsed, 3),
        "seconds_per_step": round(elapsed / 200.0, 4),
        "peak_memory_bytes": peak,
        "peak_memory_mb": round(peak / 1e6, 1),
        "exit_code": exit_code,
    }
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
