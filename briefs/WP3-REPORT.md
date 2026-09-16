# WP3 report — OpenVINO closed-loop evaluation and Tiber command

## Delivered

- `intel_eval.py`: torch-free closed-loop OpenVINO evaluator, simulator-truth A/B outcomes, swap table, per-seed/tiled video, timing and hardware metadata.
- `run_intel_demo.ps1`: fresh short-path venv, render gate, probe, all-device FP16/INT8 benchmark, best-device evaluation, result zip, optional isolated XPU smoke.
- `intel_render_test.py`, `intel_xpu_smoke.py`, and `TIBER-RUNBOOK.md`.

## What the IR actually contains

OpenVINO read the FP16 graph as:

```text
INPUTS
state [1,12] float32
tokenized_prompt [1,48] int64
tokenized_prompt_mask [1,48] boolean/char
images [3,1,3,512,512] float32
image_masks [3,1] boolean/char
OUTPUTS
action [1,50,12] float32
```

This is the full exported `policy.export(..., backend="openvino")` inference path: vision/language conditioning and all 10 flow-matching steps are inside the graph, producing the final 50-action chunk. It is not a one-step denoiser. `intel_export.py` configured `num_steps=10`; the policy schedule is Euler `dt=-0.1`, at `t=1.0, 0.9, ..., 0.1`. FP16 is OpenVINO FP16 weight compression of the FP32 graph. INT8 is NNCF asymmetric INT8 weight compression, not activation calibration.

## Implemented contract

State and action are 12 float32 values in collector order: right arm six joints, then left arm six. Every policy call captures `top`, `left_wrist`, and `right_wrist`, resizes each to 512×512, converts HWC uint8 to CHW float in `[-1,1]`, stacks as `[3,1,3,512,512]`, tokenizes the newline-terminated task with the exported OpenVINO tokenizer, and applies `stats.json` normalization. The returned `[50,12]` chunk is denormalized, actuator-limited and applied at 50 Hz; MuJoCo advances four 0.005 s steps per action. Success and faults come from `sim.episode` predicates/state. Instruction B additionally requires cup-to-relay error ≤20 mm, upright ≥0.85, right-arm home error ≤0.08 rad, and no simulator faults.

## Acceptance 1 — standalone closed loop

Command:

```powershell
C:\pai\ev\Scripts\python.exe intel_eval.py --ir C:\pai\policy_ov\fp16 --device CPU --seeds 2 --seed0 1000 --instruction A --out out\acceptance_eval --swap --video --max-steps 1500
```

Actual final output:

```text
seed=1000 given=A success=False steps=1500 wall=74.84s faults=[]
seed=1000 given=B success=False steps=1500 wall=77.28s faults=[]
seed=1001 given=A success=False steps=1500 wall=96.93s faults=[]
seed=1001 given=B success=False steps=1500 wall=107.00s faults=[]
inference p50=2140.48 ms p95=3386.57 ms calls=120 output_finite=True
instruction swap table: {"A-given/A-done": 0, "A-given/B-done": 0, "B-given/A-done": 0, "B-given/B-done": 0}
NOT_THE_REQUIRED_SILICON: true
wrote ...\out\acceptance_eval\eval_results.json
```

Exit code was 0. Four MP4s and `tiled_A.mp4` were produced. Zero success is expected from this untrained export.

## Acceptance 2 — fresh `C:\pai\dv2` end-to-end rehearsal

Command:

```powershell
C:\pai\run_intel_demo.ps1 -Seeds 2 -Repeats 1 -VenvPath C:\pai\dv2
```

Selected actual output (the full console output was observed during the run):

```text
creating venv C:\pai\dv2 with ...\Python313\python.exe
Successfully installed ... openvino-2026.3.1 ... openvino-tokenizers-2026.3.1.0 ... mujoco-3.13.0 ...
RENDER OK C:\pai\out\render_test.png shape=(256, 256, 3) dtype=uint8
OpenVINO 2026.3.1-22476-759c5a6ab8c-releases/2026/3
  CPU      AMD Ryzen 9 7940HS w/ Radeon 780M Graphics
  GPU      NVIDIA GeForce RTX 4060 Laptop GPU (dGPU)
  CPU      3003.71 ms mean | 3003.71 p95 | 0.3 fps | compile 25.79s
  GPU      FAILED: Exception from ... OpenVINO GPU plugin ...
  CPU      2698.26 ms mean | 2698.26 p95 | 0.4 fps | compile 28.70s
  GPU      10987.87 ms mean | 10987.87 p95 | 0.1 fps | compile 55.11s OUTPUT NOT FINITE
best FP16 device: CPU (3003.71 ms mean)
seed=1000 given=A success=False steps=1500 wall=91.96s faults=[]
seed=1000 given=B success=False steps=1500 wall=71.57s faults=[]
seed=1001 given=A success=False steps=1500 wall=75.81s faults=[]
seed=1001 given=B success=False steps=1500 wall=77.35s faults=[]
inference p50=2021.08 ms p95=2924.5 ms calls=120 output_finite=True
instruction swap table: {"A-given/A-done": 0, "A-given/B-done": 0, "B-given/A-done": 0, "B-given/B-done": 0}
NOT_THE_REQUIRED_SILICON: true
RESULT ZIP C:\pai\tiber_results_20260915_190352.zip bytes=12629976 size_mb=12.0
INTEL DEMO COMPLETE
```

Exit code was 0. The fresh-venv version-print invocation initially encountered a Windows Application Control DLL-load error for MuJoCo, but the immediately following render process loaded MuJoCo and passed. The script has since been hardened to print installed versions through package metadata and to fail explicitly if that probe fails; no model/runtime behavior changed.

Rehearsal artifacts are retained in `out/rehearsal/`: `devices.json`, both benchmark JSONs, `render_test.png` (40,064 bytes), `eval_fp16_best/eval_results.json`, four seed videos, `tiled_A.mp4`, and `tiber_results_20260915_190352.zip` (12,629,976 bytes).

## Bundle copied to Tiber

Measured copy list:

```text
policy_ov\fp16       800,521,024 bytes   763.44 MiB
policy_ov\int8       408,670,857 bytes   389.74 MiB
sim\                  69,732,403 bytes    66.50 MiB
seven scripts/docs        70,055 bytes     0.07 MiB
TOTAL               1,278,994,339 bytes 1,219.74 MiB
```

The seven root files are `intel_bench.py`, `intel_eval.py`, `intel_render_test.py`, `intel_xpu_smoke.py`, `run_intel_demo.ps1`, `PLAN.md`, and `INTEL-PATH.md`. Virtual environments and rehearsal output are not part of the copy bundle.

## Open risks and unfinished hardware-only work

- Tiber RDP may lack a usable OpenGL context; the render-first gate exits with the mandated fallback message.
- NPU and Arc iGPU compilation/performance cannot be tested on this AMD/NVIDIA laptop. Driver enumeration and per-device errors will be recorded on Tiber.
- MuJoCo printed `Nan, Inf or huge value in QACC at DOF 15` early in the standalone run, while `env.rec.faults` stayed empty and episodes continued with finite qpos. The new Menagerie dynamics may need tuning; the evaluator intentionally reports simulator truth rather than inventing a fault.
- The optional XPU smoke is implemented but untested here (no Intel GPU). Exact wheel command printed by the script: `python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu`.
