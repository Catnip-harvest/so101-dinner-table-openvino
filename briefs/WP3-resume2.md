# RESUME NOTICE 2 (17:30)

The previous run of this brief stopped at ~17:25 with "You have hit your usage limit" on the ChatGPT account. You now run on an OpenAI API key with a small dollar cap: be economical -- do not re-read large files you already know from what is on disk, do not repeat long commands. State on disk: intel_eval.py exists and was mid-edit (evaluation-exception handling) when the run stopped -- start with python -m py_compile intel_eval.py, then continue with the remaining deliverables and the local rehearsal.

---

# RESUME NOTICE

A previous run of this brief wrote intel_eval.py (and possibly more) but its sandbox could not start python.exe ("Access is denied"), so nothing was executed. This run has NO sandbox. Read what exists (intel_eval.py, briefs/WP3-codex-last.md if present), keep what is right, finish the remaining deliverables, and actually RUN every acceptance step in the brief, pasting the real output into briefs/WP3-REPORT.md. Note the simulator has since moved to a 0.005 s timestep / 50 Hz control and a new arm model (Menagerie); another worker (WP1) may be editing sim/episode.py concurrently -- if an import breaks mid-run, wait a minute and retry once before diagnosing.

---

# WP3 — closed-loop OpenVINO evaluation harness + the one-command Tiber demo script

You are working in `C:\Users\vieth\Documents\lablab hackathon` on Windows 11 (Git Bash and
PowerShell available). Deadline for the whole project is in ~32 h; this package must be
done and verified in **4 h**. Work only in NEW files unless told otherwise. Report with
measured output (paste the actual command output), never with claims.

## Why this exists

The challenge's hard rule: *"MuJoCo and the AI/VLA/VLM inference pipeline must execute on an
Intel Core Ultra Series 2/3 system for the final demonstration."* The Intel box we have is a
Tiber AI Cloud **Core Ultra X7 358H, Windows 11, 16 threads, 31.6 GB, CPU + Arc iGPU + NPU**,
reachable only by RDP, and the owner has about **40 minutes** on it tomorrow. So everything
must be one PowerShell command that produces: device probe, benchmark JSON (latency /
throughput / precision on every device that enumerates), a 10-seed closed-loop evaluation
with success measured from simulator truth, the frames for the demo video, and an
instruction-swap test. It must be rehearsed on this laptop first.

## Hard rules

* **Never open, print, grep, diff or copy any `.env*` file.** `.env.local` exists in the
  project root; it is out of bounds. A missing credential means stop and write it in the
  report — never invent a placeholder.
* **Do not edit** `sim/episode.py`, `sim/bimanual_scene.py`, `sim/ik_control.py` or anything
  under `sim/tools/`. Another engineer is changing those files right now (swapping the arm
  model to the Menagerie SO-101). Read them; use their public functions; if you need a hook
  they lack, implement it in your own file against `env.model` / `env.data`.
* Nothing goes public, nothing is pushed anywhere.
* The Tiber bundle must be **torch-free and lerobot-free**: `openvino`, `openvino-tokenizers`,
  `numpy`, `mujoco`, `mink`, `imageio`, `imageio-ffmpeg` only, plus this repo's `sim/` folder
  and the IR directories. `INTEL-PATH.md` §Part 1 explains why the venv must live at a short
  path (`C:\pai`) and why `openvino-tokenizers` is mandatory even when not imported.

## What exists (read these first)

* `INTEL-PATH.md` — the Tiber runbook so far (device probe + benchmark only, no closed loop).
* `intel_export.py` (199 lines) — exports the SmolVLA policy to OpenVINO IR. Determine
  **exactly** what graph it exports: inputs (names, shapes, dtypes), outputs, and whether it
  covers the whole `select_action` path (vision encoder + language + the flow-matching
  denoising loop → an action chunk) or a single denoising step. Also read `manifest.json` and
  `stats.json` in an exported bundle — look under `C:\pai` for `policy_ov\fp16` / `int8` from
  the untrained export. If none exists, the export env is `C:\pai\ev` (`lerobot` +
  `openvino`); re-run the export with the untrained `lerobot/smolvla_base` and note the time.
* `intel_bench.py` (142 lines) — device probe + per-device latency; reuse its conventions
  (p95 nearest-rank, `output_finite` check, JSON schema) rather than inventing new ones.
* `sim/episode.py` — `BimanualEnv`, `randomize_objects()`, success predicates (simulator
  truth), `TASK_STRINGS`, cameras `top`, `left_wrist`, `right_wrist`, control at 50 Hz over a
  200 Hz sim. `sim/mj_backend.py`: `select_backend()` must be called before `import mujoco`.
* `k1_kaggle_autopilot.py` — the trainer; its `RENAME` shows the camera-key mapping the
  policy will be trained with (`top→camera1`, `left_wrist→camera2`, `right_wrist→camera3`).

## Deliverables

### 1. `intel_eval.py` — closed-loop evaluation with the OpenVINO policy

```
python intel_eval.py --ir C:\pai\policy_ov\fp16 --device CPU --seeds 10 --seed0 1000 \
    --instruction A --out C:\pai\out\eval_fp16_cpu [--swap] [--video] [--max-steps 1500]
```

* Loads the IR + tokenizer + `stats.json` (normalisation), runs the **full** policy path
  each time an action chunk is needed. If the IR is a single denoising step, run the
  flow-matching loop in Python around it (read the number of steps and schedule from the
  lerobot SmolVLA config that `intel_export.py` used, and state it in the report).
* Per seed: `env.randomize_objects()` with that seed, render the three cameras at the
  resolution the policy expects, build the observation (state order must match what the
  collector records — coordinate with `briefs/WP2-collector.md`: right arm 6 then left arm 6;
  if that file's report says otherwise, follow the report), run the policy, apply the action
  chunk at 50 Hz with the env's stepping, repeat until success, fault, or `--max-steps`.
* Success from the env's own predicates; also log final cup position error to the target,
  time to success, faults.
* `--swap`: every seed is also run with the *other* instruction and the outcome table
  (A-given/A-done, A-given/B-done, …) is written — this is the language-conditioning test.
* `--video`: per-seed top-camera mp4 plus a 10-seed tiled mp4 (or a frames folder if
  encoding fails — say which).
* Timing per inference call (p50/p95) and per full episode; JSON output with the same
  field conventions as `intel_bench.py`, plus `device`, `precision`, `ir_path`, `ir_size_mb`,
  `openvino_version`, `cpu_name`, and a `NOT_THE_REQUIRED_SILICON: true` flag whenever the CPU
  string is not an Intel Core Ultra — the laptop is not, and that flag must show up in the
  laptop rehearsal output.
* Untrained model → near-zero success is the expected rehearsal result. The rehearsal proves
  the plumbing, not the policy.

### 2. `run_intel_demo.ps1` — the one command for the Tiber box

Runs from `C:\pai` on a fresh Windows 11 machine with Python 3.12–3.14 installed:

1. Creates/uses venv `C:\pai\dv`, installs the torch-free dependency list, prints versions.
2. **Render test first (must be the first thing after the venv):** builds the scene and
   renders one `top` frame to `C:\pai\out\render_test.png`. RDP sessions may have no usable
   OpenGL; if this fails, print the exact error and a clear line
   `RENDER FAILED — see fallbacks in PLAN.md §6`, then exit non-zero. Nothing else is
   attempted.
3. Device probe (`intel_bench.py` stage one) → `devices.json`.
4. Benchmark FP16 and INT8 on every device that enumerated, **3 repeats each**, → JSON.
5. `intel_eval.py` 10 seeds, FP16 on the best device from step 4, `--swap --video`.
6. Zips `C:\pai\out` to `C:\pai\tiber_results_<timestamp>.zip` and prints its size.
7. Optional `-XpuSmoke` switch at the very end (never blocks the above): installs torch with
   XPU support into a separate venv `C:\pai\xv`, runs a **200-step** SmolVLA fine-tune smoke
   on the iGPU with a tiny local dataset slice, records seconds/step and peak memory. If
   `torch.xpu.is_available()` is False, print that and stop — that is a valid result.
   Write the exact pip index/wheel command you would use and mark it "untested here (no
   Intel GPU on this laptop)".

Also write `TIBER-RUNBOOK.md`: what to copy (with sizes), the single command, what a good
run prints, what each failure looks like. Audience: the owner, who is not a roboticist.

### 3. Local rehearsal (this laptop, no Intel silicon)

Run the whole `run_intel_demo.ps1` here with `-Seeds 2 -Repeats 1`, using the untrained IR.
Expect success 0/2 and the `NOT_THE_REQUIRED_SILICON` flag. Keep the produced JSON, the
render test PNG and the video in `out/rehearsal/` and cite them in the report.

## Acceptance (all shown in the report with real output)

1. `intel_eval.py` runs 2 seeds closed-loop on CPU with the untrained FP16 IR, produces the
   JSON, the swap table and a video (or frames), with inference p50/p95 printed.
2. `run_intel_demo.ps1 -Seeds 2 -Repeats 1` completes end-to-end on this laptop from a fresh
   `C:\pai\dv2` venv (do not reuse `dv`), including the render test, and the zip exists.
3. Total bundle size for the copy to Tiber, itemised.

## Report

Write `briefs/WP3-REPORT.md`: what the IR actually is (inputs/outputs/what's inside), the
observation/action contract you implemented, measured rehearsal numbers, bundle size, the
exact copy list, open risks (RDP OpenGL, NPU driver), anything left undone and why.
