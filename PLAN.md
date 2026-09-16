# PLAN — Intel Physical AI Online Challenge, the last 33 hours

Written 15 Sep 2026 16:40 Hanoi. Supersedes the "next actions" in `HANDOFF-LABLAB.md`.
Nothing below is built until the owner has seen section 3.

## 0. Facts that bind (all checked, sources named)

| Fact | Value | Source |
|---|---|---|
| Deadline | **16 Sep 18:30 UTC = 17 Sep 01:30 Hanoi** | live lablab page, 15 Sep 16:29: "Submission deadline Sep 17, 1:30 AM IT" (Indochina Time). The 08:00 Hanoi figure in older notes was wrong by 6.5 h. |
| Time left at writing | **32 h 50 min** | measured against the page clock |
| What Intel provides | **Nothing runnable.** No scene, no dataset, no baseline, no harness. Evaluation is manual, from the video and repo. | brief read-through (research agent 1) |
| Prescribed asset | "two Menagerie SO-101 arms", MuJoCo 3.x, 200 Hz sim / 50 Hz control | brief |
| Hard rule | "MuJoCo and the AI/VLA/VLM inference pipeline **must execute on an Intel Core Ultra Series 2/3** system for the final demonstration" | brief |
| Rubric | Task 30 · VLA multi-modal reasoning 20 · OpenVINO & Core Ultra 20 · Robustness 15 · Reproducibility 10 · Innovation 5 | brief |
| Deliverables | public GitHub repo (scene, training, eval, inference), reproducible MuJoCo package with randomization, Intel benchmark script (latency / throughput / precision), **10-randomized-seed demo video**, technical README | brief |
| Intel silicon we have | Tiber Core Ultra X7 358H (Panther Lake, Series 3), Win 11, 16 threads, 31.6 GB, CPU + Arc iGPU + NPU. **RDP only — no SSH from this machine.** Owner drives it. | `INTEL-PATH.md`, memory |
| Training silicon | Kaggle T4: SmolVLA fp32 **2.352 s/step** at batch 8 (measured 13 Sep). Tiber XPU training on Win 11: 12–45 s/step estimate, open Panther Lake bug — **not** a training box. | k1 header; research agent 4 |
| Owner availability | ~1 h hands-on total, high-level decisions only | owner, 15 Sep |

### What the community already solved (so we do not invent)

* **`google-deepmind/mujoco_menagerie/robotstudio_so101`** (Apache-2.0, pushed 4 Sep 2026): the SO-101 with the jaw hull problem fixed upstream — jaw meshes split into convex parts (`maxhullvert="64"`), fingertip primitives (`fixed_jaw_box1..7`, `fixed_jaw_sph_tip1..3`, `moving_jaw_box1..3`, `moving_jaw_sph_tip1..3`), a `collision_gripper` class with `condim=6 friction="1 5e-3 5e-4" solref="0.01 1" priority=1`, position actuators kp 998.22 / forcerange ±2.94, site `gripperframe` at `0.012 -0.000218 -0.098127` quat `1 0 1 0`. **Same six joint names as our vendored TRS file**, so our IK, reach tables and episode code carry over; the site moved +20 mm along body x and the jaw geometry is new, so the grasp offset and the reach tables are re-measured, not reused.
* **PegBitStudio/two-arm-table-setter** (strongest competitor): same hull diagnosis, fixed with pads; `holding()` = both jaw geoms in `data.contact` with the object; scripted skills self-record demos, failures discarded (1,219 kept); ACT 5M trained on CPU; OpenVINO measured on an i7-1165G7.
* **peacestate/aiinfra-vla**: the only entry with a trained, language-conditioned policy — SmolVLA 8k steps, 30 % success, instruction-swap test p = 0.028 — on gym-aloha, **not SO-101**.

### Where the field is weak (checked across the six public entries on this track)

* **Nobody ran on a Core Ultra.** Hardware seen: i7-1165G7 (×2), i3-1215U, AMD EPYC, none. The hard rule and the 20 OpenVINO/Core Ultra points are unclaimed.
* **Nobody used the brief's named Menagerie asset.** Four used the raw TRS export, one hand-built, one ALOHA.
* **Only one has a trained language-conditioned policy, and it is on the wrong robot.** Three of the four SO-101 entries carry objects with a weld or a kinematic pin — what we do today.

## 1. Goals, in the owner's priority order

1. **First prize** → take every rubric point the field is leaving on the table (above), and be honest everywhere else; the strongest entries document their failures and judges read that.
2. **A working trained model** → SmolVLA (owner rejected ACT on 13 Sep: "camera is fixed", no language channel), trained on Kaggle, driving the demo closed-loop.
3. **Experience training on Intel** → a measured 200-step XPU smoke fine-tune on the Tiber Arc iGPU, reported next to the T4 number. Goal 3 never delays goals 1–2.

## 2. Strategy in one paragraph

Swap the arm to the Menagerie SO-101 so grasps are real contact, not a pose rewrite; keep our measured-reachability planner and cup relay on top of it; let the scripted relay self-record a few hundred successful demos under two different instructions; train SmolVLA on Kaggle overnight; export to OpenVINO FP16/INT8; run the **whole** closed loop — MuJoCo + OpenVINO SmolVLA — on the Tiber Core Ultra from one command that emits the 10-seed video, the benchmark JSON and an instruction-swap result; write it up with every number measured. Scope stays at the cup relay onto the plate (plate carry, drawer and pour are out — see §7); the README says so plainly.

## 3. Decisions for the owner (reply with numbers; silence = default)

| # | Decision | Default | Why the default |
|---|---|---|---|
| 1 | Swap to Menagerie `robotstudio_so101`, physics-only grasp, **gate at 21:30 Hanoi tonight** | **Yes** | It is the brief's asset and the community fix; same joint names; failure mode is bounded by the gate (fallback = keep our kinematic carry, disclosed) |
| 2 | Second instruction for language conditioning | **B: "slide the cup to the middle of the table"** (right arm places at the relay point and stops; the existing instruction A does the full relay onto the plate) | Same primitives, visibly different end state, gives a real instruction-swap test |
| 3 | Demo policy | **SmolVLA only** | Owner's 13 Sep call. Flagging once: if SmolVLA lands below ~30 % there is no second policy family in this window; mitigation is demo volume (target 600 kept episodes, not 300) + DART noise |
| 4 | Kaggle start | **Owner clicks "Save & Run All" at ~23:00 Hanoi tonight**; autopilot waits up to 6 h for the dataset | Idle GPU time burns the 30 h/week quota; a 23:00 start meets a ~00:30 dataset landing |
| 5 | Tiber session | **16 Sep 15:00–16:00 Hanoi**, one PowerShell command; optional XPU smoke step at the end if minutes remain | Leaves 9 h for video assembly, README and the form |
| 6 | Repos | Create **private** GitHub repo under `Catnip-harvest` and private HF dataset/model repos now; flip all public **only on a fresh "go" at submission time** | Standing rule: nothing public without approval |
| 7 | Cup 130 mm and table relay instead of mid-air hand-off | **Keep** | Measured limits in `FINDINGS-SIM.md` §8–9; changing them now costs the gate |

If nothing is heard by **17:10 Hanoi**, work starts on the defaults.

## 4. Schedule (Hanoi time), with gates

| When | What | Who | Gate / acceptance |
|---|---|---|---|
| 15 Sep 17:10–21:30 | **WP1** Menagerie swap + physics grasp | me | 21:30: ≥ 6/10 scripted cup relays succeed with **no** kinematic carry, placement ≤ 15 mm. Else: Menagerie asset + kinematic carry (disclosed) |
| 17:10–20:00 | **WP2** collection → LeRobot v3 dataset → private HF push; autopilot re-audit + dry run | Opus worker | 5-episode probe on HF loads in `lerobot` and the autopilot's Phase 3 timing probe runs on it |
| 17:10–22:00 | **WP3** `intel_eval.py` closed-loop OpenVINO harness + `run_intel_demo.ps1` | Opus worker | Runs locally with the untrained IR end-to-end: 2 seeds, video file, JSON, swap-test row |
| 21:30–00:30 | Volume collection (target 600 kept, 2 instructions, DART on 30 %), upload | me | dataset on HF, `STATUS.md` says "dataset found" |
| 23:00 | Start Kaggle autopilot | **owner, 5 min** | notebook running |
| 00:30–07:30 | Training (~10k steps ≈ 6.5 h, checkpoint to HF every 20 min) | Kaggle | loss curve in `STATUS.md` |
| 00:30–08:00 | README + architecture doc + benchmark plots + video assembler; local closed-loop eval of every intermediate checkpoint | me | first success-rate number by ~03:00 (2k steps) |
| 16 Sep 08:00–12:00 | Export best checkpoint FP16/INT8; 20-seed local eval; instruction swap; pick demo checkpoint | me | numbers in `RESULTS.md` |
| 12:00–15:00 | Package Tiber bundle (< 1.5 GB), rehearse `run_intel_demo.ps1` locally | me | one command, clean machine assumptions listed |
| 15:00–16:00 | **Tiber run**: device probe → benchmark → 10-seed closed loop → video frames → optional XPU smoke | **owner, ~40 min** | `C:\pai\out\*.json`, `*.mp4`, copied back |
| 16:00–23:30 | Final video, README numbers, repo public (approval), self-check against the deliverables list | me | every deliverable row ticked |
| 23:30–00:30 | Submit on lablab | **owner, 15 min** | confirmation page |
| 00:30–01:30 | Buffer | — | — |

Owner total: 5 + 5 + 40 + 15 = **65 min**.

## 5. Work packages

### WP1 — Menagerie swap (me, 4 h 20 min, hard gate)

1. Vendor `robotstudio_so101/` (so101.xml, scene.xml, assets/*.stl, LICENSE) into `sim/assets/menagerie_so101/`, record the upstream commit hash in `sim/assets/menagerie_so101/UPSTREAM.txt`. No edits to the vendored files.
2. `bimanual_scene.py`: `ARM_XML` → the new file; delete `replace_gripper_collision` and the all-links arm–object contact exclusions; keep `MOUNT_Z`, park pose, table, objects. Kinematic carry becomes a flag `--carry kinematic`, default off.
3. Re-measure with the existing tools, in this order: `measure_jaw_gap.py --raw` (expect a non-empty pinched set now — that is the whole point), then `reach_table.py` (rebuild, new site offset), `validate_reach.py`.
4. Replace `GRASP_CENTRE_SITE` and `GRASP_WINDOW` with the measured pinch centroid; add `holding(arm, obj)` = both jaw collision geoms in `data.contact` with the object (PegBitStudio's check, credited); `grasp_ok` becomes `holding` after a 0.3 s squeeze.
5. Retune `CLOSE_Q` for the 48 mm cup against the new joint range (−0.17…1.75) by sweep, and re-run `holds()`.
6. Run 10 scripted relays. Gate.

### WP2 — Data (Opus worker, then me for the volume run)

* `sim/collect.py`: N workers (16 threads → 6 workers), seeds, both instructions, DART perturbation (σ 0.02 rad on 30 % of episodes, clean label recorded, `is_perturbed` flag), three cameras 256×256 at 50 Hz control / 200 Hz sim, **failures discarded and counted**, writes LeRobotDataset v3 with camera keys already named `camera1/2/3` (kills the autopilot's rename step), pushes to a private HF repo.
* Re-audit `k1_kaggle_autopilot.py` (the four defects from the 15 Sep audit + anything new), then a dry run of Phases 1–3 against the 5-episode probe. Acceptance is the dry run, not the diff.

### WP3 — Intel closed loop (Opus worker)

* `intel_eval.py`: pure-Python bundle (openvino, openvino-tokenizers, numpy, mujoco, imageio). Loads the IR + `stats.json`, runs the full `select_action` path (verify `intel_export.py` exported the whole denoising loop; if it exported one step, loop it in Python), steps our scene closed-loop for K seeds, records success from simulator truth, per-step latency, frames → mp4, JSON. `--swap` runs each seed with the other instruction and reports the outcome table.
* `run_intel_demo.ps1`: venv at `C:\pai`, device probe, **5-line render test first** (RDP OpenGL is the one unverified dependency — see §6), benchmark FP16 + INT8 on CPU/GPU/NPU ×3 repeats, 10-seed eval, zip results. Optional `-XpuSmoke` flag: torch-xpu install + 200-step fine-tune, records s/step and peak memory.

### WP4 — Write-up (me, overnight)

README (problem, standard, procedure — numbers with their measurement method), `ARCHITECTURE.md`, `RESULTS.md`, `bench/` plots, credits (Menagerie, PegBitStudio's holding check, peacestate's swap test), honest-limitations section (cup only, 130 mm cup, table relay, what failed).

## 6. Risks and their fallbacks (each with a trigger)

| Risk | Trigger | Fallback |
|---|---|---|
| Physics grasp does not hold on the Menagerie model | 21:30 gate fails | Menagerie asset + kinematic carry gated by `grasp_ok` (today's mechanism), stated in README; no further time on contact |
| SmolVLA success too low | < 30 % at the 6k-step checkpoint (~05:00) | Increase kept demos to 1000 (collection is ~15 s/episode, cheap), restart training 06:00 → ends ~12:30; Tiber slot moves to 17:00. Below ~10 % at the end: ship it anyway with the number, scripted relay shown separately and labelled |
| Kaggle quota short | notebook refuses GPU | Colab T4 (shorter sessions, 4k steps), documented |
| MuJoCo cannot get an OpenGL context over RDP on Tiber | render test fails in the first minute of the owner's session | The policy needs camera frames, so this is not optional. Fallback in order: run from the console session via a scheduled task; WSL2 + OSMesa if nested virtualisation is on; last resort the owner installs Mesa's software `opengl32.dll` next to `python.exe` (owner's download, not mine) |
| `intel_export.py` IR is one denoising step, not `select_action` | WP3 inspection | Python loop around the IR; latency reported per full action chunk |
| Tiber instance gone / owner slot missed | 16 Sep 17:00 with no results | Submit with laptop-CPU OpenVINO numbers, labelled "not the required silicon" — the same honest flag jianwang-ntu used. Hard rule then fails; say so |

## 7. Explicitly out of scope, and why

* **Plate carry, drawer, pour** — the plate is not graspable as modelled (`FINDINGS-SIM.md` §2), near-vertical wrists are 485/205k configs; each is a day, not an hour. README states the subset.
* **RoboLab** — NVLabs Isaac Lab benchmark, RTX 48 GB+, unrelated to Intel or this brief.
* **Training on Tiber** — the brief rewards inference on Intel, not training location; a real fine-tune there is 5–20× slower than the T4 and blocked by an open Panther Lake bug. Only the 200-step smoke run (goal 3).
* **Mid-air hand-off, 90 mm cup** — measured infeasible (`FINDINGS-SIM.md` §8–9).
* **Any public push, dataset, model or repo** until the owner says go.

## 8. Standing rules carried forward

Never open any `.env*`; a missing credential stops the step and asks; verify by running and report what was seen; a number that looks too good is a failure until re-measured; no artifacts, plain files only; GitHub identity `Catnip-harvest`.
