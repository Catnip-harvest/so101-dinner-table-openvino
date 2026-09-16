# lablab submission text — Intel Physical AI Challenge

Paste-ready. Numbers marked `[R3]` are filled in once the 5090 run (run 3) is evaluated;
if run 3 does not land, delete those lines and the run-1 figures stand.

---

## Project name

**Dinner-Table Relay: two SO-101 arms, one cup, SmolVLA on Intel Core Ultra**

## One-line description (≤ 140 chars)

Bimanual SO-101 cup hand-off in MuJoCo, trained with SmolVLA, exported to OpenVINO and
benchmarked on Intel Core Ultra — with the failures measured, not hidden.

## Short description (≤ 500 chars)

Two SO-101 arms face each other across a dinner table. Given a natural-language instruction,
one arm picks up a cup, relays it across the table, and the other places it on a plate. We
built the full pipeline: physics-accurate demonstration collection (real jaw contact, not a
kinematic cheat), a 400-episode LeRobot v3 dataset, a SmolVLA fine-tune, OpenVINO
FP32/FP16/INT8 export, and a torch-free closed-loop evaluator run on a bare-metal Panther Lake
Core Ultra X7 358H with Arc B390 iGPU.

## Long description

### The task

A fixed 130 mm cup starts on the right arm's side of a dinner table. Instruction A —
*"bring the cup across the table and set it on the plate"* — requires a two-arm relay: the
right arm grasps and lifts the cup, sets it mid-table, and the left arm picks it up and places
it on the plate. Instruction B — *"slide the cup to the middle of the table"* — is the
single-arm half of the same motion. A language-conditioned policy must do the right one.

### What we built

**Simulation with honest grasps.** MuJoCo 3.x, two MuJoCo-Menagerie SO-101 arms, 200 Hz
physics / 50 Hz control. Early in the project we found the scripted demonstrations were not
grasping at all — the runner carried objects by rewriting their pose. We replaced that with
real contact: split jaw collision hulls, a measured grasp window, and a `holding()` check that
requires both jaws in contact with the cup. The scripted controller passes its gate at 7/10 and
14/20 across seeds with no predicate loosened.

**Dataset.** 749 attempts → **400 success-only episodes, 323,442 frames at 50 fps** (53.4 %
yield), three 256×256 cameras (top, left wrist, right wrist), 12-D state and action
(right arm ×6, left arm ×6), two task strings. DART noise (σ = 0.02 rad on 30 % of episodes)
with clean labels. Every failed attempt is discarded and counted; the fault taxonomy (831
events over 194 distinct strings) is published.

**Training.** SmolVLA (450 M) fine-tuned from `lerobot/smolvla_base` with LeRobot 0.6.1.
Run 1: 10,300 steps, batch 8, fp32, Tesla T4, 10.4 h. Run 3 `[R3]`: batch 64, bf16, RTX 5090,
`[R3 steps]` steps in `[R3 hours]` h.

**OpenVINO export.** FP32 1,586 MB → **FP16 800 MB** (exactly 50 %) → **INT8 409 MB** (NNCF
weight compression, 300/300 layers). The full flow-matching loop — prefix encoding, KV cache,
ten Euler steps — is inside the IR; only pre/post-processing lives outside, reproduced in NumPy.

**Evaluation on the required silicon.** A torch-free evaluator (`intel_eval.py`) drives MuJoCo
closed-loop from the OpenVINO IR alone. Run on Intel's bare-metal `bm-ptl` instance:

| Device | FP16 latency | INT8 latency |
|---|---:|---:|
| Core Ultra X7 358H (CPU) | 1,991.5 ms | 2,696.7 ms |
| **Arc B390 iGPU** | **124.6 ms** | **119.7 ms** |
| AI Boost NPU | compile failed | compile failed |

One inference is a 50-step action chunk — one second of motion at 50 Hz — so the iGPU runs
at **8× real time** and is **16× faster than the CPU**. Two findings we did not expect and
report anyway: INT8 is *35 % slower* than FP16 on the CPU (weight-only compression pays a
dequantisation cost that exceeds the bandwidth saving; its real benefit is halving the artefact
on disk), and the NPU compiler rejects a 450 M VLA with dynamic shapes and an in-graph
denoising loop in OpenVINO 2026.3.1.

### Results, stated plainly

| | Run 1 (T4, 1 epoch) | Run 3 (5090) `[R3]` |
|---|---:|---:|
| Closed-loop success on Core Ultra, 10 seeds × 2 instructions | **0 / 20** | `[R3]` / 20 |
| Best final cup error to plate | 105 mm | `[R3]` mm |
| Inference on Arc B390, FP16 | 127 ms mean | `[R3]` ms |

Run 1 did not complete the task. We localised why rather than guessing: replaying *training*
observations through the exported IR gives 0.39–0.66 rad error at 0.66–0.85 correlation — the
observation pipeline, camera order, normalisation and export are correct (a mis-wired chain
correlates near zero), but the model is underfit. 10,300 steps × batch 8 is **1.006 epochs**.
The reported loss of 0.028 flattered it: half the action dimensions sit idle in the single-arm
episodes and dominate the mean. A low imitation loss on a bimanual dataset with one idle arm is
not evidence of a learned task. `[R3: Run 3 at batch 64 / bf16 addresses exactly this defect.]`

### Why this entry

- **It runs on the required hardware.** Device table, three precisions, closed-loop success and
  latency all measured on Core Ultra + Arc, with `NOT_THE_REQUIRED_SILICON: false` in the raw
  JSON, committed to the repo.
- **The data is physically honest.** Real grasps, failures counted, DART recorded.
- **Every number has a file behind it.** Volume log, benchmark JSONs, eval JSON, videos, and
  the replay diagnostic are in `evidence/`.
- **The failure is diagnosed, not described.** We would rather submit a measured 0/20 with the
  cause isolated than an unmeasured claim.

### Limitations

Simulation only — no physical SO-101 was driven. Fixed cup size; relay via the table rather
than a mid-air hand-off. Success-only dataset. NPU unexercised. Run-1 policy underfit.

## Tech stack

MuJoCo 3 · MuJoCo Menagerie SO-101 · LeRobot 0.6.1 (dataset v3) · SmolVLA · PyTorch ·
physicalai-train · OpenVINO 2026.3.1 · NNCF · Intel Core Ultra X7 358H / Arc B390 (Panther
Lake, Intel Cloud `bm-ptl`) · Kaggle T4 · RTX 5090 (Vast.ai) `[R3]`

## Links

- Code, docs, evidence: https://github.com/Catnip-harvest/so101-dinner-table-openvino
- Results with every measurement and its evidence file: `RESULTS.md` in the repo
- Dataset: `VietHwang/dinner-table-v2` on Hugging Face (private; 6 GB of simulation video —
  available to judges on request)
- Trained policy: `VietHwang/smolvla-dinner-table` (run 1) `[R3: + run 3 repo]`

## Video

`[link]` — scripted relay demonstrations from the dataset (top + wrist cameras), the Core Ultra
benchmark run, and the closed-loop policy evaluation tiled across seeds.
