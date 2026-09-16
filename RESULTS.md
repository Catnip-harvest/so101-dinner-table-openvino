# Results

The physics collection, the SmolVLA fine-tune, the OpenVINO export and the required-silicon
benchmark are all complete and measured. The trained policy does **not** complete the task:
**0/20** on Intel Core Ultra. [Why the policy did not succeed](#why-the-policy-did-not-succeed)
measures the cause rather than guessing at it.

## Dataset result

| Measurement | Observed result | Evidence |
|---|---|---|
| Physics v2 total | **400 episodes; 323,442 frames; 50 fps** | [volume log](out/volume_v2.log) |
| Collection accounting | **749 attempts; 400 kept; 53.4% yield** | [volume log](out/volume_v2.log) |
| Attempted instruction split | **A 375; B 374** | [volume log](out/volume_v2.log) |
| Kept instruction split | **A 203; B 197** | [volume log](out/volume_v2.log) |
| chunk1 | **192 attempts; 100 kept; 81,944 frames** | [volume log](out/volume_v2.log) |
| chunk2 | **176 attempts; 100 kept; 79,984 frames** | [volume log](out/volume_v2.log) |
| chunk3 | **204 attempts; 100 kept; 80,044 frames** | [volume log](out/volume_v2.log) |
| chunk4 | **177 attempts; 100 kept; 81,470 frames** | [volume log](out/volume_v2.log) |

The first private Hub chunk passed schema and visual QA: **two task strings**, **three 256x256 camera features**, and **12-D state/action** were present; numeric ranges were finite; both wrist views looked down the fingers with visible jaws; and the cup was visible in/against the jaws in sampled mid-episode frames. `is_perturbed` was absent from both converted frame features and episode metadata. [QA record](briefs/ORCHESTRATOR-LOG.md)

## Scripted controller gates

| Carry mode | Result | Interpretation | Evidence |
|---|---:|---|---|
| Historical kinematic | **8/10** | Scripted success gate; v1 mode | [WP1 report](briefs/WP1-REPORT.md) |
| Physics, seed 1000 | **7/10** | Scripted success gate passed the owner's threshold | [WP1b report](briefs/WP1b-REPORT.md) |
| Physics, seed 2000 | **14/20** | Second scripted validation | [WP1b report](briefs/WP1b-REPORT.md) |

The collector-fix probe measured **10 kept from 18 attempts (55.6%) and 7,005 frames**. This was deliberately small and must not be reported as the long-run collection yield; the volume result is **400/749 (53.4%)**. [Probe evidence](briefs/WP2c-REPORT.md), [volume ground truth](out/volume_v2.log)

## Fault accounting

The volume log records **831 fault events across 194 exact strings**. Faults include downstream checks, so the event count is not equivalent to failed attempts. [Volume ground truth](out/volume_v2.log)

| Fault category | Events | Evidence |
|---|---:|---|
| place_jaws_closed | 343 | [volume log](out/volume_v2.log) |
| pick_jaws_closed | 172 | [volume log](out/volume_v2.log) |
| pick_axis_or_body | 130 | [volume log](out/volume_v2.log) |
| resting_height_release | 56 | [volume log](out/volume_v2.log) |
| relay_point | 51 | [volume log](out/volume_v2.log) |
| left_grasp_set | 41 | [volume log](out/volume_v2.log) |
| target_release | 30 | [volume log](out/volume_v2.log) |
| plate_centre | 6 | [volume log](out/volume_v2.log) |
| left_placing_set | 2 | [volume log](out/volume_v2.log) |

The two most common exact strings were `place jaws closed` with **343 events** and `pick jaws closed` with **172 events**. [Volume ground truth](out/volume_v2.log)

## Trained policy

| Measurement | Run 1 (shipped) | Evidence |
|---|---|---|
| Dataset snapshot at train time | **100 episodes; 81,944 frames** | `STATUS.md` in the policy repo |
| Hardware | Kaggle **Tesla T4**, fp32 (mixed precision off) | same |
| Steps / batch | **10,300** steps at batch **8** | `train_config.json` in the checkpoint |
| Throughput | **3.564 s/step**, 10.37 h wall clock | same |
| Final loss | **0.028** | same |
| Parameters | **450 M**; `model.safetensors` 906.7 MB | local checkpoint |
| Exit | `exit_code: 0`, checkpoint uploaded at step 9,900 | same |

Run 1 trained on the 100 episodes that existed when the notebook started; collection was still
running and reached 400 later that night. **10,300 steps x batch 8 = 82,400 samples against
81,944 frames, i.e. 1.006 epochs.** That single fact explains the evaluation result.

Run 2 was launched on the complete 400-episode dataset (8,800 steps, same batch and throughput)
into a separate repository, leaving run 1 intact as the shipped fallback.

## OpenVINO export

Exported with `intel_export.py` from the run-1 checkpoint. FP16 is a re-save of the FP32 IR
because `SmolVLA.extra_export_args` hardcodes `compress_to_fp16=False`; INT8 is NNCF
weight-only compression applied here, not a physicalai-train feature.

| Precision | Total bundle | Weights (`smolvla.bin`) | Export time | Note |
|---|---:|---:|---:|---|
| FP32 | 1,586.5 MB | 1,574.0 MB | 114.1 s | reference |
| FP16 | **800.5 MB** | 787.0 MB | 3.8 s | exactly 50% of FP32 |
| INT8 | **408.7 MB** | 395.1 MB | 32.6 s | NNCF `int8_asym` per-channel, 300/300 layers |

The exact 50% FP16 reduction confirms the fp32 cast worked. Without it the SmolVLM2 backbone
stays bfloat16, OpenVINO's FP16 compression skips that half, and the result is only ~20% smaller.

## Required-silicon result

Run on the reserved Intel bare-metal **`bm-ptl`** (Panther Lake) instance
`so101-bimanual-openvino`, OpenVINO **2026.3.1**. `NOT_THE_REQUIRED_SILICON: false`.
[Raw archive](out/tiber/)

| Device | Full name |
|---|---|
| CPU | Intel(R) Core(TM) Ultra X7 358H |
| GPU | Intel(R) Arc(TM) B390 GPU (iGPU) |
| NPU | Intel(R) AI Boost |

One inference is a complete 50-step action chunk, i.e. **one second of motion at 50 Hz control**.
Real-time therefore requires latency under 1,000 ms, and the last column is the headroom.

| Precision | Device | mean | p50 | p95 | fps | compile | vs real time |
|---|---|---:|---:|---:|---:|---:|---:|
| FP16 | CPU | 1,991.5 ms | 1,959.8 | 2,058.5 | 0.5 | 10.8 s | **0.5x — too slow** |
| FP16 | **GPU** | **124.6 ms** | 124.9 | 125.0 | 8.0 | 20.8 s | **8.0x** |
| FP16 | NPU | compile failed | | | | | — |
| INT8 | CPU | 2,696.7 ms | 2,699.2 | 2,699.9 | 0.4 | 10.5 s | 0.4x |
| INT8 | **GPU** | **119.7 ms** | 119.7 | 119.9 | 8.4 | 77.5 s | **8.4x** |
| INT8 | NPU | compile failed | | | | | — |

Three findings, two of them counter to the usual expectation:

1. **The integrated Arc GPU is 16x the CPU at FP16** (124.6 ms vs 1,991.5 ms). This is the
   difference between real-time control and not: the GPU runs at 8x real time, the CPU at 0.5x.
2. **INT8 is 35% *slower* than FP16 on CPU** (2,696.7 vs 1,991.5 ms) and only 4% faster on the
   GPU. NNCF applies weight-only compression, so weights are dequantized at runtime; for this
   model that cost exceeds the memory-bandwidth saving. **INT8's real benefit here is halving
   the artefact on disk (787 -> 395 MB), not latency.**
3. **The NPU cannot compile this graph.**
   `intel_npu/src/compiler_adapter/src/compiler_impl.cpp:280: Compilation failed`. A 450 M
   parameter VLA carrying dynamic shapes and an in-graph flow-matching loop is beyond what the
   AI Boost compiler accepts in OpenVINO 2026.3.1. CPU and GPU both accept it and return finite
   output.

### Closed-loop evaluation on Core Ultra

FP16 on the GPU (the benchmark's own best-device pick), 10 seeds x both instructions.

| Measurement | Result |
|---|---|
| Episodes | **20** (seeds 1000-1009, instructions A and B) |
| Successes | **0** |
| Final cup error to plate | **105 mm to 1,041 mm** (best: seed 1000/B at 105 mm) |
| Inference over the whole run | mean **127.28 ms**, p50 126.47, p95 132.44, 7.9 fps, 600 calls |
| Graph output | finite throughout |
| Faults raised | none — every episode ran its full 1,500 steps |
| Instruction swap table | all four cells **0** |

The empty swap table is itself informative: the policy completed *neither* instruction, so no
claim about instruction discrimination can be made in either direction.

A 5-seed FP16 CPU probe on the development laptop (AMD Ryzen 9 7940HS, **not** required silicon)
returned the same 0/5 with p50 2,621 ms, corroborating that the result is the policy and not the
Intel platform.

## Why the policy did not succeed

The failure was localised by replaying **training** observations through the exported IR and
comparing predicted actions with the recorded ones. In-distribution, a correct chain must
reproduce its own demonstrations. [`tools_replay_check.py`](tools_replay_check.py)

| Frame | Predicted (first 6 joints) | Recorded | MAE | Correlation |
|---:|---|---|---:|---:|
| 0 | `[-0.35 -1.35 0.46 1.08 ...]` | `[-0.00 -1.60 0.30 1.50 ...]` | 0.659 rad | 0.66 |
| 200 | `[-0.75 -0.47 0.04 1.02 ...]` | `[-0.41 -0.31 -0.07 1.14 ...]` | 0.478 rad | 0.72 |
| 400 | `[-0.20 -0.95 0.72 0.73 ...]` | `[ 0.07 -0.93 0.78 0.89 ...]` | 0.442 rad | 0.82 |
| 600 | `[-0.26 -1.26 0.26 1.17 ...]` | `[ 0.00 -1.60 0.30 1.50 ...]` | 0.391 rad | 0.85 |

**Correlation of 0.66-0.85 shows the observation pipeline, camera ordering, normalisation and
export are correct** — a mis-wired chain would correlate near zero. But the error is
**0.39-0.66 rad, i.e. 22-38 degrees per joint, on data the model was fit on.** The model is
underfit, not misconnected.

The reported training loss of 0.028 was misleading for a structural reason worth recording:
half of the 12 action dimensions sit frozen at the home pose throughout the ~50% of episodes
that are single-arm (instruction B moves only the right arm, by design). Those dimensions are
trivially predictable and dominate the mean, so the loss reflects the easy half of the problem.
**A low imitation loss on a bimanual dataset with one idle arm is not evidence of a learned task.**

Root cause: **one epoch of training.** SmolVLA fine-tunes are normally run for tens of thousands
of steps; at 3.564 s/step one overnight Kaggle session bought 10,300.

### Reproducibility note

The saved checkpoint's `config.json` carries SmolVLA **base** metadata — `observation.state: [6]`
and cameras named `camera1..3` — which contradicts its own normalizer tensors (12-D state, cameras
`top`/`left_wrist`/`right_wrist`) and its `train_config.json`. LeRobot does not rewrite those
fields on save. `intel_export.py` therefore fails with
`RuntimeError: shape '[6]' is invalid for input of size 12` until the config is corrected; the
original is preserved alongside it as `config.json.orig`. The weights are unaffected, because
SmolVLA pads state and action to `max_state_dim = 32` internally.

Two environment traps cost real time and are recorded so they do not recur: `physicalai` 0.1.1
imports `dataset_to_policy_features` from a module LeRobot moved in 0.6.0, so **LeRobot must be
pinned to 0.5.1**; and on a fresh Windows host `mujoco.dll` fails to load until the Microsoft
Visual C++ redistributable is installed.

## Limitations

The evidence covers only a simulated fixed **130 mm cup** moved via a table relay. Physics v2
retains successes only. Private v1 remains separate historical kinematic data. DART was active in
collection, while the conversion omitted its perturbation metadata. **No real-robot result
exists — every number here is simulation, benchmarked on required silicon for inference but never
run on physical SO-101 arms.** The shipped policy achieves 0/20 and is underfit at one epoch. The
NPU is unexercised because the graph will not compile for it.
[Scope](PLAN.md), [volume log](out/volume_v2.log), [v1 history and QA](briefs/ORCHESTRATOR-LOG.md),
[DART probe](briefs/WP2c-REPORT.md), [Intel archive](out/tiber/)
