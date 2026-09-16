# Dinner-table cup relay

This project is a MuJoCo bimanual SO-101 pipeline for scripted demonstration collection, LeRobot conversion, language-conditioned SmolVLA training, and OpenVINO closed-loop evaluation. The current demonstrated task is cup-only: a fixed **130 mm cup** is moved through a table relay, not a mid-air handoff. [Scope evidence](PLAN.md)

## Current dataset standard

Use the private physics dataset `VietHwang/dinner-table-v2` for training. It contains **400 success-only episodes / 323,442 frames at 50 fps**, collected from **749 attempts** for a **53.4% kept/attempt yield**. Instruction A has **203 kept from 375 attempts**; instruction B has **197 kept from 374 attempts**. [Volume ground truth](out/volume_v2.log)

| Chunk | Attempts | Kept | Frames | Evidence |
|---|---:|---:|---:|---|
| chunk1 | 192 | 100 | 81,944 | [volume log](out/volume_v2.log) |
| chunk2 | 176 | 100 | 79,984 | [volume log](out/volume_v2.log) |
| chunk3 | 204 | 100 | 80,044 | [volume log](out/volume_v2.log) |
| chunk4 | 177 | 100 | 81,470 | [volume log](out/volume_v2.log) |

Hub load-back QA passed: the dataset has exactly **two task strings**, **three 256x256 cameras** (`top`, `left_wrist`, `right_wrist`), and **12-D state/action**. The visual grid showed both wrist cameras looking down the fingers with the jaws visible, and the cup in/against the jaws mid-episode. `is_perturbed` is absent from converted frame features and episode metadata, so no converted perturbation share is claimed. [QA record](briefs/ORCHESTRATOR-LOG.md)

## Trained policy and Intel result

A SmolVLA checkpoint exists and was benchmarked on the required silicon. Both headline numbers:

| | |
|---|---|
| Training | **10,300 steps**, batch 8, Tesla T4, fp32, 10.37 h, final loss 0.028 |
| Export | FP32 1,586 MB -> **FP16 800 MB** -> **INT8 409 MB** |
| Intel device | **Core Ultra X7 358H** + **Arc B390 iGPU** + AI Boost NPU, OpenVINO 2026.3.1 |
| Inference (FP16, iGPU) | **124.6 ms** per 50-step action chunk = **8x real time** at 50 Hz |
| iGPU vs CPU | **16x faster** (124.6 ms vs 1,991.5 ms) |
| NPU | graph does not compile; CPU and GPU both do |
| Closed-loop task success | **0 / 20** (10 seeds x 2 instructions) |

The deployment path works end to end on Panther Lake. The policy does not complete the task, and
the cause is measured rather than assumed: it received **1.006 epochs** of training, and replaying
training observations through the exported IR shows 0.39-0.66 rad error in-distribution with
0.66-0.85 correlation -- the chain is correct, the fit is not. Full analysis in
[RESULTS.md](RESULTS.md#why-the-policy-did-not-succeed).

## Measured gates

The historical kinematic carry gate measured **8/10** scripted successes. The repaired physics carry measured **7/10** at one seed and **14/20** at a second seed, without relaxing the predicates. These are scripted-controller gates, not trained-policy results. [Kinematic gate](briefs/WP1-REPORT.md), [physics gates](briefs/WP1b-REPORT.md)

The collector-fix probe kept **10/18 attempts and produced 7,005 frames**. It is a small validation probe, not the volume yield above. [Collector probe](briefs/WP2c-REPORT.md)

## Honest limitations

- The demonstrated scope is simulated cup manipulation only, with a fixed **130 mm cup** and table relay. [Scope evidence](PLAN.md)
- The required-silicon benchmark is complete, but there is **no real-robot result** -- every number is simulation, with inference measured on Intel hardware. [Intel archive](out/tiber/)
- Private v1 is separate historical kinematic data: **24 episodes / 16,236 frames**. It must not be mixed up with physics v2. [Round-3 record](briefs/ORCHESTRATOR-LOG.md)
- Physics v2 is success-only collection, so it does not preserve failed trajectories. [Volume ground truth](out/volume_v2.log)
- DART was used during collection, but converted v2 omits perturbation metadata. [DART evidence](briefs/WP2c-REPORT.md), [converted QA](briefs/ORCHESTRATOR-LOG.md)
- The trained policy scores **0/20** in closed loop and is underfit at one epoch. It is shipped as an honest negative result, not a working manipulation policy. [Measurement](RESULTS.md#why-the-policy-did-not-succeed)
- The **NPU is unexercised**: OpenVINO 2026.3.1 cannot compile a 450 M VLA with dynamic shapes and an in-graph flow-matching loop. [Benchmark](out/tiber/out/bench_fp16.json)

See [ARCHITECTURE.md](ARCHITECTURE.md) for contracts, [RESULTS.md](RESULTS.md) for measurements, and [TIBER-RUNBOOK.md](TIBER-RUNBOOK.md) for the target-machine procedure.

Credits: MuJoCo Menagerie for the robot/gripper asset; PegBitStudio for the holding-check reference; peacestate for the instruction-swap test reference. These credits do not imply reproduction of their results. [Attribution notes](PLAN.md)
