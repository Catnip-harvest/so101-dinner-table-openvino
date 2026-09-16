# The Intel path — export to OpenVINO, benchmark on Core Ultra

**Written 15 Sep 2026. Submission closes 17 Sep 08:00 Hanoi.**

Everything below was run. Where a number appears, it came out of a command on the
owner's machine (Windows 11, AMD Ryzen 9 7940HS, **no Intel accelerator**) and the
command is quoted next to it. Where something was not run, it says so in bold.

**What is proven.** A SmolVLA policy of the dinner-table shape — 12-D state, 12-D
action, three cameras, 50-step action chunk — exports to OpenVINO IR in FP32, FP16 and
INT8, and the IR runs a full text-to-action inference with **no torch, no LeRobot and no
physicalai in the process**. This was done with an *untrained* model, deliberately,
because the simulation pipeline is still blocked (`FINDINGS-SIM.md` section 5) and the
export path does not care about weight values.

**Why a trained checkpoint is a drop-in and not a rewrite.** The pretrained
`lerobot/smolvla_base` is a 6-DoF single arm. Re-stated to the dinner table's 12-D
bimanual shape, its state dict has **500 tensors with the identical shapes** as the
randomly initialised 12-D model — zero mismatches, checked tensor by tensor. SmolVLA
pads state and action to `max_state_dim`/`max_action_dim` (32) inside the model, so the
real dimension only changes the IR's `state` input port and the last axis of the
`action` output. **The graph is the same, so the latency measured here is the latency a
trained checkpoint will have.**

---

## Part 1 — the twenty minutes on the Tiber box

The Tiber AI Cloud instance is a **Core Ultra X7 358H, Windows 11**. Nothing in this
part needs a GPU driver toolkit, a build tool, or a training environment. It is one
`pip install` and three commands.

### What to copy onto the box

| Item | Size | From |
|---|---|---|
| `intel_bench.py` | 6 KB | this directory |
| `policy_ov\fp16\` | 763 MB | wherever the export ran |
| `policy_ov\int8\` | 390 MB | same |

Each precision directory is a complete bundle: `smolvla.xml`, `smolvla.bin`,
`tokenizer.xml`, `tokenizer.bin`, `manifest.json`, `stats.json`. Copy `fp32\` too if
the bandwidth is free (1.5 GB), but FP16 and INT8 are the two numbers the submission
needs.

### Prerequisites, in order

**1. Put everything under a short path.** `C:\pai`, not a directory under
`Documents\...`. Windows long-path support is **off** on the owner's machine
(`HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled = 0`, checked) and
is off by default on a fresh Windows 11. A deep venv path makes `pip install` of
anything containing torch fail partway through with

```
ERROR: Could not install packages due to an OSError: [Errno 2] No such file or directory:
'...\torch\include\ATen\native\transformers\cuda\mem_eff_attention\epilogue\epilogue_thread_apply_logsumexp.h'
```

which reads like a corrupt download and is not one. **Do not turn long paths on to fix
it** — that is a system setting. Use a short path.

**2. The NPU needs its driver.** The OpenVINO NPU plugin only enumerates `NPU` if the
Intel NPU driver ("Intel AI Boost" in Device Manager) is installed. **Not verified — no
Intel silicon in this session.** Step 1 below tells you within seconds whether it is
there; if it is not, the CPU and GPU numbers still stand on their own.

**3. Python 3.12–3.14.** `physicalai-train` declares `>=3.12,<3.15`; the deploy bundle
itself works on any Python with an OpenVINO wheel. 3.13.14 was used here.

### The commands

```powershell
mkdir C:\pai
cd C:\pai
python -m venv dv
.\dv\Scripts\python.exe -m pip install --no-cache-dir openvino numpy openvino-tokenizers
```

Expect `Successfully installed numpy openvino openvino-telemetry openvino-tokenizers`.
Five packages, 299 MB on disk, and **no torch** — measured.

`openvino-tokenizers` is not optional even though `intel_bench.py` does not import it.
Plain `openvino` cannot read `tokenizer.xml`; it fails with

```
Cannot create SpecialTokensSplit layer ... from unsupported opset: extension
```

The wheel is small, torch-free, and an IR written by `openvino_tokenizers` 2026.1 loads
under 2026.3.1 — checked.

**Step 1 — what does this box expose?** Seconds. Do this first, before anything
depends on it.

```powershell
.\dv\Scripts\python.exe C:\pai\intel_bench.py --out C:\pai\devices.json
```

Expected on the Tiber box, roughly:

```
OpenVINO 2026.3.1-...
  CPU      Intel(R) Core(TM) Ultra X7 358H
  GPU      Intel(R) Arc(TM) Graphics (iGPU)
  NPU      Intel(R) AI Boost
```

If the `NPU` line is missing you get this instead, which is the driver telling you it is
not installed:

```
  NOTE: no NPU listed.  Either this box has none, or - on Panther Lake -
        the driver is not installed.  CPU and GPU benchmark fine without it.
```

That message is what this machine prints, verbatim, because it has no NPU.

**Step 2 — FP16 on all three devices.**

```powershell
.\dv\Scripts\python.exe C:\pai\intel_bench.py --ir C:\pai\policy_ov\fp16\smolvla.xml --runs 10 --out C:\pai\bench_fp16.json
```

**Step 3 — INT8 on all three devices.**

```powershell
.\dv\Scripts\python.exe C:\pai\intel_bench.py --ir C:\pai\policy_ov\int8\smolvla.xml --runs 10 --out C:\pai\bench_int8.json
```

Each line of output looks like this (this is the real FP16-on-CPU line from this
machine):

```
benchmarking C:/pai/work/pov/fp16/smolvla.xml (786.7 MB of weights)
  CPU      2191.61 ms mean | 3271.75 p95 |    0.5 fps | compile 28.85s
```

**Budget 10 runs, not the default 50.** One inference is ~2 s on a laptop CPU and
compilation alone is 20–35 s, so `--runs 10` costs about 90 s per device and `--runs 50`
costs about 4 minutes per device. Three devices × two precisions at `--runs 10` is
roughly 10 minutes, which is the whole window. If there is time left, repeat steps 2 and
3 — see the note on variance at the bottom of this file, which is the reason the
reference numbers here are not quotable.

If a device fails, the script prints `FAILED: <reason>` for that device, writes the
error into the JSON, and carries on to the next one. **A device that refuses the model
is a result, not a problem — paste the line verbatim into the writeup.** The likeliest
refusal is the NPU: 27 480 operations and 787 MB of weights is large for it, and the
plugin is stricter about dynamic shapes and int64 than CPU is. The IR here is fully
static, which is the main thing in its favour.

**Cross-check, if a number looks wrong.** OpenVINO ships its own benchmark tool in the
same venv, and it does not share a line of code with `intel_bench.py`:

```powershell
.\dv\Scripts\benchmark_app.exe -m C:\pai\policy_ov\int8\smolvla.xml -d CPU -hint latency -niter 10
```

### What to bring back

`devices.json`, `bench_fp16.json`, `bench_int8.json`. Each JSON carries the absolute IR
path, the weight size in MB, and per-device `mean_ms` / `p50_ms` / `p95_ms` / `fps` /
`compile_s` / `output_finite`. `output_finite` must be `true`; if it is `false` the
graph produced NaN and the latency means nothing.

---

## Part 2 — regenerating the IR (build side, not on the Tiber box)

Run this wherever the checkpoint is — this machine, or the Kaggle notebook that trains
it. It needs torch, LeRobot and physicalai-train, about 2 GB of venv, and roughly four
minutes.

```powershell
mkdir C:\pai
cd C:\pai
python -m venv ev
.\ev\Scripts\python.exe -m pip install --no-cache-dir "physicalai-train[smolvla]"
.\ev\Scripts\python.exe -m pip install --no-cache-dir "lerobot[dataset]==0.5.1" nncf
.\ev\Scripts\python.exe "<this directory>\intel_export.py" --out C:\pai\policy_ov
```

With a trained checkpoint, add `--checkpoint <local dir or HF repo id>`. Everything else
stays the same.

Expected tail:

```
[build] policy built in 85.8s
[build] 450.0M parameters, 3 cameras, state 12, action 12, chunk 50
[build] cast every weight to float32
[export] fp32 done in 108.6s
[export] fp16 done in 3.2s
[export] int8 done in 30.5s
```

and these sizes, which are exact halvings and the sign the precision work actually
landed:

| Precision | `smolvla.bin` | `smolvla.xml` | bundle |
|---|---|---|---|
| FP32 | 1 573.4 MB | 10.5 MB | 1.5 GB |
| FP16 | 786.7 MB | 11.4 MB | 763 MB |
| INT8 | 394.5 MB | 11.5 MB | 390 MB |

### Four traps, all of which fired here

**1. LeRobot must be pinned to 0.5.1.** `physicalai` 0.1.1 imports
`dataset_to_policy_features` from `lerobot.datasets.feature_utils`. LeRobot moved that
symbol to `lerobot.utils.feature_utils` in 0.6.0, so on LeRobot 0.6.1 — which is what
`lerobot>=0.5.1` resolves to today — `import physicalai.policies.smolvla` dies with

```
ImportError: cannot import name 'dataset_to_policy_features' from 'lerobot.datasets.feature_utils'
```

**2. `physicalai-train[smolvla]` does not pull in enough LeRobot.** Importing
`physicalai.policies.smolvla` pulls `physicalai.data`, which pulls
`lerobot.datasets`, which hard-requires `datasets` and `av`. Those live in LeRobot's
own `[dataset]` extra. Hence the second install line. Without it:

```
ImportError: 'datasets' is required but not installed. Install it with: pip install 'lerobot[dataset]'
```

**3. The SmolVLM2 backbone is bfloat16, and FP16 compression ignores bf16.** The HF
config asks for bfloat16, so 907 constants totalling **584.4 MB** come out of the
export as `bf16` while the action expert's **405.6 MB** comes out as `f32`.
`compress_to_fp16` halves the f32 half and leaves the bf16 half untouched, so "FP16"
lands at 787 MB out of 990 MB — a 20% saving instead of 50% — and the NPU may reject
bf16 outright. `intel_export.py` calls `policy.model.float()` before exporting, which
is why the table above halves cleanly. `--no-cast-fp32` reproduces the mixed graph if
you want to see it.

**4. FP16 is not an export flag, and INT8 is not in physicalai-train at all.**
`SmolVLA.extra_export_args` hardcodes `compress_to_fp16=False` in its
`OpenVINOExportParameters`, and `export_kwargs` cannot reach that field — they go to
`openvino.convert_model`. So FP16 is a re-save:

```python
openvino.save_model(openvino.Core().read_model(fp32_xml), fp16_xml, compress_to_fp16=True)
```

and INT8 is NNCF weight compression, which needs no calibration data:

```python
nncf.compress_weights(model, mode=nncf.CompressWeightsMode.INT8_ASYM)
```

NNCF reported `int8_asym, per-channel | 100% (300 / 300)` — every eligible layer
compressed. **This is weight-only INT8.** Full INT8, with activations quantised too,
needs real frames through `nncf.quantize` with a calibration set, and there is no
dataset yet. Say "INT8 weight compression" in the writeup, not "INT8 quantisation".

### One more thing the flags do not control

When `--checkpoint` is given, the checkpoint's own config wins: `chunk_size`,
`resize_imgs_with_padding` and the state/action dimensions all come from the repo, and
`--chunk-size`/`--resize` are ignored. `intel_export.py` prints the checkpoint's
shapes, and if they differ from the flags it re-states the policy to the flag shapes and
says so. That is the mechanism that turns 6-DoF `lerobot/smolvla_base` into a 12-D
dinner-table export. `--keep-checkpoint-shape` disables it.

---

## Part 3 — the deploy bundle contract

The inference-time bundle is `openvino`, `openvino-tokenizers`, `numpy`, plus `mujoco`,
`imageio` and `requests` for the rollout and video. **No LeRobot, no torch, no
physicalai, no transformers.** Verified: a process with `torch` not importable ran text
→ tokens → policy → a finite `(1, 50, 12)` action chunk in 1 975 ms on CPU.

### IR ports

| Port | Shape | dtype |
|---|---|---|
| in `state` | `[1, 12]` | `float32` |
| in `tokenized_prompt` | `[1, 48]` | `int64` |
| in `tokenized_prompt_mask` | `[1, 48]` | **`bool`** |
| in `images` | `[3, 1, 3, 512, 512]` | `float32` |
| in `image_masks` | `[3, 1]` | `bool` |
| out `action` | `[1, 50, 12]` | `float32` |

All static. `images` is **one stacked tensor of all three cameras**, not three ports;
axis 0 is the camera, in the order the cameras appear in `stats.json`.

### What the deploy code has to do itself

The IR is the model only. The preprocessing that physicalai's `SmolVLAPreprocessor`
does in torch has to be reimplemented in numpy, and these are its actual rules, read
out of `physicalai/policies/smolvla/preprocessor.py`:

- **Task string:** append `"\n"` if it does not already end with one. Then tokenize
  with `max_length=48`, truncation on, padding to max length.
- **Tokenizer output:** `tokenizer.xml` emits `input_ids` and `attention_mask` both as
  **`int64`**, but the policy wants the mask as **`bool`**. Cast it —
  `mask.astype(bool)`. This is the one silent shape mismatch in the chain.
- **Images:** come in as `float32` in `[0, 1]`, `(1, 3, H, W)` per camera. Resize with
  pad to 512×512: `ratio = max(W/512, H/512)`, bilinear to
  `(int(H/ratio), int(W/ratio))`, then pad **left and top only** with value `0`. Then
  `img * 2.0 - 1.0` to reach `[-1, 1]` for SigLIP. Stack the cameras on a new leading
  axis. With square cameras — 256×256, as configured — the ratio is exactly 0.5, so
  this is a clean 2× upscale and nothing is padded.
- **`image_masks`:** all `True`, one per camera. They mark a camera as present, and an
  all-`False` mask makes the model emit NaN.
- **State:** `(state - mean) / std` from `stats.json`'s `observation.state`.
- **Action:** `action * std + mean` from `stats.json`'s `action`.

`stats.json` holds identity stats (mean 0, std 1) for the untrained export. A trained
checkpoint carries the real numbers and `intel_export.py` writes those instead, so the
deploy code never changes.

---

## Reference numbers from this machine — NOT the submission numbers

These came from `intel_bench.py --devices CPU --runs 10` on an **AMD Ryzen 9 7940HS**
laptop with **no Intel accelerator**. They exist to prove the pipeline runs and to give
a rough sense of scale. **The submission's numbers must come off the Core Ultra X7
358H.**

| Precision | Weights | mean | p50 | p95 | fps | compile | output finite |
|---|---|---|---|---|---|---|---|
| FP32 | 1 573.4 MB | 2 470.28 ms | 2 544.94 ms | 2 671.83 ms | 0.4 | 26.24 s | yes |
| FP16 | 786.7 MB | 2 191.61 ms | 2 158.88 ms | 3 271.75 ms | 0.5 | 28.85 s | yes |
| INT8 | 394.5 MB | 1 777.12 ms | 1 825.84 ms | 2 152.75 ms | 0.6 | 33.18 s | yes |

**Do not read a precision ranking out of that table. It is inside the noise.** The same
INT8 IR, same device, same command, run three times back to back, gave

```
  CPU      1411.13 ms mean | 1487.45 p95 |    0.7 fps | compile 19.36s
  CPU      1480.82 ms mean | 1807.11 p95 |    0.7 fps | compile 19.29s
  CPU      2236.90 ms mean | 2605.48 p95 |    0.4 fps | compile 19.94s
```

— a **1.6× spread on an identical measurement**. This is a thermally throttled laptop
with other work on it, and at ~2 s per inference a 10-run sample has no chance of
averaging that out. The right conclusion from the table is only "the order of magnitude
is seconds per inference on a CPU with no Intel-specific kernels, and the IR sizes halve
and quarter as they should". AMX and the NPU are exactly what this architecture is
bottlenecked on, and neither exists on this box.

**So on the Tiber box: run every device/precision combination at least three times, or
once at `--runs 30`, and close anything else that is running.** A single `--runs 10`
sample is enough to prove a device works; it is not enough to publish a latency. If the
three repeats do not agree within a few percent, say so in the writeup rather than
picking the pretty one.

OpenVINO on this machine also enumerates a second device, `GPU — NVIDIA GeForce RTX 4060
Laptop GPU (dGPU)`, through the OpenCL path. It runs — 11 793.76 ms mean at INT8,
`--runs 3` — and the number is meaningless: the plugin's kernels are written for Intel
Xe. It is recorded only so nobody mistakes a `GPU` row in a JSON from this machine for
an Arc measurement.

### What was fixed in `intel_bench.py`

It had never run. It runs now; three things were wrong.

- **p95 was off by one and could land below the mean.** `int(n * 0.95) - 1` returns the
  median at `n = 3`, and the GPU run above proved it: `p95 11 754.12` against
  `mean 11 793.76`. Now nearest-rank.
- **Boolean inputs were filled with zeros.** `_fill` sent an all-`False` attention mask
  and an all-`False` camera-presence mask, which makes the model emit NaN. Latency is
  unaffected on a static graph, so this would have produced a healthy-looking number
  from a model computing nothing. Booleans are now `True`.
- **Nothing checked the output.** One `np.isfinite` pass on the warmup result, reported
  as `output_finite` and flagged in the console as `OUTPUT NOT FINITE`. This project has
  already been burned once by a number that was a crash in disguise.

Also: the JSON now records the IR path and its weight size, because three precisions
otherwise produce three indistinguishable result files.

## State on disk

`C:\pai\ev` (2.0 GB) is the verified export environment and `C:\pai\dv` (299 MB) the
verified deploy environment; both are rebuildable from the commands above. The IR files
themselves were **deleted** after benchmarking — they were random weights, worth nothing
but the 2.7 GB they occupied on a disk at 98%. Four minutes of `intel_export.py`
regenerates them.

The raw result JSONs behind every number in this file are in `C:\pai\work\`:
`final_fp32.json`, `final_fp16.json`, `final_int8.json` are the reference table,
`rep_1..3.json` are the three identical repeats that established the variance,
`bench_gpu.json` is the NVIDIA-through-OpenCL run, and `devices_final.json` is the
device probe.
