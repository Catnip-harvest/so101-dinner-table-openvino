# Tiber one-command runbook

Copy the following into `C:\pai` while preserving directory names:

| Item | Laptop size |
|---|---:|
| `policy_ov\fp16\` | 763.44 MiB |
| `policy_ov\int8\` | 389.74 MiB |
| `sim\` | 66.50 MiB |
| `intel_bench.py`, `intel_eval.py`, `intel_render_test.py`, `intel_xpu_smoke.py`, `run_intel_demo.ps1`, `PLAN.md`, `INTEL-PATH.md` | 0.08 MiB |
| **Total** | **1,219.76 MiB** |

Do not copy a virtual environment; the command creates the short-path `C:\pai\dv` environment. In PowerShell, run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass; C:\pai\run_intel_demo.ps1
```

A good run prints `RENDER OK`, lists CPU/GPU/NPU as provided by the installed drivers, writes FP16 and INT8 benchmark rows, prints `best FP16 device`, completes 10 seeds plus the instruction swap, prints inference p50/p95, creates `C:\pai\tiber_results_<timestamp>.zip`, and ends with `INTEL DEMO COMPLETE`.

Failure guide:

- `RENDER FAILED — see fallbacks in PLAN.md §6`: the RDP graphics context is unusable. The script exits immediately; follow PLAN §6 before retrying.
- No NPU in `devices.json`: install/update the Intel NPU driver. CPU/GPU remain usable.
- A device has `FAILED` or `OUTPUT NOT FINITE`: that device/precision is recorded but is not selected for evaluation.
- `missing ... IR`: recopy both complete precision directories, including `.xml`, `.bin`, tokenizer, manifest and stats.
- Dependency or DLL error: keep the exact console text. Confirm Python is 3.12–3.14 and Windows Application Control permits packages under `C:\pai`.
- Evaluation exits nonzero: inspect `C:\pai\out\eval_fp16_best\eval_results.json`, especially `faults` and `evaluation_error`.

Optional, after the main result only: append `-XpuSmoke -XpuDataset C:\pai\xpu_smoke_dataset`. It uses the separate `C:\pai\xv` environment and never invalidates the OpenVINO result.
