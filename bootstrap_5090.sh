#!/usr/bin/env bash
# One-shot bootstrap for a freshly rented Blackwell box (Vast / RunPod / Lambda).
# Fails in about 60 seconds on a box that cannot do this job, BEFORE any paid time
# goes into a 6 GB dataset download or a pip resolve.
set -euo pipefail

pipi() {   # PEP 668: fresh Debian/Ubuntu images refuse system-wide pip without this flag
  python -m pip install -q "$@" 2>/dev/null || python -m pip install -q --break-system-packages "$@"
}

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "=== disk (need ~40 GB: 6 GB dataset + HF cache + ~5 checkpoints at 1.3 GB) ==="
df -h "$HOME" | tail -1
FREE_GB=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
[ "$FREE_GB" -ge 40 ] || { echo "FAIL: only ${FREE_GB} GB free in \$HOME; rent a box with >=60 GB disk"; exit 1; }

echo "=== python / pip ==="
python -c "import sys; print('python', sys.version.split()[0])"
python -m pip --version

echo "=== torch ==="
if ! python -c "import torch" 2>/dev/null; then
  echo "torch missing; installing the cu128 build"
  pipi torch --index-url https://download.pytorch.org/whl/cu128
fi
python - <<'PY'
import torch, time
assert torch.cuda.is_available(), "FAIL: no CUDA device"
cap = torch.cuda.get_device_capability()
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| sm_%d%d" % cap,
      "|", torch.cuda.get_device_name(0))
assert torch.cuda.is_bf16_supported(), "FAIL: bf16 unsupported - wrong card for this plan"
if cap[0] >= 10:   # Blackwell: a torch without sm_100/sm_120 kernels imports fine and dies on first matmul
    archs = torch.cuda.get_arch_list()
    print("arch list:", archs)
    assert any(a.endswith(("100", "120")) for a in archs), \
        "FAIL: this torch has no Blackwell kernels. Run: python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128"
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize(); t = time.perf_counter()
for _ in range(50): x @ x
torch.cuda.synchronize()
print("measured bf16 matmul: %.0f TFLOPS" % (50 * 2 * 4096**3 / (time.perf_counter() - t) / 1e12))
print("PASS: box is usable")
PY

echo
echo "Next:  HF_TOKEN=<read-scope token> BUDGET_H=2.5 bash train_blackwell.sh"
