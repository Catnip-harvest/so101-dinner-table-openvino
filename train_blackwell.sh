#!/usr/bin/env bash
# SmolVLA fine-tune on a rented Blackwell box (RTX 5090 / PRO 6000 / B200).
#
# Every flag below is copied from k1_kaggle_autopilot.py, the command that demonstrably
# ran 10,300 steps on lerobot 0.6.1 against this exact dataset. Verified against the
# 0.6.1 wheel: batch_size, steps, save_freq, log_freq, output_dir, job_name,
# policy.use_amp, policy.push_to_hub, policy.device, policy.path, wandb.enable.
#
# What run 1 got wrong and this fixes: batch 8 in fp32 on a T4 (1/8 of the standard
# recipe in samples per step). Here: batch 64 in bf16. In lerobot 0.6.1
# --policy.use_amp=true autocasts to bfloat16 - that killed the T4, which has no bf16,
# and is exactly right for Blackwell. No shim needed.
#
#   HF_TOKEN=<read-scope token> BUDGET_H=2.5 bash train_blackwell.sh
#
# Nothing is pushed to the Hub. The result is packed to a single .tgz for scp.
set -euo pipefail

DATASET="${DATASET:-VietHwang/dinner-table-v2}"
BUDGET_H="${BUDGET_H:-2.5}"        # wall clock for the main training step only
BATCH="${BATCH:-64}"
AMP="${AMP:-true}"                 # bf16 autocast; auto-falls back to fp32 if the loss goes nan
STEP_FLOOR="${STEP_FLOOR:-2000}"
STEP_CEIL="${STEP_CEIL:-30000}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT:-$HOME/smolvla_run3_$STAMP}"   # fresh dir: lerobot refuses an existing output_dir
export LOG="$OUT/train.log"

: "${HF_TOKEN:?set HF_TOKEN to a READ-scope Hugging Face token (the dataset is private)}"
export HF_TOKEN
export HF_HOME="${HF_HOME:-$HOME/hf}"
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"   # lerobot-train is single-process
unset HF_HUB_ENABLE_HF_TRANSFER                             # one fewer thing that can break
mkdir -p "$OUT" "$HF_HOME"
touch "$LOG"

say()  { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
die()  { say "FATAL: $*"; exit 1; }

# pip on a fresh Debian/Ubuntu image may refuse system-wide installs (PEP 668).
pipi() {
  python -m pip install -q "$@" 2>>"$LOG" \
    || python -m pip install -q --break-system-packages "$@" 2>>"$LOG"
}

say "=== phase 0: environment ==="
say "output dir: $OUT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee -a "$LOG"
python -c "import sys; print('python', sys.version.split()[0])" | tee -a "$LOG"

pipi --upgrade pip
pipi "lerobot[smolvla,dataset]==0.6.1"
python - <<'PY' 2>&1 | tee -a "$LOG"
import torch, lerobot
print("torch", torch.__version__, "| cuda", torch.version.cuda)
print("lerobot", getattr(lerobot, "__version__", "?"))
assert torch.cuda.is_available(), "FATAL: no CUDA device visible"
assert torch.cuda.is_bf16_supported(), "FATAL: bf16 unsupported; --policy.use_amp would die like the T4 did"
cap = torch.cuda.get_device_capability()
print("device", torch.cuda.get_device_name(0), "sm_%d%d" % cap,
      "| %.0f GB" % (torch.cuda.get_device_properties(0).total_memory / 1e9))
PY
grep -q "^lerobot 0.6.1" "$LOG" || die "lerobot 0.6.1 did not install; see $LOG"

if command -v lerobot-train >/dev/null 2>&1; then
  TRAIN="lerobot-train"
else
  TRAIN="python -m lerobot.scripts.lerobot_train"
fi
say "train entrypoint: $TRAIN"

say "=== phase 1: dataset (also proves the token works, before any GPU time is spent) ==="
python - <<PY 2>&1 | tee -a "$LOG"
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("$DATASET")
n_ep, n_fr = ds.num_episodes, ds.num_frames
print("episodes", n_ep, "frames", n_fr, "fps", ds.fps)
assert n_ep >= 100 and n_fr >= 80000, "FATAL: dataset smaller than expected; wrong repo or partial download"
PY
grep -q "^episodes" "$LOG" || die "dataset did not load; see $LOG"

# Shared argument list, identical to the proven Kaggle command.
common_args() {
  echo "--policy.path=lerobot/smolvla_base --dataset.repo_id=$DATASET" \
       "--policy.device=cuda --policy.push_to_hub=false --policy.use_amp=$AMP" \
       "--wandb.enable=false"
}

# Run a short measurement at a given batch; return 0 on success, 2 on CUDA OOM, 1 otherwise.
measure() {
  local b="$1" dir="$OUT/measure_b$1" alog="$OUT/measure_b$1.log"
  rm -rf "$dir"
  say "measuring 60 steps at batch $b"
  set +e
  # shellcheck disable=SC2046
  $TRAIN $(common_args) --batch_size="$b" --steps=60 --save_freq=100000 --log_freq=10 \
         --output_dir="$dir" --job_name="measure_b$b" >"$alog" 2>&1
  local rc=$?
  set -e
  cat "$alog" >>"$LOG"
  if grep -qiE "CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED" "$alog"; then return 2; fi
  return $rc
}

say "=== phase 2: measure throughput, halving batch on OOM ==="
while :; do
  set +e
  measure "$BATCH"
  rc=$?
  set -e
  if [ "$rc" -eq 0 ]; then
    # bf16 is unproven for this model (run 1 trained in fp32). Catch a nan loss in the
    # 60-step measurement and fall back to fp32 rather than discover it two hours later.
    if grep -qiE "\bloss[:=]\s*(nan|inf)" "$OUT/measure_b$BATCH.log"; then
      [ "$AMP" = "true" ] || die "loss is nan/inf even in fp32 - stop and investigate"
      say "loss went nan under bf16 autocast -> falling back to fp32 (use_amp=false) and re-measuring"
      AMP=false
      continue
    fi
    break
  fi
  if [ "$rc" -eq 2 ]; then
    [ "$BATCH" -le 8 ] && die "OOM even at batch 8"
    say "OOM at batch $BATCH -> retrying at $((BATCH / 2))"
    BATCH=$((BATCH / 2))
  else
    die "measurement failed (rc=$rc), not an OOM; last 40 lines of $OUT/measure_b$BATCH.log:"$'\n'"$(tail -40 "$OUT/measure_b$BATCH.log")"
  fi
done

SPS=$(python - "$OUT/measure_b$BATCH.log" <<'PY'
import re, sys
log = open(sys.argv[1], encoding="utf-8", errors="replace").read()
vals = [float(x) for x in re.findall(r"updt_s[:=]\s*([\d.]+)", log)]   # proven regex (autopilot)
tail = vals[-5:] if len(vals) >= 5 else vals
print("%.4f" % (sorted(tail)[len(tail) // 2]) if tail else "")
PY
)
[ -n "$SPS" ] || die "no updt_s lines in the measurement log - cannot size the run; refusing to guess"
say "measured: $SPS s/step at batch $BATCH"

STEPS=$(python - "$SPS" "$BUDGET_H" "$STEP_FLOOR" "$STEP_CEIL" <<'PY'
import sys
sps, budget_h, lo, hi = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
steps = int(budget_h * 3600 * 0.92 / sps)      # 8% reserved for checkpoint writes
print(max(lo, min(steps, hi)))
PY
)
SAVE_FREQ=$(( STEPS / 4 > 2000 ? STEPS / 4 : 2000 ))   # at most ~4 intermediate checkpoints
EPOCHS=$(python -c "print('%.2f' % ($STEPS * $BATCH / 323442))")
say "=== phase 3: $STEPS steps x batch $BATCH = $EPOCHS epochs, use_amp=$AMP, save every $SAVE_FREQ ==="
say "projected: $(python -c "print('%.1f' % ($STEPS * $SPS / 3600))") h"

RUN="$OUT/train"
set +e
# shellcheck disable=SC2046
$TRAIN $(common_args) --batch_size="$BATCH" --steps="$STEPS" --save_freq="$SAVE_FREQ" \
       --log_freq=100 --output_dir="$RUN" --job_name="run3" 2>&1 | tee -a "$LOG" \
       | grep --line-buffered -E "step:|loss:|Checkpoint|saved|Error|error"
rc=${PIPESTATUS[0]}      # exit status of $TRAIN itself, not of tee/grep (set +e is active)
set -e
[ "$rc" -eq 0 ] || die "training exited rc=$rc; see $LOG"

say "=== phase 4: verify the artefact ==="
CKPT="$RUN/checkpoints/last/pretrained_model"
[ -f "$CKPT/model.safetensors" ] || die "no $CKPT/model.safetensors"
SZ=$(stat -c%s "$CKPT/model.safetensors")
[ "$SZ" -gt 500000000 ] || die "model.safetensors is only $SZ bytes - not a full checkpoint"
python - "$LOG" <<'PY' | tee -a "$LOG"
import re, sys
log = open(sys.argv[1], encoding="utf-8", errors="replace").read()
losses = re.findall(r"\bloss[:=]\s*(nan|inf|-?[\d.]+(?:[eE][+-]?\d+)?)", log)
bad = [x for x in losses[-20:] if x in ("nan", "inf")]
print("loss samples:", len(losses), "| first:", losses[:1], "| last:", losses[-3:])
assert not bad, "FATAL: loss diverged to nan/inf in the last 20 samples"
PY

tar -C "$RUN/checkpoints/last" -czf "$OUT/run3_pretrained_model.tgz" pretrained_model
say "=== done: $(du -h "$OUT/run3_pretrained_model.tgz" | cut -f1) ==="
say "pull it back with:"
say "  scp -P <port> <user>@<host>:$OUT/run3_pretrained_model.tgz ."
