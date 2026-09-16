#!/usr/bin/env bash
# Run-3 post-training pipeline on the laptop (Git Bash):
#   scp the checkpoint tarball back -> extract -> fix the stale base-config metadata that
#   LeRobot leaves in config.json -> export FP32/FP16/INT8 with intel_export.py -> 5-seed
#   FP16 closed-loop probe. Everything lands on C: because D: has only 2.5 GB free.
#
#   bash prep_run3.sh /root/smolvla_run3_<stamp>/run3_pretrained_model.tgz
set -euo pipefail
cd "/c/Users/vieth/Documents/lablab hackathon"

REMOTE_TGZ="${1:?path of run3_pretrained_model.tgz on the box}"
H=180.189.55.43; P=56703; K=~/.ssh/tiber_ed25519
CK=/c/Users/vieth/run3_ckpt            # bash path
CKW="C:/Users/vieth/run3_ckpt"          # what Python must see (never /c/...)
OV="C:/Users/vieth/policy_ov_run3"
LOG=out/run3_prep.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
mkdir -p out; : > "$LOG"

say "=== 1. pull checkpoint ==="
rm -rf "$CK" && mkdir -p "$CK"
scp -i "$K" -P "$P" -o StrictHostKeyChecking=no "root@$H:$REMOTE_TGZ" "$CK/run3.tgz"
tar -C "$CK" -xzf "$CK/run3.tgz" && rm "$CK/run3.tgz"
PM="$CK/pretrained_model"
[ -f "$PM/model.safetensors" ] || { say "FATAL: no model.safetensors after extract"; exit 1; }
say "model.safetensors: $(stat -c%s "$PM/model.safetensors") bytes"

say "=== 2. verify normalizers are 12-D (proves it trained on our data) and fix config.json ==="
python - "$CKW/pretrained_model" <<'PY'
import json, sys, shutil
from safetensors import safe_open
b = sys.argv[1] + "/"
with safe_open(b + "policy_preprocessor_step_5_normalizer_processor.safetensors", framework="pt") as s:
    st = tuple(s.get_slice("observation.state.mean").get_shape())
    ac = tuple(s.get_slice("action.mean").get_shape())
    cams = sorted({k.split(".")[2] for k in s.keys() if k.startswith("observation.images.")})
print("normalizer state", st, "action", ac, "cameras", cams)
assert st == (12,) and ac == (12,), "FATAL: checkpoint normalizers are not 12-D"
assert cams == ["left_wrist", "right_wrist", "top"], "FATAL: unexpected camera set %s" % cams
tc = json.load(open(b + "train_config.json"))
print("train_config: repo", tc.get("dataset", {}).get("repo_id"), "| steps", tc.get("steps"), "| batch", tc.get("batch_size"),
      "| amp", tc.get("policy", {}).get("use_amp"))
shutil.copy(b + "config.json", b + "config.json.orig")
c = json.load(open(b + "config.json"))
c["input_features"] = {"observation.state": {"type": "STATE", "shape": [12]},
                       **{f"observation.images.{cam}": {"type": "VISUAL", "shape": [3, 256, 256]} for cam in ("top", "left_wrist", "right_wrist")}}
c["output_features"] = {"action": {"type": "ACTION", "shape": [12]}}
json.dump(c, open(b + "config.json", "w"), indent=2)
print("config.json aligned to the trained shapes (original kept as config.json.orig)")
PY

say "=== 3. export FP32 / FP16 / INT8 -> $OV ==="
python intel_export.py --out "$OV" --checkpoint "$CKW/pretrained_model" 2>&1 | grep -E "^\[|TOTAL|smolvla\.bin|Traceback|Error" | tee -a "$LOG"
[ -f "$(cygpath -u "$OV")/fp16/smolvla.xml" ] || { say "FATAL: FP16 IR missing"; exit 1; }

say "=== 4. 5-seed FP16 probe, instruction A (same seeds as the run-1 probe: 3000-3004) ==="
python intel_eval.py --ir "$OV/fp16" --device CPU --seeds 5 --seed0 3000 --instruction A \
       --out out/eval_run3_probe 2>&1 | grep -E "^seed=|inference|swap|wrote|Traceback" | tee -a "$LOG"
python - <<'PY' | tee -a "$LOG"
import json, glob
d = json.load(open(glob.glob("out/eval_run3_probe/**/eval_results.json", recursive=True)[0]))
eps = d["episodes"]
print("RUN 3 PROBE: %d/%d success | cup errors (mm): %s" % (
    sum(e["success"] for e in eps), len(eps),
    [int(1000 * e["final_cup_position_error_to_plate_m"]) for e in eps]))
print("run 1 on the same seeds was 0/5 with errors [411, 880, 351, 559, 202] mm")
PY
say "=== done. If better than run 1: bundle $OV/fp16 for Tiber (764 MB) and re-run intel_eval there ==="
