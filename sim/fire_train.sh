#!/usr/bin/env bash
# Waits for chunk 2 to reach HF, then pushes (= runs) the Kaggle training kernel
# and records the outcome. No model in the loop.
cd "/c/Users/vieth/Documents/lablab hackathon"
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)
K=viethwang3i/lablab-smolvla-train
LOG=out/fire_train.log
say(){ echo "$(date +%H:%M:%S) $*" >> "$LOG"; }

say "waiting for: convert/push chunk 2 rc=0"
until grep -q "convert/push chunk 2 rc=0" out/volume_v2.driver.log 2>/dev/null; do sleep 30; done
say "chunk 2 on HF; pushing kernel"
kaggle kernels push -p out/kernel_train >> "$LOG" 2>&1

for i in $(seq 1 40); do
  s=$(kaggle kernels status $K 2>&1 | tr -d '\r')
  say "$s"
  case "$s" in
    *COMPLETE*) say "RUN COMPLETE"; break;;
    *ERROR*|*CANCEL*)
      say "RUN FAILED -- downloading log"
      rm -rf out/kout2 && mkdir -p out/kout2
      kaggle kernels output $K -p out/kout2 >> "$LOG" 2>&1
      python - <<'PY' >> "$LOG" 2>&1
import glob,re
for f in glob.glob("out/kout2/*.log"):
    t=open(f,encoding="utf-8",errors="replace").read()
    msgs=re.findall(r'"data":"([^"]*(?:Error|error|Traceback|secret)[^"]*)"',t)
    print("\n".join(msgs[-25:]))
PY
      break;;
  esac
  sleep 60
done
say "watcher done"
