#!/usr/bin/env bash
# READ-ONLY monitor for the Kaggle training run. Never pushes (a push would
# clobber the notebook's HF_TOKEN secret binding).
cd "/c/Users/vieth/Documents/lablab hackathon"
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)
K=viethwang3i/lablab-smolvla-train
LOG=out/monitor_train.log
say(){ echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
BASE="2026-09-15 13:52:19"   # my failed v2 run; anything later is the owner's run
say "monitor armed; baseline lastRunTime=$BASE"
prev=""
for i in $(seq 1 480); do          # 8 h at 60 s
  st=$(kaggle kernels status $K 2>&1 | tr -d '\r' | sed 's/.*status //')
  lr=$(kaggle kernels list -m -s lablab-smolvla-train 2>&1 | tail -1 | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9:]{8}')
  cur="$st|$lr"
  [ "$cur" != "$prev" ] && { say "status=$st lastRun=$lr"; prev="$cur"; }
  case "$st" in
    *COMPLETE*)
      if [ "$lr" \> "$BASE" ]; then
        say "RUN COMPLETE (owner run) -- pulling output"
        rm -rf out/kfinal && mkdir -p out/kfinal
        kaggle kernels output $K -p out/kfinal >> "$LOG" 2>&1
        say "output in out/kfinal"; break
      fi;;
    *ERROR*|*CANCEL*)
      if [ "$lr" \> "$BASE" ]; then
        say "OWNER RUN FAILED -- pulling log"
        rm -rf out/kfail && mkdir -p out/kfail
        kaggle kernels output $K -p out/kfail >> "$LOG" 2>&1
        python - <<'PY' >> "$LOG" 2>&1
import glob,json,re
f=glob.glob("out/kfail/*.log")[0]
raw=open(f,encoding="utf-8",errors="replace").read()
try: ev=json.loads(raw)
except Exception: ev=[json.loads(m) for m in re.findall(r'\{"stream_name".*?\}',raw)]
print("".join(e.get("data","") for e in ev)[-2000:])
PY
        break
      fi;;
  esac
  sleep 60
done
say "monitor done"
