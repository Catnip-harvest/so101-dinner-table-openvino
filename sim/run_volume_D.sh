#!/usr/bin/env bash
# Continuation of the physics collection with all heavy output on D: (10 GB free,
# C: is tight). Raw chunks and the growing LeRobot root both live on D:. Same
# serial collect -> convert -> private push -> delete-raw loop. Starts at the
# chunk number given as $1 (default 2), since chunk 1 already landed from C:.
set -u
cd "/c/Users/vieth/Documents/lablab hackathon"
export BIMANUAL_CARRY=physics
REPO=VietHwang/dinner-table-v2
ROOT=/d/lerobot/VietHwang__dinner-table-v2
LOG=out/volume_v2.log
START=${1:-2}
mkdir -p /d/lerobot /d/episodes
free_gb() { powershell -NoProfile -Command "[math]::Round((Get-PSDrive D).Free/1GB,2)" | tr -d '\r'; }
for N in $(seq "$START" 8); do
  now=$(date +%H%M)
  if [ "$now" -gt 2330 ]; then echo "$(date +%T) clock stop before chunk $N" | tee -a "$LOG"; break; fi
  f=$(free_gb); echo "$(date +%T) chunk $N start, D: free ${f} GB" | tee -a "$LOG"
  if awk "BEGIN{exit !($f < 1.5)}"; then echo "$(date +%T) D: floor, stopping" | tee -a "$LOG"; break; fi
  RAW=/d/episodes/chunk$N
  python sim/collect.py --out "$RAW" --target-kept 100 --instruction both --workers 3 --seed $((100000*N)) >> "$LOG" 2>&1; rc=$?
  echo "$(date +%T) collect chunk $N rc=$rc" | tee -a "$LOG"
  python -c "import json;m=json.load(open('$RAW/manifest.json'));print({k:v for k,v in m.items() if not isinstance(v,(list,dict))})" >> "$LOG" 2>&1
  /c/pai/ev/Scripts/python.exe sim/to_lerobot.py --raw "$RAW" --repo "$REPO" --root "$ROOT" --append --chunk-id "chunk$N" --push >> "$LOG" 2>&1; crc=$?
  echo "$(date +%T) convert/push chunk $N rc=$crc" | tee -a "$LOG"
  if [ "$crc" -eq 0 ]; then rm -rf "$RAW"; else echo "$(date +%T) keeping raw chunk $N for inspection" | tee -a "$LOG"; break; fi
  if [ "$rc" -eq 2 ]; then echo "$(date +%T) yield below 30%, stopping" | tee -a "$LOG"; break; fi
  # stop when the growing dataset would threaten D:
  if awk "BEGIN{exit !($(free_gb) < 2.0)}"; then echo "$(date +%T) D: below 2GB after chunk $N, stopping clean" | tee -a "$LOG"; break; fi
done
echo "$(date +%T) D-DRIVER DONE, D: free $(free_gb) GB" | tee -a "$LOG"
