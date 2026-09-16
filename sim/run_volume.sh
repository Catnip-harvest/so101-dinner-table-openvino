#!/usr/bin/env bash
# Serial physics-data collection: collect a chunk, convert+push it, delete the raw
# files, repeat. Three workers, because six plus a converter blew the pagefile up
# to the disk floor on 15 Sep. Stops on the disk floor, on low yield, or at 23:30.
set -u
cd "/c/Users/vieth/Documents/lablab hackathon"
export BIMANUAL_CARRY=physics
REPO=VietHwang/dinner-table-v2
ROOT=out/lerobot/VietHwang__dinner-table-v2
LOG=out/volume_v2.log
free_gb() { powershell -NoProfile -Command "[math]::Round((Get-PSDrive C).Free/1GB,2)" | tr -d '\r'; }
for N in 1 2 3 4 5 6; do
  now=$(date +%H%M)
  if [ "$now" -gt 2330 ] && [ "$N" -gt 4 ]; then echo "$(date +%T) clock stop before chunk $N" | tee -a "$LOG"; break; fi
  f=$(free_gb)
  echo "$(date +%T) chunk $N start, free ${f} GB" | tee -a "$LOG"
  if awk "BEGIN{exit !($f < 1.5)}"; then echo "$(date +%T) disk floor, stopping" | tee -a "$LOG"; break; fi
  RAW=out/episodes_v2_chunk$N
  python sim/collect.py --out "$RAW" --target-kept 100 --instruction both --workers 3 --seed $((100000*N)) >> "$LOG" 2>&1; rc=$?
  echo "$(date +%T) collect chunk $N rc=$rc" | tee -a "$LOG"
  python -c "import json;m=json.load(open('$RAW/manifest.json'));print({k:v for k,v in m.items() if not isinstance(v,(list,dict))})" >> "$LOG" 2>&1
  /c/pai/ev/Scripts/python.exe sim/to_lerobot.py --raw "$RAW" --repo "$REPO" --root "$ROOT" --append --chunk-id "chunk$N" --push >> "$LOG" 2>&1; crc=$?
  echo "$(date +%T) convert/push chunk $N rc=$crc" | tee -a "$LOG"
  if [ "$crc" -eq 0 ]; then rm -rf "$RAW"; else echo "$(date +%T) keeping raw chunk $N for inspection" | tee -a "$LOG"; break; fi
  if [ "$rc" -eq 2 ]; then echo "$(date +%T) yield below 30%, stopping" | tee -a "$LOG"; break; fi
done
echo "$(date +%T) DONE free $(free_gb) GB" | tee -a "$LOG"
