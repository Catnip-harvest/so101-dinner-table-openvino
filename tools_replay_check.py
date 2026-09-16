"""In-distribution replay: does the exported IR reproduce the demonstrations it trained on?

Feeds real training observations (episode 1 of dinner-table-v2, the relay task) through
the FP16 OpenVINO bundle and compares the predicted action chunk with the recorded one.
A healthy chain matches closely; a broken observation pipeline does not.
"""
import json, sys
import numpy as np, pandas as pd, av
from pathlib import Path
sys.argv = ["x"]
import intel_eval as E

BUNDLE = Path("D:/policy_ov/fp16")
stats = json.loads((BUNDLE / "stats.json").read_text(encoding="utf-8"))
pol = E.OpenVINOPolicy(BUNDLE, "CPU", stats)
print("compiled in %.1fs" % pol.compile_s)

df = pd.read_parquet("D:/dsprobe/data/chunk-000/file-000.parquet")
ep = df[df.episode_index == 1]
start = int(df[df.episode_index < 1].shape[0])          # frames before episode 1
state = np.stack(ep["observation.state"].values).astype(np.float32)
action = np.stack(ep["action"].values).astype(np.float32)
print("episode 1: %d frames, video offset %d" % (len(ep), start))

def frames_at(idx_set):
    out = {}
    for cam in E.CAMERAS:
        path = "D:/dsprobe/videos/observation.images.%s/chunk-000/file-000.mp4" % cam
        want = {start + i for i in idx_set}
        got = {}
        with av.open(path) as c:
            for n, f in enumerate(c.decode(video=0)):
                if n in want:
                    got[n - start] = f.to_ndarray(format="rgb24")
                if n > max(want):
                    break
        out[cam] = got
    return out

probe = [0, 200, 400, 600]
fr = frames_at(probe)
INSTR = "bring the cup across the table and set it on the plate"
print("\n  t   pred[:6]                         recorded[:6]                     MAE   corr")
for t in probe:
    frames = {cam: fr[cam][t] for cam in E.CAMERAS}
    pred = pol.predict(frames, state[t], INSTR)          # (chunk, 12)
    n = min(len(pred), len(action) - t)
    p, r = pred[:n], action[t:t + n]
    mae = float(np.abs(p - r).mean())
    corr = float(np.corrcoef(p.ravel(), r.ravel())[0, 1])
    print("%4d  %s  %s  %.4f  %.3f" % (
        t, np.array2string(p[0][:6], precision=2, suppress_small=True),
        np.array2string(r[0][:6], precision=2, suppress_small=True), mae, corr))
    if t == 0:
        print("      predicted chunk spread (std over chunk, per dim):",
              np.array2string(p.std(0), precision=3, suppress_small=True))
        print("      recorded  chunk spread:",
              np.array2string(r.std(0), precision=3, suppress_small=True))
