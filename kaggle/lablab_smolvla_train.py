# ============================================================================
#  SmolVLA on Kaggle - AUTOPILOT.  Paste in ONE cell, Save Version > Save & Run All, leave.
#  Phase 1 A/B tests fp32 vs fp16-autocast on a public dataset and keeps the winner.
#  Phase 2 waits for your dataset to appear on the Hub (bounded, to spare GPU quota).
#  Phase 3 measures s/step on the REAL schema and picks --steps to fit the time budget.
#  Phase 4 trains, uploading a checkpoint to HF every 20 min.
#  Re-running the notebook later RESUMES from the last HF checkpoint.
#  Progress is pushed to <POLICY_REPO>/STATUS.md - readable from a phone, no Kaggle needed.
#  Settings: Accelerator GPU T4 x2 | Internet ON | Add-ons > Secrets > HF_TOKEN (write)
# ============================================================================
import os, re, sys, glob, json, time, shutil, threading, subprocess, importlib

DATASET      = "VietHwang/dinner-table-v2"        # physics relay dataset; v1=24 kinematic kept separate
PROBE        = "VietHwang/dinner-table-probe"     # 5-episode probe, if it lands first
POLICY_REPO  = "VietHwang/smolvla-dinner-table-run2"
BUDGET_H     = 9.5     # Kaggle cuts the session at 12 h; leave headroom
RESERVE_H    = 0.6      # for final upload + load check
WAIT_H       = 8.0      # real run: start this notebook AT THE SAME TIME as the collection
                        # notebook and let it idle here until the dataset lands on the Hub.
ACCEPT_PROBE = False    # True only when you want a timing probe; otherwise the 5-episode
                        # probe repo would be grabbed first and burn the session on 100 steps.
RUN_AB       = False    # settled 13 Sep: fp32 2.352 s/step (two methods agreed); fp16
                        # exited 1 during import. Phase 3 re-measures on the real schema,
                        # so skipping this costs no accuracy and saves ~19 min of quota.
BATCH        = 8        # the smoke test used 3.6/15 GB at batch 8
STEP_FLOOR, STEP_CEIL = 4000, 20000
WORK, RESUME_DIR = "/kaggle/working/train/smolvla", "/kaggle/working/resume"
LOGDIR = "/kaggle/working/logs"   # /kaggle/tmp is NOT saved in the notebook output
T0 = time.time()
# our camera names -> what smolvla_base declares (camera1/2/3; a subset is allowed)
RENAME = {"observation.images.top": "observation.images.camera1",
          "observation.images.left_wrist": "observation.images.camera2",
          "observation.images.right_wrist": "observation.images.camera3"}

os.makedirs(LOGDIR, exist_ok=True)

def budget_steps(elapsed_h, seconds_per_step, budget_h=BUDGET_H,
                 reserve_h=RESERVE_H, floor=STEP_FLOOR, ceil=STEP_CEIL):
    """Return a 100-step-aligned budget that cannot cross the wall-time reserve."""
    usable_s = max(budget_h - elapsed_h - reserve_h, 0.0) * 3600.0
    capacity = int(usable_s / max(seconds_per_step, 1e-6))
    aligned = max(0, capacity // 100 * 100)
    return min(aligned, ceil) if aligned < floor else min(aligned, ceil)

def prune_keep_count(free_space_gb, normal_keep=2):
    """Pure checkpoint retention decision, split out for offline testing."""
    return 1 if free_space_gb < 5.0 else normal_keep

def write_status_file(path, state):
    body = "# SmolVLA autopilot\n\n" + "\n".join(
        "- **%s**: %s" % (key, value) for key, value in state.items()
    )
    with open(path, "w") as handle:
        handle.write(body)
    return body

def checkpoint_complete(path):
    """Only upload checkpoints after both model and resumable optimizer state exist."""
    return bool(path and
                os.path.isfile(os.path.join(path, "pretrained_model", "train_config.json")) and
                glob.glob(os.path.join(path, "pretrained_model", "*.safetensors")) and
                os.path.isfile(os.path.join(path, "training_state", "training_step.json")))

def run_selftest():
    import tempfile
    assert budget_steps(10.0, 2.0) == 700
    assert budget_steps(0.0, 1.0) == STEP_CEIL
    assert prune_keep_count(4.9) == 1 and prune_keep_count(5.0) == 2
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "STATUS.md")
        body = write_status_file(path, {"phase": "selftest", "steps": 700})
        assert open(path).read() == body and "selftest" in body
    print("SELFTEST PASS: step budgeting, checkpoint pruning, status writing")

if __name__ == "__main__" and "--selftest" in sys.argv:
    run_selftest()
    raise SystemExit(0)

def hrs():
    return (time.time() - T0) / 3600

def log(*a):
    line = "[%5.2fh] " % hrs() + " ".join(str(x) for x in a)
    print(line, flush=True)
    with open(LOGDIR + "/autopilot.log", "a") as f:
        f.write(line + "\n")

# ---------------------------------------------------------------- phase 0: env
# Every HF_* variable must be set BEFORE huggingface_hub is imported - it resolves the cache
# path once, at import time. Setting HF_HOME afterwards left the PARENT reading ~/.cache while
# the CHILD processes, which inherit the environment, read /kaggle/tmp/hf, so the prefetch warmed
# a cache nothing else read. CUDA_VISIBLE_DEVICES goes above torch for the same reason.
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
os.environ["HF_HOME"] = "/kaggle/tmp/hf"
os.makedirs("/kaggle/tmp/hf", exist_ok=True)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # Kaggle only offers T4 x2; lerobot-train is single-process

r = subprocess.run(sys.executable + ' -m pip -q install "lerobot[smolvla,dataset]==0.6.1" hf_transfer',
                   shell=True, text=True)
log("pip exit", r.returncode)
if r.returncode != 0:
    raise SystemExit("dependency installation failed; refusing to start an unattended run")
importlib.invalidate_caches()
import torch, lerobot, av
import lerobot.utils.import_utils as _iu
_iu._require_package_cache.clear()   # a failed earlier import poisons this cache for the whole kernel
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from huggingface_hub import HfApi, snapshot_download

TRAIN = "lerobot-train" if shutil.which("lerobot-train") else sys.executable + " -m lerobot.scripts.lerobot_train"

# lerobot 0.6.1 + T4: --policy.use_amp=true autocasts with torch.get_autocast_dtype("cuda"),
# which is hardcoded bfloat16 and dies at step 0 on Turing. No CLI flag reaches fp16, so we
# set the global autocast dtype inside the CHILD process before lerobot's main runs.
SHIM = "/kaggle/tmp/ampshim/run_fp16.py"
os.makedirs(os.path.dirname(SHIM), exist_ok=True)
with open(SHIM, "w") as _f:
    _f.write(
        "import sys, runpy, torch\n"
        "for _name, _args in ((\"set_autocast_dtype\", (\"cuda\", torch.float16)),\n"
        "                     (\"set_autocast_gpu_dtype\", (torch.float16,))):\n"
        "    _fn = getattr(torch, _name, None)\n"
        "    if _fn is None:\n"
        "        continue\n"
        "    try:\n"
        "        _fn(*_args)\n"
        "        print('[ampshim]', _name, '-> float16 | now',\n"
        "              torch.get_autocast_dtype('cuda'), flush=True)\n"
        "        break\n"
        "    except Exception as _e:\n"
        "        print('[ampshim]', _name, 'failed:', _e, flush=True)\n"
        "else:\n"
        "    print('[ampshim] NO SETTER FOUND - autocast stays bfloat16, expect failure on T4', flush=True)\n"
        "sys.argv = ['lerobot-train'] + sys.argv[1:]\n"
        "runpy.run_module('lerobot.scripts.lerobot_train', run_name='__main__')\n")
AMPTRAIN = sys.executable + " " + SHIM

_h = subprocess.run(TRAIN + " --help", shell=True, text=True, capture_output=True)
HELP = _h.stdout + _h.stderr

def has(flag):
    return flag in HELP

api = HfApi()
api.create_repo(POLICY_REPO, private=True, exist_ok=True)
api.update_repo_settings(POLICY_REPO, private=True)
if not api.model_info(POLICY_REPO).private:
    raise SystemExit("policy repository is not private; refusing to train or upload")
log("torch", torch.__version__, "| lerobot", getattr(lerobot, "__version__", "?"),
    "| av", av.__version__, "|", torch.cuda.get_device_name(0))

STATE = {"phase": "init", "gpu": torch.cuda.get_device_name(0)}

def status(**kw):
    STATE.update(kw)
    STATE["elapsed_h"] = round(hrs(), 2)
    STATE["utc"] = time.strftime("%F %T", time.gmtime())
    write_status_file("/kaggle/tmp/STATUS.md", STATE)
    try:
        api.upload_file(path_or_fileobj="/kaggle/tmp/STATUS.md", path_in_repo="STATUS.md", repo_id=POLICY_REPO)
    except Exception as e:
        log("status push failed:", repr(e)[:110])

def run_train(cmd, tag):
    """Stream lerobot-train, tee to a log, harvest per-step timing and loss."""
    updt, losses = [], []
    path = LOGDIR + "/" + tag + ".log"
    log("$", cmd)
    with open(path, "w") as f:
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1)
        for line in p.stdout:
            f.write(line)
            f.flush()
            m = re.search(r"updt_s[:=]\s*([\d.]+)", line)
            if m:
                updt.append(float(m.group(1)))
            m = re.search(r"\bloss[:=]\s*(nan|inf|-?[\d.]+(?:[eE][+-]?\d+)?)", line)
            if m:
                losses.append(m.group(1))
        p.wait()
    if p.returncode != 0:                      # otherwise the traceback dies with the child
        with open(path) as _f:
            log("FAILED", tag, "exit", p.returncode, "- tail of", path)
            print(_f.read()[-4000:], flush=True)   # straight to the notebook Logs tab
    return p.returncode, updt, losses

def s_per_step(updt, wall, steps):
    tail = updt[-5:] if len(updt) >= 5 else updt
    if tail:
        return sorted(tail)[len(tail) // 2]
    return max(wall - 90.0, 1.0) / steps          # ~90 s of import + checkpoint download

def diverged(losses):
    return any(x in ("nan", "inf") for x in losses[-20:])

def cmd_for(repo, steps, save_freq, amp, out, job, rename=None, log_freq=10):
    base = AMPTRAIN if amp else TRAIN     # the fp16 path goes through the autocast shim
    c = (base + " --policy.path=lerobot/smolvla_base --dataset.repo_id=" + repo +
         " --policy.device=cuda --policy.push_to_hub=false" +          # validate() throws without this
         " --batch_size=%d --steps=%d --save_freq=%d" % (BATCH, steps, save_freq) +
         " --output_dir=" + out + " --job_name=" + job + " --wandb.enable=false")
    # default log_freq is 200, so a 120-step measurement prints no updt_s at all and
    # s_per_step() silently falls back to wall-clock, which includes startup. Log often.
    if has("log_freq"):
        c += " --log_freq=%d" % log_freq
    if amp:
        c += " --policy.use_amp=true"      # T4 fp16 tensor cores ~65 TFLOPS vs ~8 fp32
        # NB: the flag alone is not enough on 0.6.1 — the shim above supplies the fp16 dtype
    if rename:
        c += " '--rename_map=" + json.dumps(rename) + "'"
    return c

# ------------------------------------------- resume immediately if a run exists
if any(f.endswith("train_config.json") for f in api.list_repo_files(POLICY_REPO)):
    snapshot_download(POLICY_REPO, local_dir=RESUME_DIR)
    cfgs = sorted(glob.glob(RESUME_DIR + "/**/train_config.json", recursive=True), key=os.path.getmtime)
    if cfgs:
        status(phase="resuming", config=cfgs[-1])
        log("RESUMING from", cfgs[-1])
        with open(cfgs[-1]) as config_handle:
            resume_amp = bool(json.load(config_handle).get("policy", {}).get("use_amp"))
        resume_train = AMPTRAIN if resume_amp else TRAIN
        rc, updt, losses = run_train(resume_train + " --config_path=" + cfgs[-1] + " --resume=true", "resume")
        completed = [path for path in glob.glob(WORK + "/checkpoints/*")
                     if os.path.isdir(path) and not os.path.islink(path) and checkpoint_complete(path)]
        completed.sort(key=os.path.getmtime)
        if rc == 0 and completed:
            try:
                api.upload_folder(folder_path=completed[-1], repo_id=POLICY_REPO,
                                  path_in_repo="checkpoints/last")
                log("resumed run checkpoint pushed:", completed[-1])
            except Exception as e:
                log("resumed run final upload failed:", repr(e)[:160])
                rc = 4
        elif rc == 0:
            log("resumed trainer exited zero but produced no complete checkpoint")
            rc = 5
        status(phase="resume finished", exit_code=rc, last_loss=losses[-1] if losses else None)
        sys.exit(rc)

# ------------------------------------------------ phase 1: precision A/B (~18 min)
# Warm the cache first: whichever arm runs second would otherwise skip the base-checkpoint
# download and look faster for a reason that has nothing to do with precision.
try:
    snapshot_download("lerobot/smolvla_base")
    log("smolvla_base cached - both A/B arms now start warm")
except Exception as e:
    log("prefetch failed (A/B may be biased toward the second arm):", repr(e)[:110])

SMOKE, smoke_rename = None, None
for cand in ([] if not RUN_AB else
             ["lerobot/svla_so101_pickplace", "lerobot/svla_so100_pickplace",
              "lerobot/aloha_sim_transfer_cube_human", "lerobot/pusht"]):
    try:
        d = LeRobotDataset(cand)
        SMOKE = cand
        keys = list(d.meta.camera_keys)
        smoke_rename = {}
        for i, k in enumerate(keys[:3]):
            want = "observation.images.camera%d" % (i + 1)
            if k != want:
                smoke_rename[k] = want
        smoke_rename = smoke_rename or None
        log("smoke dataset", cand, "|", d.num_episodes, "eps |", keys)
        break
    except Exception as e:
        log("skip", cand, repr(e)[:80])

best_amp, ab = False, {}
if SMOKE:
    for amp in (False, True):
        status(phase="A/B precision amp=%s" % amp)
        t = time.time()
        rc, updt, losses = run_train(
            cmd_for(SMOKE, 120, 999999, amp, "/kaggle/working/ab_%d" % int(amp),
                    "ab%d" % int(amp), smoke_rename), "ab_amp%d" % int(amp))
        sps = s_per_step(updt, time.time() - t, 120)
        ab["amp=%s" % amp] = {"s_per_step": round(sps, 3), "exit": rc, "diverged": diverged(losses),
                              "last_loss": losses[-1] if losses else None}
        log("amp=%s: %.2f s/step, exit %d, diverged %s, loss %s"
            % (amp, sps, rc, diverged(losses), losses[-1] if losses else "?"))
    f32, f16 = ab.get("amp=False", {}), ab.get("amp=True", {})
    if (f16 and f32 and f16.get("exit") == 0 and not f16.get("diverged")
            and f16["s_per_step"] < f32["s_per_step"] * 0.95):
        best_amp = True
    log("PRECISION WINNER:", "fp16 autocast" if best_amp else "fp32", json.dumps(ab))
status(phase="A/B done", ab=json.dumps(ab), amp=best_amp)

# ------------------------------------------------ phase 2: wait for the dataset
target = None
deadline = time.time() + WAIT_H * 3600
while time.time() < deadline:
    for rid in ((DATASET, PROBE) if ACCEPT_PROBE else (DATASET,)):
        try:
            api.dataset_info(rid)
            target = rid
            break
        except Exception:
            pass
    if target:
        break
    status(phase="waiting for %s (or %s)" % (DATASET, PROBE),
           wait_left_h=round((deadline - time.time()) / 3600, 2))
    time.sleep(max(15.0, min(300.0, deadline - time.time())))   # never overshoot the deadline

if not target:
    status(phase="STOPPED: no dataset appeared",
           note="re-run the notebook once the dataset is on the Hub; A/B result above still stands")
    log("no dataset within", WAIT_H, "h - exiting to spare GPU quota")
    sys.exit(6)

log("dataset found:", target)
ds = LeRobotDataset(target)
cams = list(ds.meta.camera_keys)
log("schema:", ds.num_episodes, "eps |", ds.num_frames, "frames | cameras", cams)
status(phase="dataset found", dataset=target, episodes=ds.num_episodes,
       frames=ds.num_frames, cameras=str(cams))
missing_cams = sorted(set(RENAME) - set(cams))
feature_shapes = {
    key: tuple(ds.meta.features.get(key, {}).get("shape", ()))
    for key in ("observation.state", "action")
}
if missing_cams or feature_shapes != {"observation.state": (12,), "action": (12,)}:
    status(phase="FAILED: incompatible dataset schema", missing_cameras=str(missing_cams),
           feature_shapes=str(feature_shapes))
    log("incompatible schema | missing cameras", missing_cams, "| shapes", feature_shapes)
    sys.exit(7)
rename = {k: v for k, v in RENAME.items() if k in cams} or None

# ------------------------------- phase 3: measure on the REAL schema, size the run
status(phase="measuring on real schema")
t = time.time()
rc, updt, losses = run_train(
    cmd_for(target, 100, 999999, best_amp, "/kaggle/working/probe", "probe", rename), "probe")
if rc != 0:
    status(phase="FAILED: probe run on the real dataset", exit_code=rc, log=LOGDIR + "/probe.log")
    log("probe failed - read the log before spending the budget")
    sys.exit(1)

sps = s_per_step(updt, time.time() - t, 100)
left = max(BUDGET_H - hrs() - RESERVE_H, 0.0) * 3600
steps = budget_steps(hrs(), sps)
if steps < STEP_FLOOR:
    status(phase="STOPPED: insufficient safe training window", affordable_steps=steps,
           required_floor=STEP_FLOOR)
    log("only", steps, "steps fit before the reserve; refusing an unfinishable run")
    sys.exit(2)
log("real schema: %.2f s/step -> %d steps in %.1f h (budget left %.1f h)"
    % (sps, steps, steps * sps / 3600, left / 3600))
status(phase="sized", s_per_step=round(sps, 3), steps=steps,
       projected_h=round(steps * sps / 3600, 2))

if target == PROBE:
    status(phase="STOPPED: probe only",
           note="%.2f s/step measured on the real schema; re-run when %s is up "
                "and it will train for %d steps" % (sps, DATASET, steps))
    log("probe dataset only - timing captured, not training on 5 episodes")
    sys.exit(0)

# Timing runs leave final smoke/probe checkpoints even with a huge save_freq. They are no
# longer useful once sizing succeeds and otherwise consume several GB of the 20 GB work disk.
for scratch in ("/kaggle/working/probe", "/kaggle/working/ab_0", "/kaggle/working/ab_1"):
    if os.path.isdir(scratch):
        shutil.rmtree(scratch)
        log("removed completed timing scratch", scratch, "| free %.1f GB" %
            (shutil.disk_usage("/kaggle/working").free / 1e9))

# ------------------------------------------------ phase 4: the real run
def ckpts():
    return [p for p in sorted(glob.glob(WORK + "/checkpoints/*"), key=os.path.getmtime)
            if os.path.isdir(p) and not os.path.islink(p)]

def newest():
    c = [path for path in ckpts() if checkpoint_complete(path)]
    return c[-1] if c else None

def free_gb():
    return shutil.disk_usage("/kaggle/working").free / 1e9

def prune(keep=2):
    """Each SmolVLA checkpoint is ~1.3 GB and /kaggle/working caps at 20 GB. Uploading
    without deleting fills the disk around step 15k - the exact failure that killed the
    7 Sep AAMAS run. Keep the newest `keep` (the newest may still be being written)."""
    keep = prune_keep_count(free_gb(), keep)
    for old in ckpts()[:-keep] if keep else ckpts():
        try:
            shutil.rmtree(old)
            log("pruned", os.path.basename(old), "| free %.1f GB" % free_gb())
        except Exception as e:
            log("prune failed:", repr(e)[:100])

def uploader():
    while True:
        time.sleep(1200)
        c = newest()
        if c:
            try:
                api.upload_folder(folder_path=c, repo_id=POLICY_REPO, path_in_repo="checkpoints/last")
                status(phase="training", last_upload=os.path.basename(c), free_gb=round(free_gb(), 1))
                log("uploaded", c, "| free %.1f GB" % free_gb())
                prune()          # only after the upload succeeds - HF is the durable copy
            except Exception as e:
                log("upload failed:", repr(e)[:120])
                if free_gb() < 3:      # upload broken AND disk nearly gone: save the run
                    log("disk critical with a failed upload - pruning anyway")
                    prune(keep=1)

threading.Thread(target=uploader, daemon=True).start()
status(phase="training", steps=steps, amp=best_amp)
save_freq = max(100, int(1200.0 / sps) // 100 * 100)
rc, updt, losses = run_train(
    cmd_for(target, steps, save_freq, best_amp, WORK, "smolvla_dinner", rename), "train")
log("training exit", rc, "| last loss", losses[-1] if losses else "?")

# ------------------------------------------------ phase 5: final push + proof
c = newest()
final_upload_ok = False
if c:
    try:
        api.upload_folder(folder_path=c, repo_id=POLICY_REPO, path_in_repo="checkpoints/last")
        log("final checkpoint pushed:", c)
        final_upload_ok = True
    except Exception as e:
        log("final upload failed:", repr(e)[:120])

loaded = None
try:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    pm = glob.glob(c + "/**/pretrained_model", recursive=True) if c else []
    p = SmolVLAPolicy.from_pretrained(pm[0] if pm else POLICY_REPO)
    loaded = round(sum(x.numel() for x in p.parameters()) / 1e6, 1)
    log("policy loads:", loaded, "M params")
except Exception as e:
    log("load check failed:", repr(e)[:160])

done = rc == 0 and loaded is not None and c is not None and final_upload_ok
status(phase="DONE" if done else "FAILED: training or final proof incomplete", exit_code=rc,
       last_loss=losses[-1] if losses else None, params_M=loaded,
       next="download checkpoints/last, export to OpenVINO, run the Intel benchmark")
if not done:
    raise SystemExit(rc or 3)

if __name__ == "__main__" and "--selftest" in sys.argv:
    run_selftest()
