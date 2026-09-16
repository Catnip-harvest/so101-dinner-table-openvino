"""Assemble the lablab demo video from real assets: scripted relay footage, the Core Ultra
policy-eval reel, and title cards carrying the measured numbers. 1280x720, 30 fps."""
import os, subprocess, textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()
ROOT = Path(r"C:\Users\vieth\Documents\lablab hackathon")
WORK = Path(r"D:\demo"); WORK.mkdir(exist_ok=True)
W, H, FPS = 1280, 720, 30
BG = (13, 17, 28); FG = (233, 237, 243); TEAL = (45, 212, 191); DIM = (140, 150, 165)
FONT = r"C:\Windows\Fonts\arialbd.ttf"; FONTR = r"C:\Windows\Fonts\arial.ttf"

def font(sz, bold=True):
    return ImageFont.truetype(FONT if bold else FONTR, sz)

def card(path, lines, seconds):
    """lines: list of (text, size, color, bold)."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 6], fill=TEAL)
    total = sum(sz + 18 for _, sz, _, _ in lines)
    y = (H - total) // 2
    for text, sz, color, bold in lines:
        f = font(sz, bold)
        w = d.textlength(text, font=f)
        d.text(((W - w) / 2, y), text, font=f, fill=color)
        y += sz + 18
    png = WORK / (path + ".png"); img.save(png)
    mp4 = WORK / (path + ".mp4")
    subprocess.run([FF, "-y", "-loop", "1", "-i", str(png), "-t", str(seconds),
                    "-r", str(FPS), "-vf", f"scale={W}:{H},format=yuv420p",
                    "-c:v", "libx264", "-preset", "medium", str(mp4)],
                   check=True, capture_output=True)
    return mp4

def relay_segment(name, ep, seconds=9, speed=1.6):
    """Top view large with the two wrist cameras as corner insets."""
    top = ROOT / f"out/episodes_v2_chunk2/episode_{ep}_top.mp4"
    lw = ROOT / f"out/episodes_v2_chunk2/episode_{ep}_left_wrist.mp4"
    rw = ROOT / f"out/episodes_v2_chunk2/episode_{ep}_right_wrist.mp4"
    mp4 = WORK / (name + ".mp4")
    fc = (
        f"[0:v]setpts=PTS/{speed},scale={W}:{H},format=yuv420p[base];"
        f"[1:v]setpts=PTS/{speed},scale=300:-1[lw];"
        f"[2:v]setpts=PTS/{speed},scale=300:-1[rw];"
        f"[base][lw]overlay=24:24[t1];"
        f"[t1][rw]overlay={W}-324:24[outv]"
    )
    subprocess.run([FF, "-y", "-i", str(top), "-i", str(lw), "-i", str(rw),
                    "-filter_complex", fc, "-map", "[outv]", "-t", str(seconds),
                    "-r", str(FPS), "-c:v", "libx264", "-preset", "medium", str(mp4)],
                   check=True, capture_output=True)
    return mp4

def norm(src, name, seconds=None, speed=1.0, label=None):
    """Normalize any clip to 1280x720/30fps, optional speed + caption bar."""
    mp4 = WORK / (name + ".mp4")
    vf = f"setpts=PTS/{speed},scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=0x0D111C,format=yuv420p"
    if label:
        vf += f",drawbox=y=ih-56:w=iw:h=56:color=0x0D111C@0.85:t=fill,drawtext=fontfile='C\\:/Windows/Fonts/arialbd.ttf':text='{label}':fontcolor=white:fontsize=26:x=(w-text_w)/2:y=h-44"
    cmd = [FF, "-y", "-i", str(src), "-vf", vf, "-r", str(FPS), "-an"]
    if seconds:
        cmd += ["-t", str(seconds)]
    cmd += ["-c:v", "libx264", "-preset", "medium", str(mp4)]
    subprocess.run(cmd, check=True, capture_output=True)
    return mp4

segs = []
segs.append(card("01_title", [
    ("DINNER-TABLE RELAY", 64, FG, True),
    ("Bimanual SO-101  ·  SmolVLA  ·  Intel Core Ultra", 30, TEAL, False),
    ("", 10, FG, False),
    ("A full sim-to-silicon pipeline, measured end to end", 24, DIM, False),
], 3.5))

segs.append(card("02_data", [
    ("STEP 1 — DATA GENERATION", 40, TEAL, True),
    ("", 8, FG, False),
    ("Physics grasps in MuJoCo: real jaw contact,", 28, FG, False),
    ("not a kinematic cheat", 28, FG, False),
], 3.0))
segs.append(relay_segment("03_relay_a", "000000", seconds=9, speed=1.6))
segs.append(relay_segment("04_relay_b", "000002", seconds=8, speed=1.6))

segs.append(card("05_dataset", [
    ("THE DATASET", 40, TEAL, True),
    ("", 8, FG, False),
    ("400 success-only episodes  ·  323,442 frames", 30, FG, False),
    ("749 attempts  ·  53.4% yield  ·  every failure counted", 26, DIM, False),
    ("3 cameras  ·  12-D bimanual state/action  ·  2 instructions", 26, DIM, False),
], 4.0))

segs.append(card("06_intel", [
    ("STEP 2 — INTEL CORE ULTRA", 40, TEAL, True),
    ("", 8, FG, False),
    ("SmolVLA  ->  OpenVINO FP16 / INT8", 30, FG, False),
    ("Panther Lake: Core Ultra X7 358H + Arc B390 + AI Boost", 24, DIM, False),
], 3.5))
segs.append(card("07_intel_nums", [
    ("MEASURED ON THE REQUIRED SILICON", 34, TEAL, True),
    ("", 10, FG, False),
    ("Arc B390 iGPU:  124.6 ms  per action chunk", 34, FG, True),
    ("16x faster than CPU   ·   8x real-time at 50 Hz", 28, FG, False),
    ("INT8 halves the model to 409 MB   ·   NPU: graph too large", 24, DIM, False),
], 4.5))

segs.append(card("08_policy", [
    ("STEP 3 — THE LEARNED POLICY", 40, TEAL, True),
    ("", 8, FG, False),
    ("Closed-loop evaluation on Core Ultra", 28, FG, False),
    ("honest, not cherry-picked", 26, DIM, False),
], 3.0))
segs.append(norm(ROOT / "out/tiber/out/eval_fp16_best/tiled_A.mp4", "09_eval",
                 seconds=12, speed=2.0, label="SmolVLA policy — 10 seeds on Core Ultra"))

segs.append(card("10_results", [
    ("THE HONEST RESULT", 40, TEAL, True),
    ("", 10, FG, False),
    ("0 / 20 task success  —  run 1 (1 epoch) and run 3 (2.5)", 30, FG, True),
    ("The arms carry the cup, stall ~10 cm short of placement", 26, FG, False),
    ("Diagnosed: structural to the imitation budget,", 24, DIM, False),
    ("not just undertraining  ·  every number has a file", 24, DIM, False),
], 5.0))
segs.append(card("11_end", [
    ("github.com/Catnip-harvest", 30, DIM, False),
    ("so101-dinner-table-openvino", 40, FG, True),
    ("", 12, FG, False),
    ("Data generation works. Deployment works.", 26, TEAL, False),
    ("The failure is measured, not hidden.", 26, TEAL, False),
], 4.5))

# concat
lst = WORK / "concat.txt"
lst.write_text("".join(f"file '{s.as_posix()}'\n" for s in segs))
out = ROOT / "out" / "lablab_demo.mp4"
subprocess.run([FF, "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                "-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(out)], check=True, capture_output=True)
print("segments:", len(segs))
print("wrote", out, round(out.stat().st_size / 1e6, 1), "MB")
