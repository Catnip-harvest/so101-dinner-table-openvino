"""Render the lablab slide deck (PDF) and cover image (PNG) from the same measured numbers
and real footage as the demo video. 1280x720. Output: out/lablab_slides.pdf, out/lablab_cover.png."""
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()
ROOT = Path(r"C:\Users\vieth\Documents\lablab hackathon")
WORK = Path(r"D:\demo\slides"); WORK.mkdir(parents=True, exist_ok=True)
W, H = 1280, 720
BG = (13, 17, 28); FG = (233, 237, 243); TEAL = (45, 212, 191); DIM = (150, 160, 175)
BOLD = r"C:\Windows\Fonts\arialbd.ttf"; REG = r"C:\Windows\Fonts\arial.ttf"

def font(sz, bold=False):
    return ImageFont.truetype(BOLD if bold else REG, sz)

def frame(video, t, name):
    png = WORK / f"{name}.png"
    subprocess.run([FF, "-y", "-ss", str(t), "-i", str(video), "-frames:v", "1", str(png)],
                   check=True, capture_output=True)
    return Image.open(png).convert("RGB")

def base(title, kicker=None):
    img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 6], fill=TEAL)
    if kicker:
        d.text((64, 44), kicker, font=font(20, True), fill=TEAL)
    d.text((64, 72 if kicker else 52), title, font=font(42, True), fill=FG)
    d.text((64, H - 40), "github.com/Catnip-harvest/so101-dinner-table-openvino", font=font(16), fill=DIM)
    return img, d

def body(d, lines, x=64, y=160, gap=14):
    """lines: (text, size, color, bold)"""
    for text, sz, color, bold in lines:
        d.text((x, y), text, font=font(sz, bold), fill=color)
        y += sz + gap

def put_image(img, pic, box):
    x0, y0, x1, y1 = box
    k = min((x1 - x0) / pic.width, (y1 - y0) / pic.height)
    pic = pic.resize((int(pic.width * k), int(pic.height * k)), Image.LANCZOS)
    img.paste(pic, (x0 + (x1 - x0 - pic.width) // 2, y0 + (y1 - y0 - pic.height) // 2))
    ImageDraw.Draw(img).rectangle([x0 - 1, y0 - 1, x1, y1], outline=(40, 48, 66), width=2)

slides = []
top0 = frame(ROOT / "out/episodes_v2_chunk2/episode_000000_top.mp4", 7, "relay_top")
top1 = frame(ROOT / "out/episodes_v2_chunk2/episode_000002_top.mp4", 12, "relay_top2")
lw = frame(ROOT / "out/episodes_v2_chunk2/episode_000000_left_wrist.mp4", 12, "relay_lw")
evalf = frame(ROOT / "out/tiber/out/eval_fp16_best/tiled_A.mp4", 15, "eval_tile")

# 1 - cover / title
img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
d.rectangle([0, 0, W, 6], fill=TEAL)
put_image(img, top0, (700, 150, 1216, 570))
body(d, [("INTEL PHYSICAL AI CHALLENGE  ·  SIM TRACK", 20, TEAL, True),
         ("", 20, FG, False),
         ("Dinner-Table", 66, FG, True), ("Relay", 66, FG, True),
         ("", 10, FG, False),
         ("Two SO-101 arms, one cup, SmolVLA,", 26, FG, False),
         ("measured on Intel Core Ultra", 26, FG, False),
         ("", 10, FG, False),
         ("A full sim-to-silicon pipeline -", 20, DIM, False),
         ("with the failures measured, not hidden", 20, DIM, False)], y=150)
img.save(ROOT / "out/lablab_cover.png"); slides.append(img)

# 2 - task
img, d = base("The task", "01")
put_image(img, top1, (700, 150, 1216, 570))
body(d, [("Two SO-101 arms face each other across a table.", 26, FG, False),
         ("", 8, FG, False),
         ("Instruction A", 22, TEAL, True),
         ("\"bring the cup across the table and set it on the plate\"", 22, FG, False),
         ("Right arm grasps, lifts, sets mid-table;", 20, DIM, False),
         ("left arm picks up and places on the plate.", 20, DIM, False),
         ("", 8, FG, False),
         ("Instruction B", 22, TEAL, True),
         ("\"slide the cup to the middle of the table\"", 22, FG, False),
         ("The single-arm half of the same motion.", 20, DIM, False),
         ("", 8, FG, False),
         ("A language-conditioned policy must pick the right one.", 22, FG, True)])
slides.append(img)

# 3 - physics-honest data generation
img, d = base("Data generation: real grasps, not a kinematic cheat", "02")
put_image(img, lw, (760, 160, 1216, 560))
body(d, [("MuJoCo 3  ·  Menagerie SO-101  ·  200 Hz physics / 50 Hz control", 22, FG, False),
         ("", 8, FG, False),
         ("What we found", 22, TEAL, True),
         ("Our early scripted demos never grasped - the runner", 22, FG, False),
         ("carried objects by rewriting their pose.", 22, FG, False),
         ("", 8, FG, False),
         ("What we did", 22, TEAL, True),
         ("Split jaw collision hulls, a measured grasp window,", 22, FG, False),
         ("and a holding() check requiring both jaws in contact.", 22, FG, False),
         ("", 8, FG, False),
         ("Scripted controller gate: 7/10 and 14/20 across seeds,", 22, FG, True),
         ("no predicate loosened.", 22, FG, True)])
slides.append(img)

# 4 - dataset
img, d = base("The dataset", "03")
body(d, [("400 success-only episodes  ·  323,442 frames at 50 fps", 34, FG, True),
         ("", 10, FG, False),
         ("749 attempts  ->  53.4 % yield.  Every failure discarded and counted.", 24, FG, False),
         ("Fault taxonomy: 831 events over 194 distinct strings, published.", 24, FG, False),
         ("", 14, FG, False),
         ("3 cameras   top · left wrist · right wrist   256x256", 24, DIM, False),
         ("12-D state and action   right arm x6, left arm x6", 24, DIM, False),
         ("2 task strings   relay and single-arm slide", 24, DIM, False),
         ("DART noise   sigma = 0.02 rad on 30 % of episodes, clean labels", 24, DIM, False),
         ("", 14, FG, False),
         ("LeRobot v3 format  ·  VietHwang/dinner-table-v2 (private, available to judges)", 20, TEAL, False)])
slides.append(img)

# 5 - training
img, d = base("Training: SmolVLA fine-tune", "04")
body(d, [("lerobot/smolvla_base (450 M)  ->  LeRobot 0.6.1", 26, FG, False)])
d.text((64, 250), "Run 1", font=font(26, True), fill=TEAL)
d.text((64, 290), "Tesla T4 (Kaggle)  ·  10,300 steps x batch 8  ·  fp32  ·  10.4 h", font=font(22), fill=FG)
d.text((64, 322), "= 1.006 epochs.  Final loss 0.028.", font=font(22), fill=DIM)
d.text((64, 400), "Run 3", font=font(26, True), fill=TEAL)
d.text((64, 440), "RTX 5090 (Vast.ai)  ·  12,593 steps x batch 64  ·  bf16  ·  1.8 h", font=font(22), fill=FG)
d.text((64, 472), "= 2.5 epochs.  Final loss 0.015.", font=font(22), fill=DIM)
d.text((64, 560), "The loss flattered both: half the action dims sit idle in single-arm episodes", font=font(20), fill=DIM)
d.text((64, 588), "and dominate the mean. Low imitation loss is not a learned task.", font=font(20), fill=DIM)
slides.append(img)

# 6 - OpenVINO + Core Ultra
img, d = base("OpenVINO on the required silicon", "05")
body(d, [("FP32 1,586 MB  ->  FP16 800 MB  ->  INT8 409 MB (NNCF, 300/300 layers)", 24, FG, False),
         ("Full flow-matching loop - prefix, KV cache, 10 Euler steps - inside the IR.", 22, DIM, False)], y=150)
tx, ty = 64, 270
cols = [(tx, "Device"), (tx + 520, "FP16"), (tx + 760, "INT8")]
for x, h in cols:
    d.text((x, ty), h, font=font(22, True), fill=TEAL)
rows = [("Core Ultra X7 358H  (CPU)", "1,991.5 ms", "2,696.7 ms", FG),
        ("Arc B390 iGPU", "124.6 ms", "119.7 ms", TEAL),
        ("AI Boost NPU", "compile failed", "compile failed", DIM)]
for i, (a, b, c, col) in enumerate(rows):
    y = ty + 44 + i * 44
    hl = col is TEAL
    d.text((cols[0][0], y), a, font=font(24, hl), fill=col)
    d.text((cols[1][0], y), b, font=font(24, hl), fill=col)
    d.text((cols[2][0], y), c, font=font(24, hl), fill=col)
d.line([tx, ty + 36, tx + 1000, ty + 36], fill=(40, 48, 66), width=2)
body(d, [("One inference = 50-step chunk = 1 s of motion.  iGPU: 8x real time, 16x faster than CPU.", 22, FG, True),
         ("INT8 is 35 % slower than FP16 on CPU - dequantisation cost exceeds the bandwidth saving.", 20, DIM, False),
         ("NPU rejects a 450 M VLA with dynamic shapes and an in-graph denoising loop (OV 2026.3.1).", 20, DIM, False),
         ("Bare-metal Intel Cloud bm-ptl  ·  NOT_THE_REQUIRED_SILICON: false in the raw JSON.", 20, TEAL, False)], y=460)
slides.append(img)

# 7 - results
img, d = base("Results, stated plainly", "06")
put_image(img, evalf, (720, 150, 1216, 520))
body(d, [("Closed-loop, 20 seed/instruction pairs", 22, TEAL, True),
         ("", 6, FG, False),
         ("Run 1 (1.0 epoch):     0 / 20     best 105 mm", 26, FG, True),
         ("Run 3 (2.5 epochs):    0 / 20     best  96 mm", 26, FG, True),
         ("", 10, FG, False),
         ("Arms carry the cup, stall ~10 cm short of placement.", 22, FG, False),
         ("", 10, FG, False),
         ("Diagnosis, not a guess", 22, TEAL, True),
         ("Replaying training obs through the IR: 0.66-0.85", 22, FG, False),
         ("correlation. The pipeline is correct; a mis-wired", 22, FG, False),
         ("chain correlates near zero. More epochs halved the", 22, FG, False),
         ("loss and changed nothing: the shortfall is structural", 22, FG, False),
         ("to the imitation budget, not undertraining.", 22, FG, False)], y=150)
d.text((720, 540), "SmolVLA FP16 on Arc B390, 10 seeds, Core Ultra", font=font(18), fill=DIM)
slides.append(img)

# 8 - why this entry
img, d = base("Why this entry", "07")
body(d, [("It runs on the required hardware.", 26, TEAL, True),
         ("Device table, three precisions, closed-loop success and latency - all on Core Ultra + Arc.", 22, FG, False),
         ("", 10, FG, False),
         ("The data is physically honest.", 26, TEAL, True),
         ("Real grasps, failures counted, DART recorded.", 22, FG, False),
         ("", 10, FG, False),
         ("Every number has a file behind it.", 26, TEAL, True),
         ("Volume log, benchmark JSONs, eval JSON, videos, replay diagnostic - in evidence/.", 22, FG, False),
         ("", 10, FG, False),
         ("The failure is diagnosed, not described.", 26, TEAL, True),
         ("We would rather submit a measured 0/20 with the cause isolated than an unmeasured claim.", 22, FG, False),
         ("", 16, FG, False),
         ("Limitations: simulation only · fixed cup size · relay via table · NPU unexercised.", 20, DIM, False)])
slides.append(img)

pdf = ROOT / "out/lablab_slides.pdf"
slides[0].save(pdf, save_all=True, append_images=slides[1:], resolution=96)
for i, s in enumerate(slides, 1):
    s.save(WORK / f"slide_{i:02d}.png")
print("cover", ROOT / "out/lablab_cover.png")
print("pdf", pdf, round(pdf.stat().st_size / 1e6, 2), "MB", len(slides), "slides")
