"""
Export a SmolVLA policy to OpenVINO IR in FP32, FP16 and INT8.  Build-side script:
it needs torch, LeRobot and Intel's physicalai-train.  The IR it produces needs none
of those - see intel_bench.py for the deploy-side bundle.

    pip install "physicalai-train[smolvla]" "lerobot[dataset]==0.5.1" nncf
    python intel_export.py --out policy_ov                      # untrained, shape-correct
    python intel_export.py --out policy_ov --checkpoint ckpt_dir # a real checkpoint

Every flag has a default matching the dinner-table task: 12-D state, 12-D action,
three cameras, 50-step action chunk.  Run with no checkpoint and you get a randomly
initialised policy of exactly the right shape, which is enough to prove the export and
benchmark path; drop a trained checkpoint in later and nothing else changes.

Three things about this pipeline are not obvious and cost a session to find:

1.  `physicalai` 0.1.1 imports `dataset_to_policy_features` from
    `lerobot.datasets.feature_utils`.  LeRobot moved that symbol to
    `lerobot.utils.feature_utils` in 0.6.0, so the import fails on any LeRobot newer
    than 0.5.1 even though the dependency is declared as `lerobot>=0.5.1`.  Pin 0.5.1.

2.  The SmolVLM2 backbone is instantiated in bfloat16, because that is what its HF
    config asks for.  Exported as-is, the IR is a bf16/f32 mix, and OpenVINO's FP16
    compression leaves the bf16 half alone - so "FP16" comes out only 20% smaller
    instead of 50%, and the NPU may refuse the graph.  `--no-cast-fp32` disables the
    cast if you want to see that; leave it on otherwise.

3.  `policy.export(..., backend="openvino")` hardcodes `compress_to_fp16=False` in
    SmolVLA.extra_export_args, and export_kwargs cannot reach it.  FP16 is therefore a
    re-save of the FP32 IR, not an export flag.  INT8 is not in physicalai-train at all;
    it is NNCF weight compression here, which needs no calibration data.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

CAMERAS = "top,left_wrist,right_wrist"


def build_stats(state_dim, action_dim, cameras, img_shape):
    """Normalisation stats in the shape physicalai's SmolVLA expects.

    Identity stats (mean 0, std 1) when there is no dataset to measure.  A trained run
    carries the real numbers in its checkpoint and overrides these.
    """
    stats = {
        "observation.state": {
            "name": "state", "shape": tuple([state_dim]), "type": "STATE",
            "mean": [0.0] * state_dim, "std": [1.0] * state_dim,
        },
        "action": {
            "name": "action", "shape": tuple([action_dim]), "type": "ACTION",
            "mean": [0.0] * action_dim, "std": [1.0] * action_dim,
        },
    }
    for camera in cameras:
        stats["observation.images.%s" % camera] = {
            "name": "images.%s" % camera, "shape": tuple(img_shape), "type": "VISUAL",
            "mean": [0.0] * 3, "std": [1.0] * 3,
        }
    return stats


def load_policy(args, stats):
    from physicalai.policies.smolvla import SmolVLA

    kwargs = dict(
        chunk_size=args.chunk_size,
        n_action_steps=args.chunk_size,
        resize_imgs_with_padding=(args.resize, args.resize),
        num_steps=args.num_steps,
        load_vlm_weights=bool(args.checkpoint),
    )
    if not args.checkpoint:
        return SmolVLA(dataset_stats=stats, **kwargs)

    # A checkpoint carries its own config and its own stats, and both win: chunk_size,
    # resize and the state/action dimensions all come from the repo, not from the flags.
    policy = SmolVLA(pretrained_name_or_path=args.checkpoint, **kwargs)
    loaded = {k: tuple(v["shape"]) for k, v in policy._dataset_stats.items()}
    wanted = {k: tuple(v["shape"]) for k, v in stats.items()}
    if loaded == wanted or args.keep_checkpoint_shape:
        print("[build] using the checkpoint's own feature shapes: %s" % loaded)
        return policy

    # lerobot/smolvla_base is a single 6-DoF arm with cameras named camera1..3, so its
    # shapes are not the dinner table's.  Re-stating is safe: the model pads state and
    # action to max_state_dim/max_action_dim (32) internally, so every weight tensor is
    # the same size either way - only the IR's `state` input and `action` output change.
    print("[build] re-stating checkpoint from %s to %s" % (loaded, wanted))
    policy._update_preprocessor_stats(stats)
    return policy


def report(directory):
    total = 0
    for path in sorted(Path(directory).iterdir()):
        if path.is_file():
            total += path.stat().st_size
            print("      %-20s %9.1f MB" % (path.name, path.stat().st_size / 1e6))
    print("      %-20s %9.1f MB" % ("TOTAL", total / 1e6))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="policy_ov", help="output directory")
    parser.add_argument("--checkpoint", default=None,
                        help="local checkpoint dir or HF repo id; omit for random weights")
    parser.add_argument("--state-dim", type=int, default=12)
    parser.add_argument("--action-dim", type=int, default=12)
    parser.add_argument("--cameras", default=CAMERAS)
    parser.add_argument("--img", default="3,256,256", help="raw camera shape, C,H,W")
    parser.add_argument("--resize", type=int, default=512, help="what the policy resizes to")
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--num-steps", type=int, default=10, help="flow-matching denoise steps")
    parser.add_argument("--precisions", default="fp32,fp16,int8")
    parser.add_argument("--keep-checkpoint-shape", action="store_true",
                        help="export a checkpoint at its own state/action dims instead of the flags")
    parser.add_argument("--no-cast-fp32", action="store_true",
                        help="leave the bfloat16 VLM weights alone (see module docstring)")
    args = parser.parse_args()

    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
    img_shape = [int(x) for x in args.img.split(",")]
    precisions = [p.strip() for p in args.precisions.split(",") if p.strip()]
    stats = build_stats(args.state_dim, args.action_dim, cameras, img_shape)

    out = Path(args.out)
    fp32_dir = out / "fp32"

    started = time.perf_counter()
    policy = load_policy(args, stats)
    print("[build] policy built in %.1fs" % (time.perf_counter() - started))
    print("[build] %.1fM parameters, %d cameras, state %d, action %d, chunk %d"
          % (sum(p.numel() for p in policy.model.parameters()) / 1e6,
             len(cameras), args.state_dim, args.action_dim, args.chunk_size))

    if not args.no_cast_fp32:
        policy.model.float()
        print("[build] cast every weight to float32")

    print("[export] fp32 -> %s" % fp32_dir)
    started = time.perf_counter()
    policy.export(fp32_dir, backend="openvino")
    print("[export] fp32 done in %.1fs" % (time.perf_counter() - started))
    (fp32_dir / "stats.json").write_text(
        json.dumps({k: {kk: list(vv) if isinstance(vv, tuple) else vv for kk, vv in v.items()}
                    for k, v in stats.items()}, indent=2))
    report(fp32_dir)

    # Everything below is pure OpenVINO and NNCF - no torch, no policy object.
    import openvino

    extras = [name for name in ("tokenizer.xml", "tokenizer.bin", "manifest.json", "stats.json")
              if (fp32_dir / name).exists()]

    def finish(target_dir, model, label, compress_to_fp16):
        target_dir.mkdir(parents=True, exist_ok=True)
        openvino.save_model(model, str(target_dir / "smolvla.xml"),
                            compress_to_fp16=compress_to_fp16)
        for name in extras:
            shutil.copy2(fp32_dir / name, target_dir / name)
        print("[export] %s done in %.1fs" % (label, time.perf_counter() - started))
        report(target_dir)

    if "fp16" in precisions:
        print("[export] fp16 -> %s" % (out / "fp16"))
        started = time.perf_counter()
        finish(out / "fp16", openvino.Core().read_model(fp32_dir / "smolvla.xml"),
               "fp16", True)

    if "int8" in precisions:
        print("[export] int8 -> %s" % (out / "int8"))
        try:
            import nncf
        except ImportError:
            sys.exit("int8 needs NNCF.  pip install nncf")
        started = time.perf_counter()
        # Weight-only compression: no calibration dataset needed, which matters because
        # there is no collected dataset yet.  Full INT8 (activations too) needs real
        # frames through nncf.quantize with a calibration set.
        compressed = nncf.compress_weights(
            openvino.Core().read_model(fp32_dir / "smolvla.xml"),
            mode=nncf.CompressWeightsMode.INT8_ASYM,
        )
        finish(out / "int8", compressed, "int8", False)

    if "fp32" not in precisions:
        shutil.rmtree(fp32_dir)
        print("[export] removed fp32 (not requested)")

    print("\nnext: python intel_bench.py --ir %s --runs 10" % (out / "fp16" / "smolvla.xml"))


if __name__ == "__main__":
    sys.exit(main())
