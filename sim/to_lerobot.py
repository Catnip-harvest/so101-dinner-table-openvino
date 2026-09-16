"""Convert raw WP2 episodes to a LeRobot v3 dataset and optionally push privately."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np

CAMERAS = ("top", "left_wrist", "right_wrist")
IMAGE_KEYS = tuple(f"observation.images.{camera}" for camera in CAMERAS)
JOINT_NAMES = (
    "right.shoulder_pan", "right.shoulder_lift", "right.elbow_flex",
    "right.wrist_flex", "right.wrist_roll", "right.gripper",
    "left.shoulder_pan", "left.shoulder_lift", "left.elbow_flex",
    "left.wrist_flex", "left.wrist_roll", "left.gripper",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--repo", default="VietHwang/dinner-table-v1")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--append", action="store_true", help="Create/resume one dataset with chunk tracking")
    parser.add_argument("--chunk-id", help="Stable unique chunk name (required with --append)")
    return parser.parse_args()


def features() -> dict:
    result = {
        "observation.state": {"dtype": "float32", "shape": (12,), "names": list(JOINT_NAMES)},
        "action": {"dtype": "float32", "shape": (12,), "names": list(JOINT_NAMES)},
    }
    for key in IMAGE_KEYS:
        result[key] = {
            "dtype": "video", "shape": (256, 256, 3),
            "names": ["height", "width", "channels"],
        }
    return result


def frame_iter(raw: Path, episode: dict):
    files = episode["files"]
    if episode["storage"] == "mp4":
        readers = [iio.imiter(raw / files[camera], plugin="FFMPEG") for camera in CAMERAS]
    else:
        readers = [
            (iio.imread(path) for path in sorted((raw / files[camera]).glob("*.png")))
            for camera in CAMERAS
        ]
    for images in zip(*readers, strict=True):
        yield images


def printable_shape(value) -> list[int]:
    shape = getattr(value, "shape", ())
    return [int(v) for v in shape]


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if args.append != bool(args.chunk_id):
        raise SystemExit("--append and --chunk-id must be supplied together")
    manifest = json.loads((args.raw / "manifest.json").read_text(encoding="utf-8"))
    kept = [episode for episode in manifest["episodes"] if episode["success"]]
    if not kept:
        raise SystemExit("raw manifest has no kept episodes")
    if tuple(manifest["state_action_order"]) != JOINT_NAMES:
        raise SystemExit("raw state/action ordering does not match the LeRobot feature schema")

    import lerobot
    from lerobot.datasets.dataset_metadata import CODEBASE_VERSION
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = args.root or (args.raw / "lerobot" / args.repo.replace("/", "__"))
    journal_path = root / "meta" / "conversion_chunks.json"
    journal = None
    chunk = None
    convert = not root.exists()
    expected_episodes = len(kept)
    expected_frames = None
    if args.append:
        fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        if root.exists():
            if not journal_path.exists():
                raise SystemExit("append requires a dataset created with --append; missing chunk journal")
            journal = json.loads(journal_path.read_text(encoding="utf-8"))
        else:
            journal = {"repo": args.repo, "chunks": {}}
        if journal["repo"] != args.repo:
            raise SystemExit("chunk journal repository mismatch")
        pending = [key for key, value in journal["chunks"].items() if value["status"] != "complete"]
        if pending:
            raise SystemExit(f"unfinished conversion {pending}; inspect/recover locally before retrying (no automatic duplication)")
        chunk = journal["chunks"].get(args.chunk_id)
        if chunk and chunk["manifest_sha256"] != fingerprint:
            raise SystemExit("chunk-id was already used with a different manifest")
        if not chunk and any(c["manifest_sha256"] == fingerprint for c in journal["chunks"].values()):
            raise SystemExit("this manifest was already converted under another chunk-id")
        convert = chunk is None
        if convert:
            frame_count = 0
            for episode in kept:
                with np.load(args.raw / episode["files"]["npz"], allow_pickle=False) as data:
                    if len(data["state"]) != len(data["action"]):
                        raise SystemExit("state/action frame counts differ")
                    frame_count += len(data["state"])
            chunk = {"manifest_sha256": fingerprint, "episodes": len(kept), "frames": frame_count, "status": "pending"}
            journal["chunks"][args.chunk_id] = chunk
        expected_episodes = sum(c["episodes"] for c in journal["chunks"].values())
        expected_frames = sum(c["frames"] for c in journal["chunks"].values())
    print(f"lerobot package={lerobot.__version__} codebase={CODEBASE_VERSION}")
    if root.exists():
        if args.append and convert:
            info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
            if (info["total_episodes"], info["total_frames"]) != (expected_episodes - chunk["episodes"], expected_frames - chunk["frames"]):
                raise SystemExit("dataset counts differ from completed chunk journal")
            write_json(journal_path, journal)
            dataset = LeRobotDataset.resume(args.repo, root=root, video_backend="pyav", image_writer_threads=3, vcodec="h264")
        elif not args.push and not args.append:
            raise SystemExit(f"dataset root already exists: {root}")
        else:
            dataset = LeRobotDataset(args.repo, root=root, video_backend="pyav")
            print(f"reusing finalized dataset root={root}")
    else:
        root.parent.mkdir(parents=True, exist_ok=True)
        dataset = LeRobotDataset.create(
            args.repo, fps=50, features=features(), root=root, robot_type="bimanual_so101",
            use_videos=True, image_writer_threads=3, vcodec="h264",
        )
        if journal is not None:
            write_json(journal_path, journal)
    if convert:
        try:
            for episode in kept:
                with np.load(args.raw / episode["files"]["npz"], allow_pickle=False) as raw_episode:
                    states = raw_episode["state"]
                    actions = raw_episode["action"]
                    task = str(raw_episode["task"].item())
                    expected = len(states)
                    count = 0
                    for count, images in enumerate(frame_iter(args.raw, episode), start=1):
                        index = count - 1
                        if index >= expected:
                            raise RuntimeError(
                                f"{episode['files']['npz']}: video has more than {expected} frames"
                            )
                        dataset.add_frame({
                            "observation.state": states[index], "action": actions[index], "task": task,
                            **{key: image for key, image in zip(IMAGE_KEYS, images, strict=True)},
                        })
                    if count != expected:
                        raise RuntimeError(
                            f"{episode['files']['npz']}: decoded {count} image frames, expected {expected}"
                        )
                    # Windows spawn imports a full Torch stack per camera; keep
                    # encoding serial to avoid pagefile growth on this machine.
                    dataset.save_episode(parallel_encoding=False)
                    print(f"saved episode attempt={episode['attempt']} frames={expected} task={task!r}")
            dataset.finalize()
        except Exception:
            dataset.finalize()
            raise

    # TorchCodec wheels in the Windows writer venv do not load against its
    # PyTorch build.  PyAV is an installed, supported LeRobot backend and gives
    # a real decoded-frame load-back without changing the dataset on disk.
    loaded = LeRobotDataset(args.repo, root=root, video_backend="pyav")
    if loaded.num_episodes != expected_episodes:
        raise RuntimeError(f"episode count mismatch: {loaded.num_episodes} != {expected_episodes}")
    if expected_frames is not None and loaded.num_frames != expected_frames:
        raise RuntimeError(f"frame count mismatch: {loaded.num_frames} != {expected_frames}")
    sample = loaded[0]
    # Decode the newest episode too, including the final appended frame.
    loaded[loaded.num_frames - 1]
    print(f"loadback episodes={loaded.num_episodes} frames={loaded.num_frames}")
    print("sample keys=" + json.dumps(sorted(sample.keys())))
    print("sample shapes=" + json.dumps({key: printable_shape(sample[key]) for key in sorted(sample)}))
    if journal is not None and convert:
        chunk["status"] = "complete"
        write_json(journal_path, journal)

    if args.push:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id=args.repo, repo_type="dataset", private=True, exist_ok=True)
        if not api.dataset_info(args.repo).private:
            raise RuntimeError(f"refusing upload: existing repository {args.repo} is not private")
        loaded.push_to_hub(private=True)
        info = api.dataset_info(args.repo)
        if not info.private:
            raise RuntimeError(f"refusing success: {args.repo} is not private")
        url = f"https://huggingface.co/datasets/{args.repo}"
        print(f"pushed private dataset: {url}")
    else:
        print("push skipped (pass --push to upload privately)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
