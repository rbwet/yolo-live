#!/usr/bin/env python3
"""Extract frames from gameplay clips at a target FPS.

Walks a clips directory (default: ./clips), decodes each video with OpenCV,
samples frames at --fps (default 2), and writes JPEGs to --out (default
./dataset/images) as <clip_stem>_<frame_index>.jpg.

Optional SSIM-style dedup: if a frame is nearly identical to the previous
kept frame (mean absolute difference < --dedup-thresh), it's skipped. This
cuts down on the "nothing moved in 0.5s" duplicates.

Usage:
    python tools/extract_frames.py                          # defaults
    python tools/extract_frames.py --clips clips --fps 2
    python tools/extract_frames.py --dedup 4.0              # stronger dedup
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def iter_videos(clips_dir: Path):
    for p in sorted(clips_dir.rglob("*")):
        if p.suffix.lower() in VIDEO_EXTS and p.is_file():
            yield p


def extract_from_video(
    video_path: Path,
    out_dir: Path,
    target_fps: float,
    dedup_thresh: float,
    jpg_quality: int,
) -> tuple[int, int]:
    """Return (frames_read, frames_saved)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [skip] could not open {video_path}")
        return (0, 0)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, int(round(src_fps / max(target_fps, 0.01))))

    stem = video_path.stem.replace(" ", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    last_kept_small: np.ndarray | None = None
    frames_read = 0
    frames_saved = 0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames_read += 1
        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        # Dedup: compare a small grayscale thumbnail against previous kept frame.
        small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
        small_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if last_kept_small is not None and dedup_thresh > 0:
            diff = float(np.mean(np.abs(small_gray - last_kept_small)))
            if diff < dedup_thresh:
                frame_idx += 1
                continue
        last_kept_small = small_gray

        out_path = out_dir / f"{stem}_{frame_idx:08d}.jpg"
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
        frames_saved += 1
        frame_idx += 1

    cap.release()
    print(
        f"  {video_path.name}: src_fps={src_fps:.1f}, frames={total_frames}, "
        f"saved={frames_saved} (stride={stride}, dedup={dedup_thresh})"
    )
    return (frames_read, frames_saved)


def main():
    parser = argparse.ArgumentParser(description="Extract frames from gameplay clips")
    parser.add_argument("--clips", type=Path, default=Path("clips"), help="Folder with videos")
    parser.add_argument("--out", type=Path, default=Path("dataset/images"), help="Output image folder")
    parser.add_argument("--fps", type=float, default=2.0, help="Frames per second to extract")
    parser.add_argument(
        "--dedup",
        type=float,
        default=2.5,
        help="Mean-abs-diff threshold on 64x36 grayscale thumbnails; 0 disables dedup. "
        "Typical: 2-5 (lower = keep more, higher = drop more near-duplicates)",
    )
    parser.add_argument("--jpg-quality", type=int, default=92)
    args = parser.parse_args()

    if not args.clips.is_dir():
        print(f"Clips dir not found: {args.clips}")
        sys.exit(1)

    videos = list(iter_videos(args.clips))
    if not videos:
        print(f"No videos found under {args.clips} (supported: {sorted(VIDEO_EXTS)})")
        sys.exit(1)

    print(f"Found {len(videos)} video(s) in {args.clips}")
    print(f"Extracting ~{args.fps} fps to {args.out}  (dedup={args.dedup})")

    total_saved = 0
    for v in videos:
        _, saved = extract_from_video(v, args.out, args.fps, args.dedup, args.jpg_quality)
        total_saved += saved

    print(f"\nDone. Saved {total_saved} frames to {args.out}")


if __name__ == "__main__":
    main()
