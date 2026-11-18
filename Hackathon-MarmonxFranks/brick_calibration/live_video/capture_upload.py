#!/usr/bin/env python3
"""Capture labelled Mac camera frames and upload them to photolab cache.

Use one physical brick per run. Press Space to begin, or q to cancel.
The full, unannotated frames and a capture manifest remain on the Mac too.
"""
from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path

import cv2


REMOTE_ROOT = "/cache/s95vkond/live_video_capture"


def capture(args, destination: Path) -> None:
    backends = {"auto": cv2.CAP_ANY, "avfoundation": cv2.CAP_AVFOUNDATION}
    camera = cv2.VideoCapture(args.camera, backends[args.backend])
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    label = f"Brick_{args.brick_id}_{args.truth}"
    print(f"Place ONLY {label} in view. Press Space in the preview to capture {args.count} frames.", flush=True)
    started = False
    next_capture = 0.0
    rows = []
    try:
        cv2.namedWindow("Labelled brick capture", cv2.WINDOW_NORMAL)
        while len(rows) < args.count:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Camera stopped delivering frames")
            if started and time.monotonic() >= next_capture:
                stamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
                filename = f"frame_{len(rows) + 1:03d}.jpg"
                if not cv2.imwrite(str(destination / filename), frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise OSError(f"Could not save {filename}")
                rows.append({"brick_id": label, "truth": args.truth,
                             "captured_at": stamp, "filename": filename})
                print(f"Saved {label} {len(rows)}/{args.count}: {filename}", flush=True)
                next_capture = time.monotonic() + args.interval
            preview = cv2.resize(frame, (min(960, frame.shape[1]),
                                         round(frame.shape[0] * min(960, frame.shape[1]) / frame.shape[1])))
            cv2.putText(preview, f"{label}  {len(rows)}/{args.count}  " +
                        ("SPACE=start, q=cancel" if not started else "capturing; q=stop"),
                        (12, 34), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
            cv2.imshow("Labelled brick capture", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                raise KeyboardInterrupt("Capture cancelled; saved frames remain on Mac")
            if key == ord(" "):
                started = True
                next_capture = time.monotonic()
        with (destination / "manifest.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("brick_id", "truth", "captured_at", "filename"))
            writer.writeheader()
            writer.writerows(rows)
    finally:
        camera.release()
        cv2.destroyAllWindows()


def upload(args, destination: Path) -> None:
    remote_parent = f"{args.remote_root}/Brick_{args.brick_id}_{args.truth}"
    login = f"{args.user}@{args.host}"
    subprocess.run(["ssh", login, f"mkdir -p {shlex.quote(remote_parent)}"], check=True)
    subprocess.run(["scp", "-r", str(destination), f"{login}:{remote_parent}/"], check=True)
    print(f"Uploaded to {login}:{remote_parent}/{destination.name}/", flush=True)


def import_existing(args, destination: Path) -> None:
    images = sorted(path for path in args.from_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png"))
    raw = [path for path in images if path.name.startswith("raw_")]
    images = raw or [path for path in images if not path.name.startswith(("prediction_", "preview"))]
    if not images:
        raise ValueError(f"No raw images found in {args.from_dir}")
    rows = []
    label = f"Brick_{args.brick_id}_{args.truth}"
    for index, source in enumerate(images, 1):
        frame = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Could not decode {source}")
        filename = f"frame_{index:03d}.jpg"
        if not cv2.imwrite(str(destination / filename), frame,
                           [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"Could not save {filename}")
        rows.append({"brick_id": label, "truth": args.truth,
                     "captured_at": datetime.fromtimestamp(source.stat().st_mtime).astimezone().isoformat(timespec="milliseconds"),
                     "filename": filename})
    with (destination / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("brick_id", "truth", "captured_at", "filename"))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Imported {len(rows)} raw images from {args.from_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brick-id", required=True, type=int,
                        help="Physical brick number, e.g. 5; use one brick per run")
    parser.add_argument("--truth", required=True, choices=("good", "bad"))
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--interval", type=float, default=3.0)
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--backend", choices=("auto", "avfoundation"), default="avfoundation")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--host", default="131.220.238.95")
    parser.add_argument("--user", default="s95vkond")
    parser.add_argument("--remote-root", default=REMOTE_ROOT)
    parser.add_argument("--local-root", type=Path, default=Path.cwd() / "live_capture")
    parser.add_argument("--from-dir", type=Path,
                        help="Upload existing raw_*.jpg live frames instead of opening camera")
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()
    if args.brick_id < 1 or args.count < 1 or args.interval <= 0:
        parser.error("brick ID and count must be positive; interval must exceed zero")
    label = f"Brick_{args.brick_id}_{args.truth}"
    session = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    destination = args.local_root / label / session
    destination.mkdir(parents=True, exist_ok=False)
    if args.from_dir is not None:
        import_existing(args, destination)
    else:
        try:
            capture(args, destination)
        except KeyboardInterrupt as error:
            print(str(error), flush=True)
            return
    print(f"Mac copy: {destination.resolve()}", flush=True)
    if not args.no_upload:
        upload(args, destination)


if __name__ == "__main__":
    main()
