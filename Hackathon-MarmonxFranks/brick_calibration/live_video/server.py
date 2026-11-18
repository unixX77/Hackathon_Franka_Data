#!/usr/bin/env python3
"""Persistent SSH JSON-lines server for the isolated live-video variant."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from model import CHECKPOINT, LiveVideoClassifier, decode_jpeg


def send(record: dict) -> None:
    sys.stdout.write(json.dumps(record, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    args = parser.parse_args()
    from transformers.utils import logging as transformers_logging
    transformers_logging.disable_progress_bar()
    print(f"Loading live-video Approach 17 from {args.checkpoint}", file=sys.stderr,
          flush=True)
    model = LiveVideoClassifier(args.checkpoint)
    send({"type": "ready", "model": "Approach 17 live_video", "device": model.classifier.device})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            started = time.monotonic()
            images = request.get("images_b64") or [request["image_b64"]]
            frames = [decode_jpeg(image) for image in images]
            result = model.predict(frames, request.get("roi"))
            response = {"type": "prediction", "frame_id": request.get("frame_id"),
                        "captured_at": request.get("captured_at", ""),
                        "truth_label": request.get("truth_label") or "",
                        "frame_width": frames[-1].shape[1],
                        "frame_height": frames[-1].shape[0],
                        "inference_seconds": round(time.monotonic() - started, 3),
                        **result}
            truth = response["truth_label"]
            response["correct"] = (int(response["label"] == truth) if truth else "")
        except Exception as error:
            response = {"type": "error", "frame_id": None,
                        "error": f"{type(error).__name__}: {error}"}
        try:
            send(response)
        except BrokenPipeError:
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
