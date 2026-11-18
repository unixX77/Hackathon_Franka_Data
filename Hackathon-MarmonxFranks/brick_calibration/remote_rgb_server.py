#!/usr/bin/env python3
"""Approach 17 inference over a persistent SSH standard-input/output stream.

Reads one JSON request per line, containing a base64 JPEG. Writes one JSON
response per line. Standard output is reserved exclusively for this protocol.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from fusion_checkpoint import FusionClassifier
from live_rgb_classifier import classify_snapshot
from threshold_policies.policy import NAMES, labels_for_all, load_policy


DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "approach_results" / "baseline_checkpoint"


def send(message: dict) -> None:
    sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def process_request(request: dict, classifier: FusionClassifier,
                    policy_name: str = "baseline_margin") -> dict:
    payload = base64.b64decode(request["image_b64"], validate=True)
    encoded = np.frombuffer(payload, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("JPEG frame could not be decoded")
    if frame.shape[0] > 4096 or frame.shape[1] > 4096:
        raise ValueError("Frame exceeds the 4096-pixel maximum dimension")
    result = classify_snapshot(frame, classifier, str(request.get("captured_at", "")),
                               request.get("truth_label"))
    record = result["record"]
    record["threshold_policy"] = policy_name
    if record["status"] == "ok":
        labels = labels_for_all(record["score_bad"])
        record["baseline_margin_label"] = labels["baseline_margin"]
        record["zero_binary_label"] = labels["zero_binary"]
        record["minus_two_binary_label"] = labels["minus_two_binary"]
        record["review_band_175_120_label"] = labels["review_band_175_120"]
        record["label"] = labels[policy_name]
        truth = record["truth_label"]
        record["correct"] = int(record["label"] == truth) if truth in ("good", "bad") else ""
    return {"type": "prediction", "frame_id": request.get("frame_id"),
            **record}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--policy", choices=NAMES, default="baseline_margin")
    args = parser.parse_args()
    load_policy(args.policy)
    from transformers.utils import logging as transformers_logging
    transformers_logging.disable_progress_bar()
    print(f"Loading Approach 17 on photolab from {args.checkpoint}...", file=sys.stderr,
          flush=True)
    classifier = FusionClassifier.load(args.checkpoint)
    send({"type": "ready", "model": "Approach 17", "device": classifier.device,
          "threshold_policy": args.policy})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            response = process_request(request, classifier, args.policy)
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
