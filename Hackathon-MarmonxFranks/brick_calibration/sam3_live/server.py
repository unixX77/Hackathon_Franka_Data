#!/usr/bin/env python3
"""Persistent SSH server: SAM3 masks -> selected brick quality classifier."""
from __future__ import annotations

import argparse
import base64
from collections import deque
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

BRICK_CALIBRATION = Path(__file__).resolve().parents[1]
if str(BRICK_CALIBRATION) not in sys.path:
    sys.path.insert(0, str(BRICK_CALIBRATION))

from offline_sam3_classifier import (  # noqa: E402
    SAM3_SNAPSHOT, BrickGeometrySegmenter, FusionClassifier,
    annotate_soft_boundary, select_bricks,
)
from local_defect_candidate import LocalDefectCandidate  # noqa: E402
from color_only_classifier import ColorOnlyClassifier  # noqa: E402
from good_brick_clusters import GoodBrickReferenceClassifier  # noqa: E402
from face_color_rotation_candidate import FaceColorRotationClassifier  # noqa: E402
from threshold_policies.policy import NAMES, labels_for_all, load_policy  # noqa: E402
sys.path.append(str(BRICK_CALIBRATION / "redness_classifier"))
from classify_brick import RednessMaskClassifier  # noqa: E402


CHECKPOINT = BRICK_CALIBRATION / "approach_results/baseline_checkpoint"


def box_iou(first: list[int], second: list[int]) -> float:
    x0, y0 = max(first[0], second[0]), max(first[1], second[1])
    x1, y1 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, x1 - x0 + 1) * max(0, y1 - y0 + 1)
    area_first = (first[2] - first[0] + 1) * (first[3] - first[1] + 1)
    area_second = (second[2] - second[0] + 1) * (second[3] - second[1] + 1)
    return intersection / max(1, area_first + area_second - intersection)


class TrackState:
    """Stabilize scores for stationary bricks across three recent frames."""

    def __init__(self):
        self.tracks: dict[int, dict] = {}
        self.next_id = 1
        self.frame = 0

    def update(self, detections: list[dict]) -> None:
        self.frame += 1
        available = set(self.tracks)
        for detection in detections:
            ranked = sorted(((box_iou(detection["box_xyxy"], self.tracks[key]["box"]), key)
                             for key in available), reverse=True)
            if ranked and ranked[0][0] >= .25:
                key = ranked[0][1]
                available.remove(key)
            else:
                key = self.next_id
                self.next_id += 1
                self.tracks[key] = {"scores": deque(maxlen=3)}
            track = self.tracks[key]
            track["box"] = detection["box_xyxy"]
            track["seen"] = self.frame
            track["scores"].append(detection["score_bad"])
            detection["track_id"] = key
            detection["raw_score_bad"] = detection["score_bad"]
            detection["score_bad"] = float(np.median(track["scores"]))
            detection["label"] = "bad" if detection["score_bad"] > 0 else "good"
            detection["temporal_frames"] = len(track["scores"])
        self.tracks = {key: track for key, track in self.tracks.items()
                       if self.frame - track["seen"] <= 2}


def send(record: dict) -> None:
    sys.stdout.write(json.dumps(record, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def process_request(request: dict, segmenter: BrickGeometrySegmenter,
                    classifier: FusionClassifier | LocalDefectCandidate | ColorOnlyClassifier | GoodBrickReferenceClassifier | FaceColorRotationClassifier | RednessMaskClassifier,
                    policy_name: str, model_name: str,
                    outline_width: int, tracker: TrackState | None = None) -> dict:
    payload = base64.b64decode(request["image_b64"], validate=True)
    frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if frame is None or max(frame.shape[:2]) > 4096:
        raise ValueError("Invalid JPEG or frame exceeds 4096 pixels")
    started = time.monotonic()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    instance, records, rejected = segmenter(rgb)
    selected = select_bricks(instance, records, max_bricks=10)
    truth = request.get("truth_label") or ""
    result = {"type": "prediction", "frame_id": request.get("frame_id"),
              "captured_at": request.get("captured_at", ""),
              "pipeline": {"baseline": "sam3", "local": "sam3_local",
                           "color": "sam3_color", "reference": "sam3_reference",
                           "face_color": "sam3_face_color", "redness": "sam3_redness"}[model_name],
              "threshold_policy": {"baseline": policy_name, "local": "local_candidate_zero",
                                   "color": "color_only_zero", "reference": "good_reference_distance",
                                   "face_color": "face_color_rotation_threshold",
                                   "redness": "redness_window"}[model_name],
              "truth_label": truth, "correct": "", "label": "", "score_bad": "",
              "approach5_score": "", "dinov2_score": "",
              "baseline_margin_label": "", "zero_binary_label": "",
              "minus_two_binary_label": "", "review_band_175_120_label": "",
              "status": "no_reliable_sam3_brick",
              "sam_candidates": len(records), "sam_rejected": len(rejected),
              "sam_score": "", "sam_touches_border": "", "mask_area_px": "",
              "brick_count": 0, "good_count": 0, "bad_count": 0,
              "uncertain_count": 0, "detections": [],
              "box_x": "", "box_y": "", "box_width": "", "box_height": "",
              "frame_width": frame.shape[1], "frame_height": frame.shape[0],
              "error": ""}
    annotated = frame.copy()
    masks = []
    for brick_id, (mask, sam_record) in enumerate(selected, 1):
        prediction = classifier.classify(rgb, mask=mask)
        score = float(prediction["score_bad"])
        if np.isnan(score):  # redness model: perforated face not visible / over-exposed
            score = ""
        labels = labels_for_all(score) if model_name == "baseline" else {}
        label = labels[policy_name] if model_name == "baseline" else prediction["label"]
        x, y, width, height = cv2.boundingRect(cv2.findNonZero(mask))
        detection = {"id": brick_id, "label": label, "score_bad": score,
                     "approach5_score": prediction.get("approach5_score", ""),
                     "dinov2_score": prediction.get("dinov2_score", ""),
                     "baseline_margin_label": labels.get("baseline_margin", ""),
                     "zero_binary_label": labels.get("zero_binary", ""),
                     "minus_two_binary_label": labels.get("minus_two_binary", ""),
                     "review_band_175_120_label": labels.get("review_band_175_120", ""),
                     "local_defect_features": prediction.get("local_defect_features", []),
                     "reference_distance": prediction.get("distance", ""),
                     "reference_threshold": prediction.get("reference_threshold", ""),
                     "nearest_good_reference": prediction.get("nearest_good_reference", ""),
                     "view_proxy": prediction.get("view_proxy", {}),
                     "candidate_raw_score": prediction.get("raw_score", ""),
                     "candidate_threshold": prediction.get("candidate_threshold", ""),
                     "no_verdict_reason": prediction.get("reason", ""),
                     "perforations": prediction.get("perforations", ""),
                     "face_colour": prediction.get("face_colour", {}),
                     "sam_score": sam_record["score"],
                     "sam_touches_border": sam_record["touches_border"],
                     "mask_area_px": int(mask.sum()),
                     "box_xyxy": [x, y, x + width - 1, y + height - 1],
                     "centroid_xy": sam_record["centroid_xy"],
                     "min_area_rect": sam_record["min_area_rect"]}
        result["detections"].append(detection)
        masks.append(mask)
    if tracker is not None:
        tracker.update(result["detections"])
    for detection, mask in zip(result["detections"], masks):
        label = detection["label"]
        result[f"{label}_count"] += 1
        score_text = (f"{detection['score_bad']:+.2f}" if detection["score_bad"] != ""
                      else f"({detection.get('no_verdict_reason') or 'no score'})")
        annotated = annotate_soft_boundary(
            annotated, mask, label, detection["score_bad"] or 0.0, outline_width,
            caption=f"#{detection.get('track_id', detection['id'])} "
                    f"{label.upper()} {score_text}")
    result["brick_count"] = len(result["detections"])
    if result["brick_count"]:
        result["status"] = "ok"
    if result["brick_count"] == 1:
        detection = result["detections"][0]
        x, y, x1, y1 = detection["box_xyxy"]
        result.update({"label": detection["label"],
                       "correct": int(detection["label"] == truth)
                       if truth in ("good", "bad") else "",
                       "score_bad": detection["score_bad"],
                       "approach5_score": detection["approach5_score"],
                       "dinov2_score": detection["dinov2_score"],
                       "baseline_margin_label": detection["baseline_margin_label"],
                       "zero_binary_label": detection["zero_binary_label"],
                       "minus_two_binary_label": detection["minus_two_binary_label"],
                       "review_band_175_120_label": detection["review_band_175_120_label"],
                       "sam_score": detection["sam_score"],
                       "sam_touches_border": detection["sam_touches_border"],
                       "mask_area_px": detection["mask_area_px"],
                       "box_x": x, "box_y": y,
                       "box_width": x1 - x + 1, "box_height": y1 - y + 1})
    if not result["brick_count"]:
        cv2.putText(annotated, "NO RELIABLE SAM3 BRICK", (25, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 190, 255), 2, cv2.LINE_AA)
    result["inference_seconds"] = round(time.monotonic() - started, 3)
    ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise ValueError("Could not encode annotated frame")
    result["annotated_b64"] = base64.b64encode(encoded).decode("ascii")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--sam-model", type=Path, default=SAM3_SNAPSHOT)
    parser.add_argument("--prompt", default="brick")
    parser.add_argument("--sam-threshold", type=float, default=.35)
    parser.add_argument("--outline-width", type=int, default=3)
    parser.add_argument("--policy", choices=NAMES, default="review_band_175_120")
    parser.add_argument("--model", choices=("baseline", "local", "color", "reference", "face_color", "redness"), default="baseline")
    args = parser.parse_args()
    load_policy(args.policy)
    if not args.sam_model.is_dir():
        parser.error(f"Cached SAM3 checkpoint not found: {args.sam_model}")
    if not 1 <= args.outline_width <= 8:
        parser.error("--outline-width must be 1 to 8 pixels")
    from transformers.utils import logging as transformers_logging
    transformers_logging.disable_progress_bar()
    sam_args = argparse.Namespace(model=str(args.sam_model), prompt=args.prompt,
                                  threshold=args.sam_threshold, mask_threshold=.5,
                                  max_overlap=.5, min_area=1200, min_rect=.55)
    print(f"Loading cached SAM3 and {args.model} classifier on photolab...", file=sys.stderr,
          flush=True)
    segmenter = BrickGeometrySegmenter(sam_args)
    classifier = (FusionClassifier.load(args.checkpoint) if args.model == "baseline"
                  else LocalDefectCandidate(baseline=args.checkpoint) if args.model == "local"
                  else ColorOnlyClassifier() if args.model == "color"
                  else GoodBrickReferenceClassifier(baseline=args.checkpoint) if args.model == "reference"
                  else RednessMaskClassifier(sam_model=segmenter.model, sam_processor=segmenter.processor)
                  if args.model == "redness"
                  else FaceColorRotationClassifier(baseline=args.checkpoint))
    send({"type": "ready", "model": {"baseline": "SAM3 + Approach 17",
          "local": "SAM3 + local defect candidate", "color": "SAM3 + color-only classifier",
          "reference": "SAM3 + good-brick reference classifier",
          "face_color": "SAM3 + face/color/rotation candidate",
          "redness": "SAM3 + perforated-face redness window"}[args.model],
          "device": segmenter.device,
          "pipeline": {"baseline": "sam3", "local": "sam3_local", "color": "sam3_color",
                       "reference": "sam3_reference", "face_color": "sam3_face_color",
                       "redness": "sam3_redness"}[args.model],
          "threshold_policy": {"baseline": args.policy, "local": "local_candidate_zero",
                               "color": "color_only_zero", "reference": "good_reference_distance",
                               "face_color": "face_color_rotation_threshold",
                               "redness": "redness_window"}[args.model]})
    tracker = TrackState() if args.model == "local" else None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            response = process_request(request, segmenter, classifier,
                                       args.policy, args.model, args.outline_width, tracker)
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
