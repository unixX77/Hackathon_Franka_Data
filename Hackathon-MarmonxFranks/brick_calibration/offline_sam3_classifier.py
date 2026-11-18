#!/usr/bin/env python3
"""SAM3 whole-brick mask -> frozen Approach 17 -> saved threshold policy.

Offline, folder-wise evaluation only. Input images are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BRICK_CALIBRATION = Path(__file__).resolve().parent
SEGMENTATION = ROOT / "segmentation_sam3_bounding_box"
sys.path.insert(0, str(SEGMENTATION))

from segment_lego_sam3_single_prompt import BrickGeometrySegmenter  # noqa: E402
from fusion_checkpoint import FusionClassifier  # noqa: E402
from threshold_policies.policy import label_for, labels_for_all, load_policy  # noqa: E402


DEFAULT_INPUT = BRICK_CALIBRATION / "brick_splits" / "images"
DEFAULT_OUTPUT = BRICK_CALIBRATION / "offline_results"
SAM3_SNAPSHOT = Path("/cache/s95vkond/huggingface/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7")
EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
COLOURS = {"good": (0, 180, 0), "bad": (0, 0, 225),
           "uncertain": (0, 190, 255)}  # OpenCV BGR


def dominant_brick(instance: np.ndarray, records: list[dict]) -> tuple[np.ndarray, dict] | None:
    """Select one central, non-border whole-brick SAM3 instance."""
    height, width = instance.shape
    candidates = []
    for record in records:
        mask = (instance == record["id"]).astype(np.uint8)
        count, components, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if count <= 1:
            continue
        index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (components == index).astype(np.uint8)
        area = int(mask.sum())
        area_fraction = area / (height * width)
        if not .02 <= area_fraction <= .75:
            continue
        points = cv2.findNonZero(mask)
        x, y, box_width, box_height = cv2.boundingRect(points)
        centre_distance = np.hypot((x + box_width / 2 - width / 2) / width,
                                   (y + box_height / 2 - height / 2) / height)
        if centre_distance > .48:
            continue
        # A real brick may be cut off by one image edge. Accept only a large,
        # central, high-confidence instance in that case; tiny border objects
        # remain excluded by the area test above.
        if record["touches_border"] and not (
                area_fraction >= .08 and centre_distance <= .35 and
                float(record["score"]) >= .65):
            continue
        rank = area * (1 - centre_distance) * max(.1, float(record["score"]))
        candidates.append((rank, mask, record))
    if not candidates:
        return None
    _, mask, record = max(candidates, key=lambda item: item[0])
    return mask, record


def select_bricks(instance: np.ndarray, records: list[dict],
                  max_bricks: int = 10) -> list[tuple[np.ndarray, dict]]:
    """Keep up to ten credible separate SAM3 masks across the whole frame."""
    height, width = instance.shape
    candidates = []
    for record in records:
        component = (instance == record["id"]).astype(np.uint8)
        count, components, stats, _ = cv2.connectedComponentsWithStats(component, 8)
        if count <= 1:
            continue
        index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (components == index).astype(np.uint8)
        area = int(mask.sum())
        area_fraction = area / (height * width)
        if not .008 <= area_fraction <= .75:
            continue
        x, y, box_width, box_height = cv2.boundingRect(cv2.findNonZero(mask))
        if min(box_width, box_height) < 35:
            continue
        centre_distance = np.hypot((x + box_width / 2 - width / 2) / width,
                                   (y + box_height / 2 - height / 2) / height)
        if record["touches_border"]:
            # A small border object is usually background. Keep a large,
            # credible brick that is partly outside the frame.
            if not (area_fraction >= .05 and centre_distance <= .55 and
                    float(record["score"]) >= .65):
                continue
        elif centre_distance > .71:
            continue
        rank = area * max(.1, float(record["score"]))
        candidates.append((rank, mask, record))
    candidates.sort(key=lambda item: item[0], reverse=True)
    candidates = candidates[:max_bricks]
    # Image-local IDs only: sort visually from upper-left to lower-right.
    candidates.sort(key=lambda item: (item[2]["centroid_xy"][1],
                                      item[2]["centroid_xy"][0]))
    return [(mask, record) for _, mask, record in candidates]


def annotate_soft_boundary(frame: np.ndarray, mask: np.ndarray,
                           label: str, score: float,
                           outline_width: int = 3,
                           caption: str | None = None) -> np.ndarray:
    """Subtle interior tint and a softly blended SAM3 contour."""
    if outline_width < 1 or outline_width > 8:
        raise ValueError("outline_width must be between 1 and 8 pixels")
    colour = np.asarray(COLOURS[label], np.float32)
    selected = mask.astype(bool)
    edge = np.zeros(mask.shape, np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(edge, contours, -1, 255, outline_width, cv2.LINE_AA)
    soft = cv2.GaussianBlur(edge, (0, 0), sigmaX=max(1.3, outline_width * .5))
    alpha = np.maximum(soft.astype(np.float32) / 255 * .45,
                       edge.astype(np.float32) / 255 * .65)
    alpha[selected] = np.maximum(alpha[selected], .055)
    out = (frame.astype(np.float32) * (1 - alpha[..., None]) +
           colour[None, None, :] * alpha[..., None]).clip(0, 255).astype(np.uint8)
    x, y, width, _ = cv2.boundingRect(cv2.findNonZero(mask))
    text = caption or f"{label.upper()}  {score:+.3f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(.55, min(frame.shape[1] / 1600, .85))
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    tx = min(x, max(0, frame.shape[1] - tw - 12))
    ty = y - 9 if y >= th + baseline + 13 else y + th + 10
    cv2.rectangle(out, (tx, ty - th - 5), (tx + tw + 8, ty + baseline + 4),
                  (20, 20, 20), -1)
    cv2.putText(out, text, (tx + 4, ty), font, scale, COLOURS[label],
                thickness, cv2.LINE_AA)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sam-model", type=Path, default=SAM3_SNAPSHOT)
    parser.add_argument("--prompt", default="brick")
    parser.add_argument("--sam-threshold", type=float, default=.35)
    parser.add_argument("--policy", choices=("minus_two_binary", "zero_binary", "baseline_margin"),
                        default="minus_two_binary")
    parser.add_argument("--limit", type=int, help="Images per folder for a preview run")
    args = parser.parse_args()
    if not args.input.is_dir():
        parser.error(f"Input folder is missing: {args.input}")
    if not args.sam_model.is_dir():
        parser.error(f"Cached SAM3 snapshot is missing: {args.sam_model}")
    folders = sorted((path for path in args.input.iterdir()
                      if path.is_dir() and path.name.startswith("Brick_") and
                      path.name.rsplit("_", 1)[-1] in ("good", "bad")),
                     key=lambda path: int(path.name.split("_")[1]))
    if not folders:
        parser.error(f"No labelled Brick_N_good/bad folders in {args.input}")
    load_policy(args.policy)
    sam_args = argparse.Namespace(model=str(args.sam_model), prompt=args.prompt,
                                  threshold=args.sam_threshold, mask_threshold=.5,
                                  max_overlap=.5, min_area=1200, min_rect=.55)
    print("Loading SAM3 and frozen Approach 17 once...", flush=True)
    segmenter = BrickGeometrySegmenter(sam_args)
    classifier = FusionClassifier.load(BRICK_CALIBRATION / "approach_results/baseline_checkpoint")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for folder in folders:
        images = sorted(path for path in folder.iterdir()
                        if path.is_file() and path.suffix.lower() in EXTENSIONS)
        if args.limit is not None:
            images = images[:args.limit]
        destination = args.output / folder.name
        mask_dir = destination / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        folder_rows = []
        for path in images:
            started = time.monotonic()
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"Could not decode {path}")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            instance, records, rejected = segmenter(rgb)
            selected = dominant_brick(instance, records)
            result = {"folder": folder.name, "image": path.name,
                      "truth": folder.name.rsplit("_", 1)[-1],
                      "status": "ok" if selected else "no_reliable_sam3_brick",
                      "label": "", "correct": "", "score_bad": "",
                      "approach5_score": "", "dinov2_score": "",
                      "baseline_margin_label": "", "zero_binary_label": "",
                      "minus_two_binary_label": "", "sam_score": "",
                      "mask_area_px": "", "box_xyxy": "", "rotated_rect": "",
                      "sam_touches_border": "",
                      "sam_candidates": len(records), "sam_rejected": len(rejected),
                      "policy": args.policy, "seconds": ""}
            mask = np.zeros(frame.shape[:2], np.uint8)
            annotated = frame.copy()
            if selected:
                mask, sam_record = selected
                prediction = classifier.classify(rgb, mask=mask)
                score = float(prediction["score_bad"])
                labels = labels_for_all(score)
                x, y, box_width, box_height = cv2.boundingRect(cv2.findNonZero(mask))
                result.update({"label": labels[args.policy],
                               "correct": int(labels[args.policy] == result["truth"]),
                               "score_bad": round(score, 6),
                               "approach5_score": round(float(prediction["approach5_score"]), 6),
                               "dinov2_score": round(float(prediction["dinov2_score"]), 6),
                               "baseline_margin_label": labels["baseline_margin"],
                               "zero_binary_label": labels["zero_binary"],
                               "minus_two_binary_label": labels["minus_two_binary"],
                               "sam_score": sam_record["score"],
                               "sam_touches_border": sam_record["touches_border"],
                               "mask_area_px": int(mask.sum()),
                               "box_xyxy": [x, y, x + box_width - 1, y + box_height - 1],
                               "rotated_rect": sam_record["min_area_rect"]})
                annotated = annotate_soft_boundary(frame, mask, result["label"], score)
            else:
                cv2.putText(annotated, "NO RELIABLE SAM3 BRICK MASK", (25, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, .9, (0, 190, 255), 2, cv2.LINE_AA)
            result["seconds"] = round(time.monotonic() - started, 3)
            if not cv2.imwrite(str(destination / path.name), annotated):
                raise OSError(f"Could not save annotated image for {path}")
            if not cv2.imwrite(str(mask_dir / f"{path.stem}.png"), mask * 255):
                raise OSError(f"Could not save mask for {path}")
            (destination / f"{path.stem}.json").write_text(json.dumps(result, indent=2) + "\n")
            rows.append(result)
            folder_rows.append(result)
            print(f"{folder.name} {path.name}: {result['label'] or result['status']} "
                  f"score={result['score_bad']} in {result['seconds']}s", flush=True)
        write_csv(destination / "predictions.csv", folder_rows)
    write_csv(args.output / "all_predictions.csv", rows)
    summary = {"input": str(args.input), "output": str(args.output),
               "sam_model": str(args.sam_model), "sam_prompt": args.prompt,
               "threshold_policy": args.policy, "total_images": len(rows),
               "classified": sum(row["status"] == "ok" for row in rows),
               "correct": sum(row["correct"] == 1 for row in rows),
               "per_folder": {folder.name: {
                   "images": sum(row["folder"] == folder.name for row in rows),
                   "classified": sum(row["folder"] == folder.name and row["status"] == "ok" for row in rows),
                   "correct": sum(row["folder"] == folder.name and row["correct"] == 1 for row in rows),
               } for folder in folders},
               "note": "These are the same physical bricks used to fit the frozen checkpoint; this is an in-sample SAM3-mask check, not brick-held-out generalisation."}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
