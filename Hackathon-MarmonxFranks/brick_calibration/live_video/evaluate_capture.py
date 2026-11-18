#!/usr/bin/env python3
"""Evaluate labelled Mac camera captures with the frozen and guarded decisions."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2

from model import LiveVideoClassifier


def metrics(rows: list[dict], label_key: str) -> dict:
    decided = [row for row in rows if row[label_key] in ("good", "bad")]
    bad = [row for row in rows if row["truth"] == "bad"]
    return {
        "images": len(rows),
        "decided": len(decided),
        "coverage": len(decided) / len(rows) if rows else None,
        "accuracy_all_uncertain_as_wrong": sum(row[label_key] == row["truth"] for row in rows) / len(rows) if rows else None,
        "accuracy_when_decided": sum(row[label_key] == row["truth"] for row in decided) / len(decided) if decided else None,
        "bad_recall_all_uncertain_as_missed": sum(row[label_key] == "bad" for row in bad) / len(bad) if bad else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/cache/s95vkond/live_video_capture"))
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    args = parser.parse_args()
    paths = sorted(path for path in args.data.glob("Brick_*_*/*/frame_*.jpg") if path.is_file())
    if not paths:
        parser.error(f"No labelled capture frames under {args.data}")
    model = LiveVideoClassifier()
    rows = []
    for index, path in enumerate(paths, 1):
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Could not decode {path}")
        truth = path.parent.parent.name.rsplit("_", 1)[-1]
        if truth not in ("good", "bad"):
            raise ValueError(f"Unknown truth label for {path}")
        result = model.predict([frame], args.roi)
        raw_label = (model.classifier._decide(result["score_bad"])
                     if "score_bad" in result else "uncertain")
        rows.append({"brick": path.parent.parent.name, "session": path.parent.name,
                     "file": str(path), "truth": truth,
                     "baseline_label": raw_label, "live_label": result["label"],
                     "score_bad": result.get("score_bad", ""),
                     "approach5_score": result.get("approach5_score", ""),
                     "dinov2_score": result.get("dinov2_score", ""),
                     "mask_fraction": result.get("mask_fraction", ""),
                     "quality_flags": ";".join(result["quality_flags"])})
        print(f"{index}/{len(paths)} {path.parent.parent.name} {result['label']}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "camera_predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["brick"]].append(row)
    report = {"source": str(args.data), "roi": args.roi,
              "baseline_on_same_crops": metrics(rows, "baseline_label"),
              "guarded_live_video": metrics(rows, "live_label"),
              "per_brick": {brick: metrics(items, "live_label")
                            for brick, items in sorted(grouped.items())}}
    (args.output / "camera_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
