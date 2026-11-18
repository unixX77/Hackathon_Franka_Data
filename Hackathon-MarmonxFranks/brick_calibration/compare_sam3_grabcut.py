#!/usr/bin/env python3
"""Compare frozen Approach 17 scores on matched SAM3 and cached GrabCut inputs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import numpy as np

from evaluate_approaches import discover_images
from threshold_policies.policy import label_for, load_policy


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "offline_results"
DATA = ROOT / "brick_splits/images"
CACHE = ROOT / "approach_results"
CHECKPOINT = CACHE / "baseline_checkpoint"
POLICY = load_policy("minus_two_binary")


def main() -> None:
    images = discover_images(DATA)
    a5_cache = np.load(CACHE / "extracted_features.npz")
    dino_cache = np.load(CACHE / "dino_feature_cache/dinov2_features.npz")
    manifest = list(csv.DictReader((CACHE / "dino_feature_cache/dinov2_input_manifest.csv").open()))
    if len(images) != len(manifest) or len(images) != len(a5_cache["groups"]):
        raise ValueError("Feature caches do not match the image dataset length")
    for image, group, entry in zip(images, a5_cache["groups"], manifest):
        if (image["brick_id"] != group or image["brick_id"] != entry["brick_id"] or
                image["path"].name != Path(entry["file"]).name):
            raise ValueError("Feature cache and image order differ")
    a5 = joblib.load(CHECKPOINT / "approach5_svm.joblib")
    dino = joblib.load(CHECKPOINT / "dinov2_patch_svm.joblib")
    grabcut_scores = (a5.decision_function(a5_cache["masked_histogram_texture"]) +
                      dino.decision_function(dino_cache["patch_distribution"])) / 2
    sam_rows = {(row["folder"], row["image"]): row for row in
                csv.DictReader((RESULTS / "all_predictions.csv").open())}
    rows = []
    for image, old_score in zip(images, grabcut_scores):
        key = (image["brick_id"], image["path"].name)
        new = sam_rows[key]
        truth = image["label_name"]
        old_label = label_for(float(old_score), POLICY)
        new_label = new["label"]
        rows.append({"folder": key[0], "image": key[1], "truth": truth,
                     "grabcut_score_bad": round(float(old_score), 6),
                     "grabcut_label": old_label,
                     "grabcut_correct": int(old_label == truth),
                     "sam3_score_bad": new["score_bad"],
                     "sam3_label": new_label,
                     "sam3_correct": int(new_label == truth),
                     "sam3_mask_path": f"{key[0]}/masks/{image['path'].stem}.png"})
    with (RESULTS / "mask_input_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    folders = sorted({row["folder"] for row in rows}, key=lambda name: int(name.split("_")[1]))
    report = {
        "note": "Same 258 familiar photos, same frozen full-data checkpoint and -2 policy. GrabCut uses previously cached features. These are not held-out estimates.",
        "images": len(rows),
        "grabcut_correct": sum(row["grabcut_correct"] for row in rows),
        "sam3_correct": sum(row["sam3_correct"] for row in rows),
        "changed_labels": sum(row["grabcut_label"] != row["sam3_label"] for row in rows),
        "per_folder": {folder: {
            "images": sum(row["folder"] == folder for row in rows),
            "grabcut_correct": sum(row["folder"] == folder and row["grabcut_correct"] for row in rows),
            "sam3_correct": sum(row["folder"] == folder and row["sam3_correct"] for row in rows),
        } for folder in folders},
    }
    (RESULTS / "mask_input_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
