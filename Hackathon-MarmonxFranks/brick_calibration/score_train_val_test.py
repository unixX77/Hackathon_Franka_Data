#!/usr/bin/env python3
"""Train, validation and test accuracy of Approaches 5 and 17 on every split.

Uses the same C selection as ``evaluate_on_splits.py``:

* train score      - model fitted on the training bricks (chosen C), scored on them;
* validation score - that same model scored on the two validation bricks;
* test score       - model refitted on train + validation, scored on the test bricks.

Approach 17 is the mean of the Approach 5 and DINOv2 decision scores at each
stage. Accuracy uses threshold 0; balanced accuracy averages good and bad images.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate_approaches import discover_images, write_csv
from evaluate_improvements import svm_model
from evaluate_on_splits import MEMBERS, fit_member


def accuracy(scores, labels):
    predicted = scores >= 0
    truth = labels.astype(bool)
    return (float(np.mean(predicted == truth)),
            float(np.mean([np.mean(~predicted[~truth]), np.mean(predicted[truth])])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    records = discover_images(args.data)
    labels = np.asarray([r["label"] for r in records])
    groups = np.asarray([r["brick_id"] for r in records])
    files = {str(r["path"]): i for i, r in enumerate(records)}
    approach5 = np.load(args.results / "extracted_features.npz")
    dino = np.load(args.results / "dino_feature_cache" / "dinov2_features.npz")
    features = {"masked_histogram_texture": approach5["masked_histogram_texture"],
                "patch_distribution": dino["patch_distribution"]}
    output = args.results / "split_evaluation"
    summary = {}
    for family in ("without_brick6", "with_brick6"):
        manifest = list(csv.DictReader((args.splits / family / "split_manifest.csv").open()))
        assignment = json.loads((args.splits / family / "split_assignment.json").read_text())
        rows = []
        for split in assignment:
            subset = {part: np.asarray(sorted(files[r["source_file"]] for r in manifest
                                              if r["split"] == split["split"] and r["subset"] == part), int)
                      for part in ("train", "val", "test")}
            stage1, final = {}, {}
            for member, (key, pca) in MEMBERS.items():
                x = features[key]
                refit, c, _ = fit_member(x, pca, labels, groups, subset["train"], subset["val"])
                train_model = svm_model(pca).set_params(svc__C=c).fit(x[subset["train"]], labels[subset["train"]])
                stage1[member] = train_model.decision_function(x)
                final[member] = refit.decision_function(x)
            stage1["approach17"] = (stage1["approach5"] + stage1["dinov2_patch"]) / 2
            final["approach17"] = (final["approach5"] + final["dinov2_patch"]) / 2
            record = {"split": split["split"]}
            for approach in ("approach5", "approach17"):
                for part, scores in (("train", stage1), ("val", stage1), ("test", final)):
                    acc, bal = accuracy(scores[approach][subset[part]], labels[subset[part]])
                    record[f"{approach}_{part}_accuracy"] = acc
                    record[f"{approach}_{part}_balanced_accuracy"] = bal
                    record[f"{approach}_{part}_correct"] = int(round(acc * len(subset[part])))
                    record[f"{approach}_{part}_images"] = len(subset[part])
            rows.append(record)
        write_csv(output / family / "train_val_test_scores.csv", rows)
        summary[family] = {}
        for approach in ("approach5", "approach17"):
            entry = {}
            for part in ("train", "val", "test"):
                correct = sum(r[f"{approach}_{part}_correct"] for r in rows)
                total = sum(r[f"{approach}_{part}_images"] for r in rows)
                per_split = [r[f"{approach}_{part}_accuracy"] for r in rows]
                entry[part] = {"pooled_accuracy": correct / total, "correct": correct, "images": total,
                               "mean_balanced_accuracy": float(np.mean([r[f"{approach}_{part}_balanced_accuracy"] for r in rows])),
                               "split_min": float(np.min(per_split)), "split_max": float(np.max(per_split))}
            summary[family][approach] = entry
    (output / "train_val_test_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
