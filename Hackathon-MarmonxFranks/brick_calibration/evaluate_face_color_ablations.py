#!/usr/bin/env python3
"""Evaluate four feature additions separately using identical grouped folds.

The shared control is a single supervised SVM over original Approach 5
features and the original-orientation frozen DINO distribution. It is NOT the
official two-SVM Approach 17, and each model selects its threshold inside the
training folds. These exploratory comparisons do not select the final model.
"""
from __future__ import annotations

import json
import numpy as np
from threadpoolctl import threadpool_limits

from evaluate_approaches import discover_images, write_csv
from face_color_rotation_candidate import DATA, OUTPUT, evaluate, feature_cache
from local_defect_candidate import BASELINE, HERE, feature_cache as old_features


def main() -> None:
    rows = discover_images(DATA)
    features, diagnostics = feature_cache(rows, OUTPUT)
    old = old_features(rows, HERE / "approach_results/local_defect_candidate/sam3_mask_features.npz", BASELINE)
    if not np.allclose(features[:, :52], old[:, :52], atol=1e-4, rtol=1e-5):
        raise ValueError("Original SAM3-mask feature caches do not match")
    y = np.asarray([row["label"] for row in rows], dtype=int)
    groups = np.asarray([row["brick_id"] for row in rows])
    base = features.copy()
    base[:, 52:133] = 0
    base[:, 133:] = old[:, 68:]
    inputs = {"baseline_control": base}
    for name, section in (("view_proxy", slice(52, 60)),
                          ("material_color", slice(60, 113)),
                          ("local_color", slice(113, 133))):
        x = base.copy(); x[:, section] = features[:, section]; inputs[name] = x
    rotated = base.copy(); rotated[:, 133:] = features[:, 133:]; inputs["rotation_dino"] = rotated
    inputs["all_four"] = features
    matrix = []
    with threadpool_limits(limits=1):
        for name, x in inputs.items():
            print(f"Ablation: {name}", flush=True)
            existing = (OUTPUT / "evaluation.json" if name == "all_four"
                        else OUTPUT / "ablations" / name / "evaluation.json")
            if existing.is_file():
                result = json.loads(existing.read_text())
                print(f"Reusing completed evaluation: {existing}", flush=True)
            else:
                result = evaluate(x, y, groups, rows, diagnostics, OUTPUT / "ablations" / name,
                                  save_images=False)
            matrix.append({"approach": name, "correct": result["image_correct"],
                           "total": result["n_images"], "accuracy": result["image_accuracy"],
                           "good_recall": result["good_recall"], "bad_recall": result["bad_recall"],
                           "brick_correct": result["brick_correct"]})
    write_csv(OUTPUT / "feature_ablation_matrix.csv", matrix)
    print(json.dumps(matrix, indent=2), flush=True)


if __name__ == "__main__":
    main()
