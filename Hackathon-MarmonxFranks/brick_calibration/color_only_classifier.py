#!/usr/bin/env python3
"""SAM3-mask brick quality classification using only LAB color histograms.

The supervised decision uses no texture, crack, shape, depth, or DINO features.
SAM3 is used upstream to obtain the mask; this module classifies that mask.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from evaluate_approaches import discover_images, lab_histogram, usable_pixels, write_csv

HERE = Path(__file__).resolve().parent
DATA = HERE / "brick_splits/images"
MASKS = HERE / "offline_results"
OUTPUT = HERE / "approach_results/color_only"
C_VALUES = (.01, .1, 1.0, 10.0)
VARIANTS = ("lab_histogram", "ab_histogram")


def extract_color(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    lab, usable, _ = usable_pixels(rgb, mask)
    return lab_histogram(lab, usable).astype(np.float32)


def select_variant(x: np.ndarray, variant: str) -> np.ndarray:
    if variant == "lab_histogram":
        return x
    if variant == "ab_histogram":
        return x[..., 12:]
    raise ValueError(f"Unknown color variant {variant}")


def sample_weights(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    weights = np.empty(len(y), dtype=float)
    for label in (0, 1):
        group_ids = np.unique(groups[y == label])
        for group in group_ids:
            indices = (y == label) & (groups == group)
            weights[indices] = 1.0 / (2 * len(group_ids) * indices.sum())
    return weights * len(y)


def fit(x: np.ndarray, y: np.ndarray, groups: np.ndarray, variant: str, c: float):
    model = make_pipeline(StandardScaler(), SVC(kernel="linear", C=c))
    model.fit(select_variant(x, variant), y, svc__sample_weight=sample_weights(y, groups))
    return model


def choose(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[str, float]:
    """Choose color features and regularization using only training bricks."""
    folds = list(GroupKFold(n_splits=3).split(x, y, groups))
    candidates = []
    for variant in VARIANTS:
        for c in C_VALUES:
            accuracy_by_brick = []
            for train, val in folds:
                model = fit(x[train], y[train], groups[train], variant, c)
                prediction = model.decision_function(select_variant(x[val], variant)) > 0
                for group in np.unique(groups[val]):
                    subset = groups[val] == group
                    accuracy_by_brick.append(float(np.mean(prediction[subset] == y[val][subset])))
            candidates.append((float(np.mean(accuracy_by_brick)), -c,
                               variant == "ab_histogram", variant, c))
    _, _, _, variant, c = max(candidates)
    return variant, c


def color_features(rows: list[dict], data_root: Path, masks_root: Path) -> np.ndarray:
    # The local-defect feature cache contains the exact same first 44 color
    # histograms. Validate source signatures before reusing them.
    cache_path = HERE / "approach_results/local_defect_candidate/sam3_mask_features.npz"
    if cache_path.exists():
        with np.load(cache_path) as cache:
            paths = np.asarray([str(row["path"].resolve()) for row in rows])
            signatures = []
            for row in rows:
                img = row["path"]
                mask = masks_root / row["brick_id"] / "masks" / img.name
                a, b = img.stat(), mask.stat()
                signatures.append(f"{a.st_mtime_ns}:{a.st_size}:{b.st_mtime_ns}:{b.st_size}")
            if ("signatures" in cache and np.array_equal(paths, cache["paths"])
                    and np.array_equal(np.asarray(signatures), cache["signatures"])):
                return np.asarray(cache["features"][:, :44], dtype=np.float32)
    features = []
    for index, row in enumerate(rows, 1):
        image = cv2.imread(str(row["path"]), cv2.IMREAD_COLOR)
        mask_path = masks_root / row["brick_id"] / "masks" / row["path"].name
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None or image.shape[:2] != mask.shape:
            raise ValueError(f"Missing or mismatched image/mask: {row['path']} {mask_path}")
        features.append(extract_color(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), mask))
        if index % 25 == 0 or index == len(rows):
            print(f"Extracted color features {index}/{len(rows)}", flush=True)
    return np.vstack(features)


def evaluate(x: np.ndarray, y: np.ndarray, groups: np.ndarray, rows: list[dict], output: Path) -> dict:
    scores = np.empty(len(y), dtype=float)
    folds = []
    for train, test in LeaveOneGroupOut().split(x, y, groups):
        variant, c = choose(x[train], y[train], groups[train])
        model = fit(x[train], y[train], groups[train], variant, c)
        scores[test] = model.decision_function(select_variant(x[test], variant))
        folds.append({"held_out_brick": groups[test][0], "color_variant": variant, "C": c})
        print(groups[test][0], 'correct', int(np.sum((scores[test] > 0) == y[test])), '/', len(test),
              'variant', variant, 'C', c, flush=True)
    predictions = scores > 0
    matrix = confusion_matrix(y, predictions, labels=[0, 1])
    per_brick = []
    for group in np.unique(groups):
        chosen = groups == group
        actual = int(y[chosen][0])
        median_score = float(np.median(scores[chosen]))
        per_brick.append({"brick_id": group, "truth": "bad" if actual else "good",
                          "n_images": int(chosen.sum()),
                          "correct_images": int(np.sum(predictions[chosen] == y[chosen])),
                          "predicted_good": int(np.sum(~predictions[chosen])),
                          "predicted_bad": int(np.sum(predictions[chosen])),
                          "median_score": median_score,
                          "brick_prediction": "bad" if median_score > 0 else "good"})
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "held_out_image_predictions.csv", [
        {"brick_id": row["brick_id"], "image": row["path"].name,
         "truth": row["label_name"], "score_bad": float(scores[i]),
         "prediction": "bad" if predictions[i] else "good",
         "correct": int(predictions[i] == y[i])} for i, row in enumerate(rows)])
    write_csv(output / "held_out_brick_predictions.csv", per_brick)
    write_csv(output / "selected_parameters.csv", folds)
    result = {"evaluation": "leave-one-physical-brick-out; inner grouped selection",
              "n_images": len(y), "n_bricks": len(np.unique(groups)),
              "correct_images": int(np.sum(predictions == y)),
              "image_accuracy": float(np.mean(predictions == y)),
              "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
              "confusion_matrix": matrix.tolist(),
              "good_recall": float(matrix[0, 0] / matrix[0].sum()),
              "bad_recall": float(matrix[1, 1] / matrix[1].sum()),
              "brick_correct": sum(r["truth"] == r["brick_prediction"] for r in per_brick),
              "threshold": 0.0}
    (output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


class ColorOnlyClassifier:
    def __init__(self, checkpoint: Path = OUTPUT):
        self.model = joblib.load(checkpoint / "color_svm.joblib")
        self.metadata = json.loads((checkpoint / "metadata.json").read_text())

    def classify(self, rgb: np.ndarray, mask: np.ndarray) -> dict:
        color = extract_color(rgb, mask)[None]
        score = float(self.model.decision_function(select_variant(color, self.metadata["variant"]))[0])
        return {"score_bad": score, "label": "bad" if score > 0 else "good"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--masks", type=Path, default=MASKS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    rows = discover_images(args.data)
    x = color_features(rows, args.data, args.masks)
    y = np.asarray([row["label"] for row in rows], dtype=int)
    groups = np.asarray([row["brick_id"] for row in rows])
    result = evaluate(x, y, groups, rows, args.output)
    variant, c = choose(x, y, groups)
    model = fit(x, y, groups, variant, c)
    joblib.dump(model, args.output / "color_svm.joblib")
    metadata = {"name": "color_only", "variant": variant, "C": c, "threshold": 0.0,
                "features": "LAB color histograms inside SAM3 mask; no texture or DINOv2",
                "n_images": len(y), "n_physical_bricks": len(np.unique(groups)),
                "evaluation_file": "evaluation.json"}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
