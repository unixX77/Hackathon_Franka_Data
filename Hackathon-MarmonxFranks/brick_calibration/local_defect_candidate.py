#!/usr/bin/env python3
"""Separate SAM3-mask quality candidate with local defect and DINOv2 features.

The Approach 17 checkpoint is used only as a source of frozen DINOv2 encoder
weights. Its fitted SVM scores are never used when evaluating this candidate.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from evaluate_approaches import (discover_images, lab_histogram,
                                 local_defect_features, texture_features,
                                 usable_pixels, write_csv)
from evaluate_dino_approaches import prepare_crop_rgb
from fusion_checkpoint import FusionClassifier


HERE = Path(__file__).resolve().parent
DATA = HERE / "brick_splits/images"
MASKS = HERE / "offline_results"
BASELINE = HERE / "approach_results/baseline_checkpoint"
OUTPUT = HERE / "approach_results/local_defect_candidate"
HANDCRAFTED_DIM = 68  # LAB 44 + texture 8 + local defects 16
C_VALUES = (.03, .3, 3.0)


def masked_features(rgb: np.ndarray, mask: np.ndarray, encoder: FusionClassifier) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    lab, usable, _ = usable_pixels(rgb, mask)
    base = np.r_[lab_histogram(lab, usable), texture_features(rgb, usable),
                 local_defect_features(rgb, mask)]
    if len(base) != HANDCRAFTED_DIM:
        raise ValueError(f"Unexpected handcrafted dimension: {len(base)}")
    dino = encoder._dino_features(prepare_crop_rgb(rgb, mask))
    return np.r_[base, dino].astype(np.float32)


def model_for_c(c: float):
    transformer = ColumnTransformer([
        ("handcrafted", "passthrough", slice(0, HANDCRAFTED_DIM)),
        ("dino", PCA(n_components=32, whiten=True, random_state=7),
         slice(HANDCRAFTED_DIM, None)),
    ])
    return make_pipeline(transformer, StandardScaler(),
                         SVC(kernel="linear", C=c, class_weight="balanced"))


def sample_weights(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Give each physical brick equal weight within each quality class."""
    weights = np.empty(len(y), np.float64)
    for label in (0, 1):
        class_groups = np.unique(groups[y == label])
        for group in class_groups:
            selected = (y == label) & (groups == group)
            weights[selected] = 1.0 / (2 * len(class_groups) * selected.sum())
    return weights * len(y)


def fit(x: np.ndarray, y: np.ndarray, groups: np.ndarray, c: float):
    model = model_for_c(c)
    model.fit(x, y, svc__sample_weight=sample_weights(y, groups))
    return model


def choose_c(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    """Select regularization using only the current outer training bricks."""
    folds = list(GroupKFold(n_splits=3).split(x, y, groups))
    ranked = []
    for c in C_VALUES:
        brick_accuracies = []
        for train, test in folds:
            model = fit(x[train], y[train], groups[train], c)
            scores = model.decision_function(x[test])
            for brick in np.unique(groups[test]):
                selected = groups[test] == brick
                brick_accuracies.append(float(np.mean((scores[selected] > 0) == y[test][selected])))
        ranked.append((float(np.mean(brick_accuracies)), -c, c))
    return max(ranked)[2]


def mask_path_for(row: dict) -> Path:
    return MASKS / row["brick_id"] / "masks" / row["path"].name


def feature_cache(rows: list[dict], cache: Path, baseline: Path) -> np.ndarray:
    paths = np.asarray([str(row["path"].resolve()) for row in rows])
    signatures = np.asarray([
        f"{row['path'].stat().st_mtime_ns}:{row['path'].stat().st_size}:"
        f"{mask_path_for(row).stat().st_mtime_ns}:{mask_path_for(row).stat().st_size}"
        for row in rows])
    if cache.is_file():
        saved = np.load(cache)
        if ("signatures" in saved and np.array_equal(saved["paths"], paths)
                and np.array_equal(saved["signatures"], signatures)):
            return saved["features"]
    encoder = FusionClassifier.load(baseline)
    features = []
    for index, row in enumerate(rows, 1):
        image = cv2.imread(str(row["path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode {row['path']}")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_path = mask_path_for(row)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != rgb.shape[:2]:
            raise ValueError(f"Missing or mismatched SAM3 mask: {mask_path}")
        features.append(masked_features(rgb, mask, encoder))
        if index % 25 == 0 or index == len(rows):
            print(f"Features: {index}/{len(rows)}", flush=True)
    result = np.vstack(features)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, features=result, paths=paths, signatures=signatures)
    return result


def evaluation(x: np.ndarray, y: np.ndarray, groups: np.ndarray,
               rows: list[dict], output: Path) -> dict:
    scores = np.full(len(y), np.nan)
    selected_c = []
    for train, test in LeaveOneGroupOut().split(x, y, groups):
        c = choose_c(x[train], y[train], groups[train])
        model = fit(x[train], y[train], groups[train], c)
        scores[test] = model.decision_function(x[test])
        selected_c.append({"held_out_brick": str(groups[test][0]), "C": c})
        print(f"Held out {groups[test][0]}: {sum((scores[test] > 0) == y[test])}/{len(test)}", flush=True)
    pred = scores > 0
    matrix = confusion_matrix(y, pred, labels=[0, 1])
    per_brick = []
    for group in np.unique(groups):
        chosen = groups == group
        median = float(np.median(scores[chosen]))
        per_brick.append({"brick_id": group, "true_label": "bad" if y[chosen][0] else "good",
                          "median_score": median, "predicted_label": "bad" if median > 0 else "good",
                          "correct_images": int(sum(pred[chosen] == y[chosen])),
                          "image_count": int(chosen.sum())})
    payload = {"evaluation": "leave-one-physical-brick-out; inner group folds choose C",
               "threshold": 0.0, "image_accuracy": float(np.mean(pred == y)),
               "image_correct": int(sum(pred == y)), "image_count": len(y),
               "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
               "good_recall": float(matrix[0, 0] / matrix[0].sum()),
               "bad_recall": float(matrix[1, 1] / matrix[1].sum()),
               "confusion_matrix": matrix.tolist(),
               "brick_correct": sum(row["true_label"] == row["predicted_label"] for row in per_brick),
               "brick_count": len(per_brick),
               "baseline_approach17_different_mask_image_accuracy": 212 / 258}
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "held_out_image_predictions.csv", [
        {"brick_id": row["brick_id"], "image": row["path"].name,
         "truth": row["label_name"], "score_bad": float(scores[i]),
         "prediction": "bad" if pred[i] else "good", "correct": int(pred[i] == y[i])}
        for i, row in enumerate(rows)])
    write_csv(output / "held_out_brick_predictions.csv", per_brick)
    write_csv(output / "selected_regularization.csv", selected_c)
    (output / "evaluation.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


class LocalDefectCandidate:
    """Load the separately trained candidate for one masked live brick."""

    def __init__(self, checkpoint: Path = OUTPUT, baseline: Path = BASELINE):
        self.model = joblib.load(checkpoint / "candidate_svm.joblib")
        self.metadata = json.loads((checkpoint / "metadata.json").read_text())
        self.encoder = FusionClassifier.load(baseline)
        self.device = self.encoder.device

    def classify(self, rgb: np.ndarray, mask: np.ndarray) -> dict:
        features = masked_features(rgb, mask, self.encoder)
        score = float(self.model.decision_function(features[None])[0])
        threshold = float(self.metadata["threshold"])
        return {"score_bad": score, "label": "bad" if score > threshold else "good",
                "threshold": threshold, "local_defect_features": features[52:68].tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    args = parser.parse_args()
    rows = discover_images(args.data)
    x = feature_cache(rows, args.output / "sam3_mask_features.npz", args.baseline)
    y = np.asarray([row["label"] for row in rows], int)
    groups = np.asarray([row["brick_id"] for row in rows])
    result = evaluation(x, y, groups, rows, args.output)
    c = choose_c(x, y, groups)
    model = fit(x, y, groups, c)
    joblib.dump(model, args.output / "candidate_svm.joblib")
    metadata = {"name": "local_defect_candidate", "feature_dimension": int(x.shape[1]),
                "handcrafted_dimension": HANDCRAFTED_DIM, "threshold": 0.0,
                "C": c, "n_images": len(y), "n_physical_bricks": len(np.unique(groups)),
                "feature_source": "SAM3 masks; LAB histogram, texture, local defects, frozen DINOv2 patch distribution",
                "label_limit": "Only brick-level labels exist; a hidden defect may make a bad-brick image look good.",
                "evaluation_file": "evaluation.json"}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
