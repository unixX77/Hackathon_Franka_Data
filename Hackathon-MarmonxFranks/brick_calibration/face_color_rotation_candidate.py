#!/usr/bin/env python3
"""Experimental SAM3-mask face, material color, local color and rotation model.

Face geometry is an unvalidated view proxy, not a trained physical-face label.
The four image rotations are in-plane, not views of hidden physical faces.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from canonical_surface import canonicalize_masked_object
from evaluate_approaches import (discover_images, lab_histogram, texture_features,
                                 usable_pixels, write_csv)
from evaluate_dino_approaches import prepare_crop_rgb
from fusion_checkpoint import FusionClassifier
from good_brick_clusters import optimize_threshold
from local_defect_candidate import BASELINE, DATA, MASKS, sample_weights

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "approach_results/face_color_rotation_candidate"
FEATURE_CACHE = Path("/cache/s95vkond/brick_feature_cache/face_color_rotation_features.npz")
VERSION = "face-color-rotation-v2"
HAND_DIM = 133  # 52 original + 8 view + 53 material LAB + 20 local color
DINO_DIM = 1152
C_VALUES = (.03, .3, 3.0)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    return lab


def appearance_features(rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict, np.ndarray, np.ndarray]:
    """Extract baseline, view proxy, and two groups of material color features."""
    mask = (mask > 0).astype(np.uint8)
    lab_full, usable, _ = usable_pixels(rgb, mask)
    original = np.r_[lab_histogram(lab_full, usable), texture_features(rgb, usable)]
    points = cv2.findNonZero(mask)
    if points is None:
        raise ValueError("Empty SAM3 mask")
    (_, _), (w, h), angle = cv2.minAreaRect(points)
    long_axis, short_axis = max(w, h), min(w, h)
    if long_axis <= 1 or short_axis <= 1:
        raise ValueError("Degenerate SAM3 mask")
    angle = float(angle + (90 if w < h else 0))
    crop, crop_mask = canonicalize_masked_object(rgb, mask)
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    inner = cv2.erode(crop_mask, np.ones((3, 3), np.uint8), iterations=1) > 0
    if inner.sum() < 100:
        inner = crop_mask > 0
    # Heuristic for clean brick material: rejects dark holes/shadows, very
    # pale background/highlights and pixels at the mask perimeter.
    # Relative brightness helps exclude hole interiors across brighter and
    # darker bricks. This is heuristic and can also exclude shaded material.
    value_floor = max(35.0, .85 * float(np.median(hsv[..., 2][inner])))
    material = inner & (hsv[..., 2] >= value_floor) & (hsv[..., 2] < 245) & (hsv[..., 1] >= 25)
    fallback = int(material.sum()) < 500
    if fallback:
        material = inner & (hsv[..., 2] > 18) & (hsv[..., 2] < 248)
    if material.sum() < 100:
        raise ValueError("Too few pixels for material color")
    lab = rgb_to_lab(crop)
    histogram = lab_histogram(lab, material)
    percentiles = np.percentile(lab[material], [10, 50, 90], axis=0).ravel()
    material_features = np.r_[histogram, percentiles]

    # Coarse aligned regions are summarized orderlessly: percentiles across
    # local LAB medians preserve local color variation without binding the
    # classifier to the left/right direction of a rotated brick.
    medians = []
    for y0, y1 in zip(np.linspace(0, crop.shape[0], 3, dtype=int)[:-1],
                      np.linspace(0, crop.shape[0], 3, dtype=int)[1:]):
        for x0, x1 in zip(np.linspace(0, crop.shape[1], 5, dtype=int)[:-1],
                          np.linspace(0, crop.shape[1], 5, dtype=int)[1:]):
            valid = material[y0:y1, x0:x1]
            if valid.sum() >= 30:
                medians.append(np.median(lab[y0:y1, x0:x1][valid], axis=0))
    if not medians:
        medians = [np.median(lab[material], axis=0)]
    medians = np.asarray(medians)
    local = np.r_[np.percentile(medians, [10, 25, 50, 75, 90], axis=0).ravel(),
                  np.std(medians, axis=0), len(medians) / 8, float(material.sum() / max(1, inner.sum()))]

    # Dark enclosed components can be holes, surface patches or shadow. Their
    # counts/coverage are proxies for visible face geometry, not face labels.
    dark = (inner & (hsv[..., 2] < value_floor)).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, count)
             if 12 <= stats[i, cv2.CC_STAT_AREA] <= inner.sum() * .15]
    theta = np.deg2rad(angle)
    view = np.asarray([short_axis / long_axis, float(mask.sum() / max(1, w * h)),
                       float(dark.sum() / max(1, inner.sum())), min(len(areas), 12) / 12,
                       max(areas, default=0) / max(1, inner.sum()),
                       sum(areas) / max(1, inner.sum()), np.cos(2 * theta), np.sin(2 * theta)])
    result = np.r_[original, view, material_features, local].astype(np.float32)
    if len(result) != HAND_DIM or not np.all(np.isfinite(result)):
        raise ValueError(f"Invalid handcrafted feature vector: {len(result)}")
    metadata = {"axis_angle_deg": angle, "aspect_ratio": float(short_axis / long_axis),
                "dark_region_count": len(areas), "material_fraction": float(material.sum() / max(1, inner.sum())),
                "material_value_floor": value_floor,
                "material_mask_fallback": fallback}
    return result, metadata, crop, material.astype(np.uint8)


@torch.inference_mode()
def rotation_features(rgb: np.ndarray, mask: np.ndarray, encoder: FusionClassifier) -> np.ndarray:
    crop = prepare_crop_rgb(rgb, (mask > 0).astype(np.uint8))
    rotated = [np.ascontiguousarray(np.rot90(crop, k=k)) for k in range(4)]
    pixels = encoder.processor(images=rotated, return_tensors="pt")["pixel_values"].to(encoder.device)
    with torch.autocast(device_type="cuda", dtype=torch.float16,
                        enabled=encoder.device.startswith("cuda")):
        hidden = encoder.encoder(pixel_values=pixels).last_hidden_state
    patches = hidden.float().cpu().numpy()[:, 1:]
    per_rotation = np.concatenate((patches.mean(axis=1), patches.std(axis=1),
                                   np.percentile(patches, 90, axis=1)), axis=1)
    return per_rotation.mean(axis=0).astype(np.float32)


def extract_features(rgb: np.ndarray, mask: np.ndarray, encoder: FusionClassifier) -> tuple[np.ndarray, dict, np.ndarray, np.ndarray]:
    hand, view, crop, material = appearance_features(rgb, mask)
    dino = rotation_features(rgb, mask, encoder)
    return np.r_[hand, dino].astype(np.float32), view, crop, material


def feature_cache(rows: list[dict], output: Path) -> tuple[np.ndarray, list[dict]]:
    cache = FEATURE_CACHE
    paths = np.asarray([str(row["path"].resolve()) for row in rows])
    signatures = []
    for row in rows:
        image = row["path"]
        mask = MASKS / row["brick_id"] / "masks" / image.name
        a, b = image.stat(), mask.stat()
        signatures.append(f"{a.st_mtime_ns}:{a.st_size}:{b.st_mtime_ns}:{b.st_size}")
    if cache.is_file() and (output / "view_diagnostics.json").exists():
        with np.load(cache) as data:
            if (str(data["version"]) == VERSION and np.array_equal(paths, data["paths"])
                    and np.array_equal(np.asarray(signatures), data["signatures"])):
                return data["features"], json.loads((output / "view_diagnostics.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    from transformers.utils import logging as transformers_logging
    transformers_logging.disable_progress_bar()
    encoder = FusionClassifier.load(BASELINE)
    features, diagnostics = [], []
    saved = set()
    for i, row in enumerate(rows, 1):
        bgr = cv2.imread(str(row["path"]))
        mask_path = MASKS / row["brick_id"] / "masks" / row["path"].name
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if bgr is None or mask is None or bgr.shape[:2] != mask.shape:
            raise ValueError(f"Missing or invalid SAM3 mask for {row['path']}")
        feat, view, crop, material = extract_features(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), mask, encoder)
        features.append(feat)
        diagnostics.append({"brick_id": row["brick_id"], "image": row["path"].name, **view})
        if row["brick_id"] not in saved:
            tint = crop.copy(); tint[material > 0] = (tint[material > 0] * .7 + np.array([30, 200, 40]) * .3).astype(np.uint8)
            preview = np.hstack((crop, tint))
            target = output / "material_mask_examples" / f"{row['brick_id']}.jpg"
            target.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(target), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)):
                raise OSError(f"Could not save material-mask example: {target}")
            saved.add(row["brick_id"])
        if i % 25 == 0 or i == len(rows):
            print(f"Extracted four-feature vectors: {i}/{len(rows)}", flush=True)
    x = np.vstack(features)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, features=x, version=VERSION, paths=paths,
                        signatures=np.asarray(signatures))
    (output / "view_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    return x, diagnostics


def make_model(c: float):
    transform = ColumnTransformer([
        ("appearance", "passthrough", slice(0, HAND_DIM)),
        ("rotation_dino", PCA(n_components=16, whiten=True, random_state=7,
                              svd_solver="randomized"), slice(HAND_DIM, None)),
    ])
    return make_pipeline(transform, StandardScaler(), SVC(kernel="linear", C=c))


def fit(x: np.ndarray, y: np.ndarray, groups: np.ndarray, c: float):
    model = make_model(c)
    model.fit(x, y, svc__sample_weight=sample_weights(y, groups))
    return model


def select_parameters(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    """Select C and threshold solely from grouped training-fold predictions."""
    # Every physical brick belongs to one class. Distribute class groups
    # across three folds so both labels remain represented in train/validation.
    by_class = [np.unique(groups[y == label]) for label in (0, 1)]
    if min(map(len, by_class)) < 3:
        raise ValueError("At least three physical bricks per class are required for inner folds")
    splits = []
    for fold in range(3):
        validation_groups = np.concatenate([values[fold::3] for values in by_class])
        val = np.flatnonzero(np.isin(groups, validation_groups))
        train = np.flatnonzero(~np.isin(groups, validation_groups))
        splits.append((train, val))
    candidates = []
    for c in C_VALUES:
        scores = np.empty(len(y))
        for train, val in splits:
            model = fit(x[train], y[train], groups[train], c)
            scores[val] = model.decision_function(x[val])
        threshold, metrics = optimize_threshold(scores, y, groups)
        candidates.append({"C": c, "threshold": threshold, "inner_metrics": metrics})
    return max(candidates, key=lambda z: (z["inner_metrics"]["balanced_group_accuracy"],
                                          z["inner_metrics"]["bad_group_accuracy"], -z["C"]))


def evaluate(x: np.ndarray, y: np.ndarray, groups: np.ndarray, rows: list[dict],
             diagnostics: list[dict], output: Path, save_images: bool = True) -> dict:
    scores, thresholds = np.empty(len(y)), np.empty(len(y))
    folds = []
    for train, test in LeaveOneGroupOut().split(x, y, groups):
        chosen = select_parameters(x[train], y[train], groups[train])
        model = fit(x[train], y[train], groups[train], chosen["C"])
        scores[test] = model.decision_function(x[test])
        thresholds[test] = chosen["threshold"]
        folds.append({"held_out_brick": groups[test][0], "C": chosen["C"],
                      "threshold": chosen["threshold"],
                      "inner_balanced_group_accuracy": chosen["inner_metrics"]["balanced_group_accuracy"]})
        print(f"Held out {groups[test][0]}: {int(np.sum((scores[test] > thresholds[test]) == y[test]))}/{len(test)}", flush=True)
    pred = scores > thresholds
    matrix = confusion_matrix(y, pred, labels=[0, 1])
    result = {"method": "outer leave-one-physical-brick-out; inner grouped C and threshold selection",
              "n_images": len(y), "n_bricks": len(np.unique(groups)),
              "image_correct": int(np.sum(pred == y)), "image_accuracy": float(np.mean(pred == y)),
              "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
              "good_recall": float(matrix[0, 0] / matrix[0].sum()),
              "bad_recall": float(matrix[1, 1] / matrix[1].sum()),
              "confusion_matrix_good_bad": matrix.tolist(),
              "material_mask_fallbacks": sum(int(r["material_mask_fallback"]) for r in diagnostics)}
    image_rows = []
    for i, row in enumerate(rows):
        image_rows.append({"brick_id": row["brick_id"], "image": row["path"].name,
                           "truth": row["label_name"], "raw_score": float(scores[i]),
                           "threshold": float(thresholds[i]),
                           "score_bad": float(scores[i] - thresholds[i]),
                           "prediction": "bad" if pred[i] else "good", "correct": int(pred[i] == y[i]),
                           **{k: diagnostics[i][k] for k in ("axis_angle_deg", "aspect_ratio", "dark_region_count", "material_fraction", "material_mask_fallback")}})
    brick_rows = []
    for group in np.unique(groups):
        sel = groups == group
        median = float(np.median(scores[sel] - thresholds[sel]))
        brick_rows.append({"brick_id": group, "truth": "bad" if y[sel][0] else "good",
                           "images": int(sel.sum()), "correct_images": int(np.sum(pred[sel] == y[sel])),
                           "predicted_good": int(np.sum(~pred[sel])), "predicted_bad": int(np.sum(pred[sel])),
                           "median_score_bad": median, "brick_prediction": "bad" if median > 0 else "good"})
    result["brick_correct"] = sum(r["truth"] == r["brick_prediction"] for r in brick_rows)
    output.mkdir(parents=True, exist_ok=True)
    if save_images:
        write_csv(output / "held_out_image_predictions.csv", image_rows)
    write_csv(output / "held_out_brick_predictions.csv", brick_rows)
    write_csv(output / "held_out_fold_parameters.csv", folds)
    (output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


class FaceColorRotationClassifier:
    def __init__(self, checkpoint: Path = OUTPUT, baseline: Path = BASELINE):
        self.model = joblib.load(checkpoint / "candidate_svm.joblib")
        self.metadata = json.loads((checkpoint / "metadata.json").read_text())
        self.encoder = FusionClassifier.load(baseline)

    def classify(self, rgb: np.ndarray, mask: np.ndarray) -> dict:
        features, view, _, _ = extract_features(rgb, mask, self.encoder)
        raw = float(self.model.decision_function(features[None])[0])
        threshold = float(self.metadata["selection"]["threshold"])
        score = raw - threshold
        return {"score_bad": score, "label": "bad" if score > 0 else "good",
                "raw_score": raw, "candidate_threshold": threshold, "view_proxy": view}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    rows = discover_images(args.data)
    x, diagnostics = feature_cache(rows, args.output)
    y = np.asarray([row["label"] for row in rows], dtype=int)
    groups = np.asarray([row["brick_id"] for row in rows])
    with threadpool_limits(limits=1):
        result = evaluate(x, y, groups, rows, diagnostics, args.output)
        selection = select_parameters(x, y, groups)
        model = fit(x, y, groups, selection["C"])
    joblib.dump(model, args.output / "candidate_svm.joblib")
    metadata = {"name": VERSION, "selection": selection, "n_images": len(y),
                "n_physical_bricks": len(np.unique(groups)),
                "features": {"baseline_appearance": 52, "view_geometry_proxy": 8,
                             "filtered_material_lab": 53, "local_lab": 20,
                             "rotation_averaged_dino": DINO_DIM},
                "view_limitation": "No labeled physical face types; dark components and geometry are proxies only.",
                "rotation_limitation": "0,90,180,270 are in-plane rotations, not hidden faces.",
                "decision": "bad if SVM raw score > training-selected threshold"}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"evaluation": result, "selection": selection}, indent=2), flush=True)


if __name__ == "__main__":
    main()
