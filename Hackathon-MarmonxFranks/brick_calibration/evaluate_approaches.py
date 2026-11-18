#!/usr/bin/env python3
"""Compare lightweight brick-quality classifiers with grouped validation.

Every outer fold holds out one complete physical brick.  Hyperparameters are
selected only from the remaining physical bricks, preventing photographs of a
held-out brick from leaking into training.  Results are written separately for
each approach plus a common comparison matrix.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
BRICK_DIR = re.compile(r"^Brick_(\d+)_(good|bad)$", re.IGNORECASE)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def discover_images(root: Path):
    records = []
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        match = BRICK_DIR.fullmatch(folder.name)
        if not match:
            continue
        label_name = match.group(2).lower()
        label = int(label_name == "bad")
        for path in sorted(p for p in folder.rglob("*")
                           if p.is_file() and p.suffix.lower() in EXTENSIONS):
            records.append({"path": path, "brick_id": folder.name,
                            "label": label, "label_name": label_name})
    if not records:
        raise ValueError(f"No Brick_N_good / Brick_N_bad images below {root}")
    return sorted(records, key=lambda row: (row["brick_id"], row["path"].name))


def center_roi(shape, fraction):
    height, width = shape
    mask = np.zeros(shape, np.uint8)
    roi_width, roi_height = int(width * fraction), int(height * fraction)
    x0, y0 = (width - roi_width) // 2, (height - roi_height) // 2
    mask[y0:y0 + roi_height, x0:x0 + roi_width] = 1
    return mask


def material_mask(rgb):
    """Conservative automatic foreground mask for a centred brick on paper.

    GrabCut supplies object/background separation.  Colour distance from the
    border prevents white paper from entering the foreground.  Dark holes and
    clipped highlights are removed later by ``usable_pixels`` rather than
    being learned as brick colour.
    """
    height, width = rgb.shape[:2]
    # GrabCut at the native 1920x1080 capture resolution is unnecessarily
    # expensive for a coarse material mask.  Estimate it at 640 px and restore
    # it with nearest-neighbour interpolation so no fractional labels appear.
    if width > 640:
        scale = 640.0 / width
        reduced = cv2.resize(rgb, (640, max(1, round(height * scale))),
                             interpolation=cv2.INTER_AREA)
        reduced_mask = material_mask(reduced)
        return cv2.resize(reduced_mask, (width, height),
                          interpolation=cv2.INTER_NEAREST)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gc = np.full((height, width), cv2.GC_PR_BGD, np.uint8)
    border = max(8, int(min(height, width) * .025))
    gc[:border] = gc[-border:] = cv2.GC_BGD
    gc[:, :border] = gc[:, -border:] = cv2.GC_BGD
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    border_pixels = np.concatenate((lab[:border].reshape(-1, 3),
                                    lab[-border:].reshape(-1, 3),
                                    lab[:, :border].reshape(-1, 3),
                                    lab[:, -border:].reshape(-1, 3)))
    background = np.median(border_pixels, axis=0)
    distance = np.linalg.norm(lab - background, axis=2)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    likely_material = (distance > 18) | (hsv[..., 1] > 40)
    central = center_roi((height, width), .80).astype(bool)
    gc[likely_material & central] = cv2.GC_PR_FGD
    strong = likely_material & central & (distance > 30)
    gc[strong] = cv2.GC_FGD
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, gc, None, bg_model, fg_model, 4, cv2.GC_INIT_WITH_MASK)
    mask = np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    count, components, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        raise ValueError("automatic mask has no foreground component")
    candidates = []
    image_center = np.array([width / 2, height / 2])
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        component_center = np.array([x + w / 2, y + h / 2])
        distance_to_center = np.linalg.norm((component_center - image_center) /
                                            np.array([width, height]))
        candidates.append((area / (1 + 4 * distance_to_center), index))
    mask = (components == max(candidates)[1]).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.erode(mask, np.ones((7, 7), np.uint8), iterations=1)
    fraction = float(mask.mean())
    if not .03 <= fraction <= .75:
        raise ValueError(f"implausible automatic mask fraction {fraction:.3f}")
    return mask


def usable_pixels(rgb, mask):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    value = hsv[..., 2]
    usable = mask.astype(bool) & (value > 18) & (value < 248)
    if usable.sum() < 500:
        raise ValueError("too few usable material pixels")
    # OpenCV LAB is used consistently within all folds and is much faster than
    # repeatedly converting full-resolution images through scikit-image.
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    return lab, usable, value


def summary_features(lab, usable):
    pixels = lab[usable]
    return np.r_[np.median(pixels, axis=0), np.std(pixels, axis=0),
                 np.percentile(pixels, 10, axis=0),
                 np.percentile(pixels, 90, axis=0)]


def robust_percentiles(lab, usable):
    pixels = lab[usable]
    percentiles = np.percentile(pixels, [5, 10, 25, 50, 75, 90, 95], axis=0)
    return np.r_[percentiles.ravel(), np.std(pixels, axis=0)]


def lab_histogram(lab, usable):
    pixels = lab[usable]
    parts = []
    for channel, bins, limits in ((0, 12, (0, 100)),
                                  (1, 16, (-80, 80)),
                                  (2, 16, (-80, 80))):
        histogram, _ = np.histogram(pixels[:, channel], bins=bins, range=limits)
        parts.append(histogram.astype(float) / max(1, histogram.sum()))
    return np.concatenate(parts)


def texture_features(rgb, usable):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)[usable]
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F))[usable]
    return np.r_[np.percentile(gradient, [25, 50, 75, 90]),
                 np.percentile(laplacian, [50, 75, 90]),
                 float(np.std(gray[usable]))]


def local_defect_features(rgb, material):
    """Spatial anomaly descriptors intended to retain small crack signals.

    These features are deliberately generic and are calculated identically for
    every brick.  The mask is eroded so the brick silhouette cannot become the
    longest detected line. Hole edges may still contribute and are therefore
    documented as a limitation rather than tuned away using Brick 6.
    """
    height, width = material.shape
    scale = min(1.0, 640.0 / width)
    if scale < 1:
        size = (round(width * scale), round(height * scale))
        image = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(material, size, interpolation=cv2.INTER_NEAREST)
    else:
        image, mask = rgb, material.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    radius = max(3, round(min(gray.shape) * .012))
    if radius % 2 == 0:
        radius += 1
    inner = cv2.erode(mask, np.ones((radius, radius), np.uint8), iterations=1).astype(bool)
    if inner.sum() < 500:
        inner = mask.astype(bool)
    smooth = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(2, radius / 2))
    dark_contrast = np.maximum(smooth.astype(np.float32) - gray.astype(np.float32), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    dark_values = dark_contrast[inner]
    gradient_values = gradient[inner]

    # Connected anomalous dark regions: area, elongation and count. Threshold
    # is image-adaptive and independent of quality labels.
    threshold = max(8.0, float(np.percentile(dark_values, 95)))
    anomaly = ((dark_contrast >= threshold) & inner).astype(np.uint8)
    anomaly = cv2.morphologyEx(anomaly, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(anomaly, 8)
    component_areas, elongations = [], []
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area < 4:
            continue
        component_areas.append(area / inner.sum())
        elongations.append(max(w, h) / max(1, min(w, h)))

    edges = cv2.Canny(gray, 50, 140)
    edges[~inner] = 0
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25,
                            minLineLength=max(10, round(min(gray.shape) * .035)),
                            maxLineGap=max(3, round(min(gray.shape) * .012)))
    diagonal = float(np.hypot(*gray.shape))
    lengths = [] if lines is None else [
        float(np.hypot(line[0][2] - line[0][0], line[0][3] - line[0][1])) / diagonal
        for line in lines]

    # Patch anomalies preserve the most unusual local region rather than
    # averaging it into the whole brick descriptor.
    ys, xs = np.nonzero(inner)
    x_edges = np.linspace(xs.min(), xs.max() + 1, 7).astype(int)
    y_edges = np.linspace(ys.min(), ys.max() + 1, 4).astype(int)
    patch_means, patch_stds, patch_edges, patch_dark = [], [], [], []
    for y0, y1 in zip(y_edges[:-1], y_edges[1:]):
        for x0, x1 in zip(x_edges[:-1], x_edges[1:]):
            selected = inner[y0:y1, x0:x1]
            if selected.sum() < 30:
                continue
            patch_gray = gray[y0:y1, x0:x1][selected]
            patch_means.append(float(np.mean(patch_gray)))
            patch_stds.append(float(np.std(patch_gray)))
            patch_edges.append(float(np.mean(edges[y0:y1, x0:x1][selected] > 0)))
            patch_dark.append(float(np.percentile(
                dark_contrast[y0:y1, x0:x1][selected], 95)))
    mean_median = np.median(patch_means)
    return np.asarray([
        *np.percentile(dark_values, [90, 95, 99]),
        *np.percentile(gradient_values, [90, 95, 99]),
        float(max(component_areas, default=0)),
        float(max(elongations, default=0)),
        float(len(component_areas)),
        float(max(lengths, default=0)),
        float(sum(length >= .08 for length in lengths)),
        float(np.mean(edges[inner] > 0)),
        float(max((abs(value - mean_median) for value in patch_means), default=0)),
        float(max(patch_stds, default=0)),
        float(max(patch_edges, default=0)),
        float(max(patch_dark, default=0)),
    ], np.float32)


def classifier(kind):
    if kind == "logreg":
        model = make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=3000, class_weight="balanced", solver="liblinear"))
        grid = {"logisticregression__C": [.03, .1, .3, 1.0, 3.0]}
    else:
        model = make_pipeline(StandardScaler(), SVC(
            kernel="linear", class_weight="balanced"))
        grid = {"svc__C": [.01, .03, .1, .3, 1.0, 3.0]}
    return model, grid


def metrics(true, predicted):
    matrix = confusion_matrix(true, predicted, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "accuracy": float(accuracy_score(true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
        "bad_precision": float(precision_score(true, predicted, zero_division=0)),
        "bad_recall": float(recall_score(true, predicted, zero_division=0)),
        "bad_f1": float(f1_score(true, predicted, zero_division=0)),
        "false_accept_bad_as_good": int(fn),
        "false_reject_good_as_bad": int(fp),
        "confusion_matrix": matrix.tolist(),
    }


def grouped_evaluation(features, labels, groups, rows, kind, output):
    outer = LeaveOneGroupOut()
    predictions = np.full(len(labels), -1, int)
    scores = np.full(len(labels), np.nan)
    selected = []
    for train, test in outer.split(features, labels, groups):
        model, grid = classifier(kind)
        # GridSearchCV may dispatch folds to worker processes; materialize the
        # split iterator so it is serializable and reproducible.
        inner = list(LeaveOneGroupOut().split(
            features[train], labels[train], groups[train]))
        # Each inner validation fold contains one physical brick and therefore
        # one class. Accuracy is well-defined for that fold; GridSearchCV then
        # gives every held-out brick equal weight when averaging fold scores.
        search = GridSearchCV(model, grid, scoring="accuracy", cv=inner,
                              n_jobs=-1, error_score="raise")
        search.fit(features[train], labels[train])
        fitted = search.best_estimator_
        predictions[test] = fitted.predict(features[test])
        scores[test] = fitted.decision_function(features[test])
        selected.append({"held_out_brick": groups[test][0],
                         "best_parameters": json.dumps(search.best_params_),
                         "inner_mean_held_out_brick_accuracy": float(search.best_score_)})
    image_rows = []
    for index, row in enumerate(rows):
        image_rows.append({"brick_id": row["brick_id"], "file": str(row["path"]),
                           "true_label": row["label_name"],
                           "predicted_label": "bad" if predictions[index] else "good",
                           "decision_score_bad": float(scores[index]),
                           "correct": int(predictions[index] == labels[index])})
    brick_rows = []
    brick_true, brick_pred = [], []
    for brick in dict.fromkeys(groups):
        indices = np.flatnonzero(groups == brick)
        score = float(np.mean(scores[indices]))
        prediction = int(score >= 0)
        truth = int(labels[indices[0]])
        brick_true.append(truth)
        brick_pred.append(prediction)
        brick_rows.append({"brick_id": brick,
                           "true_label": "bad" if truth else "good",
                           "predicted_label": "bad" if prediction else "good",
                           "mean_decision_score_bad": score,
                           "correct": int(truth == prediction),
                           "n_images": len(indices)})
    image_metrics = metrics(labels, predictions)
    brick_metrics = metrics(np.asarray(brick_true), np.asarray(brick_pred))
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "held_out_image_predictions.csv", image_rows)
    write_csv(output / "held_out_brick_predictions.csv", brick_rows)
    write_csv(output / "selected_hyperparameters.csv", selected)
    payload = {"image_level": image_metrics, "brick_level": brick_metrics,
               "n_images": len(labels), "n_bricks": len(set(groups))}
    (output / "metrics.json").write_text(json.dumps(payload, indent=2))
    matrix = np.asarray(brick_metrics["confusion_matrix"])
    fig, axis = plt.subplots(figsize=(4.5, 4))
    image = axis.imshow(matrix, cmap="Blues", vmin=0)
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["good", "bad"],
             yticklabels=["good", "bad"], xlabel="Predicted", ylabel="Actual",
             title="Held-out physical bricks")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(output / "brick_confusion_matrix.png", dpi=160)
    plt.close(fig)
    return payload


def save_final_checkpoint(features, labels, groups, rows, kind, output,
                          approach, feature_description):
    """Select regularization by grouped CV and fit the deployable all-data model."""
    model, grid = classifier(kind)
    folds = list(LeaveOneGroupOut().split(features, labels, groups))
    search = GridSearchCV(model, grid, scoring="accuracy", cv=folds,
                          n_jobs=-1, error_score="raise")
    search.fit(features, labels)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_estimator_, output / "model.joblib")
    metadata = {
        "approach": approach,
        "feature_description": feature_description,
        "feature_dimension": int(features.shape[1]),
        "classifier": kind,
        "selected_parameters": search.best_params_,
        "grouped_cv_mean_image_accuracy": float(search.best_score_),
        "training_images": len(rows),
        "training_physical_bricks": len(set(groups)),
        "class_counts_images": {"good": int(np.sum(labels == 0)),
                                "bad": int(np.sum(labels == 1))},
        "label_mapping": {"good": 0, "bad": 1},
        "decision_rule": "decision_function >= 0 means bad",
        "extractor_source": str(Path(__file__).resolve()),
        "dataset_root": str(rows[0]["path"].parents[1]),
        "warning": "Checkpoint is trained on ten physical bricks and requires the identical feature extractor. Validate on robot-domain data before deployment.",
    }
    (output / "checkpoint_metadata.json").write_text(json.dumps(metadata, indent=2))
    write_csv(output / "training_images.csv", [
        {"file": str(row["path"]), "brick_id": row["brick_id"],
         "label": row["label_name"]} for row in rows])
    return metadata


def save_mask_overlays(rows, masks, output):
    output.mkdir(parents=True, exist_ok=True)
    seen = {}
    for row, mask in zip(rows, masks):
        number = seen.get(row["brick_id"], 0)
        if number >= 2:
            continue
        rgb = cv2.cvtColor(cv2.imread(str(row["path"])), cv2.COLOR_BGR2RGB)
        overlay = rgb.copy()
        overlay[mask == 0] = (overlay[mask == 0] * .25).astype(np.uint8)
        height = 360
        width = int(rgb.shape[1] * height / rgb.shape[0])
        preview = np.hstack((cv2.resize(rgb, (width, height)),
                             cv2.resize(overlay, (width, height))))
        destination = output / f"{row['brick_id']}_{number + 1}.jpg"
        cv2.imwrite(str(destination), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        seen[row["brick_id"]] = number + 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path,
                        default=Path("brick_calibration/approach_results"))
    parser.add_argument("--roi-fraction", type=float, default=.40)
    args = parser.parse_args()
    rows = discover_images(args.data)
    labels = np.asarray([row["label"] for row in rows])
    groups = np.asarray([row["brick_id"] for row in rows])
    feature_sets = {name: [] for name in
                    ("center_summary", "masked_summary", "masked_percentiles",
                     "masked_histogram", "masked_histogram_texture",
                     "masked_histogram_texture_local")}
    masks, qc_rows, errors = [], [], []
    valid_rows = []
    for index, row in enumerate(rows, 1):
        try:
            bgr = cv2.imread(str(row["path"]), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("could not decode image")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            automatic = material_mask(rgb)
            centre = center_roi(rgb.shape[:2], args.roi_fraction)
            lab_auto, usable_auto, value = usable_pixels(rgb, automatic)
            lab_centre, usable_centre, _ = usable_pixels(rgb, centre)
            histogram = lab_histogram(lab_auto, usable_auto)
            feature_sets["center_summary"].append(summary_features(lab_centre, usable_centre))
            feature_sets["masked_summary"].append(summary_features(lab_auto, usable_auto))
            feature_sets["masked_percentiles"].append(robust_percentiles(lab_auto, usable_auto))
            feature_sets["masked_histogram"].append(histogram)
            feature_sets["masked_histogram_texture"].append(
                np.r_[histogram, texture_features(rgb, usable_auto)])
            feature_sets["masked_histogram_texture_local"].append(
                np.r_[histogram, texture_features(rgb, usable_auto),
                      local_defect_features(rgb, automatic)])
            masks.append(automatic)
            valid_rows.append(row)
            selected = automatic.astype(bool)
            qc_rows.append({"brick_id": row["brick_id"], "file": str(row["path"]),
                            "mask_fraction": float(selected.mean()),
                            "usable_fraction_of_mask": float(usable_auto.sum() / selected.sum()),
                            "dark_pct": float(100 * np.mean(value[selected] <= 18)),
                            "clipped_pct": float(100 * np.mean(value[selected] >= 248))})
        except Exception as error:
            errors.append(f"{row['path']}: {error}")
        if index % 25 == 0:
            print(f"Features: {index}/{len(rows)}", flush=True)
    if errors:
        (args.out / "skipped_images.txt").parent.mkdir(parents=True, exist_ok=True)
        (args.out / "skipped_images.txt").write_text("\n".join(errors))
    if len(valid_rows) != len(rows):
        print(f"WARNING: skipped {len(rows) - len(valid_rows)} images")
    rows = valid_rows
    labels = np.asarray([row["label"] for row in rows])
    groups = np.asarray([row["brick_id"] for row in rows])
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out / "extracted_features.npz", labels=labels, groups=groups,
        **{name: np.asarray(values) for name, values in feature_sets.items()})
    write_csv(args.out / "mask_quality.csv", qc_rows)
    save_mask_overlays(rows, masks, args.out / "mask_overlays")
    approaches = [
        ("01_center_roi_summary_logreg", "center_summary", "logreg"),
        ("02_masked_summary_logreg", "masked_summary", "logreg"),
        ("03_masked_percentiles_logreg", "masked_percentiles", "logreg"),
        ("04_masked_lab_hist_linear_svm", "masked_histogram", "svm"),
        ("05_masked_lab_hist_texture_linear_svm", "masked_histogram_texture", "svm"),
        ("06_masked_lab_hist_texture_local_linear_svm",
         "masked_histogram_texture_local", "svm"),
    ]
    comparison = []
    for folder, feature_name, model_name in approaches:
        print(f"Evaluating {folder}", flush=True)
        result = grouped_evaluation(np.asarray(feature_sets[feature_name]), labels,
                                    groups, rows, model_name, args.out / folder)
        for level in ("image_level", "brick_level"):
            comparison.append({"approach": folder, "feature_set": feature_name,
                               "classifier": model_name, "level": level,
                               **{key: value for key, value in result[level].items()
                                  if key != "confusion_matrix"},
                               "confusion_matrix": json.dumps(
                                   result[level]["confusion_matrix"])})
    write_csv(args.out / "evaluation_matrix.csv", comparison)
    (args.out / "evaluation_matrix.json").write_text(json.dumps(comparison, indent=2))
    baseline_checkpoint = save_final_checkpoint(
        np.asarray(feature_sets["masked_histogram_texture"]), labels, groups,
        rows, "svm", args.out / "05_approach_checkpoint", "Approach 5",
        "automatic material mask; per-channel LAB histograms; global gradient, Laplacian and intensity texture statistics")
    brick_rows = [row for row in comparison if row["level"] == "brick_level"]
    best = sorted(brick_rows, key=lambda row: (
        row["bad_recall"], row["balanced_accuracy"], row["accuracy"]), reverse=True)[0]
    summary = {"data": str(args.data.resolve()), "n_images": len(rows),
               "n_physical_bricks": len(set(groups)), "skipped_images": len(errors),
               "selection_rule": "brick bad recall, then balanced accuracy, then accuracy",
               "recommended_approach_by_rule": best["approach"],
               "frozen_approach_5_checkpoint": "05_approach_checkpoint/model.joblib",
               "approach_5_checkpoint_parameters": baseline_checkpoint["selected_parameters"],
               "warning": "Only ten physical bricks; results are exploratory, not a deployment guarantee."}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
