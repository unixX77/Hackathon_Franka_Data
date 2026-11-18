#!/usr/bin/env python3
"""Evaluate frozen DINOv2 and hybrid brick representations.

The encoder is never fine-tuned. Every outer fold holds out one physical brick;
PCA, scaling, SVM regularization selection, and SVM fitting use only training
bricks. Results extend the existing approach_results evaluation matrix.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from transformers import AutoImageProcessor, AutoModel

from evaluate_approaches import discover_images, material_mask, write_csv


def prepare_crop(path: Path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not decode {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return prepare_crop_rgb(rgb, material_mask(rgb))


def prepare_crop_rgb(rgb, mask):
    ys, xs = np.nonzero(mask)
    margin = round(.04 * max(xs.max() - xs.min(), ys.max() - ys.min()))
    x0, x1 = max(0, xs.min() - margin), min(rgb.shape[1], xs.max() + margin + 1)
    y0, y1 = max(0, ys.min() - margin), min(rgb.shape[0], ys.max() + margin + 1)
    crop, selected = rgb[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1].astype(bool)
    material = crop[selected]
    neutral = np.median(material, axis=0).astype(np.uint8)
    crop[~selected] = neutral
    return crop


@torch.inference_mode()
def extract_dino(rows, model_name, batch_size, device, output):
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    cls_features, patch_features = [], []
    records = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        crops = [prepare_crop(row["path"]) for row in batch_rows]
        inputs = processor(images=crops, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        if device.startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                hidden = model(pixel_values=pixel_values).last_hidden_state
        else:
            hidden = model(pixel_values=pixel_values).last_hidden_state
        hidden = hidden.float().cpu().numpy()
        cls_features.append(hidden[:, 0])
        patches = hidden[:, 1:]
        # Orderless patch-distribution summary retains more localized variation
        # than the CLS token while remaining invariant to small viewpoint shifts.
        patch_features.append(np.concatenate((
            patches.mean(axis=1), patches.std(axis=1),
            np.percentile(patches, 90, axis=1)), axis=1))
        for row, crop in zip(batch_rows, crops):
            records.append({"file": str(row["path"]), "brick_id": row["brick_id"],
                            "label": row["label_name"],
                            "crop_height": crop.shape[0], "crop_width": crop.shape[1]})
        print(f"DINO features: {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    cls_features = np.concatenate(cls_features).astype(np.float32)
    patch_features = np.concatenate(patch_features).astype(np.float32)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "dinov2_features.npz", cls=cls_features,
                        patch_distribution=patch_features)
    write_csv(output / "dinov2_input_manifest.csv", records)
    return cls_features, patch_features


def metrics(true, predicted):
    matrix = confusion_matrix(true, predicted, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {"accuracy": float(accuracy_score(true, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
            "bad_precision": float(precision_score(true, predicted, zero_division=0)),
            "bad_recall": float(recall_score(true, predicted, zero_division=0)),
            "bad_f1": float(f1_score(true, predicted, zero_division=0)),
            "false_accept_bad_as_good": int(fn),
            "false_reject_good_as_bad": int(fp),
            "confusion_matrix": matrix.tolist()}


def make_model(handcrafted_dimension, pca_components):
    if handcrafted_dimension:
        transform = ColumnTransformer([
            ("handcrafted", "passthrough", slice(0, handcrafted_dimension)),
            ("dino", PCA(n_components=pca_components, whiten=True, random_state=7),
             slice(handcrafted_dimension, None)),
        ])
        model = make_pipeline(transform, StandardScaler(),
                              SVC(kernel="linear", class_weight="balanced"))
    else:
        model = make_pipeline(
            PCA(n_components=pca_components, whiten=True, random_state=7),
            StandardScaler(), SVC(kernel="linear", class_weight="balanced"))
    return model, {"svc__C": [.01, .03, .1, .3, 1.0, 3.0]}


def evaluate(features, labels, groups, rows, output, handcrafted_dimension,
             pca_components):
    predictions = np.full(len(labels), -1, int)
    scores = np.full(len(labels), np.nan)
    selected = []
    for train, test in LeaveOneGroupOut().split(features, labels, groups):
        model, grid = make_model(handcrafted_dimension, pca_components)
        inner = list(LeaveOneGroupOut().split(
            features[train], labels[train], groups[train]))
        search = GridSearchCV(model, grid, scoring="accuracy", cv=inner,
                              n_jobs=-1, error_score="raise")
        search.fit(features[train], labels[train])
        predictions[test] = search.best_estimator_.predict(features[test])
        scores[test] = search.best_estimator_.decision_function(features[test])
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
    brick_rows, brick_true, brick_pred = [], [], []
    for brick in dict.fromkeys(groups):
        indices = np.flatnonzero(groups == brick)
        score = float(np.mean(scores[indices]))
        prediction, truth = int(score >= 0), int(labels[indices[0]])
        brick_true.append(truth); brick_pred.append(prediction)
        brick_rows.append({"brick_id": brick, "true_label": "bad" if truth else "good",
                           "predicted_label": "bad" if prediction else "good",
                           "mean_decision_score_bad": score,
                           "correct": int(truth == prediction), "n_images": len(indices)})
    payload = {"image_level": metrics(labels, predictions),
               "brick_level": metrics(np.asarray(brick_true), np.asarray(brick_pred)),
               "n_images": len(labels), "n_bricks": len(set(groups)),
               "pca_components": pca_components,
               "handcrafted_dimension": handcrafted_dimension}
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "held_out_image_predictions.csv", image_rows)
    write_csv(output / "held_out_brick_predictions.csv", brick_rows)
    write_csv(output / "selected_hyperparameters.csv", selected)
    (output / "metrics.json").write_text(json.dumps(payload, indent=2))
    matrix = np.asarray(payload["brick_level"]["confusion_matrix"])
    fig, axis = plt.subplots(figsize=(4.5, 4))
    plot = axis.imshow(matrix, cmap="Blues", vmin=0)
    for y in range(2):
        for x in range(2): axis.text(x, y, str(matrix[y, x]), ha="center", va="center")
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["good", "bad"],
             yticklabels=["good", "bad"], xlabel="Predicted", ylabel="Actual",
             title="Held-out physical bricks")
    fig.colorbar(plot, ax=axis); fig.tight_layout()
    fig.savefig(output / "brick_confusion_matrix.png", dpi=160); plt.close(fig)
    return payload


def anomaly_scores(train_features, train_labels, test_features, components):
    """Distance to the nearest frozen-DINO references from good bricks only."""
    good = train_features[train_labels == 0]
    scaler = StandardScaler().fit(good)
    good_scaled = scaler.transform(good)
    count = min(components, len(good_scaled) - 1, good_scaled.shape[1])
    pca = PCA(n_components=count, whiten=True, random_state=7).fit(good_scaled)
    references = pca.transform(good_scaled)
    queries = pca.transform(scaler.transform(test_features))
    distances = np.sqrt(((queries[:, None, :] - references[None, :, :]) ** 2).sum(axis=2))
    nearest = min(5, references.shape[0])
    return np.partition(distances, nearest - 1, axis=1)[:, :nearest].mean(axis=1)


def choose_grouped_threshold(scores, labels, groups):
    group_scores, group_labels = [], []
    for group in dict.fromkeys(groups):
        selected = groups == group
        group_scores.append(float(np.mean(scores[selected])))
        group_labels.append(int(labels[np.flatnonzero(selected)[0]]))
    unique = np.unique(group_scores)
    candidates = np.r_[unique[0] - 1e-6, (unique[:-1] + unique[1:]) / 2,
                       unique[-1] + 1e-6]
    best = None
    for threshold in candidates:
        predicted = (np.asarray(group_scores) >= threshold).astype(int)
        score = (balanced_accuracy_score(group_labels, predicted),
                 recall_score(group_labels, predicted, zero_division=0),
                 -int(np.sum((np.asarray(group_labels) == 0) & (predicted == 1))))
        if best is None or score > best[0]:
            best = (score, float(threshold))
    return best[1]


def evaluate_good_reference_anomaly(features, labels, groups, rows, output,
                                    pca_components):
    predictions = np.full(len(labels), -1, int)
    scores = np.full(len(labels), np.nan)
    selected_rows = []
    for outer_train, outer_test in LeaveOneGroupOut().split(features, labels, groups):
        train_x, train_y, train_g = (features[outer_train], labels[outer_train],
                                     groups[outer_train])
        inner_scores = np.full(len(outer_train), np.nan)
        for inner_train, inner_test in LeaveOneGroupOut().split(train_x, train_y, train_g):
            inner_scores[inner_test] = anomaly_scores(
                train_x[inner_train], train_y[inner_train], train_x[inner_test],
                pca_components)
        threshold = choose_grouped_threshold(inner_scores, train_y, train_g)
        test_scores = anomaly_scores(train_x, train_y, features[outer_test], pca_components)
        scores[outer_test] = test_scores - threshold
        predictions[outer_test] = (test_scores >= threshold).astype(int)
        selected_rows.append({"held_out_brick": groups[outer_test][0],
                              "anomaly_threshold": threshold,
                              "pca_components_max": pca_components,
                              "good_reference_images": int(np.sum(train_y == 0))})
    image_rows = [{"brick_id": row["brick_id"], "file": str(row["path"]),
                   "true_label": row["label_name"],
                   "predicted_label": "bad" if predictions[i] else "good",
                   "decision_score_bad": float(scores[i]),
                   "correct": int(predictions[i] == labels[i])}
                  for i, row in enumerate(rows)]
    brick_rows, brick_true, brick_pred = [], [], []
    for brick in dict.fromkeys(groups):
        indices = np.flatnonzero(groups == brick)
        score = float(np.mean(scores[indices])); prediction = int(score >= 0)
        truth = int(labels[indices[0]])
        brick_true.append(truth); brick_pred.append(prediction)
        brick_rows.append({"brick_id": brick, "true_label": "bad" if truth else "good",
                           "predicted_label": "bad" if prediction else "good",
                           "mean_decision_score_bad": score,
                           "correct": int(truth == prediction), "n_images": len(indices)})
    payload = {"image_level": metrics(labels, predictions),
               "brick_level": metrics(np.asarray(brick_true), np.asarray(brick_pred)),
               "n_images": len(labels), "n_bricks": len(set(groups)),
               "method": "mean distance to five nearest good-image references",
               "pca_components_max": pca_components}
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "held_out_image_predictions.csv", image_rows)
    write_csv(output / "held_out_brick_predictions.csv", brick_rows)
    write_csv(output / "selected_hyperparameters.csv", selected_rows)
    (output / "metrics.json").write_text(json.dumps(payload, indent=2))
    matrix = np.asarray(payload["brick_level"]["confusion_matrix"])
    fig, axis = plt.subplots(figsize=(4.5, 4)); plot = axis.imshow(matrix, cmap="Blues", vmin=0)
    for y in range(2):
        for x in range(2): axis.text(x, y, str(matrix[y, x]), ha="center", va="center")
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["good", "bad"],
             yticklabels=["good", "bad"], xlabel="Predicted", ylabel="Actual",
             title="Held-out physical bricks")
    fig.colorbar(plot, ax=axis); fig.tight_layout()
    fig.savefig(output / "brick_confusion_matrix.png", dpi=160); plt.close(fig)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--model", default="facebook/dinov2-small")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pca-components", type=int, default=32)
    parser.add_argument("--reuse-features", action="store_true")
    args = parser.parse_args()
    rows = discover_images(args.data)
    labels = np.asarray([row["label"] for row in rows])
    groups = np.asarray([row["brick_id"] for row in rows])
    cache_path = args.results / "dino_feature_cache" / "dinov2_features.npz"
    if args.reuse_features and cache_path.is_file():
        cached = np.load(cache_path)
        cls_features, patch_features = cached["cls"], cached["patch_distribution"]
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cls_features, patch_features = extract_dino(
            rows, args.model, args.batch_size, device, cache_path.parent)
    handcrafted_cache = np.load(args.results / "extracted_features.npz")
    handcrafted = handcrafted_cache["masked_histogram_texture"].astype(np.float32)
    if not (len(cls_features) == len(patch_features) == len(handcrafted) == len(rows)):
        raise ValueError("Feature cache and dataset sizes do not match")
    approaches = [
        ("07_frozen_dinov2_cls_linear_svm", cls_features, 0, "dinov2_cls"),
        ("08_frozen_dinov2_patch_distribution_linear_svm", patch_features, 0,
         "dinov2_patch_distribution"),
        ("09_approach5_plus_dinov2_cls_linear_svm",
         np.c_[handcrafted, cls_features], handcrafted.shape[1], "approach5+dinov2_cls"),
        ("10_approach5_plus_dinov2_patch_distribution_linear_svm",
         np.c_[handcrafted, patch_features], handcrafted.shape[1],
         "approach5+dinov2_patch_distribution"),
    ]
    anomaly_name = "11_frozen_dinov2_good_reference_anomaly"
    matrix_path = args.results / "evaluation_matrix.csv"
    existing = list(csv.DictReader(matrix_path.open()))
    names = {item[0] for item in approaches} | {anomaly_name}
    existing = [row for row in existing if row["approach"] not in names]
    additions = []
    for folder, features, handcrafted_dimension, feature_name in approaches:
        print(f"Evaluating {folder}", flush=True)
        result = evaluate(features, labels, groups, rows, args.results / folder,
                          handcrafted_dimension, args.pca_components)
        for level in ("image_level", "brick_level"):
            additions.append({"approach": folder, "feature_set": feature_name,
                              "classifier": "svm", "level": level,
                              **{key: value for key, value in result[level].items()
                                 if key != "confusion_matrix"},
                              "confusion_matrix": json.dumps(
                                  result[level]["confusion_matrix"])})
    print(f"Evaluating {anomaly_name}", flush=True)
    anomaly = evaluate_good_reference_anomaly(
        patch_features, labels, groups, rows, args.results / anomaly_name,
        args.pca_components)
    for level in ("image_level", "brick_level"):
        additions.append({"approach": anomaly_name,
                          "feature_set": "dinov2_patch_good_reference_distance",
                          "classifier": "nearest_good_anomaly", "level": level,
                          **{key: value for key, value in anomaly[level].items()
                             if key != "confusion_matrix"},
                          "confusion_matrix": json.dumps(
                              anomaly[level]["confusion_matrix"])})
    combined = existing + additions
    write_csv(matrix_path, combined)
    (args.results / "evaluation_matrix.json").write_text(json.dumps(combined, indent=2))
    summary = {"model": args.model, "encoder_frozen": True,
               "pca_components": args.pca_components,
               "n_images": len(rows), "n_physical_bricks": len(set(groups)),
               "approaches": [item[0] for item in approaches] + [anomaly_name]}
    (args.results / "dino_experiment_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
