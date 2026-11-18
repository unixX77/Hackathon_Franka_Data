#!/usr/bin/env python3
"""Approaches 17-20: score fusion and richer material texture for Approach 5.

Every approach uses the same protocol as Approaches 01-16:

* outer folds hold out one complete physical brick;
* scaling, PCA, SVM regularisation and fitting use training bricks only;
* inner leave-one-brick-out folds select C (mean per-brick accuracy);
* a brick decision averages the held-out image decision scores.

Late fusion averages linear-SVM decision values with a fixed equal weight, so it
introduces no extra parameter that could be tuned on held-out bricks.

Required inputs (all produced by earlier scripts, aligned by image order):
  approach_results/extracted_features.npz              (Approach 5 features)
  approach_results/dino_feature_cache/dinov2_features.npz  (Approach 8 features)
  approach_results/improvement_feature_cache/improvement_features.npz
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from evaluate_approaches import discover_images, metrics, write_csv
from evaluate_canonical_a5 import burst_analysis, uncertainty_analysis

C_GRID = [.01, .03, .1, .3, 1.0, 3.0]


def svm_model(pca_components=None):
    steps = [PCA(n_components=pca_components, whiten=True, random_state=7)] \
        if pca_components else []
    return make_pipeline(*steps, StandardScaler(),
                         SVC(kernel="linear", class_weight="balanced"))


def fit_svm(features, labels, groups, train, pca_components=None):
    """Fit with nested brick-grouped C selection.

    A 3-D ``features`` array (rotations, images, dims) is rotation augmentation:
    every rotation of a training image is a training sample, and all rotations of
    one brick stay in the same inner fold.
    """
    if features.ndim == 3:
        copies = features.shape[0]
        x = features[:, train].reshape(-1, features.shape[2])
        y, g = np.tile(labels[train], copies), np.tile(groups[train], copies)
    else:
        x, y, g = features[train], labels[train], groups[train]
    inner = list(LeaveOneGroupOut().split(x, y, g))
    search = GridSearchCV(svm_model(pca_components), {"svc__C": C_GRID},
                          scoring="accuracy", cv=inner, n_jobs=-1, error_score="raise")
    search.fit(x, y)
    return search


def decision(model, features, test):
    if features.ndim == 3:
        return np.mean([model.decision_function(view[test]) for view in features], axis=0)
    return model.decision_function(features[test])


def evaluate(members, labels, groups, rows, output, description):
    """``members`` maps a name to (features, pca_components); scores are averaged."""
    scores = np.full(len(labels), np.nan)
    member_scores = {name: np.full(len(labels), np.nan) for name in members}
    selected = []
    for train, test in LeaveOneGroupOut().split(labels, labels, groups):
        record = {"held_out_brick": groups[test][0]}
        for name, (features, pca_components) in members.items():
            search = fit_svm(features, labels, groups, train, pca_components)
            member_scores[name][test] = decision(search.best_estimator_, features, test)
            record[f"{name}_C"] = search.best_params_["svc__C"]
            record[f"{name}_inner_brick_accuracy"] = float(search.best_score_)
        selected.append(record)
    scores = np.mean([member_scores[name] for name in members], axis=0)
    predictions = (scores >= 0).astype(int)

    image_rows = []
    for index, row in enumerate(rows):
        image_rows.append({
            "brick_id": row["brick_id"], "file": str(row["path"]),
            "true_label": row["label_name"],
            "predicted_label": "bad" if predictions[index] else "good",
            "decision_score_bad": float(scores[index]),
            **{f"score_{name}": float(member_scores[name][index]) for name in members},
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
                           "image_accuracy": float(np.mean(predictions[indices] == truth)),
                           "correct": int(truth == prediction), "n_images": len(indices)})
    payload = {"description": description,
               "members": {name: {"dimension": int(features.shape[-1]),
                                  "pca_components": pca_components,
                                  "rotation_views": int(features.shape[0])
                                  if features.ndim == 3 else 1}
                           for name, (features, pca_components) in members.items()},
               "fusion_rule": "mean of member linear-SVM decision values; >= 0 means bad",
               "image_level": metrics(labels, predictions),
               "brick_level": metrics(np.asarray(brick_true), np.asarray(brick_pred)),
               "n_images": len(labels), "n_bricks": len(set(groups))}
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "held_out_image_predictions.csv", image_rows)
    write_csv(output / "held_out_brick_predictions.csv", brick_rows)
    write_csv(output / "selected_hyperparameters.csv", selected)
    (output / "metrics.json").write_text(json.dumps(payload, indent=2))
    matrix = np.asarray(payload["brick_level"]["confusion_matrix"])
    fig, axis = plt.subplots(figsize=(4.5, 4))
    plot = axis.imshow(matrix, cmap="Blues", vmin=0)
    for y in range(2):
        for x in range(2):
            axis.text(x, y, str(matrix[y, x]), ha="center", va="center")
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["good", "bad"],
             yticklabels=["good", "bad"], xlabel="Predicted", ylabel="Actual",
             title="Held-out physical bricks")
    fig.colorbar(plot, ax=axis); fig.tight_layout()
    fig.savefig(output / "brick_confusion_matrix.png", dpi=160); plt.close(fig)
    burst_analysis(output / "held_out_image_predictions.csv", output)
    uncertainty_analysis(output / "held_out_image_predictions.csv", output)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--only", nargs="*", help="approach folder names to run")
    args = parser.parse_args()
    rows = discover_images(args.data)
    labels = np.asarray([row["label"] for row in rows])
    groups = np.asarray([row["brick_id"] for row in rows])

    approach5 = np.load(args.results / "extracted_features.npz")
    dino = np.load(args.results / "dino_feature_cache" / "dinov2_features.npz")
    extra = np.load(args.results / "improvement_feature_cache" / "improvement_features.npz")
    for name, cache in (("Approach 5", approach5), ("improvement", extra)):
        if not np.array_equal(cache["groups"], groups):
            raise ValueError(f"{name} feature cache is not aligned with the dataset order")
    a5 = approach5["masked_histogram_texture"]
    dino_patch = dino["patch_distribution"]
    lbp = extra["lbp"]
    paper_wb = extra["paper_wb_a5"]
    rotated = np.load(args.results / "improvement_feature_cache" /
                      "dinov2_rotated_patch_features.npz")
    if not np.array_equal(rotated["groups"], groups):
        raise ValueError("rotated DINOv2 feature cache is not aligned with the dataset order")
    dino_rotated = rotated["patch_distribution"]

    approaches = {
        "17_late_fusion_approach5_dinov2_patch": (
            {"approach5": (a5, None), "dinov2_patch": (dino_patch, 32)},
            "Equal-weight late fusion of Approach 5 and Approach 8 decision scores"),
        "18_approach5_plus_rotation_invariant_lbp": (
            {"approach5_lbp": (np.c_[a5, lbp], None)},
            "Approach 5 + scale-normalised rotation-invariant uniform LBP (P,R)=(8,1),(16,2),(24,3)"),
        "19_paper_white_balanced_approach5": (
            {"paper_wb_approach5": (paper_wb, None)},
            "Approach 5 features after per-image white balance to the paper background"),
        "20_late_fusion_approach5_lbp_dinov2_patch": (
            {"approach5_lbp": (np.c_[a5, lbp], None), "dinov2_patch": (dino_patch, 32)},
            "Equal-weight late fusion of Approach 18 and Approach 8 decision scores"),
        "21_dinov2_patch_rotation_augmented": (
            {"dinov2_patch_rot4": (dino_rotated, 32)},
            "Approach 8 trained on 0/90/180/270-degree crops; test score averaged over the four rotations"),
        "22_late_fusion_approach5_dinov2_patch_rotation_augmented": (
            {"approach5": (a5, None), "dinov2_patch_rot4": (dino_rotated, 32)},
            "Equal-weight late fusion of Approach 5 and Approach 21 decision scores"),
    }
    summary = {}
    for folder, (members, description) in approaches.items():
        if args.only and folder not in args.only:
            continue
        print(f"Evaluating {folder}", flush=True)
        result = evaluate(members, labels, groups, rows, args.results / folder, description)
        brick6 = next(row for row in csv.DictReader(
            (args.results / folder / "held_out_brick_predictions.csv").open())
            if row["brick_id"] == "Brick_6_bad")
        summary[folder] = {"image_accuracy": result["image_level"]["accuracy"],
                           "image_bad_recall": result["image_level"]["bad_recall"],
                           "brick_accuracy": result["brick_level"]["accuracy"],
                           "brick_6_image_accuracy": float(brick6["image_accuracy"]),
                           "brick_6_mean_score": float(brick6["mean_decision_score_bad"])}
        print(json.dumps(summary[folder], indent=2), flush=True)
        update_matrix(args.results, folder, "+".join(members), result)
    write_comparison(args.results)
    return summary


def update_matrix(results, folder, feature_set, payload):
    path = results / "evaluation_matrix.csv"
    rows = [row for row in csv.DictReader(path.open()) if row["approach"] != folder]
    for level in ("image_level", "brick_level"):
        metric = payload[level]
        rows.append({"approach": folder, "feature_set": feature_set,
                     "classifier": "svm_late_fusion" if "+" in feature_set else "svm",
                     "level": level,
                     **{key: value for key, value in metric.items() if key != "confusion_matrix"},
                     "confusion_matrix": json.dumps(metric["confusion_matrix"])})
    write_csv(path, rows)
    (results / "evaluation_matrix.json").write_text(json.dumps(rows, indent=2))


COMPARED = ["05_masked_lab_hist_texture_linear_svm",
            "08_frozen_dinov2_patch_distribution_linear_svm",
            "17_late_fusion_approach5_dinov2_patch",
            "18_approach5_plus_rotation_invariant_lbp",
            "19_paper_white_balanced_approach5",
            "20_late_fusion_approach5_lbp_dinov2_patch",
            "21_dinov2_patch_rotation_augmented",
            "22_late_fusion_approach5_dinov2_patch_rotation_augmented"]


def write_comparison(results):
    """Per-brick held-out image accuracy plus baseline burst/uncertainty tables."""
    output = results / "improvement_comparison"
    output.mkdir(parents=True, exist_ok=True)
    table = []
    for folder in COMPARED:
        if not (results / folder / "held_out_image_predictions.csv").exists():
            continue
        images = list(csv.DictReader((results / folder / "held_out_image_predictions.csv").open()))
        bricks = {row["brick_id"]: row for row in csv.DictReader(
            (results / folder / "held_out_brick_predictions.csv").open())}
        record = {"approach": folder,
                  "images_correct": sum(int(row["correct"]) for row in images),
                  "images_total": len(images),
                  "bricks_correct": sum(int(row["correct"]) for row in bricks.values())}
        for brick in sorted(bricks, key=lambda name: int(name.split("_")[1])):
            subset = [int(row["correct"]) for row in images if row["brick_id"] == brick]
            record[f"{brick}_correct"] = f"{sum(subset)}/{len(subset)}"
            record[f"{brick}_mean_score"] = round(float(bricks[brick]["mean_decision_score_bad"]), 3)
        table.append(record)
    write_csv(output / "per_brick_comparison.csv", table)
    baseline = output / "approach5_baseline_deployment_metrics"
    baseline.mkdir(exist_ok=True)
    source = results / COMPARED[0] / "held_out_image_predictions.csv"
    burst_analysis(source, baseline)
    uncertainty_analysis(source, baseline)


if __name__ == "__main__":
    main()
