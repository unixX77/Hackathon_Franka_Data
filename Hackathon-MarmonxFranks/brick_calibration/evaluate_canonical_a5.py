#!/usr/bin/env python3
"""Rotation, canonical-crop, burst, and uncertainty evaluation for Approach 5."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, recall_score

from canonical_surface import canonicalize_masked_object
from evaluate_approaches import (discover_images, grouped_evaluation, lab_histogram,
                                 material_mask, texture_features, usable_pixels,
                                 write_csv)


def approach5_features(rgb, mask):
    lab, usable, _ = usable_pixels(rgb, mask)
    histogram = lab_histogram(lab, usable)
    return np.r_[histogram, texture_features(rgb, usable)].astype(np.float32)


def rotate_image(rgb, angle):
    height, width = rgb.shape[:2]
    transform = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    border = np.median(np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1])),
                       axis=0).tolist()
    return cv2.warpAffine(rgb, transform, (width, height), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def rotation_test(rows, checkpoint, output, angles):
    records, summary = [], []
    for angle in angles:
        truth, predicted, scores = [], [], []
        for index, row in enumerate(rows, 1):
            bgr = cv2.imread(str(row["path"]), cv2.IMREAD_COLOR)
            rgb = rotate_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), angle)
            feature = approach5_features(rgb, material_mask(rgb))
            score = float(checkpoint.decision_function(feature.reshape(1, -1))[0])
            prediction = int(score >= 0)
            truth.append(row["label"]); predicted.append(prediction); scores.append(score)
            records.append({"angle_deg": angle, "brick_id": row["brick_id"],
                            "file": str(row["path"]), "true_label": row["label_name"],
                            "predicted_label": "bad" if prediction else "good",
                            "decision_score_bad": score,
                            "correct": int(prediction == row["label"])})
        summary.append({"angle_deg": angle, "image_accuracy": accuracy_score(truth, predicted),
                        "image_bad_recall": recall_score(truth, predicted, zero_division=0),
                        "predicted_bad": int(np.sum(predicted)),
                        "mean_score": float(np.mean(scores))})
        print(f"Rotation {angle:g} degrees complete", flush=True)
    write_csv(output / "rotation_image_predictions.csv", records)
    write_csv(output / "rotation_metrics.csv", summary)
    return summary


def canonical_features(rows, output):
    features, previews = [], []
    preview_counts = {}
    for index, row in enumerate(rows, 1):
        bgr = cv2.imread(str(row["path"]), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask = material_mask(rgb)
        crop, crop_mask = canonicalize_masked_object(rgb, mask)
        features.append(approach5_features(crop, crop_mask))
        count = preview_counts.get(row["brick_id"], 0)
        if count < 2:
            shown = crop.copy(); shown[crop_mask == 0] = (shown[crop_mask == 0] * .2).astype(np.uint8)
            destination = output / "canonical_previews" / f"{row['brick_id']}_{count + 1}.jpg"
            destination.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(destination), cv2.cvtColor(shown, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            preview_counts[row["brick_id"]] = count + 1
        if index % 25 == 0: print(f"Canonical features: {index}/{len(rows)}", flush=True)
    features = np.asarray(features)
    np.savez_compressed(output / "canonical_features.npz", features=features)
    return features


def burst_analysis(prediction_path, output, sizes=(1, 3, 5, 9), repetitions=1000):
    rows = list(csv.DictReader(prediction_path.open()))
    groups = {}
    for row in rows: groups.setdefault(row["brick_id"], []).append(row)
    rng = np.random.default_rng(7); results = []
    for size in sizes:
        accuracies, recalls = [], []
        for _ in range(repetitions):
            truth, predicted = [], []
            for brick_rows in groups.values():
                chosen = rng.choice(len(brick_rows), min(size, len(brick_rows)), replace=False)
                score = np.mean([float(brick_rows[i]["decision_score_bad"]) for i in chosen])
                truth.append(int(brick_rows[0]["true_label"] == "bad")); predicted.append(int(score >= 0))
            accuracies.append(accuracy_score(truth, predicted))
            recalls.append(recall_score(truth, predicted, zero_division=0))
        results.append({"frames_per_brick": size, "repetitions": repetitions,
                        "mean_brick_accuracy": float(np.mean(accuracies)),
                        "std_brick_accuracy": float(np.std(accuracies)),
                        "mean_bad_recall": float(np.mean(recalls)),
                        "std_bad_recall": float(np.std(recalls))})
    write_csv(output / "burst_aggregation_metrics.csv", results)
    return results


def uncertainty_analysis(prediction_path, output, margins=(0, .1, .2, .3, .5, .75, 1.0)):
    rows = list(csv.DictReader(prediction_path.open())); results = []
    for margin in margins:
        certain = [row for row in rows if abs(float(row["decision_score_bad"])) >= margin]
        uncertain = [row for row in rows if row not in certain]
        truth = np.asarray([int(row["true_label"] == "bad") for row in certain])
        pred = np.asarray([int(float(row["decision_score_bad"]) >= 0) for row in certain])
        results.append({"symmetric_margin": margin, "coverage": len(certain) / len(rows),
                        "certain_images": len(certain), "uncertain_images": len(uncertain),
                        "certain_accuracy": float(accuracy_score(truth, pred)) if len(truth) else "",
                        "certain_bad_recall": float(recall_score(truth, pred, zero_division=0)) if len(truth) else "",
                        "bad_images_marked_uncertain": sum(row["true_label"] == "bad" for row in uncertain),
                        "good_images_marked_uncertain": sum(row["true_label"] == "good" for row in uncertain)})
    write_csv(output / "uncertainty_metrics.csv", results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args(); args.results.mkdir(parents=True, exist_ok=True)
    rows = discover_images(args.data); checkpoint = joblib.load(args.checkpoint)
    rotation = rotation_test(rows, checkpoint, args.results, [0, 15, 30, 45, 60, 75, 90])
    features = canonical_features(rows, args.results)
    labels = np.asarray([row["label"] for row in rows]); groups = np.asarray([row["brick_id"] for row in rows])
    evaluation = grouped_evaluation(features, labels, groups, rows, "svm",
                                    args.results / "16_canonical_surface_approach5")
    prediction_path = args.results / "16_canonical_surface_approach5" / "held_out_image_predictions.csv"
    burst = burst_analysis(prediction_path, args.results)
    uncertainty = uncertainty_analysis(prediction_path, args.results)
    summary = {"rotation": rotation, "canonical_evaluation": evaluation,
               "burst_aggregation": burst, "uncertainty": uncertainty,
               "warning": "Burst and uncertainty tables use existing held-out photographs; validate thresholds on new robot-domain data before deployment."}
    (args.results / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(evaluation, indent=2))


if __name__ == "__main__":
    main()
