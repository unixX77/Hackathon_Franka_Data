#!/usr/bin/env python3
"""Train and evaluate a group-validated good-brick prototype classifier.

Each training good brick is represented by one or two learned prototypes.
A threshold on normalized distance to the nearest good prototype is chosen
using only held-out training groups. The final evaluation holds out one whole
physical brick at a time, including all of its photographs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from evaluate_approaches import (discover_images, lab_histogram, texture_features,
                                 usable_pixels, write_csv)
from evaluate_dino_approaches import prepare_crop_rgb
from fusion_checkpoint import FusionClassifier
from local_defect_candidate import BASELINE, DATA, feature_cache

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "approach_results/good_brick_clusters"
CACHE = HERE / "approach_results/local_defect_candidate/sam3_mask_features.npz"
VARIANTS = ("color_texture", "dino", "mixed")
PROTOTYPE_COUNTS = (1, 2)


def cache_features(rows: list[dict]) -> np.ndarray:
    # The feature cache is deterministic SAM3-mask inputs, not model outputs.
    # No Approach 17 fitted SVM or test labels enter these representations.
    x = feature_cache(rows, CACHE, BASELINE)
    return np.concatenate((x[:, :52], x[:, 68:]), axis=1).astype(np.float32)


def extract_features(rgb: np.ndarray, mask: np.ndarray, encoder: FusionClassifier | None) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    lab, usable, _ = usable_pixels(rgb, mask)
    appearance = np.r_[lab_histogram(lab, usable), texture_features(rgb, usable)]
    if encoder is None:
        return appearance.astype(np.float32)
    dino = encoder._dino_features(prepare_crop_rgb(rgb, mask))
    return np.r_[appearance, dino].astype(np.float32)


def select(x: np.ndarray, variant: str) -> np.ndarray:
    if variant == "color_texture":
        return x[:, :52]
    if variant == "dino":
        return x[:, 52:]
    if variant == "mixed":
        return x
    raise ValueError(f"Unknown feature variant {variant}")


def block_weights(variant: str) -> np.ndarray:
    if variant == "color_texture":
        return np.full(52, 1 / np.sqrt(52))
    if variant == "dino":
        return np.full(1152, 1 / np.sqrt(1152))
    return np.r_[np.full(52, np.sqrt(.5 / 52)), np.full(1152, np.sqrt(.5 / 1152))]


def scaled(x: np.ndarray, model: dict) -> np.ndarray:
    data = model["scaler"].transform(select(x, model["variant"]))
    return np.clip(data, -5, 5) * model["weights"]


def distance(x: np.ndarray, model: dict) -> tuple[np.ndarray, np.ndarray]:
    z = scaled(x, model)
    centers = model["centers"]
    # Avoid allocating an images x prototypes x features array.
    sq = (np.sum(z * z, axis=1)[:, None] + np.sum(centers * centers, axis=1)[None, :]
          - 2 * z @ centers.T)
    sq = np.maximum(sq, 0)
    nearest = np.argmin(sq, axis=1)
    return np.sqrt(sq[np.arange(len(z)), nearest]) / model["distance_scale"], nearest


def fit_reference(x: np.ndarray, y: np.ndarray, groups: np.ndarray,
                  variant: str, prototypes_per_brick: int) -> dict:
    good = y == 0
    good_groups = np.unique(groups[good])
    if len(good_groups) < 2:
        raise ValueError("At least two independent good training bricks are required")
    # Equal total weight for every physical good brick in feature scaling.
    group_sizes = {group: int(np.sum(groups[good] == group)) for group in good_groups}
    weights = np.asarray([1 / group_sizes[group] for group in groups[good]])
    scaler = StandardScaler().fit(select(x[good], variant), sample_weight=weights)
    model = {"scaler": scaler, "variant": variant, "weights": block_weights(variant),
             "distance_scale": 1.0}
    z = scaled(x[good], model)
    good_ids = groups[good]
    centers, owners = [], []
    for group in good_groups:
        points = z[good_ids == group]
        count = min(prototypes_per_brick, len(points))
        if count == 1:
            centroids = points.mean(axis=0, keepdims=True)
        else:
            centroids = KMeans(n_clusters=count, n_init=3, random_state=7,
                               max_iter=100).fit(points).cluster_centers_
        centers.extend(centroids)
        owners.extend([str(group)] * len(centroids))
    model["centers"] = np.asarray(centers, np.float64)
    model["owners"] = owners
    # Normalize fold-dependent distances with the median training-good
    # residual, so inner validation scores have a comparable relative scale.
    raw, _ = distance(x[good], model)
    model["distance_scale"] = max(float(np.median(raw)), 1e-6)
    model["prototypes_per_brick"] = prototypes_per_brick
    return model


def optimize_threshold(scores: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[float, dict]:
    unique = np.unique(scores)
    midpoints = (unique[:-1] + unique[1:]) / 2
    candidates = np.r_[unique[0] - 1e-8, midpoints, unique[-1] + 1e-8]
    group_names = np.unique(groups)
    group_labels = np.asarray([int(y[groups == g][0]) for g in group_names])
    rankings = []
    for threshold in candidates:
        predictions = scores > threshold
        accuracies = np.asarray([np.mean(predictions[groups == g] == y[groups == g])
                                 for g in group_names])
        good_acc = float(accuracies[group_labels == 0].mean())
        bad_acc = float(accuracies[group_labels == 1].mean())
        # Each brick and then each class has equal weight; prioritize detecting
        # bad bricks when balanced scores tie. Threshold must use inner data only.
        rankings.append(((good_acc + bad_acc) / 2, bad_acc, good_acc, float(threshold)))
    best = max(rankings)
    return best[3], {"balanced_group_accuracy": best[0], "bad_group_accuracy": best[1],
                     "good_group_accuracy": best[2]}


def plot_per_brick(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ordered = sorted(rows, key=lambda row: int(row["brick_id"].split("_")[1]))
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(ordered))
    correct = [int(row["correct_images"]) for row in ordered]
    wrong = [int(row["images"]) - n for row, n in zip(ordered, correct)]
    ax.barh(y_pos, correct, color="#2e9c61", label="Correct photo")
    ax.barh(y_pos, wrong, left=correct, color="#d15757", label="Incorrect photo")
    ax.set_yticks(y_pos, [row["brick_id"] for row in ordered])
    ax.invert_yaxis()
    ax.set_xlabel("Number of photos")
    ax.set_title("Good-reference model: held-out physical-brick results")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.14), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def choose(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    """Select representation, prototypes and threshold using inner brick folds."""
    candidates = []
    folds = list(LeaveOneGroupOut().split(x, y, groups))
    for variant in VARIANTS:
        for count in PROTOTYPE_COUNTS:
            inner = np.empty(len(y), np.float64)
            for train, val in folds:
                model = fit_reference(x[train], y[train], groups[train], variant, count)
                inner[val], _ = distance(x[val], model)
            threshold, metrics = optimize_threshold(inner, y, groups)
            candidates.append({"variant": variant, "prototypes_per_brick": count,
                               "threshold": threshold, "inner_metrics": metrics})
    # Stable ordering chooses simpler single prototypes on exact ties.
    return max(candidates, key=lambda c: (c["inner_metrics"]["balanced_group_accuracy"],
                    c["inner_metrics"]["bad_group_accuracy"], -c["prototypes_per_brick"]))


def evaluate(x: np.ndarray, y: np.ndarray, groups: np.ndarray,
             rows: list[dict], output: Path) -> dict:
    scores = np.empty(len(y), np.float64)
    distance_values = np.empty(len(y), np.float64)
    reference = np.empty(len(y), dtype=object)
    thresholds = np.empty(len(y), np.float64)
    fold_rows = []
    for train, test in LeaveOneGroupOut().split(x, y, groups):
        selection = choose(x[train], y[train], groups[train])
        model = fit_reference(x[train], y[train], groups[train], selection["variant"],
                              selection["prototypes_per_brick"])
        measured, nearest = distance(x[test], model)
        threshold = selection["threshold"]
        scores[test] = measured - threshold
        distance_values[test] = measured
        reference[test] = np.asarray(model["owners"], dtype=object)[nearest]
        thresholds[test] = threshold
        correct = int(np.sum((scores[test] > 0) == y[test]))
        fold_rows.append({"held_out_brick": groups[test][0], "feature_variant": selection["variant"],
                          "prototypes_per_brick": selection["prototypes_per_brick"],
                          "threshold": threshold, "inner_balanced_accuracy":
                          selection["inner_metrics"]["balanced_group_accuracy"]})
        print(f"Held out {groups[test][0]}: {correct}/{len(test)}; "
              f"{selection['variant']} k={selection['prototypes_per_brick']} threshold={threshold:.3f}", flush=True)
    pred = scores > 0
    matrix = confusion_matrix(y, pred, labels=[0, 1])
    per_brick = []
    for group in np.unique(groups):
        test = groups == group
        median = float(np.median(scores[test]))
        truth = "bad" if y[test][0] else "good"
        per_brick.append({"brick_id": group, "truth": truth, "images": int(test.sum()),
                          "predicted_good": int(np.sum(~pred[test])),
                          "predicted_bad": int(np.sum(pred[test])),
                          "correct_images": int(np.sum(pred[test] == y[test])),
                          "median_score_bad": median,
                          "brick_prediction": "bad" if median > 0 else "good"})
    result = {"method": "per-good-brick clustered prototypes; threshold optimized with inner grouped validation",
              "n_images": len(y), "n_bricks": len(np.unique(groups)),
              "correct_images": int(np.sum(pred == y)), "image_accuracy": float(np.mean(pred == y)),
              "confusion_matrix_good_bad": matrix.tolist(),
              "good_recall": float(matrix[0, 0] / matrix[0].sum()),
              "bad_recall": float(matrix[1, 1] / matrix[1].sum()),
              "correct_bricks_median": sum(r["brick_prediction"] == r["truth"] for r in per_brick)}
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "held_out_image_predictions.csv", [
        {"brick_id": row["brick_id"], "image": row["path"].name,
         "truth": row["label_name"], "distance": float(distance_values[i]),
         "threshold": float(thresholds[i]), "score_bad": float(scores[i]),
         "nearest_good_reference": reference[i], "prediction": "bad" if pred[i] else "good",
         "correct": int(pred[i] == y[i])} for i, row in enumerate(rows)])
    write_csv(output / "held_out_brick_predictions.csv", per_brick)
    write_csv(output / "held_out_fold_parameters.csv", fold_rows)
    (output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    plot_per_brick(per_brick, output / "held_out_per_brick.png")
    return result


def evaluate_known_bricks(x: np.ndarray, y: np.ndarray, groups: np.ndarray,
                          rows: list[dict], output: Path) -> dict:
    """Separate ordered photos per brick for a known-brick demo experiment.

    This intentionally tests the same physical bricks; filenames provide an
    ordering, not verified independent camera sessions or face labels.
    """
    parts = np.empty(len(y), dtype=object)
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        n_train = int(round(len(indices) * .6))
        n_val = int(round(len(indices) * .2))
        parts[indices[:n_train]] = "train"
        parts[indices[n_train:n_train + n_val]] = "validation"
        parts[indices[n_train + n_val:]] = "test"
    train, val, test = parts == "train", parts == "validation", parts == "test"
    choices = []
    for variant in VARIANTS:
        for count in PROTOTYPE_COUNTS:
            model = fit_reference(x[train], y[train], groups[train], variant, count)
            values, _ = distance(x[val], model)
            threshold, metrics = optimize_threshold(values, y[val], groups[val])
            choices.append({"variant": variant, "prototypes_per_brick": count,
                            "threshold": threshold, "validation_metrics": metrics})
    chosen = max(choices, key=lambda c: (c["validation_metrics"]["balanced_group_accuracy"],
                 c["validation_metrics"]["bad_group_accuracy"], -c["prototypes_per_brick"]))
    model = fit_reference(x[train], y[train], groups[train], chosen["variant"],
                          chosen["prototypes_per_brick"])
    model["threshold"] = chosen["threshold"]
    measured, nearest = distance(x[test], model)
    pred = measured > model["threshold"]
    truth = y[test]
    matrix = confusion_matrix(truth, pred, labels=[0, 1])
    result = {"evaluation": "ordered photo holdout from same known physical bricks; not new-brick generalization",
              "counts": {part: int(np.sum(parts == part)) for part in ("train", "validation", "test")},
              "selection": chosen, "test_correct": int(np.sum(pred == truth)),
              "test_accuracy": float(np.mean(pred == truth)),
              "test_confusion_matrix_good_bad": matrix.tolist(),
              "test_good_recall": float(matrix[0, 0] / matrix[0].sum()),
              "test_bad_recall": float(matrix[1, 1] / matrix[1].sum())}
    write_csv(output / "known_brick_split_manifest.csv", [
        {"brick_id": row["brick_id"], "image": row["path"].name,
         "truth": row["label_name"], "split": parts[i]} for i, row in enumerate(rows)])
    test_indices = np.flatnonzero(test)
    write_csv(output / "known_brick_test_predictions.csv", [
        {"brick_id": rows[idx]["brick_id"], "image": rows[idx]["path"].name,
         "truth": rows[idx]["label_name"], "distance": float(measured[pos]),
         "threshold": model["threshold"], "nearest_good_reference": model["owners"][int(nearest[pos])],
         "prediction": "bad" if pred[pos] else "good", "correct": int(pred[pos] == truth[pos])}
        for pos, idx in enumerate(test_indices)])
    (output / "known_brick_evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    joblib.dump(model, output / "known_brick_demo_model.joblib")
    return result


class GoodBrickReferenceClassifier:
    def __init__(self, checkpoint: Path = OUTPUT, baseline: Path = BASELINE):
        self.model = joblib.load(checkpoint / "reference_model.joblib")
        self.encoder = (None if self.model["variant"] == "color_texture"
                        else FusionClassifier.load(baseline))

    def classify(self, rgb: np.ndarray, mask: np.ndarray) -> dict:
        features = extract_features(rgb, mask, self.encoder)[None]
        values, nearest = distance(features, self.model)
        normalized_distance = float(values[0])
        threshold = float(self.model["threshold"])
        score = normalized_distance - threshold
        return {"score_bad": score, "label": "bad" if score > 0 else "good",
                "distance": normalized_distance, "reference_threshold": threshold,
                "nearest_good_reference": self.model["owners"][int(nearest[0])]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    rows = discover_images(args.data)
    x = cache_features(rows)
    y = np.asarray([row["label"] for row in rows], dtype=int)
    groups = np.asarray([row["brick_id"] for row in rows])
    with threadpool_limits(limits=1):
        result = evaluate(x, y, groups, rows, args.output)
        known_result = evaluate_known_bricks(x, y, groups, rows, args.output)
        selection = choose(x, y, groups)
        model = fit_reference(x, y, groups, selection["variant"], selection["prototypes_per_brick"])
    model["threshold"] = selection["threshold"]
    joblib.dump(model, args.output / "reference_model.joblib")
    metadata = {"name": "good_brick_clusters", "selection": selection,
                "reference_bricks": sorted(set(model["owners"])),
                "n_prototypes": len(model["owners"]), "n_training_images": len(y),
                "decision_rule": "bad if nearest normalized good-reference distance > threshold",
                "scaler_fit": "training good images only; equal weight per physical good brick",
                "score_scale": "distance divided by median training-good prototype residual",
                "evaluation": "evaluation.json"}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"evaluation": result, "known_brick_evaluation": known_result,
                      "final_selection": selection}, indent=2), flush=True)


if __name__ == "__main__":
    main()
