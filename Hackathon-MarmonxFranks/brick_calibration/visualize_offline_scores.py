#!/usr/bin/env python3
"""Visualize Approach 17 offline SAM3-mask scores and linear SVM contributions."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate_approaches import lab_histogram, texture_features, usable_pixels

HERE = Path(__file__).resolve().parent
METRICS = ("color_contribution", "texture_contribution", "approach5_score",
           "dinov2_score", "score_bad")
TITLES = ("Color contribution\nto A5 score", "Texture contribution\nto A5 score",
          "Approach 5\ncolor + texture", "DINOv2\nSVM score", "Combined\nA17 score")


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(row["status"] != "ok" for row in rows):
        raise ValueError("Input must contain successful offline score records")
    return sorted(rows, key=lambda row: (int(row["folder"].split("_")[1]), row["image"]))


def get_feature_cache(cache: Path) -> dict:
    if not cache.exists():
        return {}
    with np.load(cache) as data:
        if "signatures" not in data:
            return {}
        return {(Path(str(path)).parent.name, Path(str(path)).name):
                (str(path), str(sig), np.asarray(features[:52], dtype=np.float64))
                for path, sig, features in zip(data["paths"], data["signatures"], data["features"])}


def image_features(row: dict, data_dir: Path, mask_dir: Path, cached: dict) -> np.ndarray:
    image_path = data_dir / row["folder"] / row["image"]
    mask_path = mask_dir / row["folder"] / "masks" / row["image"]
    image_stat, mask_stat = image_path.stat(), mask_path.stat()
    signature = f"{image_stat.st_mtime_ns}:{image_stat.st_size}:{mask_stat.st_mtime_ns}:{mask_stat.st_size}"
    item = cached.get((row["folder"], row["image"]))
    if item and Path(item[0]).resolve() == image_path.resolve() and item[1] == signature:
        return item[2]
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if bgr is None or mask is None or bgr.shape[:2] != mask.shape:
        raise ValueError(f"Missing image or SAM3 mask for {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    lab, usable, _ = usable_pixels(rgb, (mask > 0).astype(np.uint8))
    return np.r_[lab_histogram(lab, usable), texture_features(rgb, usable)]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", type=Path, default=HERE / "offline_results")
    parser.add_argument("--data", type=Path, default=HERE / "brick_splits/images")
    parser.add_argument("--checkpoint", type=Path, default=HERE / "approach_results/baseline_checkpoint")
    parser.add_argument("--output", type=Path, default=HERE / "score_visualization")
    args = parser.parse_args()
    rows = load_rows(args.offline / "all_predictions.csv")
    svm = joblib.load(args.checkpoint / "approach5_svm.joblib")
    scaler, svc = svm.named_steps["standardscaler"], svm.named_steps["svc"]
    if svc.kernel != "linear" or svc.coef_.shape != (1, 52) or list(svc.classes_) != [0, 1]:
        raise ValueError("Expected binary linear 52-feature Approach 5 SVM")
    cache = get_feature_cache(HERE / "approach_results/local_defect_candidate/sam3_mask_features.npz")
    image_rows = []
    errors = []
    for row in rows:
        features = image_features(row, args.data, args.offline, cache)
        standardized = scaler.transform(features[None])[0]
        contributions = standardized * svc.coef_[0]
        color, texture = float(contributions[:44].sum()), float(contributions[44:].sum())
        intercept = float(svc.intercept_[0])
        reconstructed = color + texture + intercept
        saved_a5 = float(row["approach5_score"])
        error = abs(reconstructed - saved_a5)
        errors.append(error)
        if error > 1e-4:
            raise ValueError(f"Score mismatch for {row['folder']}/{row['image']}: {error}")
        dino, fused = float(row["dinov2_score"]), float(row["score_bad"])
        if abs(fused - (saved_a5 + dino) / 2) > 2e-6:
            raise ValueError(f"Fusion mismatch for {row['folder']}/{row['image']}")
        image_rows.append({"folder": row["folder"], "image": row["image"], "truth": row["truth"],
                           "color_contribution": color, "texture_contribution": texture,
                           "approach5_intercept": intercept, "approach5_score": saved_a5,
                           "dinov2_score": dino, "score_bad": fused})
    folders = list(dict.fromkeys(row["folder"] for row in image_rows))
    aggregate = []
    for folder in folders:
        group = [row for row in image_rows if row["folder"] == folder]
        result = {"folder": folder, "truth": group[0]["truth"], "n_images": len(group)}
        for key in METRICS:
            values = np.asarray([row[key] for row in group])
            result.update({f"{key}_mean": float(values.mean()), f"{key}_std": float(values.std()),
                           f"{key}_min": float(values.min()), f"{key}_max": float(values.max())})
        aggregate.append(result)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "per_image_scores.csv", image_rows)
    write_csv(args.output / "folder_score_statistics.csv", aggregate)
    (args.output / "metadata.json").write_text(json.dumps({"n_images": len(rows), "n_folders": len(folders),
        "approach5_intercept": float(svc.intercept_[0]), "maximum_reconstruction_error": max(errors),
        "color_feature_count": 44, "texture_feature_count": 8,
        "note": "Color and texture values are additive feature contributions, not separate classifiers. Scores are SVM decision values, not probabilities."}, indent=2) + "\n")

    fig, axes = plt.subplots(1, len(METRICS), figsize=(21, 8.7), sharey=True)
    rng = np.random.default_rng(27)
    for ax, key, title in zip(axes, METRICS, TITLES):
        for i, folder in enumerate(folders):
            group = [r for r in image_rows if r["folder"] == folder]
            values = np.asarray([r[key] for r in group])
            color = "#16834b" if group[0]["truth"] == "good" else "#c33232"
            ax.scatter(values, i + rng.uniform(-.22, .22, len(values)), s=17,
                       color=color, alpha=.36, edgecolors="none")
            ax.scatter([values.mean()], [i], marker="D", s=42, color="#111827",
                       zorder=5, edgecolors="white", linewidths=.5)
        ax.axvline(0, color="#78828c", linewidth=.8, linestyle=":")
        if key == "score_bad":
            ax.axvspan(-1.75, -1.20, color="#d89f1d", alpha=.18)
            ax.axvline(-1.75, color="#b77c00", linewidth=.8, linestyle="--")
            ax.axvline(-1.20, color="#b77c00", linewidth=.8, linestyle="--")
        ax.set_title(title, fontsize=11, weight="bold")
        ax.set_xlabel("SVM score / contribution")
        ax.grid(axis="x", alpha=.2)
        ax.set_ylim(len(folders) - .5, -.5)
    axes[0].set_yticks(range(len(folders)))
    axes[0].set_yticklabels([f"{f.replace('Brick_', 'Brick ')}  (n={a['n_images']})"
                              for f, a in zip(folders, aggregate)], fontsize=10)
    fig.suptitle("Approach 17: folder score distributions", fontsize=17, weight="bold")
    fig.text(.5, .025, "Each colored dot = one photo; black diamond = folder mean. Green/red = ground-truth brick class. Amber band = review range for combined score.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .055, 1, .95))
    fig.savefig(args.output / "folder_score_distributions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {len(image_rows)} photos and {len(aggregate)} folder summaries to {args.output}")
    print(f"Max reconstruction error: {max(errors):.9f}")


if __name__ == "__main__":
    main()
