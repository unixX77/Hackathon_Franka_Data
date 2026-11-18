#!/usr/bin/env python3
"""Approach 23: does any colour space separate Brick 6 from the good bricks?

For ten colour spaces, the same masked brick pixels (Approach 5 automatic mask,
holes and clipped pixels removed) are described by per-channel histograms and
percentiles. Three analyses, all reported for every space:

1. separability - is Brick 6's mean colour description closer to the good or
   the bad bricks (Brick 6 excluded from the bad set)?
2. classification - leave-one-physical-brick-out linear SVM with nested C
   selection, colour-only and colour + Approach 5 texture statistics;
3. single-feature screen - which channel percentiles/spread put Brick 6 outside
   the range of the four good bricks, measured against photo-to-photo noise
   (exploratory; 180 statistics are screened, so some will separate Brick 6 by
   chance).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import LeaveOneGroupOut

from evaluate_approaches import (discover_images, material_mask, metrics,
                                 usable_pixels, write_csv)
from evaluate_improvements import fit_svm

SAMPLES = 30000
HIST_BINS = 16
PERCENTILES = (5, 25, 50, 75, 95)


def sample_pixels(path, seed):
    rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    mask = material_mask(rgb)
    _, usable, _ = usable_pixels(rgb, mask)
    pixels = rgb[usable]
    rng = np.random.default_rng(seed)
    return pixels[rng.choice(len(pixels), min(SAMPLES, len(pixels)), replace=False)]


def opencv_space(pixels, code):
    return cv2.cvtColor(pixels[None].astype(np.uint8), code)[0].astype(np.float32)


def convert(pixels, space):
    """RGB uint8 (N, 3) -> (N, 3) float channels of the named colour space."""
    rgb = pixels.astype(np.float32)
    eps = 1.0
    if space == "rgb":
        return rgb
    if space == "hsv":
        hsv = opencv_space(pixels, cv2.COLOR_RGB2HSV)
        hsv[:, 0] = (hsv[:, 0] + 90) % 180  # move red away from the 0/180 wrap
        return hsv
    if space == "hls":
        hls = opencv_space(pixels, cv2.COLOR_RGB2HLS)
        hls[:, 0] = (hls[:, 0] + 90) % 180
        return hls
    if space == "lab":
        return opencv_space(pixels, cv2.COLOR_RGB2LAB)
    if space == "luv":
        return opencv_space(pixels, cv2.COLOR_RGB2LUV)
    if space == "ycrcb":
        return opencv_space(pixels, cv2.COLOR_RGB2YCrCb)
    if space == "xyz":
        return opencv_space(pixels, cv2.COLOR_RGB2XYZ)
    total = rgb.sum(axis=1, keepdims=True) + eps
    if space == "rgb_chromaticity":   # intensity-free r, g, b proportions
        return rgb / total
    if space == "opponent":
        r, g, b = rgb.T
        return np.c_[(r - g) / np.sqrt(2), (r + g - 2 * b) / np.sqrt(6), (r + g + b) / np.sqrt(3)]
    if space == "log_chromaticity":   # shading-invariant log ratios + log intensity
        r, g, b = (rgb + eps).T
        return np.c_[np.log(r / g), np.log(b / g), np.log(total[:, 0] / 3)]
    raise ValueError(space)


SPACES = ("rgb", "hsv", "hls", "lab", "luv", "ycrcb", "xyz",
          "rgb_chromaticity", "opponent", "log_chromaticity")
CHANNELS = {"rgb": "R G B", "hsv": "H S V", "hls": "H L S", "lab": "L a b",
            "luv": "L u v", "ycrcb": "Y Cr Cb", "xyz": "X Y Z",
            "rgb_chromaticity": "r g b", "opponent": "O1 O2 O3",
            "log_chromaticity": "log(R/G) log(B/G) logI"}


def colour_features(samples, space):
    converted = [convert(pixels, space) for pixels in samples]
    stacked = np.concatenate(converted)
    # Label-free histogram ranges shared by every image.
    low, high = np.percentile(stacked, .5, axis=0), np.percentile(stacked, 99.5, axis=0)
    features, names = [], []
    for values in converted:
        row = []
        for c in range(3):
            histogram, _ = np.histogram(values[:, c], bins=HIST_BINS, range=(low[c], high[c]))
            row += list(histogram / max(1, histogram.sum()))
            row += list(np.percentile(values[:, c], PERCENTILES)) + [float(np.std(values[:, c]))]
        features.append(row)
    for channel in CHANNELS[space].split():
        names += [f"{channel}_hist{i:02d}" for i in range(HIST_BINS)]
        names += [f"{channel}_p{p}" for p in PERCENTILES] + [f"{channel}_std"]
    return np.asarray(features, np.float32), names


def separability(features, labels, groups):
    scaled = (features - features.mean(0)) / (features.std(0) + 1e-9)
    bricks = sorted(set(groups), key=lambda b: int(b.split("_")[1]))
    centre = {b: scaled[groups == b].mean(0) for b in bricks}
    label = {b: int(labels[groups == b][0]) for b in bricks}
    result = {}
    for brick in bricks:
        others = [b for b in bricks if b != brick and b != "Brick_6_bad"]
        good = min(np.linalg.norm(centre[brick] - centre[b]) for b in others if label[b] == 0)
        bad = min(np.linalg.norm(centre[brick] - centre[b]) for b in others if label[b] == 1)
        result[brick] = {"nearest_good": float(good), "nearest_bad": float(bad),
                         "closer_to": "good" if good < bad else "bad"}
    return result


def lobo(features, labels, groups):
    scores = np.full(len(labels), np.nan)
    for train, test in LeaveOneGroupOut().split(features, labels, groups):
        search = fit_svm(features, labels, groups, train)
        scores[test] = search.best_estimator_.decision_function(features[test])
    predictions = (scores >= 0).astype(int)
    bricks = sorted(set(groups), key=lambda b: int(b.split("_")[1]))
    brick_scores = {b: float(scores[groups == b].mean()) for b in bricks}
    brick_correct = sum((brick_scores[b] >= 0) == bool(labels[groups == b][0]) for b in bricks)
    return {"image": metrics(labels, predictions),
            "bricks_correct": int(brick_correct),
            "per_brick_images_correct": {b: f"{int(np.sum(predictions[groups == b] == labels[groups == b]))}"
                                            f"/{int(np.sum(groups == b))}" for b in bricks},
            "brick6_images_correct": int(np.sum(predictions[groups == "Brick_6_bad"] == 1)),
            "brick6_mean_score": brick_scores["Brick_6_bad"]}


def single_feature_screen(features, names, labels, groups, space):
    """Continuous channel statistics only (histogram bins are often empty for
    all good bricks, which makes a range test meaningless). Distance is measured
    in units of the typical photo-to-photo spread of that statistic within a
    brick, so values below 1 are inside ordinary measurement variation."""
    bricks = sorted(set(groups), key=lambda b: int(b.split("_")[1]))
    medians = {b: np.median(features[groups == b], axis=0) for b in bricks}
    noise = np.median([features[groups == b].std(0) for b in bricks], axis=0) + 1e-9
    good = np.asarray([medians[b] for b in bricks if labels[groups == b][0] == 0])
    other_bad = [b for b in bricks if labels[groups == b][0] == 1 and b != "Brick_6_bad"]
    lo, hi = good.min(0), good.max(0)
    rows = []
    for j, name in enumerate(names):
        if "_hist" in name:
            continue
        value = medians["Brick_6_bad"][j]
        side = "above" if value > hi[j] else "below" if value < lo[j] else "inside"
        distance = (value - hi[j]) if side == "above" else (lo[j] - value) if side == "below" else 0.0
        same_side = sum((medians[b][j] > hi[j]) if side == "above" else (medians[b][j] < lo[j])
                        for b in other_bad) if side != "inside" else 0
        rows.append({"space": space, "feature": name, "brick6_position": side,
                     "distance_outside_good_range_in_photo_noise_units": float(distance / noise[j]),
                     "other_bad_bricks_on_same_side": f"{same_side}/{len(other_bad)}"})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=12)
    args = parser.parse_args()
    records = discover_images(args.data)
    labels = np.asarray([r["label"] for r in records])
    groups = np.asarray([r["brick_id"] for r in records])
    texture = np.load(args.results / "extracted_features.npz")["masked_histogram_texture"][:, 44:]
    output = args.results / "23_colour_space_comparison"
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "pixel_samples.npz"
    if cache.exists():
        samples = list(np.load(cache)["samples"])
    else:
        samples = Parallel(n_jobs=args.jobs)(delayed(sample_pixels)(r["path"], i)
                                             for i, r in enumerate(records))
        if any(len(pixels) < SAMPLES for pixels in samples):
            raise ValueError(f"an image has fewer than {SAMPLES} usable pixels")
        np.savez_compressed(cache, samples=np.stack(samples).astype(np.uint8))
    table, screen, details = [], [], {}
    for space in SPACES:
        features, names = colour_features(samples, space)
        colour_only = lobo(features, labels, groups)
        with_texture = lobo(np.c_[features, texture], labels, groups)
        position = separability(features, labels, groups)
        details[space] = {"colour_only": colour_only, "colour_plus_texture": with_texture,
                          "separability": position}
        screen += single_feature_screen(features, names, labels, groups, space)
        misplaced = [b for b, v in position.items()
                     if b != "Brick_6_bad" and v["closer_to"] != ("bad" if b.endswith("bad") else "good")]
        table.append({
            "colour_space": space, "channels": CHANNELS[space],
            "brick6_nearest_good_distance": round(position["Brick_6_bad"]["nearest_good"], 2),
            "brick6_nearest_bad_distance": round(position["Brick_6_bad"]["nearest_bad"], 2),
            "brick6_closer_to": position["Brick_6_bad"]["closer_to"],
            "other_bricks_closer_to_wrong_class": " ".join(misplaced) or "none",
            "colour_only_image_accuracy": round(colour_only["image"]["accuracy"], 3),
            "colour_only_bricks_correct": colour_only["bricks_correct"],
            "colour_only_brick6_images_correct": colour_only["brick6_images_correct"],
            "colour_only_brick6_mean_score": round(colour_only["brick6_mean_score"], 3),
            "colour_texture_image_accuracy": round(with_texture["image"]["accuracy"], 3),
            "colour_texture_bricks_correct": with_texture["bricks_correct"],
            "colour_texture_brick6_images_correct": with_texture["brick6_images_correct"],
            "colour_texture_brick6_mean_score": round(with_texture["brick6_mean_score"], 3)})
        print(json.dumps(table[-1]), flush=True)
    write_csv(output / "colour_space_comparison.csv", table)
    write_csv(output / "brick6_single_feature_screen.csv",
              sorted(screen, key=lambda r: -r["distance_outside_good_range_in_photo_noise_units"]))
    (output / "details.json").write_text(json.dumps(details, indent=2))


if __name__ == "__main__":
    main()
