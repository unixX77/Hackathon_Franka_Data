#!/usr/bin/env python3
"""Extract the additional per-image features used by Approaches 17-21.

All features are computed identically for every image and never use labels.
They are cached so that model comparisons do not repeat GrabCut masking.

Feature groups
--------------
paper_wb_a5   Approach 5 features after normalising each image so the white
              paper background becomes neutral grey (per-image white balance).
lbp           Rotation-invariant uniform LBP histograms at three radii on the
              material mask, after resizing so the brick's long side is fixed
              (scale-normalised texture).
silhouette    Rotation-invariant outline-damage measures: solidity, convexity
              defect depth/count, perimeter roughness and rectangularity.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from joblib import Parallel, delayed
from skimage.feature import local_binary_pattern

from evaluate_approaches import (discover_images, lab_histogram, material_mask,
                                 texture_features, usable_pixels)

LBP_SETTINGS = ((8, 1), (16, 2), (24, 3))
TEXTURE_LONG_SIDE = 600


def paper_white_balance(rgb, mask):
    """Scale RGB channels so the surrounding paper is neutral at value 220."""
    dilated = cv2.dilate(mask, np.ones((61, 61), np.uint8))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    paper = (dilated == 0) & (hsv[..., 1] < 50) & (hsv[..., 2] > 120)
    if paper.sum() < 5000:
        raise ValueError("too little visible paper for white balance")
    reference = np.median(rgb[paper].astype(np.float32), axis=0)
    gain = 220.0 / np.maximum(reference, 1.0)
    balanced = np.clip(rgb.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return balanced, reference


def lbp_features(rgb, mask):
    ys, xs = np.nonzero(mask)
    long_side = max(xs.max() - xs.min(), ys.max() - ys.min()) + 1
    scale = TEXTURE_LONG_SIDE / long_side
    size = (max(1, round(rgb.shape[1] * scale)), max(1, round(rgb.shape[0] * scale)))
    gray = cv2.cvtColor(cv2.resize(rgb, size, interpolation=cv2.INTER_AREA),
                        cv2.COLOR_RGB2GRAY)
    small_mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    value = gray
    # Exclude dark perforation interiors and the mask boundary, as Approach 5 does.
    inner = cv2.erode(small_mask, np.ones((7, 7), np.uint8)).astype(bool)
    inner &= (value > 18) & (value < 248)
    parts = []
    for points, radius in LBP_SETTINGS:
        codes = local_binary_pattern(gray, points, radius, method="uniform")
        histogram, _ = np.histogram(codes[inner], bins=points + 2,
                                    range=(0, points + 2))
        parts.append(histogram / max(1, histogram.sum()))
    return np.concatenate(parts)


def silhouette_features(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    hull_points = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull_points)
    perimeter = cv2.arcLength(contour, True)
    hull_perimeter = cv2.arcLength(hull_points, True)
    scale = np.sqrt(max(area, 1.0))
    hull_index = cv2.convexHull(contour, returnPoints=False)
    depths = np.zeros(0)
    if hull_index is not None and len(hull_index) > 3:
        defects = cv2.convexityDefects(contour, np.sort(hull_index, axis=0))
        if defects is not None:
            depths = defects[:, 0, 3] / 256.0 / scale
    _, (width, height), _ = cv2.minAreaRect(contour)
    polygon = cv2.approxPolyDP(contour, .01 * perimeter, True)
    return np.asarray([
        area / max(hull_area, 1.0),
        float(depths.max(initial=0)),
        float(np.sort(depths)[-3:].sum()) if depths.size else 0.0,
        float(np.sum(depths > .02)),
        perimeter / max(hull_perimeter, 1.0),
        area / max(width * height, 1.0),
        float(len(polygon)),
    ], np.float32)


def extract_one(path: Path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mask = material_mask(rgb)
    balanced, reference = paper_white_balance(rgb, mask)
    lab, usable, _ = usable_pixels(balanced, mask)
    paper_wb_a5 = np.r_[lab_histogram(lab, usable), texture_features(balanced, usable)]
    return (paper_wb_a5.astype(np.float32), lbp_features(rgb, mask).astype(np.float32),
            silhouette_features(mask), reference.astype(np.float32))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=12)
    args = parser.parse_args()
    rows = discover_images(args.data)
    results = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(extract_one)(row["path"]) for row in rows)
    paper_wb_a5, lbp, silhouette, paper_rgb = map(np.asarray, zip(*results))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, paper_wb_a5=paper_wb_a5, lbp=lbp,
                        silhouette=silhouette, paper_rgb=paper_rgb,
                        labels=np.asarray([row["label"] for row in rows]),
                        groups=np.asarray([row["brick_id"] for row in rows]),
                        files=np.asarray([str(row["path"]) for row in rows]))
    with args.out.with_suffix(".paper_reference.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["file", "brick_id", "paper_r", "paper_g", "paper_b"])
        for row, reference in zip(rows, paper_rgb):
            writer.writerow([row["path"], row["brick_id"], *np.round(reference, 2)])
    print({"paper_wb_a5": paper_wb_a5.shape, "lbp": lbp.shape,
           "silhouette": silhouette.shape})


if __name__ == "__main__":
    main()
