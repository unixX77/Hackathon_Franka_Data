#!/usr/bin/env python3
"""Locate the perforated (cored) face of a brick with SAM3 and cache its colour.

The perforated face is the one surface whose fired red is visible on every
brick; the other faces carry soot, lime bloom or dust that also occurs on
good bricks. SAM3 is prompted with "hole" inside the brick crop; confident
detections are the perforations. The face is their convex hull grown by half
a perforation, minus the perforations themselves. A photo with fewer than
``MIN_HOLES`` perforations has no measurable face and is flagged.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

from extract_features import (DEFAULT_IMAGES, DEFAULT_MASKS, discover,
                              paper_colour, white_balance)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "segmentation_sam3_bounding_box"))
from segment_lego_sam3_single_prompt import dedupe, run_sam3  # noqa: E402

SAM3_SNAPSHOT = Path("/cache/s95vkond/huggingface/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7")
HOLE_SCORE = 0.8
MIN_HOLE_AREA = 0.003  # fraction of the brick mask; smaller detections are surface pits
MIN_HOLES = 2
PIXELS_PER_IMAGE = 20000


class PerforationFinder:
    def __init__(self, snapshot: Path = SAM3_SNAPSHOT, model=None, processor=None):
        """Pass an already loaded SAM3 ``model``/``processor`` to share it."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.processor = processor or Sam3Processor.from_pretrained(snapshot)
        self.model = model or Sam3Model.from_pretrained(snapshot, torch_dtype=self.dtype).to(self.device).eval()
        self.dtype = next(self.model.parameters()).dtype

    def __call__(self, rgb: np.ndarray, brick: np.ndarray):
        """Return (face mask, perforation mask, number of perforations), full frame."""
        ys, xs = np.nonzero(brick)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        crop_brick = brick[y0:y1, x0:x1]
        masks, scores = run_sam3(self.model, self.processor, Image.fromarray(rgb[y0:y1, x0:x1]),
                                 "hole", HOLE_SCORE, 0.5, self.device, self.dtype)
        holes = []
        for i in dedupe(masks, scores, 0.5) if len(scores) else []:
            area = masks[i].sum()
            if area >= MIN_HOLE_AREA * crop_brick.sum() and (masks[i] & crop_brick).sum() >= .8 * area:
                holes.append(masks[i])
        face = np.zeros_like(brick)
        perforations = np.zeros_like(brick)
        if len(holes) >= MIN_HOLES:
            union = np.any(holes, axis=0)
            hull = cv2.convexHull(cv2.findNonZero(union.astype(np.uint8)))
            grown = np.zeros(union.shape, np.uint8)
            cv2.fillConvexPoly(grown, hull, 1)
            size = int(np.sqrt(np.median([h.sum() for h in holes])))
            grown = cv2.dilate(grown, np.ones((size, size), np.uint8))  # ~half a hole each side
            walls = cv2.dilate(union.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            face[y0:y1, x0:x1] = (grown > 0) & crop_brick & ~walls
            perforations[y0:y1, x0:x1] = union
        return face, perforations, len(holes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--masks", type=Path, default=DEFAULT_MASKS)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    (args.out / "face_masks").mkdir(parents=True, exist_ok=True)
    finder = PerforationFinder()
    rng = np.random.default_rng(2026)
    rows = discover(args.images)
    n_holes, face_px, samples = [], [], []
    for brick_id, _, path in rows:
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        brick = cv2.imread(str(args.masks / brick_id / "masks" / path.name), cv2.IMREAD_GRAYSCALE) > 0
        face, perforations, count = finder(rgb, brick)
        wb = white_balance(rgb, paper_colour(rgb, brick))
        idx = np.flatnonzero(face)
        if len(idx):
            idx = rng.choice(idx, PIXELS_PER_IMAGE, replace=len(idx) < PIXELS_PER_IMAGE)
            samples.append(wb.reshape(-1, 3)[idx])
        else:
            samples.append(np.zeros((PIXELS_PER_IMAGE, 3), np.uint8))
        n_holes.append(count)
        face_px.append(int(face.sum()))
        cv2.imwrite(str(args.out / "face_masks" / f"{brick_id}__{path.name}"),
                    (face * 255 + perforations * 120).astype(np.uint8))
        print(f"{brick_id}/{path.name} perforations={count} face_px={face.sum()}", flush=True)
    np.savez_compressed(args.out / "face_samples.npz", wb=np.array(samples),
                        n_holes=np.array(n_holes), face_px=np.array(face_px))


if __name__ == "__main__":
    main()
