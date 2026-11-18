#!/usr/bin/env python3
"""Redraw saved SAM3 contours without re-running SAM3 or the classifier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from offline_sam3_classifier import annotate_soft_boundary


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=ROOT / "brick_splits/images")
    parser.add_argument("--results", type=Path, default=ROOT / "offline_results")
    parser.add_argument("--outline-width", type=int, default=3)
    args = parser.parse_args()
    count = 0
    for metadata_path in sorted(args.results.glob("Brick_*/*.json")):
        metadata = json.loads(metadata_path.read_text())
        if metadata["status"] != "ok":
            continue
        folder = metadata_path.parent.name
        image_path = args.images / folder / metadata["image"]
        mask_path = metadata_path.parent / "masks" / f"{metadata_path.stem}.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None or image.shape[:2] != mask.shape:
            raise ValueError(f"Missing or mismatched input for {metadata_path}")
        rendered = annotate_soft_boundary(image, (mask > 0).astype("uint8"),
                                          metadata["label"], float(metadata["score_bad"]),
                                          args.outline_width)
        if not cv2.imwrite(str(metadata_path.with_suffix(Path(metadata["image"]).suffix)), rendered):
            raise OSError(f"Could not write annotation for {metadata_path}")
        count += 1
    print(f"Redrew {count} annotated images with {args.outline_width}-pixel contours")


if __name__ == "__main__":
    main()
