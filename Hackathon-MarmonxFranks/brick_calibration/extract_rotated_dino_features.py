#!/usr/bin/env python3
"""Frozen DINOv2 patch-distribution features for 0/90/180/270-degree crops.

Uses the exact masked crop and patch summary (mean, std, P90) of Approach 8;
only the in-plane rotation of the crop differs. Output shape is
(4 rotations, images, 1152).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

from evaluate_approaches import discover_images
from evaluate_dino_approaches import prepare_crop

ROTATIONS = (0, 90, 180, 270)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="facebook/dinov2-small")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    rows = discover_images(args.data)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    crops = [prepare_crop(row["path"]) for row in rows]
    features = np.zeros((len(ROTATIONS), len(rows), 3 * model.config.hidden_size), np.float32)
    for r, angle in enumerate(ROTATIONS):
        for start in range(0, len(rows), args.batch_size):
            batch = [np.ascontiguousarray(np.rot90(crop, k=angle // 90))
                     for crop in crops[start:start + args.batch_size]]
            pixels = processor(images=batch, return_tensors="pt")["pixel_values"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=device == "cuda"):
                hidden = model(pixel_values=pixels).last_hidden_state
            patches = hidden.float().cpu().numpy()[:, 1:]
            features[r, start:start + len(batch)] = np.concatenate((
                patches.mean(axis=1), patches.std(axis=1),
                np.percentile(patches, 90, axis=1)), axis=1)
        print(f"rotation {angle} done", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, patch_distribution=features,
                        rotations=np.asarray(ROTATIONS), model=args.model,
                        groups=np.asarray([row["brick_id"] for row in rows]))


if __name__ == "__main__":
    main()
