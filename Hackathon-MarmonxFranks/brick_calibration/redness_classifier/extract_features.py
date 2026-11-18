#!/usr/bin/env python3
"""Cache per-image colour samples and frozen DINOv2 embeddings.

Every photo is white-balanced against the white paper it lies on, so the
classifier sees brick colour rather than the lighting of the capture session
(all good bricks were photographed before all bad bricks). Brick pixels come
from the saved SAM3 masks in ``offline_results/<brick>/masks``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoModel

HERE = Path(__file__).resolve().parent
CALIBRATION = HERE.parent
DEFAULT_IMAGES = CALIBRATION / "brick_splits" / "images"
DEFAULT_MASKS = CALIBRATION / "offline_results"
DINO_MODELS = ("facebook/dinov2-small", "facebook/dinov2-base", "facebook/dinov2-large")
PIXELS_PER_IMAGE = 40000
DINO_SIDE = 448  # 32 x 32 patches of 14 px
PAPER_GREY = 0.8  # linear reflectance the paper is mapped to


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def paper_colour(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Median sRGB of the brighter half of the paper around the brick."""
    h, w = mask.shape
    near_brick = cv2.dilate(mask.astype(np.uint8), np.ones((61, 61), np.uint8)) > 0
    region = np.zeros_like(mask)
    region[int(.1 * h):int(.9 * h), int(.15 * w):int(.85 * w)] = True  # avoids wood corners
    paper = rgb[region & ~near_brick].astype(np.float32)
    luma = paper.mean(1)
    return np.median(paper[luma > np.percentile(luma, 50)], axis=0)


def white_balance(rgb: np.ndarray, paper: np.ndarray) -> np.ndarray:
    gain = PAPER_GREY / srgb_to_linear(paper / 255.0)
    out = linear_to_srgb(srgb_to_linear(rgb.astype(np.float32) / 255.0) * gain)
    return (out * 255 + 0.5).astype(np.uint8)


def brick_crop(rgb: np.ndarray, mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    margin = round(.04 * max(np.ptp(xs), np.ptp(ys)))
    x0, x1 = max(0, xs.min() - margin), min(rgb.shape[1], xs.max() + margin + 1)
    y0, y1 = max(0, ys.min() - margin), min(rgb.shape[0], ys.max() + margin + 1)
    crop, selected = rgb[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1]
    crop[~selected] = 128
    # Pad to a square so the brick keeps its aspect ratio.
    side = max(crop.shape[:2])
    square = np.full((side, side, 3), 128, np.uint8)
    square_mask = np.zeros((side, side), bool)
    oy, ox = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
    square[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    square_mask[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = selected
    return square, square_mask


def discover(images: Path):
    rows = []
    for folder in sorted(images.glob("Brick_*"), key=lambda p: int(p.name.split("_")[1])):
        for path in sorted(folder.glob("*.png")):
            rows.append((folder.name, folder.name.rsplit("_", 1)[1], path))
    return rows


@torch.inference_mode()
def dino_embeddings(model_name: str, crops, masks, device: str) -> np.ndarray:
    """CLS token and mask-weighted mean patch token, averaged over 4 rotations."""
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    grid = DINO_SIDE // 14
    out = []
    for crop, mask in zip(crops, masks):
        image = cv2.resize(crop, (DINO_SIDE, DINO_SIDE), interpolation=cv2.INTER_AREA)
        weight = cv2.resize(mask.astype(np.float32), (grid, grid), interpolation=cv2.INTER_AREA)
        images = np.stack([np.rot90(image, k) for k in range(4)])
        weights = np.stack([np.rot90(weight, k) for k in range(4)]).reshape(4, -1, 1)
        x = torch.from_numpy(images.copy()).to(device).permute(0, 3, 1, 2).float() / 255
        hidden = model(pixel_values=(x - mean) / std).last_hidden_state
        cls = hidden[:, 0]
        patches = hidden[:, -grid * grid:]  # skips register tokens if present
        w = torch.from_numpy(weights.copy()).to(device)
        pooled = (patches * w).sum(1) / w.sum(1).clamp_min(1e-6)
        out.append(torch.cat([cls, pooled], 1).mean(0).cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.stack(out).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--masks", type=Path, default=DEFAULT_MASKS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", nargs="*", default=list(DINO_MODELS))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = discover(args.images)
    rng = np.random.default_rng(2026)
    papers, raw_pixels, wb_pixels, crops_raw, crops_wb, crop_masks = [], [], [], [], [], []
    for brick, _, path in rows:
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(args.masks / brick / "masks" / path.name), cv2.IMREAD_GRAYSCALE) > 0
        paper = paper_colour(rgb, mask)
        wb = white_balance(rgb, paper)
        inner = cv2.erode(mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        idx = np.flatnonzero(inner)
        idx = rng.choice(idx, PIXELS_PER_IMAGE, replace=len(idx) < PIXELS_PER_IMAGE)
        papers.append(paper)
        raw_pixels.append(rgb.reshape(-1, 3)[idx])
        wb_pixels.append(wb.reshape(-1, 3)[idx])
        crop, crop_mask = brick_crop(rgb, mask)
        crops_raw.append(crop)
        crops_wb.append(brick_crop(wb, mask)[0])
        crop_masks.append(crop_mask)
        print(f"{brick}/{path.name}", flush=True)

    np.savez_compressed(
        args.out / "colour_samples.npz",
        brick=np.array([r[0] for r in rows]), label=np.array([r[1] for r in rows]),
        image=np.array([r[2].name for r in rows]), paper=np.array(papers),
        raw=np.array(raw_pixels), wb=np.array(wb_pixels))

    for name in args.models:
        tag = name.split("/")[-1]
        for variant, crops in (("wb", crops_wb), ("raw", crops_raw)):
            features = dino_embeddings(name, crops, crop_masks, device)
            np.save(args.out / f"{tag}_{variant}.npy", features)
            print(f"saved {tag}_{variant} {features.shape}", flush=True)


if __name__ == "__main__":
    main()
