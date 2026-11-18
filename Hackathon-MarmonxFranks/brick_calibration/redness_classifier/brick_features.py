"""Feature sets built from the cached colour samples and DINOv2 embeddings."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MIN_L = 20.0  # drops perforation interiors and deep shadow
L_RANGE, A_RANGE, B_RANGE = (20, 90), (-5, 40), (0, 45)
PERCENTILES = (5, 25, 50, 75, 95)


def to_lab(pixels: np.ndarray) -> np.ndarray:
    """uint8 sRGB (..., N, 3) -> float CIELAB with L in 0..100."""
    shape = pixels.shape
    lab = cv2.cvtColor((pixels.reshape(-1, 1, 3) / 255.0).astype(np.float32), cv2.COLOR_RGB2Lab)
    return lab.reshape(shape)


def _hist(x, bins, rng):
    h, _ = np.histogram(x, bins=bins, range=rng)
    return h / max(1, len(x))


def colour_vector(lab: np.ndarray, kind: str) -> np.ndarray:
    lab = lab[lab[:, 0] > MIN_L]
    L, a, b = lab.T
    chroma = np.hypot(a, b)
    hue = np.degrees(np.arctan2(b, a))
    stats = np.concatenate([np.percentile(c, PERCENTILES) for c in (L, a, b, chroma, hue)]
                           + [[c.std() for c in (L, a, b, chroma, hue)]])
    if kind == "stats":
        return stats
    hists = np.concatenate([_hist(L, 14, L_RANGE), _hist(a, 15, A_RANGE),
                            _hist(b, 15, B_RANGE), _hist(chroma, 15, (0, 50)),
                            _hist(hue, 15, (20, 80))])
    if kind == "hist":
        return np.concatenate([stats, hists])
    if kind == "joint":
        # Joint lightness x hue x chroma occupancy: "how much of the brick is
        # this exact shade of red", which 1-D marginals cannot express.
        joint, _ = np.histogramdd(np.c_[L, hue, chroma], bins=(5, 5, 5),
                                  range=(L_RANGE, (20, 80), (0, 50)))
        return np.concatenate([stats, hists, joint.ravel() / len(L)])
    raise ValueError(kind)


def load(cache: Path):
    data = np.load(cache / "colour_samples.npz")
    meta = {k: data[k] for k in ("brick", "label", "image", "paper")}
    lab_wb = to_lab(data["wb"])
    lab_raw = to_lab(data["raw"])
    sets = {}
    for kind in ("stats", "hist", "joint"):
        sets[f"colour_{kind}_wb"] = np.stack([colour_vector(x, kind) for x in lab_wb])
    sets["colour_hist_raw"] = np.stack([colour_vector(x, "hist") for x in lab_raw])
    sets["paper_only"] = meta["paper"].astype(np.float64)  # confound check, must be ~chance
    for path in sorted(cache.glob("dinov2-*.npy")):
        sets[path.stem] = np.load(path).astype(np.float64)
    sets["dinov2-base_wb+colour_joint_wb"] = np.hstack([sets["dinov2-base_wb"], sets["colour_joint_wb"]])
    sets["dinov2-small_wb+colour_joint_wb"] = np.hstack([sets["dinov2-small_wb"], sets["colour_joint_wb"]])
    return meta, sets
