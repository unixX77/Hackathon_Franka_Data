"""Camera-frame adapter around the frozen Approach 17 brick classifier.

The saved Approach 17 SVMs and DINOv2 encoder are never modified here.
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path

import cv2
import numpy as np

PARENT = Path(__file__).resolve().parents[1]
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from evaluate_approaches import material_mask  # noqa: E402
from fusion_checkpoint import FusionClassifier  # noqa: E402


CHECKPOINT = PARENT / "approach_results" / "baseline_checkpoint"
# Conservative limits, wider than the fitted checkpoint's training ranges
# (A5 -8.17..4.52; DINO -11.52..9.41). Outside them, abstain.
SCORE_LIMITS = {"approach5": 20.0, "dinov2": 24.0}


def _roi(frame: np.ndarray, roi: list[int] | None) -> tuple[np.ndarray, tuple[int, int]]:
    if roi is None:
        return frame, (0, 0)
    if len(roi) != 4:
        raise ValueError("ROI must contain x, y, width, height")
    x, y, width, height = map(int, roi)
    if (x < 0 or y < 0 or width < 64 or height < 64 or
            x + width > frame.shape[1] or y + height > frame.shape[0]):
        raise ValueError("ROI is outside the camera frame or smaller than 64 pixels")
    return frame[y:y + height, x:x + width], (x, y)


def _mask_quality(rgb: np.ndarray, mask: np.ndarray) -> tuple[list[str], dict]:
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    height, width = mask.shape
    coverage = float(mask.mean())
    box_fraction = float((x1 - x0 + 1) * (y1 - y0 + 1) / (width * height))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    values = hsv[..., 2][mask.astype(bool)]
    clipped = float(np.mean((values <= 18) | (values >= 248)))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    border = max(4, round(min(height, width) * .02))
    border_pixels = np.concatenate((lab[:border].reshape(-1, 3),
                                    lab[-border:].reshape(-1, 3),
                                    lab[:, :border].reshape(-1, 3),
                                    lab[:, -border:].reshape(-1, 3)))
    background = np.median(border_pixels, axis=0)
    background_like = float(np.mean(
        np.linalg.norm(lab[mask.astype(bool)] - background, axis=1) < 18))
    flags = []
    if x0 < 3 or y0 < 3 or x1 >= width - 3 or y1 >= height - 3:
        flags.append("mask_touches_crop_edge")
    if coverage < .035 or coverage > .72 or box_fraction > .85:
        flags.append("implausible_mask_size")
    if clipped > .25:
        flags.append("material_over_or_underexposed")
    if background_like > .20:
        flags.append("mask_contains_background_colours")
    return flags, {"mask_fraction": round(coverage, 4),
                   "box_fraction": round(box_fraction, 4),
                   "clipped_material_fraction": round(clipped, 4),
                   "background_like_fraction": round(background_like, 4),
                   "local_box": [x0, y0, x1 - x0 + 1, y1 - y0 + 1]}


class LiveVideoClassifier:
    def __init__(self, checkpoint: Path = CHECKPOINT, device: str | None = None):
        self.classifier = FusionClassifier.load(checkpoint, device=device)

    def predict(self, frames: list[np.ndarray], roi: list[int] | None = None) -> dict:
        """Score a fresh burst; no predictions persist across different bricks."""
        if not 1 <= len(frames) <= 5:
            raise ValueError("One to five frames are required per prediction")
        scores, last, failed = [], None, []
        for frame_index, frame in enumerate(frames):
            crop, (offset_x, offset_y) = _roi(frame, roi)
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            try:
                mask = material_mask(rgb)
                flags, quality = _mask_quality(rgb, mask)
                if flags:
                    failed.extend(flags)
                    continue
                prediction = self.classifier.classify(rgb, mask=mask)
                x, y, width, height = quality.pop("local_box")
                encoded, png = cv2.imencode(".png", mask * 255)
                if not encoded:
                    raise ValueError("Could not encode brick mask")
                last = {"box_x": x + offset_x, "box_y": y + offset_y,
                        "box_width": width, "box_height": height,
                        "mask_png_b64": base64.b64encode(png).decode("ascii"),
                        "mask_origin_x": offset_x, "mask_origin_y": offset_y,
                        "frame_index_used": frame_index,
                        **quality}
                scores.append((prediction["score_bad"],
                               prediction["approach5_score"],
                               prediction["dinov2_score"]))
            except ValueError as error:
                failed.append(f"mask_failed:{error}")
        if not scores:
            return {"status": "uncertain", "label": "uncertain",
                    "quality_flags": sorted(set(failed)), "frames_used": 0}
        values = np.median(np.asarray(scores), axis=0)
        fused, a5, dino = map(float, values)
        flags = sorted(set(failed))
        if abs(a5) > SCORE_LIMITS["approach5"]:
            flags.append("approach5_score_outside_training_range")
        if abs(dino) > SCORE_LIMITS["dinov2"]:
            flags.append("dinov2_score_outside_training_range")
        # Keep Approach 17's original score decision for valid, familiar images.
        label = self.classifier._decide(fused)
        if flags or len(scores) < (len(frames) + 1) // 2:
            label = "uncertain"
        return {"status": "ok", "label": label, "score_bad": fused,
                "approach5_score": a5, "dinov2_score": dino,
                "quality_flags": flags, "frames_used": len(scores),
                "frames_received": len(frames), **last}


def decode_jpeg(image_b64: str) -> np.ndarray:
    encoded = np.frombuffer(base64.b64decode(image_b64, validate=True), np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None or max(frame.shape[:2]) > 4096:
        raise ValueError("Invalid JPEG or frame exceeds 4096 pixels")
    return frame
