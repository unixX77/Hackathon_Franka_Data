"""One-class "target redness" model for the perforated brick face.

Good bricks share one fired red. Bad bricks miss it in different directions
(duller, more yellow, darker, paler), so a two-class boundary learned from a
few bad bricks does not cover an unseen failure direction. The model instead
learns the good colour only: a window around the good-brick mean, scaled by
the good-brick spread, and a threshold on the distance from it.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from sklearn.covariance import LedoitWolf

FEATURE_NAMES = ("L", "a", "b", "chroma", "hue", "r", "g")
FEATURE_SETS = {
    "L,a,b": ("L", "a", "b"),
    "L,chroma,hue": ("L", "chroma", "hue"),
    "L,a,b,hue": ("L", "a", "b", "hue"),
}
DISTANCES = ("rms", "max", "mahalanobis")
OVEREXPOSED_PAPER = 230  # paper channel near clipping: white balance is unreliable


def face_colour_features(wb_pixels: np.ndarray) -> np.ndarray:
    """Median colour of white-balanced face pixels (uint8 sRGB, N x 3)."""
    x = wb_pixels.astype(np.float32) / 255.0
    lab = cv2.cvtColor(x.reshape(-1, 1, 3), cv2.COLOR_RGB2Lab).reshape(-1, 3)
    L, a, b = np.median(lab, axis=0)
    linear = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    total = linear.sum(1) + 1e-6
    r, g = np.median(linear[:, 0] / total), np.median(linear[:, 1] / total)
    return np.array([L, a, b, np.hypot(a, b), np.degrees(np.arctan2(b, a)), r, g])


def class_balanced_brick_accuracy(scores, threshold, y, groups) -> float:
    """Mean of (mean good-brick accuracy, mean bad-brick accuracy)."""
    per_brick = {g: (y[groups == g][0], np.mean((scores[groups == g] > threshold) == y[groups == g]))
                 for g in np.unique(groups)}
    return float(np.mean([np.mean([acc for label, acc in per_brick.values() if label == c])
                          for c in (0, 1)]))


@dataclass
class RednessWindow:
    features: tuple
    distance: str
    mean: list | None = None
    scale: list | None = None
    precision: list | None = None
    threshold: float = 0.0

    def _columns(self, X: np.ndarray) -> np.ndarray:
        return X[:, [FEATURE_NAMES.index(f) for f in self.features]]

    def _fit_window(self, good: np.ndarray) -> "RednessWindow":
        self.mean = good.mean(0).tolist()
        self.scale = good.std(0).tolist()
        self.precision = LedoitWolf().fit(good).precision_.tolist()
        return self

    def _distance(self, Z: np.ndarray) -> np.ndarray:
        d = Z - np.array(self.mean)
        if self.distance == "rms":
            return np.sqrt(((d / np.array(self.scale)) ** 2).mean(1))
        if self.distance == "max":
            return np.abs(d / np.array(self.scale)).max(1)
        return np.sqrt(np.einsum("ij,jk,ik->i", d, np.array(self.precision), d))

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> "RednessWindow":
        """X: all features; y: 1 = bad; groups: physical brick ids.

        The threshold is set where held-out training bricks are classified
        best, so it reflects how far an unseen good brick may drift from the
        others rather than how tightly the training photos cluster.
        """
        Z = self._columns(X)
        inner = np.zeros(len(y))
        for g in np.unique(groups):
            held = groups == g
            probe = RednessWindow(self.features, self.distance)._fit_window(Z[~held & (y == 0)])
            inner[held] = probe._distance(Z[held])
        candidates = np.unique(inner)
        self.threshold = float(max(candidates, key=lambda t: class_balanced_brick_accuracy(inner, t, y, groups)))
        self._fit_window(Z[y == 0])
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Positive = outside the good-redness window = bad."""
        return self._distance(self._columns(X)) - self.threshold

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "RednessWindow":
        data = json.loads(Path(path).read_text())
        data["features"] = tuple(data["features"])
        return cls(**data)
