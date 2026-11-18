"""Saved decision policies for a frozen Approach 17 score."""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NAMES = ("baseline_margin", "zero_binary", "minus_two_binary",
         "review_band_175_120")


@lru_cache(maxsize=len(NAMES))
def load_policy(name: str) -> dict:
    if name not in NAMES:
        raise ValueError(f"Unknown threshold policy: {name}")
    policy = json.loads((ROOT / f"{name}.json").read_text())
    if policy.get("name") != name:
        raise ValueError(f"Policy file name mismatch: {name}")
    return policy


def label_for(score: float, policy: dict) -> str:
    score = float(score)
    if not math.isfinite(score):
        raise ValueError("Nonfinite classifier score")
    if policy["kind"] == "two_threshold":
        good_max = float(policy["good_max"])
        bad_min = float(policy["bad_min_exclusive"])
        if not good_max < bad_min:
            raise ValueError("Two-threshold policy must have good_max < bad_min_exclusive")
        return "good" if score <= good_max else "bad" if score > bad_min else "uncertain"
    threshold = float(policy["threshold"])
    if policy["kind"] == "margin":
        margin = float(policy["margin"])
        if abs(score - threshold) < margin:
            return "uncertain"
        return "bad" if score >= threshold else "good"
    if policy["kind"] == "binary":
        if score == threshold:
            return policy["equal_label"]
        return "bad" if score > threshold else "good"
    raise ValueError(f"Unknown policy kind: {policy['kind']}")


def labels_for_all(score: float) -> dict[str, str]:
    return {name: label_for(score, load_policy(name)) for name in NAMES}
