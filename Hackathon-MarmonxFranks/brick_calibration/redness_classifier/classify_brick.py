#!/usr/bin/env python3
"""Classify brick photos as good / bad by the redness of the perforated face.

Pipeline per photo: SAM3 "brick" mask -> SAM3 "hole" perforations -> face
pixels -> white balance against the paper -> median L*, chroma, hue ->
distance from the good-redness window. A photo gets ``no_verdict`` when the
perforated face is not visible or the paper is over-exposed; take another view.

    python classify_brick.py photo1.png photo2.png --annotate out_dir/

Several photos of the same brick can be combined with ``--one-brick``: the
brick verdict uses the median score of the photos that have a verdict.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from extract_features import paper_colour, white_balance  # noqa: E402
from offline_sam3_classifier import SAM3_SNAPSHOT, dominant_brick  # noqa: E402
from perforated_face import PerforationFinder  # noqa: E402
from redness_model import OVEREXPOSED_PAPER, RednessWindow, face_colour_features  # noqa: E402
from segment_lego_sam3_single_prompt import BrickGeometrySegmenter  # noqa: E402

DEFAULT_MODEL = HERE / "results" / "redness_window.json"
COLOURS = {"good": (0, 180, 0), "bad": (0, 0, 225), "no_verdict": (0, 190, 255)}  # BGR


class RednessMaskClassifier:
    """Score one brick given its mask; ``classify`` matches the sam3_live server interface."""

    def __init__(self, model_path: Path = DEFAULT_MODEL, sam_model=None, sam_processor=None):
        self.window = RednessWindow.load(model_path)
        self.perforations = PerforationFinder(model=sam_model, processor=sam_processor)

    def __call__(self, rgb: np.ndarray, brick: np.ndarray) -> dict:
        brick = brick.astype(bool)
        face, _, n_holes = self.perforations(rgb, brick)
        paper = paper_colour(rgb, brick)
        result = {"brick_mask": brick, "face_mask": face, "perforations": n_holes}
        if n_holes < 2:
            return result | {"verdict": "no_verdict", "reason": "perforated face not visible"}
        if paper.max() >= OVEREXPOSED_PAPER:
            return result | {"verdict": "no_verdict", "reason": "over-exposed: paper near white clip"}
        features = face_colour_features(white_balance(rgb, paper)[face])
        score = float(self.window.decision_function(features[None])[0])
        named = dict(zip(("L", "a", "b", "chroma", "hue", "r", "g"), np.round(features.astype(float), 3).tolist()))
        return result | {"verdict": "bad" if score > 0 else "good", "score": round(score, 3),
                         "face_colour": named}

    def classify(self, rgb: np.ndarray, mask: np.ndarray) -> dict:
        """Live-server form: no verdict is reported as label "uncertain" with a NaN score."""
        result = self(rgb, mask)
        verdict = result["verdict"]
        return {"label": "uncertain" if verdict == "no_verdict" else verdict,
                "score_bad": result.get("score", float("nan")),
                "reason": result.get("reason", ""), "perforations": result["perforations"],
                "face_colour": result.get("face_colour", {})}


class RednessClassifier:
    """Whole photo: SAM3 picks the central brick, then RednessMaskClassifier scores it."""

    def __init__(self, model_path: Path = DEFAULT_MODEL):
        args = argparse.Namespace(model=str(SAM3_SNAPSHOT), prompt="brick", threshold=.35,
                                  mask_threshold=.5, max_overlap=.5, min_area=1200, min_rect=.55)
        self.bricks = BrickGeometrySegmenter(args)
        self.scorer = RednessMaskClassifier(model_path, self.bricks.model, self.bricks.processor)

    def __call__(self, rgb: np.ndarray) -> dict:
        instance, records, _ = self.bricks(rgb)
        selected = dominant_brick(instance, records)
        if selected is None:
            return {"verdict": "no_verdict", "reason": "no brick found"}
        return self.scorer(rgb, selected[0])


def annotate(bgr: np.ndarray, result: dict) -> np.ndarray:
    out = bgr.copy()
    colour = COLOURS[result["verdict"]]
    for key, thickness in (("brick_mask", 6), ("face_mask", 3)):
        if key in result:
            contours, _ = cv2.findContours(result[key].astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, colour, thickness)
    text = result["verdict"].upper()
    if "score" in result:
        text += f"  score {result['score']:+.2f}"
    elif "reason" in result:
        text += f"  ({result['reason']})"
    cv2.putText(out, text, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, colour, 5)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--annotate", type=Path, help="folder for annotated copies")
    parser.add_argument("--one-brick", action="store_true", help="all photos show the same brick")
    args = parser.parse_args()
    classifier = RednessClassifier(args.model)
    if args.annotate:
        args.annotate.mkdir(parents=True, exist_ok=True)
    scores = []
    for path in args.images:
        bgr = cv2.imread(str(path))
        result = classifier(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if args.annotate:
            cv2.imwrite(str(args.annotate / path.name), annotate(bgr, result))
        public = {k: v for k, v in result.items() if not k.endswith("_mask")}
        print(json.dumps({"image": str(path)} | public))
        if "score" in result:
            scores.append(result["score"])
    if args.one_brick:
        if scores:
            median = float(np.median(scores))
            print(json.dumps({"brick_verdict": "bad" if median > 0 else "good",
                              "median_score": round(median, 3), "photos_used": len(scores)}))
        else:
            print(json.dumps({"brick_verdict": "no_verdict", "photos_used": 0}))


if __name__ == "__main__":
    main()
