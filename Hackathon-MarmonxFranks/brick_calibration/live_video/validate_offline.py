#!/usr/bin/env python3
"""Audit the live score guard on existing brick-held-out Approach 17 scores."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from model import SCORE_LIMITS


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "approach_results/17_late_fusion_approach5_dinov2_patch/held_out_image_predictions.csv"
OUTPUT = Path(__file__).resolve().parent / "results/offline_guard_metrics.json"


def main() -> None:
    rows = list(csv.DictReader(SOURCE.open()))
    accepted = []
    outliers = []
    for row in rows:
        a5 = float(row["score_approach5"])
        dino = float(row["score_dinov2_patch"])
        score = float(row["decision_score_bad"])
        if abs(a5) > SCORE_LIMITS["approach5"] or abs(dino) > SCORE_LIMITS["dinov2"]:
            outliers.append(row["file"])
        elif abs(score) >= .3:
            accepted.append(row)
    report = {
        "source": str(SOURCE),
        "note": "Held-out photo scores only; does not test camera mask quality or live-domain accuracy.",
        "baseline_sign_accuracy": sum(row["correct"] == "1" for row in rows) / len(rows),
        "total_photos": len(rows),
        "accepted_photos": len(accepted),
        "accepted_accuracy": sum(row["correct"] == "1" for row in accepted) / len(accepted),
        "uncertain_count": len(rows) - len(accepted),
        "extreme_score_files": outliers,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
