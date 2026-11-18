#!/usr/bin/env python3
"""Evaluate the perforated-face redness window and save the final model.

Protocols (photos of one physical brick never appear in both train and test):
  * leave-one-brick-out over the 10 bricks; the feature set, distance and
    threshold are all chosen inside each outer fold from the 9 training
    bricks only (fully nested, so the table of candidates does not leak);
  * the 24 provided ``brick_splits/with_brick6`` splits (train+val -> test).
Photos without a visible perforated face or with over-exposed paper get no
verdict; they are reported as coverage and, in the strict metric, as errors.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from redness_model import (DISTANCES, FEATURE_SETS, OVEREXPOSED_PAPER,
                           RednessWindow, class_balanced_brick_accuracy,
                           face_colour_features)

CALIBRATION = Path(__file__).resolve().parent.parent
SPLITS = CALIBRATION / "brick_splits" / "with_brick6"
CANDIDATES = [(f, d) for f, d in itertools.product(FEATURE_SETS, DISTANCES)]


def select_and_fit(X, y, groups) -> tuple[RednessWindow, dict]:
    """Choose (feature set, distance) by inner leave-one-brick-out, then fit."""
    scores = {}
    for name, distance in CANDIDATES:
        inner = np.zeros(len(y))
        for g in np.unique(groups):
            held = groups == g
            model = RednessWindow(FEATURE_SETS[name], distance).fit(X[~held], y[~held], groups[~held])
            inner[held] = model.decision_function(X[held])
        scores[(name, distance)] = class_balanced_brick_accuracy(inner, 0, y, groups)
    name, distance = max(scores, key=scores.get)
    return RednessWindow(FEATURE_SETS[name], distance).fit(X, y, groups), \
        {"features": name, "distance": distance, "inner_score": scores[(name, distance)]}


def metrics(scores, y, groups, valid):
    """Image metrics on photos with a verdict, plus coverage and brick votes."""
    pred = scores > 0
    v = valid
    good, bad = v & (y == 0), v & (y == 1)
    bricks = {}
    for g in np.unique(groups):
        m = v & (groups == g)
        bricks[str(g)] = {"label": "bad" if y[groups == g][0] else "good",
                          "photos_with_verdict": int(m.sum()), "photos": int((groups == g).sum()),
                          "image_accuracy": float(np.mean(pred[m] == y[m])),
                          "mean_score": float(scores[m].mean()),
                          "brick_correct": bool((scores[m].mean() > 0) == y[m][0])}
    return {"image_accuracy": float(np.mean(pred[v] == y[v])),
            "balanced_accuracy": float((np.mean(~pred[good]) + np.mean(pred[bad])) / 2),
            "good_recall": float(np.mean(~pred[good])), "bad_recall": float(np.mean(pred[bad])),
            "coverage": float(v.mean()),
            "strict_accuracy_no_verdict_counts_wrong": float(np.sum(pred[v] == y[v]) / len(y)),
            "brick_accuracy": float(np.mean([b["brick_correct"] for b in bricks.values()])),
            "per_brick": bricks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True,
                        help="folder with colour_samples.npz and face_samples.npz")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    colour = np.load(args.cache / "colour_samples.npz")
    face = np.load(args.cache / "face_samples.npz")
    groups, images = colour["brick"], colour["image"]
    y = (colour["label"] == "bad").astype(int)
    face_visible = face["n_holes"] >= 2
    exposed_ok = colour["paper"].max(1) < OVEREXPOSED_PAPER
    valid = face_visible & exposed_ok
    X = np.full((len(y), 7), np.nan)
    for i in np.flatnonzero(face_visible):
        X[i] = face_colour_features(face["wb"][i])

    # Fully nested leave-one-brick-out.
    scores, chosen = np.zeros(len(y)), {}
    for g in np.unique(groups):
        test, train = groups == g, (groups != g) & valid
        model, choice = select_and_fit(X[train], y[train], groups[train])
        scores[test & valid] = model.decision_function(X[test & valid])
        chosen[str(g)] = choice | {"threshold": model.threshold}
        print(f"held out {g:13s} -> {choice['features']:13s} {choice['distance']:11s} "
              f"acc {np.mean((scores[test & valid] > 0) == y[test & valid]):.2f}", flush=True)
    lobo = metrics(scores, y, groups, valid) | {"chosen_per_fold": chosen}
    no6 = groups != "Brick_6_bad"
    lobo["excluding_brick6"] = {k: v for k, v in metrics(scores[no6], y[no6], groups[no6], valid[no6]).items()
                                if k != "per_brick"}

    # Provided brick-disjoint splits: fit on train+val bricks, test on test bricks.
    split_rows = []
    for split in sorted(p for p in SPLITS.glob("split_*") if p.is_dir()):
        test_bricks = {p.name for p in (split / "test").iterdir()}
        test = np.isin(groups, list(test_bricks))
        train = ~test & valid
        model, choice = select_and_fit(X[train], y[train], groups[train])
        m = test & valid
        pred = model.decision_function(X[m]) > 0
        split_rows.append({"split": split.name, "test_bricks": "+".join(sorted(test_bricks)),
                           "features": choice["features"], "distance": choice["distance"],
                           "test_photos": int(m.sum()), "test_accuracy": float(np.mean(pred == y[m]))})
    split_acc = np.array([r["test_accuracy"] for r in split_rows])
    split_no6 = np.array([r["test_accuracy"] for r in split_rows if "Brick_6" not in r["test_bricks"]])
    splits = {"n_splits": len(split_rows), "mean_test_accuracy": float(split_acc.mean()),
              "std_test_accuracy": float(split_acc.std()),
              "mean_test_accuracy_splits_without_brick6_in_test": float(split_no6.mean())}

    # Final model on all ten bricks.
    final, final_choice = select_and_fit(X[valid], y[valid], groups[valid])
    final.save(args.out / "redness_window.json")

    with open(args.out / "lobo_image_predictions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["brick", "image", "truth", "verdict", "score", "correct", *("L a b chroma hue r g".split())])
        for i in range(len(y)):
            if not valid[i]:
                verdict = "no_face_visible" if not face_visible[i] else "overexposed"
                w.writerow([groups[i], images[i], colour["label"][i], verdict, "", "", *[""] * 7])
                continue
            verdict = "bad" if scores[i] > 0 else "good"
            w.writerow([groups[i], images[i], colour["label"][i], verdict, f"{scores[i]:.3f}",
                        int(verdict == colour["label"][i]), *np.round(X[i], 4)])
    with open(args.out / "provided_splits.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(split_rows[0]))
        w.writeheader()
        w.writerows(split_rows)
    summary = {"lobo_nested": lobo, "provided_splits": splits,
               "final_model": final_choice | {"threshold": final.threshold}}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps({k: v for k, v in lobo.items() if k not in ("per_brick", "chosen_per_fold")}, indent=1))
    for b, r in lobo["per_brick"].items():
        print(f"  {b:13s} {r['image_accuracy']:.2f} ({r['photos_with_verdict']}/{r['photos']} photos) "
              f"brick {'OK' if r['brick_correct'] else 'WRONG'}")
    print("provided splits:", json.dumps(splits))
    print("final model:", final_choice)


if __name__ == "__main__":
    main()
