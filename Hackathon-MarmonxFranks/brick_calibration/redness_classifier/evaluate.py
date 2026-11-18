#!/usr/bin/env python3
"""Leave-one-physical-brick-out evaluation of the redness classifiers.

Only 10 physical bricks exist, so photos of one brick are never split across
train and test. Regularisation is chosen by an inner leave-one-brick-out loop
on the training bricks only. ``--nested-selection`` also chooses the feature
set inside each outer fold, which gives an estimate free of the optimism of
picking the best row of the table after the fact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from brick_features import load

CS = (0.001, 0.01, 0.1, 1.0)


def make_model(c: float, n_features: int):
    steps = [StandardScaler()]
    if n_features > 200:
        steps.append(PCA(n_components=32, random_state=0))
    steps.append(LogisticRegression(C=c, class_weight="balanced", max_iter=5000))
    return make_pipeline(*steps)


def brick_balanced_accuracy(scores, y, groups):
    """Mean over bricks of image accuracy, so big brick folders don't dominate."""
    return np.mean([np.mean((scores[groups == g] > 0) == y[groups == g]) for g in np.unique(groups)])


def lobo_scores(X, y, groups, c):
    scores = np.zeros(len(y))
    for g in np.unique(groups):
        test = groups == g
        model = make_model(c, X.shape[1]).fit(X[~test], y[~test])
        scores[test] = model.decision_function(X[test])
    return scores


def select(sets, names, y, groups):
    """Inner LOBO choice of (feature set, C) on the given bricks."""
    best = None
    for name in names:
        for c in CS:
            score = brick_balanced_accuracy(lobo_scores(sets[name], y, groups, c), y, groups)
            if best is None or score > best[0] + 1e-9:
                best = (score, name, c)
    return best[1], best[2]


def outer(sets, names, y, groups):
    scores, chosen = np.zeros(len(y)), {}
    for g in np.unique(groups):
        test = groups == g
        train_sets = {n: sets[n][~test] for n in names}
        name, c = select(train_sets, names, y[~test], groups[~test])
        model = make_model(c, sets[name].shape[1]).fit(sets[name][~test], y[~test])
        scores[test] = model.decision_function(sets[name][test])
        chosen[str(g)] = {"features": name, "C": c}
    return scores, chosen


def summarise(scores, y, groups):
    pred = scores > 0
    per_brick = {}
    for g in np.unique(groups):
        m = groups == g
        per_brick[str(g)] = {"image_acc": float(np.mean(pred[m] == y[m])),
                             "mean_score": float(scores[m].mean()),
                             "brick_correct": bool((scores[m].mean() > 0) == y[m][0])}
    good, bad = y == 0, y == 1
    return {"image_accuracy": float(np.mean(pred == y)),
            "balanced_accuracy": float((np.mean(~pred[good]) + np.mean(pred[bad])) / 2),
            "good_recall": float(np.mean(~pred[good])), "bad_recall": float(np.mean(pred[bad])),
            "brick_accuracy": float(np.mean([v["brick_correct"] for v in per_brick.values()])),
            "per_brick": per_brick}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sets", nargs="*")
    parser.add_argument("--nested-selection", nargs="*", metavar="SET",
                        help="candidate feature sets chosen inside each outer fold")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    meta, sets = load(args.cache)
    y = (meta["label"] == "bad").astype(int)  # positive class = bad
    groups = meta["brick"]
    results = {}
    for name in args.sets or sets:
        scores, chosen = outer(sets, [name], y, groups)
        results[name] = summarise(scores, y, groups) | {"chosen": chosen}
        r = results[name]
        wrong = [b for b, v in r["per_brick"].items() if not v["brick_correct"]]
        print(f"{name:34s} img {r['image_accuracy']:.3f}  bal {r['balanced_accuracy']:.3f}  "
              f"bricks {r['brick_accuracy']:.1f}  wrong={wrong}", flush=True)
        np.save(args.out / f"scores_{name}.npy", scores)
    if args.nested_selection:
        scores, chosen = outer(sets, args.nested_selection, y, groups)
        r = summarise(scores, y, groups) | {"chosen": chosen, "candidates": args.nested_selection}
        results["NESTED_SELECTION"] = r
        print(f"{'NESTED_SELECTION':34s} img {r['image_accuracy']:.3f}  bal {r['balanced_accuracy']:.3f}  "
              f"bricks {r['brick_accuracy']:.1f}")
        print(json.dumps(chosen, indent=0))
        np.save(args.out / "scores_NESTED_SELECTION.npy", scores)
    (args.out / "lobo_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
