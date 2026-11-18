#!/usr/bin/env python3
"""Same-SAM3-mask grouped control: separate A5 and DINOv2 SVMs, no local features."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from evaluate_approaches import discover_images
from local_defect_candidate import DATA, OUTPUT, sample_weights


def build(kind: str, c: float):
    if kind == "a5":
        return make_pipeline(StandardScaler(), SVC(kernel="linear", C=c,
                                                   class_weight="balanced"))
    return make_pipeline(PCA(n_components=32, whiten=True, random_state=7),
                         StandardScaler(), SVC(kernel="linear", C=c,
                                               class_weight="balanced"))


def main() -> None:
    rows = discover_images(DATA)
    saved = np.load(OUTPUT / "sam3_mask_features.npz")
    paths = np.asarray([str(row["path"].resolve()) for row in rows])
    if not np.array_equal(saved["paths"], paths):
        raise ValueError("Feature cache does not match dataset order")
    x = saved["features"]
    y = np.asarray([row["label"] for row in rows])
    groups = np.asarray([row["brick_id"] for row in rows])
    parts = (("a5", x[:, :52]), ("dino", x[:, 68:]))
    scores = np.zeros((len(y), 2), np.float64)

    def fit(kind, c, indices, features):
        model = build(kind, c)
        model.fit(features[indices], y[indices],
                  svc__sample_weight=sample_weights(y[indices], groups[indices]))
        return model

    for train, test in LeaveOneGroupOut().split(x, y, groups):
        for member, (kind, features) in enumerate(parts):
            ranked = []
            for c in (.03, .3, 3):
                brick_accuracy = []
                for inner_train, inner_test in GroupKFold(3).split(
                        train, y[train], groups[train]):
                    a, b = train[inner_train], train[inner_test]
                    prediction = fit(kind, c, a, features).decision_function(features[b]) > 0
                    for brick in np.unique(groups[b]):
                        selected = groups[b] == brick
                        brick_accuracy.append(float(np.mean(prediction[selected] == y[b][selected])))
                ranked.append((float(np.mean(brick_accuracy)), -c, c))
            c = max(ranked)[2]
            scores[test, member] = fit(kind, c, train, features).decision_function(features[test])
        print(groups[test][0], int(sum((scores[test].mean(axis=1) > 0) == y[test])),
              len(test), flush=True)
    prediction = scores.mean(axis=1) > 0
    result = {"sam3_mask_a17_like_correct": int(sum(prediction == y)),
              "n": len(y), "accuracy": float(np.mean(prediction == y)),
              "bad_recall": float(np.mean(prediction[y == 1])),
              "good_recall": float(np.mean(~prediction[y == 0]))}
    path = OUTPUT / "sam3_mask_a17_like_control.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
