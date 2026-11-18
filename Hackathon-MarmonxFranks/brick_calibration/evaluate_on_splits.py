#!/usr/bin/env python3
"""Evaluate Approaches 5 and 17 on the brick-disjoint train/val/test splits.

Protocol for every split (splits come from ``make_brick_splits.py``):

1. fit each candidate SVM regularisation C on the training bricks only;
2. choose C by validation balanced accuracy (mean of the two validation bricks'
   image accuracies; ties go to the smaller, more regularised C);
3. refit with that C on training + validation bricks;
4. score the test bricks once (and the set-aside Brick 6, if any).

Approach 5 uses the 52-d colour-histogram + texture vector. Approach 17 averages
the Approach 5 score and the DINOv2 patch-distribution score (PCA 32), each with
its own C chosen as above. The decision threshold stays at 0 (>= 0 means bad);
validation is not used to move it, matching the leave-one-brick-out results.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate_approaches import discover_images, write_csv
from evaluate_improvements import C_GRID, svm_model

MEMBERS = {"approach5": ("masked_histogram_texture", None),
           "dinov2_patch": ("patch_distribution", 32)}


def per_brick_accuracy(predictions, labels, groups, indices):
    return {brick: float(np.mean(predictions[indices][groups[indices] == brick] ==
                                 labels[indices][groups[indices] == brick]))
            for brick in dict.fromkeys(groups[indices])}


def fit_member(x, pca_components, labels, groups, train, val):
    validation = []
    for c in C_GRID:
        model = svm_model(pca_components).set_params(svc__C=c)
        model.fit(x[train], labels[train])
        predictions = (model.decision_function(x) >= 0).astype(int)
        validation.append(np.mean(list(
            per_brick_accuracy(predictions, labels, groups, val).values())))
    best = int(np.argmax(validation))  # first maximum = smallest C among ties
    model = svm_model(pca_components).set_params(svc__C=C_GRID[best])
    both = np.r_[train, val]
    model.fit(x[both], labels[both])
    return model, C_GRID[best], float(validation[best])


def summarise(rows, approach):
    truth = np.asarray([r["true_label"] == "bad" for r in rows])
    predicted = np.asarray([r[f"pred_{approach}"] == "bad" for r in rows])
    good, bad = ~truth, truth
    return {"images": len(rows), "correct": int(np.sum(predicted == truth)),
            "accuracy": float(np.mean(predicted == truth)),
            "good_image_accuracy": float(np.mean(~predicted[good])) if good.any() else None,
            "bad_image_recall": float(np.mean(predicted[bad])) if bad.any() else None,
            "balanced_accuracy": float(np.mean([np.mean(~predicted[good]), np.mean(predicted[bad])]))
            if good.any() and bad.any() else None}


def brick_decisions(rows, approach):
    decisions = {}
    for brick in dict.fromkeys(r["brick_id"] for r in rows):
        scores = [float(r[f"score_{approach}"]) for r in rows if r["brick_id"] == brick]
        truth = rows[[r["brick_id"] for r in rows].index(brick)]["true_label"]
        decisions[brick] = {"mean_score": float(np.mean(scores)),
                            "predicted": "bad" if np.mean(scores) >= 0 else "good",
                            "correct": int((np.mean(scores) >= 0) == (truth == "bad"))}
    return decisions


def evaluate_family(family_dir, features, labels, groups, files, output):
    index_of = {f: i for i, f in enumerate(files)}
    assignment = json.loads((family_dir / "split_assignment.json").read_text())
    manifest = list(csv.DictReader((family_dir / "split_manifest.csv").open()))
    image_rows, split_rows = [], []
    for split in assignment:
        name = split["split"]
        subsets = {part: np.asarray(sorted(index_of[r["source_file"]] for r in manifest
                                           if r["split"] == name and r["subset"] == part), int)
                   for part in ("train", "val", "test", "set_aside")}
        for part, indices in subsets.items():
            expected = set(split[part])
            if set(groups[indices]) != expected and not (len(indices) == 0 and not expected):
                raise ValueError(f"{name}/{part}: manifest bricks do not match assignment")
        used = np.concatenate(list(subsets.values()))
        if len(used) != len(set(used)) or len(used) != len(labels):
            raise ValueError(f"{name}: subsets overlap or miss images")
        scores, record = {}, {"split": name, "primary": split["primary"],
                              "train_bricks": " ".join(split["train"]),
                              "val_bricks": " ".join(split["val"]),
                              "test_bricks": " ".join(split["test"]),
                              "set_aside_bricks": " ".join(split["set_aside"])}
        for member, (key, pca_components) in MEMBERS.items():
            model, c, val_accuracy = fit_member(features[key], pca_components, labels, groups,
                                                subsets["train"], subsets["val"])
            scores[member] = model.decision_function(features[key])
            record[f"{member}_C"] = c
            record[f"{member}_val_balanced_accuracy"] = val_accuracy
        scores["approach17"] = (scores["approach5"] + scores["dinov2_patch"]) / 2
        split_images = []
        for part in ("test", "set_aside"):
            for i in subsets[part]:
                split_images.append({
                    "split": name, "subset": part, "brick_id": groups[i], "file": files[i],
                    "true_label": "bad" if labels[i] else "good",
                    "score_approach5": float(scores["approach5"][i]),
                    "score_dinov2_patch": float(scores["dinov2_patch"][i]),
                    "score_approach17": float(scores["approach17"][i]),
                    "pred_approach5": "bad" if scores["approach5"][i] >= 0 else "good",
                    "pred_approach17": "bad" if scores["approach17"][i] >= 0 else "good"})
        image_rows += split_images
        test_rows = [r for r in split_images if r["subset"] == "test"]
        for approach in ("approach5", "approach17"):
            summary = summarise(test_rows, approach)
            bricks = brick_decisions(test_rows, approach)
            record[f"{approach}_test_correct"] = f"{summary['correct']}/{summary['images']}"
            record[f"{approach}_test_accuracy"] = summary["accuracy"]
            record[f"{approach}_test_bricks_correct"] = sum(b["correct"] for b in bricks.values())
            for brick in bricks:
                kind = "good" if brick.endswith("good") else "bad"
                record[f"{approach}_{kind}_test_brick_image_acc"] = summarise(
                    [r for r in test_rows if r["brick_id"] == brick], approach)["accuracy"]
            aside = [r for r in split_images if r["subset"] == "set_aside"]
            if aside:
                record[f"{approach}_brick6_aside_correct"] = summarise(aside, approach)["correct"]
                record[f"{approach}_brick6_aside_mean_score"] = brick_decisions(aside, approach)["Brick_6_bad"]["mean_score"]
        split_rows.append(record)
        print(f"{family_dir.name} {name}: A5 {record['approach5_test_correct']}  "
              f"A17 {record['approach17_test_correct']}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "test_image_predictions.csv", image_rows)
    write_csv(output / "per_split_results.csv", split_rows)
    return image_rows, split_rows


def family_summary(image_rows, split_rows, approach):
    test = [r for r in image_rows if r["subset"] == "test"]
    overall = summarise(test, approach)
    split_accuracy = [r[f"{approach}_test_accuracy"] for r in split_rows]
    brick_instances = [r[f"{approach}_test_bricks_correct"] for r in split_rows]
    per_brick = {}
    for brick in sorted({r["brick_id"] for r in test}, key=lambda b: int(b.split("_")[1])):
        rows = [r for r in test if r["brick_id"] == brick]
        per_brick[brick] = {"times_in_test": len({r["split"] for r in rows}),
                            "image_accuracy": summarise(rows, approach)["accuracy"],
                            "brick_correct_rate": float(np.mean([
                                brick_decisions([r for r in rows if r["split"] == s], approach)[brick]["correct"]
                                for s in dict.fromkeys(r["split"] for r in rows)]))}
    primary = next(r for r in split_rows if r["primary"])
    result = {"pooled_test_image_accuracy": overall["accuracy"],
              "pooled_test_balanced_accuracy": overall["balanced_accuracy"],
              "pooled_test_bad_image_recall": overall["bad_image_recall"],
              "pooled_test_good_image_accuracy": overall["good_image_accuracy"],
              "split_test_accuracy_mean": float(np.mean(split_accuracy)),
              "split_test_accuracy_min": float(np.min(split_accuracy)),
              "split_test_accuracy_max": float(np.max(split_accuracy)),
              "brick_level_accuracy": float(np.sum(brick_instances) / (2 * len(split_rows))),
              "splits_with_both_test_bricks_correct": int(np.sum(np.asarray(brick_instances) == 2)),
              "splits": len(split_rows),
              "primary_split": {"split": primary["split"], "test_bricks": primary["test_bricks"],
                                "test_correct": primary[f"{approach}_test_correct"],
                                "test_bricks_correct": primary[f"{approach}_test_bricks_correct"]},
              "per_brick": per_brick}
    aside = [r for r in image_rows if r["subset"] == "set_aside"]
    if aside:
        result["brick6_set_aside"] = {
            "models": len(split_rows),
            "image_accuracy": summarise(aside, approach)["accuracy"],
            "models_calling_brick6_bad": int(sum(r[f"{approach}_brick6_aside_mean_score"] >= 0
                                                 for r in split_rows)),
            "mean_score": float(np.mean([r[f"{approach}_brick6_aside_mean_score"] for r in split_rows]))}
    return result


def lobo_summary(results, folder):
    rows = list(csv.DictReader((results / folder / "held_out_image_predictions.csv").open()))
    bricks = list(csv.DictReader((results / folder / "held_out_brick_predictions.csv").open()))
    truth = np.asarray([r["true_label"] == "bad" for r in rows])
    predicted = np.asarray([r["predicted_label"] == "bad" for r in rows])
    brick6 = [r for r in rows if r["brick_id"] == "Brick_6_bad"]
    return {"image_accuracy": float(np.mean(predicted == truth)),
            "balanced_accuracy": float(np.mean([np.mean(~predicted[~truth]), np.mean(predicted[truth])])),
            "bad_image_recall": float(np.mean(predicted[truth])),
            "brick_accuracy": float(np.mean([int(b["correct"]) for b in bricks])),
            "brick6_image_accuracy": float(np.mean([int(r["correct"]) for r in brick6])),
            "per_brick": {b: float(np.mean([int(r["correct"]) for r in rows if r["brick_id"] == b]))
                          for b in sorted({r["brick_id"] for r in rows}, key=lambda x: int(x.split("_")[1]))}}


def matched_comparisons(results, output):
    """Like-for-like numbers that exclude Brick 6 from every test set.

    * leave-one-brick-out restricted to the 234 non-Brick-6 images;
    * the 20 test pairs shared by both families, which have identical
      validation bricks, so the only difference is Brick 6 in training or not.
    Balanced accuracy is reported because the families test good and bad
    bricks in different proportions from leave-one-brick-out.
    """
    def score(rows, key):
        truth = np.asarray([r["true_label"] == "bad" for r in rows])
        predicted = np.asarray([r[key] == "bad" for r in rows])
        return (float(np.mean(predicted == truth)),
                float(np.mean([np.mean(~predicted[~truth]), np.mean(predicted[truth])])))

    def tests(family):
        rows = [r for r in csv.DictReader(
            (output / family / "test_image_predictions.csv").open()) if r["subset"] == "test"]
        with_brick6 = {r["split"] for r in rows if r["brick_id"] == "Brick_6_bad"}
        return [r for r in rows if r["split"] not in with_brick6]

    shared_with, shared_without = tests("with_brick6"), tests("without_brick6")
    folders = {"approach5": "05_masked_lab_hist_texture_linear_svm",
               "approach17": "17_late_fusion_approach5_dinov2_patch"}
    matched = {}
    for approach, folder in folders.items():
        lobo = [r for r in csv.DictReader((results / folder / "held_out_image_predictions.csv").open())
                if r["brick_id"] != "Brick_6_bad"]
        entry = {}
        for name, rows, key in (("lobo_excl_brick6", lobo, "predicted_label"),
                                ("shared_pairs_brick6_in_training", shared_with, f"pred_{approach}"),
                                ("shared_pairs_brick6_set_aside", shared_without, f"pred_{approach}")):
            entry[f"{name}_accuracy"], entry[f"{name}_balanced_accuracy"] = score(rows, key)
            entry[f"{name}_images"] = len(rows)
        for brick in ("Brick_5_bad", "Brick_8_bad", "Brick_10_bad", "Brick_2_good", "Brick_3_good"):
            for name, rows in (("in_training", shared_with), ("set_aside", shared_without)):
                subset = [r for r in rows if r["brick_id"] == brick]
                entry[f"{brick}_brick6_{name}"] = score(subset, f"pred_{approach}")[0]
        matched[approach] = entry
    return matched


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    records = discover_images(args.data)
    labels = np.asarray([r["label"] for r in records])
    groups = np.asarray([r["brick_id"] for r in records])
    files = [str(r["path"]) for r in records]
    approach5 = np.load(args.results / "extracted_features.npz")
    dino = np.load(args.results / "dino_feature_cache" / "dinov2_features.npz")
    if not np.array_equal(approach5["groups"], groups):
        raise ValueError("feature caches are not aligned with the dataset order")
    features = {"masked_histogram_texture": approach5["masked_histogram_texture"],
                "patch_distribution": dino["patch_distribution"]}
    output = args.results / "split_evaluation"
    summary = {"protocol": __doc__.strip(), "splits_root": str(args.splits),
               "leave_one_brick_out": {
                   "approach5": lobo_summary(args.results, "05_masked_lab_hist_texture_linear_svm"),
                   "approach17": lobo_summary(args.results, "17_late_fusion_approach5_dinov2_patch")}}
    for family in ("without_brick6", "with_brick6"):
        image_rows, split_rows = evaluate_family(args.splits / family, features, labels,
                                                 groups, files, output / family)
        summary[family] = {approach: family_summary(image_rows, split_rows, approach)
                           for approach in ("approach5", "approach17")}
    matched = matched_comparisons(args.results, output)
    summary["matched_comparisons"] = matched
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    write_csv(output / "matched_comparison.csv", [{"approach": a, **v} for a, v in matched.items()])
    comparison = []
    for approach in ("approach5", "approach17"):
        lobo = summary["leave_one_brick_out"][approach]
        out, inc = summary["without_brick6"][approach], summary["with_brick6"][approach]
        comparison.append({
            "approach": approach,
            "lobo_image_accuracy": lobo["image_accuracy"],
            "lobo_balanced_accuracy": lobo["balanced_accuracy"],
            "lobo_brick_accuracy": lobo["brick_accuracy"],
            "lobo_brick6_image_accuracy": lobo["brick6_image_accuracy"],
            "without_b6_test_image_accuracy": out["pooled_test_image_accuracy"],
            "without_b6_test_balanced_accuracy": out["pooled_test_balanced_accuracy"],
            "without_b6_split_accuracy_range": f"{out['split_test_accuracy_min']:.3f}-{out['split_test_accuracy_max']:.3f}",
            "without_b6_brick_accuracy": out["brick_level_accuracy"],
            "without_b6_brick6_aside_image_accuracy": out["brick6_set_aside"]["image_accuracy"],
            "without_b6_models_calling_brick6_bad": f"{out['brick6_set_aside']['models_calling_brick6_bad']}/{out['brick6_set_aside']['models']}",
            "with_b6_test_image_accuracy": inc["pooled_test_image_accuracy"],
            "with_b6_test_balanced_accuracy": inc["pooled_test_balanced_accuracy"],
            "with_b6_split_accuracy_range": f"{inc['split_test_accuracy_min']:.3f}-{inc['split_test_accuracy_max']:.3f}",
            "with_b6_brick_accuracy": inc["brick_level_accuracy"],
            "with_b6_brick6_test_image_accuracy": inc["per_brick"]["Brick_6_bad"]["image_accuracy"]})
    write_csv(output / "comparison.csv", comparison)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
