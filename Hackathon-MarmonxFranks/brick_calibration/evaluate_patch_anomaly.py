#!/usr/bin/env python3
"""True dense-patch anomaly evaluation for frozen DINO encoders.

For every held-out physical brick, only patches from good training bricks form
the reference memory. Each query patch receives its nearest-good cosine
distance. The image score is the mean of its top 5% valid patch distances.
Thresholds are selected using grouped inner validation only.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import LeaveOneGroupOut
from transformers import AutoImageProcessor, AutoModel

from evaluate_approaches import discover_images, material_mask, write_csv


def crop_brick(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not decode {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mask = material_mask(rgb)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    usable = mask.astype(bool) & (hsv[..., 2] > 18) & (hsv[..., 2] < 248)
    ys, xs = np.nonzero(mask)
    margin = round(.04 * max(xs.max() - xs.min(), ys.max() - ys.min()))
    x0, x1 = max(0, xs.min() - margin), min(rgb.shape[1], xs.max() + margin + 1)
    y0, y1 = max(0, ys.min() - margin), min(rgb.shape[0], ys.max() + margin + 1)
    crop, valid = rgb[y0:y1, x0:x1].copy(), usable[y0:y1, x0:x1]
    neutral = np.median(crop[valid], axis=0).astype(np.uint8)
    crop[~valid] = neutral
    crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
    valid = cv2.resize(valid.astype(np.uint8), (224, 224),
                       interpolation=cv2.INTER_AREA)
    return crop, valid


@torch.inference_mode()
def extract_dense(rows, model_name, batch_size, device, cache_path, token):
    processor = AutoImageProcessor.from_pretrained(model_name, token=token,
                                                    use_fast=False)
    model = AutoModel.from_pretrained(model_name, token=token).to(device).eval()
    patch_size = int(model.config.patch_size)
    grid = 224 // patch_size
    all_tokens, all_valid, manifest = [], [], []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        prepared = [crop_brick(row["path"]) for row in batch_rows]
        crops, masks = zip(*prepared)
        inputs = processor(images=list(crops), return_tensors="pt",
                           do_resize=False, do_center_crop=False)
        pixels = inputs["pixel_values"].to(device)
        if device.startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                hidden = model(pixel_values=pixels).last_hidden_state
        else:
            hidden = model(pixel_values=pixels).last_hidden_state
        # Taking the final grid-sized block safely excludes CLS and any register
        # tokens in both DINOv2 and DINOv3.
        tokens = hidden[:, -(grid * grid):].float()
        tokens = torch.nn.functional.normalize(tokens, dim=-1).cpu().numpy()
        all_tokens.append(tokens.astype(np.float16))
        for row, mask in zip(batch_rows, masks):
            coverage = cv2.resize(mask.astype(np.float32), (grid, grid),
                                  interpolation=cv2.INTER_AREA)
            all_valid.append((coverage >= .55).reshape(-1))
            manifest.append({"file": str(row["path"]), "brick_id": row["brick_id"],
                             "label": row["label_name"], "grid": grid,
                             "valid_patches": int(np.sum(coverage >= .55))})
        print(f"{model_name} dense features: {min(start + batch_size, len(rows))}/{len(rows)}",
              flush=True)
    tokens = np.concatenate(all_tokens)
    valid = np.asarray(all_valid, bool)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, tokens=tokens, valid=valid,
                        model=np.asarray(model_name), grid=np.asarray(grid))
    write_csv(cache_path.with_suffix(".manifest.csv"), manifest)
    return tokens, valid, grid


def make_memory(tokens, valid, indices, labels, max_patches, seed):
    patches = [tokens[index][valid[index]] for index in indices if labels[index] == 0]
    if not patches:
        raise ValueError("No good training patches available")
    memory = np.concatenate(patches).astype(np.float32)
    if len(memory) > max_patches:
        rng = np.random.default_rng(seed)
        memory = memory[rng.choice(len(memory), max_patches, replace=False)]
    return memory


def score_images(tokens, valid, indices, memory, top_fraction):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Embeddings were L2-normalized at extraction, so cosine distance to the
    # nearest reference is exactly 1 - max(query @ reference.T). GPU matrix
    # multiplication is mathematically equivalent to sklearn's brute-force
    # cosine search and makes nested grouped evaluation tractable.
    references = torch.from_numpy(memory).to(device)
    image_scores, patch_scores = [], {}
    for index in indices:
        selected = tokens[index][valid[index]].astype(np.float32)
        queries = torch.from_numpy(selected).to(device)
        distances = (1.0 - torch.mm(queries, references.T).amax(dim=1)).cpu().numpy()
        count = max(1, int(np.ceil(len(distances) * top_fraction)))
        image_scores.append(float(np.mean(np.partition(distances, -count)[-count:])))
        full = np.full(valid.shape[1], np.nan, np.float32)
        full[valid[index]] = distances
        patch_scores[index] = full
    return np.asarray(image_scores), patch_scores


def choose_threshold(scores, labels, groups):
    grouped = []
    for group in dict.fromkeys(groups):
        selected = groups == group
        grouped.append((float(np.mean(scores[selected])),
                        int(labels[np.flatnonzero(selected)[0]])))
    unique = np.unique([item[0] for item in grouped])
    candidates = np.r_[unique[0] - 1e-8, (unique[:-1] + unique[1:]) / 2,
                       unique[-1] + 1e-8]
    best = None
    truth = np.asarray([item[1] for item in grouped])
    values = np.asarray([item[0] for item in grouped])
    for threshold in candidates:
        predicted = (values >= threshold).astype(int)
        rank = (balanced_accuracy_score(truth, predicted),
                recall_score(truth, predicted, zero_division=0),
                -int(np.sum((truth == 0) & (predicted == 1))))
        if best is None or rank > best[0]: best = (rank, float(threshold))
    return best[1]


def metrics(truth, predicted):
    matrix = confusion_matrix(truth, predicted, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {"accuracy": float(accuracy_score(truth, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
            "bad_precision": float(precision_score(truth, predicted, zero_division=0)),
            "bad_recall": float(recall_score(truth, predicted, zero_division=0)),
            "bad_f1": float(f1_score(truth, predicted, zero_division=0)),
            "false_accept_bad_as_good": int(fn),
            "false_reject_good_as_bad": int(fp),
            "confusion_matrix": matrix.tolist()}


def render_heatmap(path, patch_scores, grid, threshold, destination):
    crop, valid = crop_brick(path)
    heat = patch_scores.reshape(grid, grid)
    finite = np.isfinite(heat)
    normalized = np.zeros_like(heat)
    if finite.any():
        normalized[finite] = np.clip(heat[finite] / max(threshold, 1e-6), 0, 2) / 2
    heat_large = cv2.resize(normalized, (224, 224), interpolation=cv2.INTER_CUBIC)
    colour = cv2.applyColorMap(np.uint8(255 * heat_large), cv2.COLORMAP_TURBO)
    colour = cv2.cvtColor(colour, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(crop, .55, colour, .45, 0)
    valid_large = cv2.resize(valid, (224, 224), interpolation=cv2.INTER_NEAREST) > 0
    overlay[~valid_large] = (overlay[~valid_large] * .25).astype(np.uint8)
    comparison = np.hstack((crop, overlay))
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 90])


def evaluate(tokens, valid, grid, rows, output, top_fraction, max_memory):
    labels = np.asarray([row["label"] for row in rows])
    groups = np.asarray([row["brick_id"] for row in rows])
    scores = np.full(len(rows), np.nan); predictions = np.full(len(rows), -1, int)
    thresholds, test_patch_scores = [], {}
    all_indices = np.arange(len(rows))
    for fold, (outer_train, outer_test) in enumerate(
            LeaveOneGroupOut().split(all_indices, labels, groups)):
        inner_scores = np.full(len(outer_train), np.nan)
        train_y, train_g = labels[outer_train], groups[outer_train]
        for inner_fold, (inner_train_pos, inner_test_pos) in enumerate(
                LeaveOneGroupOut().split(outer_train, train_y, train_g)):
            inner_train, inner_test = outer_train[inner_train_pos], outer_train[inner_test_pos]
            memory = make_memory(tokens, valid, inner_train, labels, max_memory,
                                 seed=1000 + fold * 20 + inner_fold)
            values, _ = score_images(tokens, valid, inner_test, memory, top_fraction)
            inner_scores[inner_test_pos] = values
        threshold = choose_threshold(inner_scores, train_y, train_g)
        memory = make_memory(tokens, valid, outer_train, labels, max_memory,
                             seed=7000 + fold)
        values, patches = score_images(tokens, valid, outer_test, memory, top_fraction)
        scores[outer_test] = values - threshold
        predictions[outer_test] = (values >= threshold).astype(int)
        test_patch_scores.update(patches)
        thresholds.append({"held_out_brick": groups[outer_test][0],
                           "threshold": threshold, "good_memory_patches": len(memory),
                           "top_fraction": top_fraction})
    image_rows = []
    threshold_by_brick = {row["held_out_brick"]: row["threshold"] for row in thresholds}
    for index, row in enumerate(rows):
        image_rows.append({"brick_id": row["brick_id"], "file": str(row["path"]),
                           "true_label": row["label_name"],
                           "predicted_label": "bad" if predictions[index] else "good",
                           "decision_score_bad": float(scores[index]),
                           "correct": int(predictions[index] == labels[index])})
        render_heatmap(row["path"], test_patch_scores[index], grid,
                       threshold_by_brick[row["brick_id"]],
                       output / "anomaly_heatmaps" / row["brick_id"] /
                       f"{row['path'].stem}.jpg")
    brick_rows, brick_true, brick_pred = [], [], []
    for brick in dict.fromkeys(groups):
        indices = np.flatnonzero(groups == brick); score = float(np.mean(scores[indices]))
        prediction, truth = int(score >= 0), int(labels[indices[0]])
        brick_true.append(truth); brick_pred.append(prediction)
        brick_rows.append({"brick_id": brick, "true_label": "bad" if truth else "good",
                           "predicted_label": "bad" if prediction else "good",
                           "mean_decision_score_bad": score,
                           "correct": int(truth == prediction), "n_images": len(indices)})
    payload = {"image_level": metrics(labels, predictions),
               "brick_level": metrics(np.asarray(brick_true), np.asarray(brick_pred)),
               "method": "per-patch nearest-good cosine anomaly; mean top 5%",
               "top_fraction": top_fraction, "grid": [grid, grid],
               "n_images": len(rows), "n_bricks": len(set(groups))}
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "held_out_image_predictions.csv", image_rows)
    write_csv(output / "held_out_brick_predictions.csv", brick_rows)
    write_csv(output / "selected_thresholds.csv", thresholds)
    (output / "metrics.json").write_text(json.dumps(payload, indent=2))
    matrix = np.asarray(payload["brick_level"]["confusion_matrix"])
    fig, axis = plt.subplots(figsize=(4.5, 4)); plot = axis.imshow(matrix, cmap="Blues")
    for y in range(2):
        for x in range(2): axis.text(x, y, str(matrix[y, x]), ha="center", va="center")
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["good", "bad"],
             yticklabels=["good", "bad"], xlabel="Predicted", ylabel="Actual",
             title="Held-out physical bricks")
    fig.colorbar(plot, ax=axis); fig.tight_layout()
    fig.savefig(output / "brick_confusion_matrix.png", dpi=160); plt.close(fig)
    return payload


def update_matrix(results_root, approach, feature_set, payload):
    path = results_root / "evaluation_matrix.csv"
    existing = [row for row in csv.DictReader(path.open()) if row["approach"] != approach]
    for level in ("image_level", "brick_level"):
        existing.append({"approach": approach, "feature_set": feature_set,
                         "classifier": "nearest_good_patch_anomaly", "level": level,
                         **{key: value for key, value in payload[level].items()
                            if key != "confusion_matrix"},
                         "confusion_matrix": json.dumps(payload[level]["confusion_matrix"])})
    write_csv(path, existing)
    (results_root / "evaluation_matrix.json").write_text(json.dumps(existing, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--approach", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-memory-patches", type=int, default=12000)
    parser.add_argument("--top-fraction", type=float, default=.05)
    parser.add_argument("--reuse-features", action="store_true")
    parser.add_argument("--token-file", type=Path)
    args = parser.parse_args()
    rows = discover_images(args.data)
    cache_path = args.cache / f"{args.approach}_dense_features.npz"
    token = args.token_file.read_text().strip() if args.token_file else None
    if args.reuse_features and cache_path.is_file():
        cached = np.load(cache_path); tokens, valid = cached["tokens"], cached["valid"]
        grid = int(cached["grid"])
    else:
        tokens, valid, grid = extract_dense(
            rows, args.model, args.batch_size,
            "cuda" if torch.cuda.is_available() else "cpu", cache_path, token)
    output = args.results / args.approach
    payload = evaluate(tokens, valid, grid, rows, output, args.top_fraction,
                       args.max_memory_patches)
    (output / "method_config.json").write_text(json.dumps({
        "model": args.model, "encoder_frozen": True, "grid": [grid, grid],
        "top_fraction": args.top_fraction,
        "max_memory_patches": args.max_memory_patches,
        "feature_cache": str(cache_path)}, indent=2))
    update_matrix(args.results, args.approach,
                  f"{args.model}:per_patch_nearest_good", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
