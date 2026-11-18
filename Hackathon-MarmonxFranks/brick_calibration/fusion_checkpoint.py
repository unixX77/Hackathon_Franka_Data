#!/usr/bin/env python3
"""Frozen Approach 17 checkpoint: Approach 5 + DINOv2 patch-distribution fusion.

Build (fits on all 258 images after grouped evaluation, like Approach 5):

    HF_HOME=/cache/s95vkond/huggingface python fusion_checkpoint.py \
        --data /cache/s95vkond/test_data/rgb_bricks \
        --results approach_results --out approach_results/17_approach_checkpoint

Use:

    from fusion_checkpoint import FusionClassifier
    classifier = FusionClassifier.load("approach_results/17_approach_checkpoint")
    result = classifier.classify(rgb)               # one RGB uint8 image
    result = classifier.classify_burst([rgb, ...])  # several stationary frames
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from evaluate_approaches import (discover_images, lab_histogram, material_mask,
                                 texture_features, usable_pixels)
from evaluate_improvements import fit_svm

DINO_MODEL = "facebook/dinov2-small"
DINO_PCA_COMPONENTS = 32


class FusionClassifier:
    def __init__(self, approach5, dino_svm, metadata, device=None, encoder_source=None):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.approach5, self.dino_svm, self.metadata = approach5, dino_svm, metadata
        self.margin = float(metadata["uncertainty_margin"])
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        source = encoder_source or metadata["dino_model"]
        revision = None if encoder_source else metadata.get("dino_revision")
        self.processor = AutoImageProcessor.from_pretrained(source, revision=revision)
        self.encoder = AutoModel.from_pretrained(source, revision=revision).to(self.device).eval()

    @classmethod
    def load(cls, directory, device=None):
        directory = Path(directory)
        metadata = json.loads((directory / "checkpoint_metadata.json").read_text())
        bundled = metadata.get("dino_bundled_directory")
        encoder_source = directory / bundled if bundled and (directory / bundled).is_dir() else None
        return cls(joblib.load(directory / "approach5_svm.joblib"),
                   joblib.load(directory / "dinov2_patch_svm.joblib"), metadata, device,
                   encoder_source)

    def _dino_features(self, crop):
        import torch
        with torch.inference_mode():
            pixels = self.processor(images=[crop], return_tensors="pt")["pixel_values"]
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=self.device.startswith("cuda")):
                hidden = self.encoder(pixel_values=pixels.to(self.device)).last_hidden_state
        patches = hidden.float().cpu().numpy()[0, 1:]
        return np.r_[patches.mean(0), patches.std(0), np.percentile(patches, 90, axis=0)]

    def scores(self, rgb, mask=None):
        """Return (fused, approach5, dinov2) decision scores; >= 0 means bad."""
        from evaluate_dino_approaches import prepare_crop_rgb
        mask = material_mask(rgb) if mask is None else mask.astype(np.uint8)
        lab, usable, _ = usable_pixels(rgb, mask)
        a5_features = np.r_[lab_histogram(lab, usable), texture_features(rgb, usable)]
        a5 = float(self.approach5.decision_function(a5_features[None])[0])
        dino = float(self.dino_svm.decision_function(
            self._dino_features(prepare_crop_rgb(rgb, mask))[None])[0])
        return (a5 + dino) / 2, a5, dino

    def _decide(self, score):
        if abs(score) < self.margin:
            return "uncertain"
        return "bad" if score >= 0 else "good"

    def classify(self, rgb, mask=None):
        fused, a5, dino = self.scores(rgb, mask)
        return {"label": self._decide(fused), "score_bad": fused,
                "approach5_score": a5, "dinov2_score": dino, "frames": 1,
                "uncertainty_margin": self.margin}

    def classify_burst(self, frames, masks=None):
        masks = masks or [None] * len(frames)
        scores = np.asarray([self.scores(frame, mask) for frame, mask in zip(frames, masks)])
        fused = float(np.median(scores[:, 0]))
        return {"label": self._decide(fused), "score_bad": fused,
                "frame_scores": scores[:, 0].tolist(), "frames": len(frames),
                "uncertainty_margin": self.margin}


def bundle_encoder(out, revision):
    """Save the exact DINOv2 weights and preprocessing next to the SVMs."""
    from transformers import AutoImageProcessor, AutoModel
    target = out / "dinov2_small"
    AutoImageProcessor.from_pretrained(DINO_MODEL, revision=revision).save_pretrained(target)
    AutoModel.from_pretrained(DINO_MODEL, revision=revision).save_pretrained(target)
    return target.name


def build(data, results, out, uncertainty_margin, dino_revision=None, bundle=False,
          extra_metadata=None):
    rows = discover_images(data)
    labels = np.asarray([row["label"] for row in rows])
    groups = np.asarray([row["brick_id"] for row in rows])
    approach5 = np.load(results / "extracted_features.npz")
    dino = np.load(results / "dino_feature_cache" / "dinov2_features.npz")
    if not np.array_equal(approach5["groups"], groups):
        raise ValueError("feature caches are not aligned with the dataset order")
    everything = np.arange(len(labels))
    a5_search = fit_svm(approach5["masked_histogram_texture"], labels, groups, everything)
    dino_search = fit_svm(dino["patch_distribution"], labels, groups, everything,
                          DINO_PCA_COMPONENTS)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(a5_search.best_estimator_, out / "approach5_svm.joblib")
    joblib.dump(dino_search.best_estimator_, out / "dinov2_patch_svm.joblib")
    metadata = {
        "approach": "Approach 17: equal-weight late fusion of Approach 5 and Approach 8",
        "decision_rule": "score_bad = mean(approach5 SVM decision, DINOv2 SVM decision); >= 0 means bad",
        "uncertainty_margin": uncertainty_margin,
        "uncertainty_rule": "|score_bad| < margin -> uncertain (never an automatic good)",
        "burst_rule": "median of per-frame fused scores",
        "approach5_features": "automatic material mask; LAB histograms 12/16/16 bins; 8 gradient/Laplacian/intensity texture statistics (52-d)",
        "dino_model": DINO_MODEL,
        "dino_revision": dino_revision,
        "dino_features": "masked bounding-box crop, background filled with median material colour; patch-token mean, std, P90 (1152-d); PCA 32 whitened",
        "selected_C": {"approach5": a5_search.best_params_["svc__C"],
                       "dinov2": dino_search.best_params_["svc__C"]},
        "training_images": len(rows), "training_physical_bricks": len(set(groups)),
        "label_mapping": {"good": 0, "bad": 1},
        "held_out_performance": "see 17_late_fusion_approach5_dinov2_patch/metrics.json",
        "warning": ("Fitted on ten physical bricks photographed on paper. The uncertainty "
                    "margin was read from the held-out sweep on the same bricks and must be "
                    "re-selected on RealSense top-surface validation bricks before deployment."),
    }
    if bundle:
        metadata["dino_bundled_directory"] = bundle_encoder(out, dino_revision)
    metadata.update(extra_metadata or {})
    (out / "checkpoint_metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--uncertainty-margin", type=float, default=0.3)
    parser.add_argument("--dino-revision", default=None,
                        help="pin the Hugging Face commit of facebook/dinov2-small")
    parser.add_argument("--bundle-encoder", action="store_true",
                        help="save the DINOv2 weights inside the checkpoint directory")
    parser.add_argument("--extra-metadata", type=Path, default=None,
                        help="JSON file whose keys are added to checkpoint_metadata.json")
    args = parser.parse_args()
    extra = json.loads(args.extra_metadata.read_text()) if args.extra_metadata else None
    print(json.dumps(build(args.data, args.results, args.out, args.uncertainty_margin,
                           args.dino_revision, args.bundle_encoder, extra), indent=2))


if __name__ == "__main__":
    main()
