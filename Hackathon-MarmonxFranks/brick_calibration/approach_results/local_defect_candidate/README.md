# Experimental local-defect classifier

This is a **separate candidate**. `../baseline_checkpoint/` and the original
Approach 17 threshold policies were not changed.

## Model

The 258 saved SAM3 masks in `../../../offline_results/` are used to remove
the background from each of the ten physical bricks. Each masked photo yields:

- 44 LAB histogram values and eight global texture values;
- 16 local dark-line, connected-region, edge, and patch anomaly values;
- 1,152 frozen DINOv2 patch-distribution values, reduced to 32 PCA dimensions
  **inside each training fold**.

A standardized, class-balanced linear SVM produces a `score_bad`. Score > 0
is bad; score <= 0 is good. Training weights give each physical brick equal
weight within its class. The final model is `candidate_svm.joblib`; the frozen
DINOv2 encoder comes from the unchanged Approach 17 bundle. The feature cache
and model are separate from the baseline checkpoint.

The evaluation holds out one complete physical brick at a time. An inner
three-fold group split chooses the SVM regularization. A final checkpoint is
then trained on all ten bricks for **experimental** live use.

| Evaluation | Image correct | Bad recall | Physical bricks correct |
|---|---:|---:|---:|
| Local candidate, SAM3 masks | 158/258 (61.2%) | 61.1% | 5/10 |
| Approach 17-like global features, same SAM3 masks | 157/258 (60.9%) | 55.0% | Not calculated |
| Existing Approach 17, its original masks | 212/258 (82.2%) | 75.2% | 9/10 |

The two rows use different masks, so the difference is not a clean ablation of
the local features. The same-mask control shows only a one-photo gain from the
new candidate. The local candidate currently performs **worse** than the
existing baseline and must not replace it as the default.

## Face information and live behavior

`face_review_template.csv` lists all 258 photos with blank `visible_face` and
`defect_visible` fields. The current photos have only whole-brick quality
labels; they do not state which face or defect is visible. No face-aware model
has been trained. Those labels need review before a model can learn a reliable
face-specific decision or claim accuracy at every orientation. For review,
record the dominant visible surface as `holed`, `flat`, `long_side`,
`short_end`, `mixed`, or `unknown`; record defect visibility as `yes`, `no`,
or `unclear`. Do not infer `defect_visible=yes` from the bad folder name.

The Mac live viewer can select this candidate with `--pipeline sam3_local`.
Photolab keeps up to ten SAM3 masks, scores each separately, and matches boxes
between successive frames using intersection-over-union. For a matched brick,
the displayed candidate score is the median of up to three recent scores.
This reduces frame-to-frame flicker; it does **not** establish that different
physical faces have been inspected. `track_id` and `raw_score_bad` are saved
in each frame's `detections_*.json`. A brick moved far enough to break box
overlap starts a new track.

From the Mac runtime directory, after copying the updated
`remote_rgb_client.py` from photolab:

```bash
python brick_calibration/remote_rgb_client.py \
  --host 131.220.238.95 --camera 1 --backend avfoundation \
  --pipeline sam3_local --interval 3 --save-frames
```

Use `--pipeline sam3 --policy minus_two_binary` to run the unchanged baseline
SAM3/Approach 17 live path. The candidate's score is on a different scale, so
the baseline threshold policies do not apply to it.

To regenerate the candidate and grouped evaluation on photolab:

```bash
HF_HOME=/cache/s95vkond/huggingface \
/home_domuser2/s95vkond/miniconda3/envs/sam3/bin/python \
  brick_calibration/local_defect_candidate.py
```
