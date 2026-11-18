# Approach 17 threshold policies

These JSON files change **only the score-to-label rule** used by
`remote_rgb_client.py` with the original or SAM3/Approach 17 live server. Approach 17's
checkpoint, foreground mask, feature extraction, and fused score are unchanged.

| Policy name | Good | Uncertain | Bad |
|---|---|---|---|
| `baseline_margin` (historical) | score ≤ −0.3 | −0.3 < score < +0.3 | score ≥ +0.3 |
| `zero_binary` | score < 0 | never | score ≥ 0 |
| `minus_two_binary` | score ≤ −2.0 | never | score > −2.0 |
| `review_band_175_120` (live baseline) | score ≤ −1.75 | −1.75 < score ≤ −1.20 | score > −1.20 |

The single-brick prediction CSV records the chosen `threshold_policy`, the
original `score_bad`, and labels for all four policies. For multi-brick SAM3
frames the individual scores and policy labels are in `detections_*.json`.
The window draws the label for the chosen policy.
This lets you compare policies on identical frames without rerunning the camera.
Scores are SVM decision values, not probabilities. Under
`review_band_175_120`, uncertain means **no good/bad decision** for that brick.

On the saved 258 SAM3-mask photos, the review-band policy made 215 decisions:
194 were correct (90.2%) and 43 were uncertain. Of the 109 good photos, 74
were called good, 10 bad, and 25 uncertain. Of the 149 bad photos, 120 were
called bad, 11 good, and 18 uncertain. Those photos are from the same ten
physical bricks used to fit Approach 17; this is not a live-camera accuracy
estimate.

## Run the review-band policy in the SAM3 live viewer

```bash
cd /Users/varunteja/Downloads/brick_live_runtime
ssh s95vkond@131.220.238.95 \
  "cd '/ipb70/home_domuser2/s95vkond/Desktop/Franks robotics/Hackathon-MarmonxFranks' && tar -cf - brick_calibration/remote_rgb_client.py" \
  | tar -xf -

python brick_calibration/remote_rgb_client.py \
  --host 131.220.238.95 --camera 1 --backend avfoundation \
  --pipeline sam3 --policy review_band_175_120 \
  --interval 3 --save-frames
```

Omit `--truth` when several bricks are in view. The Mac output is written to
`live_results/<session>/`; `predictions.csv` contains frame counts and
`detections_*.json` contains individual brick scores and labels. Green is
good, red is bad, and amber is uncertain.

## Run the −2.0 policy from the Mac

Copy the **updated client** into your existing Mac runtime folder. The JSON
policies and server remain on photolab:

```bash
cd /Users/varunteja/Downloads/brick_live_runtime
ssh s95vkond@131.220.238.95 \
  "cd '/ipb70/home_domuser2/s95vkond/Desktop/Franks robotics/Hackathon-MarmonxFranks' && tar -cf - brick_calibration/remote_rgb_client.py" \
  | tar -xf -

python brick_calibration/remote_rgb_client.py \
  --host 131.220.238.95 --camera 1 --backend avfoundation \
  --interval 3 --truth good --save-frames \
  --pipeline sam3 \
  --policy minus_two_binary
```

Use `--truth bad` for a known-bad brick. Start a new session for each brick;
keep only one brick in view, and press `q` to quit. The Mac saves
`live_results/<session>/predictions.csv`. Its `label` column uses the chosen
policy, while the additional label columns show all four policies.

To show the original policy instead, use `--policy baseline_margin`. To test
the zero cutoff, use `--policy zero_binary`. If you omit `--policy` in the
updated Mac client, the live baseline `review_band_175_120` is used.

These threshold values are test policies. The uploaded audit at
`/cache/s95vkond/live_score_audits/20261026_123809/` marked every frame bad,
including sample frames with no brick and multiple bricks; it cannot validate
which threshold is accurate. Collect correctly labelled, one-brick sessions
for the four good and six bad physical bricks before selecting a final policy.
