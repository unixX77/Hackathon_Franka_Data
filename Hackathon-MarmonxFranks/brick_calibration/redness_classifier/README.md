# Perforated-face redness classifier

Good bricks have one specific fired red (expert criterion). This classifier
measures that red on the **perforated face** and accepts a brick only if the
colour falls inside a tolerance window learned from the 4 good bricks.

## Pipeline

1. SAM3 `"brick"` mask (saved masks in `../offline_results/*/masks`).
2. SAM3 `"hole"` inside the brick crop → perforations (score ≥ 0.8). Their
   convex hull, grown by half a hole and minus the holes, is the perforated face.
3. White balance against the paper the brick lies on (linear RGB).
4. Median L\*, chroma, hue of the face → max per-axis z-distance from the
   good-brick mean → `bad` if above the threshold.
5. `no_verdict` if the perforated face is not visible (< 2 perforations) or the
   paper is near clipping (any channel ≥ 230). Take another view.

Final model (`results/redness_window.json`, fitted on all 10 bricks):

| Axis | Good window |
|---|---|
| L\* | 48.4 – 57.5 |
| Chroma | 33.7 – 41.7 |
| Hue | 48.4° – 52.2° |

## Why this design

- **Face, not whole brick.** Side faces carry soot, lime bloom and dust on good
  and bad bricks alike. Whole-brick colour varies more between views of one
  brick than between bricks (chroma ±5 vs ±1 on the face).
- **One-class window, not a two-class boundary.** Bad bricks miss the target red
  in different directions: 5 and 7 are duller and more yellow, 8 is darker,
  9 and 10 are paler. A two-class model never sees "too dark" when Brick 8 is
  held out and accepts it (0 % in every supervised run).
- **White balance on the paper.** All good bricks were photographed 11:34–12:08,
  all bad bricks 12:13–12:45. The paper is measurably greener in the bad session;
  a classifier on paper colour **alone** reaches 66 % leave-one-brick-out.
  Features use only pixels inside the brick mask, after white balance.

## Results

All numbers hold out complete physical bricks. Feature set, distance and
threshold are selected inside each outer fold (fully nested), so picking the
best candidate does not inflate them.

**Leave-one-brick-out (10 folds)**

| Metric | All bricks | Excluding Brick 6 |
|---|---:|---:|
| Image accuracy (photos with a verdict) | **80.9 %** | **89.4 %** |
| Balanced accuracy | 83.0 % | 89.7 % |
| Good recall / bad recall | 91.6 % / 74.5 % | 91.6 % / 87.8 % |
| Coverage (photos with a verdict) | 85.3 % | 84.6 % |
| Strict accuracy (no verdict = wrong) | 69.0 % | 75.6 % |
| Brick accuracy (mean score of all views) | 8/10 | 8/9 |

Per brick (image accuracy): 1: 0.73, 2: 0.96, 3: 0.96, 4: 0.95, 5: 1.00,
6: 0.05, 7: 1.00, 8: 1.00, 9: 0.91, 10: 0.54. Brick 10 is borderline at
brick level (mean score −0.11, median +0.03).

**Provided splits** (`brick_splits/with_brick6`, 24 splits, train+val → test):
mean test accuracy 76.5 % (std 20.5); 83.4 % on the 20 splits without Brick 6
in test. See `results/provided_splits.csv`.

**Baselines on the same protocol** (`evaluate.py`, whole-brick features, linear
model; image accuracy over all 258 photos): colour histograms 55–61 %;
frozen DINOv2 small/base/large (CLS + mask-pooled patches, 4 rotations)
56–71 %; DINOv2 + colour 61–65 %; paper colour only 66 %.

## Brick 6

Its perforated face is inside the good range on every colour axis
(L 53.3, a\* 24.1, b\* 29.2; Brick 3 is 53.4, 23.9, 29.2). No colour feature
can reject it. `../approach_results/brick6_label_audit` shows its crack is
also present on good Bricks 2 and 4. Confirm with the expert why Brick 6 is bad.

## Limits

- Ten physical bricks. One brick changes brick accuracy by 10 points.
- L\* depends on the face's angle to the light. A fixed camera and light at the
  robot cell would narrow the good window.
- The window was fitted on 4 good bricks; good bricks with a slightly different
  red may be rejected. Recalibrate with more good bricks when available.

## Robustness to view and lighting

Out-of-sample (nested leave-one-brick-out) accuracy by placement:

| Placement | Photos | With verdict | Accuracy | Excl. Brick 6 |
|---|---:|---:|---:|---:|
| Horizontal (0–20°) | 81 | 96 % | 80.8 % | 90.0 % |
| Diagonal (20–60°) | 117 | 98 % | 79.1 % | 87.4 % |
| Vertical / end-on (60–90°) | 60 | 45 % | 88.9 % | 96.0 % |
| Small face visible (< 30 % of brick) | 73 | 100 % | 76.7 % | 83.3 % |
| Large face visible (> 33 %) | 74 | 100 % | 82.4 % | 93.8 % |

Invariance through the live server (30 photos, final model): verdict unchanged
for 100 % of mirrored, 97 % of 180°-rotated and 93 % of 90°-rotated,
−20 % / +15 % exposure and warm-light versions. Every change was on a photo
whose score was already within ±0.4 of the threshold.

## Live camera (Mac camera → photolab)

The model runs in `sam3_live/server.py --model redness`; the Mac client selects
it with `--pipeline sam3_redness`. Setup and SSH requirements are in
`../REMOTE_RGB.md`. On the Mac, refresh the client once (it gained the new
pipeline), then run:

```bash
cd /Users/varunteja/Downloads/brick_live_runtime
ssh s95vkond@131.220.238.95 \
  "cd '/ipb70/home_domuser2/s95vkond/Desktop/Franks robotics/Hackathon-MarmonxFranks' && tar -cf - brick_calibration/remote_rgb_client.py" \
  | tar -xf -
python brick_calibration/remote_rgb_client.py --host 131.220.238.95 \
  --camera 1 --backend avfoundation --interval 3 \
  --pipeline sam3_redness --save-frames --truth good   # --truth optional
```

Contours: green good, red bad, amber **uncertain** = no verdict (caption says
why: perforated face not visible, or over-exposed). Results go to
`live_results/<timestamp>/`: `predictions.csv` (`score_bad` per frame, > 0 =
bad) and `detections_*.json` (face colour, perforation count, reason).
`--policy` is ignored by this pipeline. Check the transport without a camera:
`--image /path/to/brick.png --no-window`.

For reliable live scores: keep the brick on the white paper, show the
perforated face to the camera, avoid exposure that pushes the paper to white,
and judge a brick by the median `score_bad` of several views.

## Reproduce

```bash
PY=/home_domuser2/s95vkond/miniconda3/envs/sam3/bin/python
CACHE=/path/to/cache   # ~60 MB
$PY extract_features.py --out $CACHE            # colour samples (+ DINOv2 for baselines; --models to skip)
$PY perforated_face.py  --out $CACHE            # SAM3 perforations, face colour samples
$PY evaluate_redness.py --cache $CACHE          # writes results/
$PY evaluate.py --cache $CACHE --out $CACHE/baselines   # optional whole-brick baselines
```

Classify new photos (`--one-brick` combines several views of one brick):

```bash
$PY classify_brick.py photo1.png photo2.png --annotate annotated/ --one-brick
```
