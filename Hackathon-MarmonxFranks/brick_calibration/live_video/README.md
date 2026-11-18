# Approach 17 live_video

This is a **separate live-camera variant**. The original Approach 17 checkpoint
in `../approach_results/baseline_checkpoint/` and the original remote client and
server are unchanged. `live_video` loads that same frozen checkpoint. It does
not claim a newly trained classifier until labelled Mac camera frames are
available.

## What changes

1. On the Mac, select a region containing **one brick** and a little plain
   background. The original viewer scored the entire frame. Use a steady,
   diffuse light and lock exposure/white balance if your camera software
   allows it.
2. Send a fresh burst of three frames for each prediction. The server computes
   both Approach 5 and DINOv2 scores for each valid frame and takes their
   medians. It never mixes scores across different bricks.
3. Check foreground area, crop boundaries, background contamination, and
   over/underexposure. The viewer
   shades the **actual material mask in pale cyan**; the thin box is green for
   good, red for bad, or amber for uncertain.
4. Approach 5 and DINOv2 retain their original weights and decision rule.
   Scores outside conservative ranges derived from the frozen checkpoint's
   training features become **uncertain**, rather than falsely confident good.
   A5 scores beyond ±20 or DINOv2 scores beyond ±24 trigger this guard. The
   usual `|score_bad| < 0.3` uncertainty margin also remains.
5. Each CSV row records both component scores, mask area, clipping, frame
   count, and quality flags. Raw frames can be retained for diagnosis.

The score guard is an abstention rule, not a correction to a bad prediction.
An uncertain brick should be inspected again. A tight rectangle alone does
not prove the mask follows the brick, which is why the mask is displayed.

## Existing offline evidence

Run `python brick_calibration/live_video/validate_offline.py` on photolab. Its
report is saved in `results/offline_guard_metrics.json`. On the existing
leave-one-physical-brick-out photo predictions, original Approach 17 sign
accuracy is **212/258 = 82.2%**. With the existing ±0.3 uncertainty margin and
the new extreme-score guard, **206/258 photos receive a good/bad decision**;
**177/206 = 85.9%** of those decisions are correct. The remaining **52 are
uncertain**, including three extreme scores. This is a coverage/accuracy
tradeoff on the old photos, **not a measured Mac live-camera accuracy**. The
ROI and mask quality checks are not represented by that saved-score audit.

## Mac live preview

First copy the standalone Mac client from photolab:

```bash
mkdir -p /Users/varunteja/Downloads/brick_live_runtime
cd /Users/varunteja/Downloads/brick_live_runtime
ssh s95vkond@131.220.238.95 \
  "cd '/ipb70/home_domuser2/s95vkond/Desktop/Franks robotics/Hackathon-MarmonxFranks' && tar -cf - brick_calibration/live_video/client.py brick_calibration/live_video/capture_upload.py brick_calibration/live_video/audit_live_scores.py" \
  | tar -xf -
python -m pip install opencv-python numpy
```

If SSH or SCP does not connect, use the university VPN or the host name
`photolab95.ipb.uni-bonn.de`. Then, with one labelled physical brick in view:

```bash
python brick_calibration/live_video/client.py \
  --host 131.220.238.95 --camera 1 --backend avfoundation \
  --truth bad --burst 3 --interval 3 --save-frames
```

At startup, draw a rectangle around **one brick plus a little background** and
press Enter. `--roi X Y W H` reuses a fixed rectangle; `--full-frame` skips
selection only when the whole frame contains one isolated brick. Use
`--truth good` for a known-good session. Press `q` to stop. The Mac writes
`live_results/<timestamp>/predictions.csv` and, with `--save-frames`, raw and
annotated JPEGs. The model stays on photolab.

## Transfer labelled Mac camera frames directly to photolab cache

The transfer command above includes the standalone capture script.

## Audit scores already logged by the live viewer

After running known-good and known-bad sessions with `--truth good` or
`--truth bad`, run this on the Mac from `brick_live_runtime`:

```bash
python brick_calibration/live_video/audit_live_scores.py \
  --root live_results --upload
```

This uses Python's standard library. It gathers every `predictions.csv` below
`live_results`, records the good and bad score ranges and each component's
range, copies up to three representative `raw_*.jpg` images per session when
available, and uploads the audit to
`/cache/s95vkond/live_score_audits/<timestamp>/` on photolab. It also writes
`live_score_audit/<timestamp>/threshold_report.json` on the Mac. Run without
`--upload` if you only want the local report. A midpoint threshold is reported
only if the observed good and bad score ranges do not overlap; it is an
exploratory value from those same frames, not a validated deployment setting.
Use `--frames-per-session 9999` to include all saved raw frames instead of
three examples per session.

If the original viewer was run without `--truth`, the CSV cannot identify
which brick is good or bad; collect new labelled sessions. Keep one physical
brick in each session and use its actual label.

To **capture new photos** of physical Brick 5, known bad:

```bash
python brick_calibration/live_video/capture_upload.py \
  --camera 1 --brick-id 5 --truth bad --count 12 --interval 3
```

Press Space to begin. Run separately for each physical brick and use its true
ID and label. The script preserves a Mac copy in `live_capture/` and uploads
the images and manifest to
`/cache/s95vkond/live_video_capture/Brick_5_bad/<session>/` on photolab.

To transfer `raw_*.jpg` files **already saved** by the live viewer, without
opening the camera:

```bash
python brick_calibration/live_video/capture_upload.py \
  --brick-id 5 --truth bad \
  --from-dir live_results/REPLACE_WITH_SESSION_DIRECTORY
```

## Measure the camera accuracy

After uploading labelled sessions, run on photolab from the repository root:

```bash
/home_domuser2/s95vkond/miniconda3/envs/sam3/bin/python \
  brick_calibration/live_video/evaluate_capture.py \
  --data /cache/s95vkond/live_video_capture
```

This writes `results/camera_predictions.csv` and `results/camera_metrics.json`
under this folder. It compares the original Approach 17 decision and the
guarded live-video decision on the **same camera crops** and reports accuracy,
coverage, bad-brick recall, and results by physical brick. If you use a fixed
viewer ROI, pass the same `--roi X Y W H` here. These frames are from bricks
the checkpoint has already seen in training, so this measures camera-domain
performance on those bricks; it does not establish accuracy on new bricks.

## Next checkpoint update

The camera capture set will reveal whether low live accuracy comes from the
mask, lighting, Approach 5, DINOv2, or fusion. Retrain or calibrate only after
checking those labelled frames, keeping whole physical bricks apart when
testing generalisation. Brick 6 remains a known hard case.
