# Live SAM3 → Approach 17 viewer

The Mac camera sends one RGB JPEG every three seconds over a persistent SSH
connection. Photolab loads cached SAM3 and the frozen Approach 17 checkpoint
once, selects up to ten separate whole-brick masks, classifies each mask, and
returns an annotated image and per-brick metadata. The Mac window shows its
live camera on the left and the latest SAM3 prediction on the right.

The returned boundary follows the SAM3 mask with a **soft 3-pixel contour**.
The live baseline `review_band_175_120` shows green for good at score ≤ −1.75,
amber for uncertain at −1.75 < score ≤ −1.20, and red for bad above −1.20.
The original `baseline_margin`, `zero_binary`, and `minus_two_binary` policies
remain available. The viewer records frame counts in its CSV and each brick's
scores and policy labels in a separate JSON file.

## On the Mac

Copy the updated lightweight client. SAM3 and both model checkpoints remain
on photolab:

```bash
mkdir -p /Users/varunteja/Downloads/brick_live_runtime
cd /Users/varunteja/Downloads/brick_live_runtime
ssh s95vkond@131.220.238.95 \
  "cd '/ipb70/home_domuser2/s95vkond/Desktop/Franks robotics/Hackathon-MarmonxFranks' && tar -cf - brick_calibration/remote_rgb_client.py" \
  | tar -xf -

python -m pip install opencv-python numpy
```

Start a session with **one to ten bricks** in view:

```bash
python brick_calibration/remote_rgb_client.py \
  --host 131.220.238.95 --camera 1 --backend avfoundation \
  --pipeline sam3 --policy review_band_175_120 \
  --interval 3 --save-frames
```

For a session containing exactly one known-good brick, add `--truth good`;
for one known-bad brick, add `--truth bad`. Omit `--truth` for mixed or multiple
bricks. Stop with `q` or Esc. Leave a visible gap between adjacent bricks so
SAM3 can return separate masks; touching or heavily overlapping bricks may be
merged. Keep each brick large enough in the image to show its defects.
Try `--camera 0` if camera 1 does not open. The first connection can take
longer while photolab loads SAM3 and Approach 17; later frames reuse them.
If SSH fails, connect through the university VPN or use the reachable host
that worked for earlier transfers.

The Mac creates `live_results/<timestamp>/predictions.csv` with the number
of detected, good, and bad bricks for each frame, plus one
`detections_<timestamp>.json` per frame with each brick's image-local ID,
classification scores, SAM3 mask area, bounding box, and centroid. With
`--save-frames`, it also saves raw and annotated JPEGs. The right panel uses
the actual SAM3 contours returned by photolab. A frame without a reliable
brick mask receives no forced good/bad label. IDs can change between frames;
this viewer does not track physical bricks over time. `--pipeline baseline`
selects the previous GrabCut box viewer without modifying this SAM3 mode.

An experimental `--pipeline sam3_local` option uses a separate local-defect
candidate and three-frame score smoothing. Its leave-one-brick-out result is
158/258, below Approach 17, so `--pipeline sam3` remains the recommended
path. See `../approach_results/local_defect_candidate/README.md` for details.

For quality decisions based on color alone, `--pipeline sam3_color` runs the
separate LAB histogram classifier. Its held-out result is 136/258; see
`../approach_results/color_only/README.md` for its limitations and score rule.

`--pipeline sam3_reference` selects the experimental good-brick clustering
model. Its held-out result is 130/258; see
`../approach_results/good_brick_clusters/README.md` before using it.

For one saved Mac photo, use `--image /path/to/photo.jpg --no-window` with
the same `--pipeline sam3 --policy review_band_175_120` options. It writes a
two-panel `preview.jpg` and `prediction.json` under `live_results/`.

## Offline check performed here

The server/client protocol was tested with saved good and bad photos, plus
four- and ten-brick synthetic layouts. An empty-table photo produced no
quality label. The saved smoke images are in this folder. Real crowded or
touching-brick layouts still need camera testing. The folder-wise
offline experiment is in `../offline_results/`: 258/258 photos received a
SAM3 mask. Under the live review band, 194/215 decided photos matched their
folder label and 43/258 were uncertain. Those photos are familiar training
bricks used to tune the policy, so this is not a measurement of live-camera
or new-brick accuracy. The previous −2.0 binary policy gave 207/258 correct
with a decision for every photo.
