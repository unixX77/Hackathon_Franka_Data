# RealSense validation protocol

This experiment cannot substitute for data from the robot camera. Use the
following protocol when the camera is available.

1. Freeze exposure, gain, white balance, working distance, illumination, and
   camera intrinsics. Record those settings with the run.
2. For every independent physical brick, capture top-facing examples at the
   expected robot orientations and a smaller challenge set of tilted/upright
   examples. Keep physical brick IDs in the filenames.
3. Run RGB-D top-layer segmentation and retain the exact top-surface mask,
   ordered four image corners, depth validity fraction, and pose metadata.
4. Perspective-warp those corners with `canonicalize_detected_surface`. Do not
   infer a photographic foreground silhouette when the segmentation mask exists.
5. Keep all frames from one physical brick in the same train or test split.
   Reserve new bricks as the final test set.
6. Capture stationary bursts of 3--9 consecutive frames. Average classifier
   decision scores, not hard labels, and measure latency and score variance.
7. Choose the bad/good and uncertainty thresholds using validation bricks with
   priority on bad recall. Never tune thresholds on the final test bricks.

For safety, an uncertain classification should prevent an automatic `good`
decision. A practical robot output is `bad`, `good`, or `uncertain`, together
with the score, threshold, mask/depth quality, and number of frames aggregated.
