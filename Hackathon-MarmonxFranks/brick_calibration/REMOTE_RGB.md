# Mac camera → photolab Approach 17 → Mac preview

This is the workflow for the USB RGB camera attached to the Mac. The Mac runs
`remote_rgb_client.py` with only OpenCV and NumPy. One persistent SSH process
launches `sam3_live/server.py` on photolab, where SAM3, the Approach 17
checkpoint and DINOv2 weights remain. Every three seconds the Mac sends a JPEG
frame and receives quality and scores for each detected brick, plus an
annotated image. The Mac draws the result in a two-panel window: live camera
left, latest prediction right. Contours are **green good**, **red bad**, or
**amber uncertain** under the `review_band_175_120` policy.

The lab currently reports IP `131.220.238.95` and DNS name
`photolab95.ipb.uni-bonn.de`. The short name `photolab95` is not resolvable
from the Mac, as observed in the failed transfer. The Mac must have a working
SSH route to the IP or full DNS name, possibly through the university VPN.

## On the Mac

First verify the connection and accept the SSH host key if prompted:

```bash
ssh s95vkond@131.220.238.95 'hostname'
```

It should print `photolab95`. If it does not connect, try
`photolab95.ipb.uni-bonn.de` or connect to the network/VPN used for your
previous SCP transfers. A connection failure needs to be resolved before live
classification can work.

Copy only the small Mac client into a separate working folder. No model
checkpoint or dataset needs to be copied to the Mac:

```bash
mkdir -p /Users/varunteja/Downloads/brick_live_runtime
cd /Users/varunteja/Downloads/brick_live_runtime
ssh s95vkond@131.220.238.95 \
  "cd '/ipb70/home_domuser2/s95vkond/Desktop/Franks robotics/Hackathon-MarmonxFranks' && tar -cf - brick_calibration/remote_rgb_client.py" \
  | tar -xf -
python -m pip install opencv-python numpy
```

Then run the viewer. Camera 0 is often the built-in Mac webcam, while the
connected USB camera may be camera 1:

```bash
python brick_calibration/remote_rgb_client.py \
  --host 131.220.238.95 \
  --camera 1 --backend avfoundation --interval 3 \
  --pipeline sam3 --policy review_band_175_120 --save-frames
```

Try `--camera 0` if camera 1 does not open. For a single brick with known
quality, add `--truth bad` or `--truth good`; omit it for mixed bricks. Focus
the preview window and press
**q** or **Esc** to quit. macOS may ask for camera access for Terminal or your
Python executable; allow it in System Settings if necessary.

The client creates `live_results/<session timestamp>/` under the current
directory. `predictions.csv` records frame counts and timing, and each
`detections_*.json` records individual labels, scores, and bounding boxes.
For a single brick, the CSV also records its score and optional truth.
`--save-frames` also writes raw and
annotated JPEGs; omit it during long sessions to save disk space. Scores are
SVM decision values, not calibrated probabilities.

If a known image is on the Mac, the remote transport can be checked without
opening a camera:

```bash
python brick_calibration/remote_rgb_client.py \
  --host 131.220.238.95 --image /path/to/brick.png --no-window
```

The image check writes a two-panel `preview.jpg` and `prediction.json`.

## Limitations

The current model was trained on single centered bricks photographed against
a light background. The SAM3 whole-brick mask is a visual aid; it is not a
robot pose or a validated top-surface mask. The live preview is not an
accuracy measurement by itself. Use known-good and known-bad sessions to
calculate camera-domain accuracy. If inference plus network transfer exceeds
three seconds, the client skips scheduled frames instead of queuing old ones.

The model remains on photolab. `sam3_live/server.py` starts automatically
inside the SSH connection; there is no public HTTP service or separate server
terminal to manage.

Selectable score thresholds (`baseline_margin`, `zero_binary`,
`minus_two_binary`, and `review_band_175_120`) are saved under `threshold_policies/`. See
`threshold_policies/README.md` for exact boundaries and the `--policy` command.

The SAM3 whole-brick mask pipeline is the Mac client's default; see
`sam3_live/README.md` for the updated transfer and run commands. Use
`--pipeline baseline --policy baseline_margin` for the original GrabCut
box viewer and original policy.
