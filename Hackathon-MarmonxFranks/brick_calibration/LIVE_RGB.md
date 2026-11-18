# Live RGB brick inspection (Approach 17)

For a camera on the Mac with classification performed on photolab, use
[`REMOTE_RGB.md`](REMOTE_RGB.md). The instructions below are only for running
the entire checkpoint locally on the camera computer.

`live_rgb_classifier.py` reads a USB webcam frame every three seconds and runs
the saved `approach_results/baseline_checkpoint/` model. The left panel is the
current RGB camera view. The right panel is the most recent classified frame.
Its thin bounding box is **green = good, red = bad, amber = uncertain**. The box
surrounds the automatic foreground mask; it is a visual guide, not an RGB-D
top-face or robot pose estimate.

This pipeline is designed for **one centered brick** photographed against a
plain, preferably matte light background. It does not detect several bricks
separately. If it cannot find a plausible foreground, it reports an error
without a quality prediction. The model was trained on the original still
photos, so camera-domain accuracy must be measured with new labelled sessions.

## Run on the computer with the USB camera

The self-contained checkpoint includes both SVMs and the 85 MB DINOv2 encoder.
The viewer also needs these Python files in `brick_calibration/`:
`live_rgb_classifier.py`, `fusion_checkpoint.py`, `evaluate_approaches.py`,
`evaluate_improvements.py`, `evaluate_dino_approaches.py`,
`evaluate_canonical_a5.py`, and `canonical_surface.py`.

On the Mac used earlier in this project, open Terminal and run:

```bash
cd /Users/varunteja/Downloads/Hackathon-MarmonxFranks
conda create -n brick-live python=3.12 -y
conda activate brick-live
python -m pip install torch torchvision
python -m pip install -r brick_calibration/requirements-live.txt
python brick_calibration/live_rgb_classifier.py --camera 0 --backend avfoundation --interval 3 --truth bad --save-frames
```

Change `--truth bad` to `--truth good` when switching bricks, or omit it if the
ground truth is unknown. Run a separate session per physical brick. Camera 0
may be the built-in webcam: use `--camera 1` or `--camera 2` for the USB camera.
On Windows, replace `--backend avfoundation` with `--backend dshow`; on Linux
use `--backend v4l2` or omit `--backend`. Press **q** or **Esc** in the preview
window to stop.

The default output is `live_results/<session timestamp>/` under the current
working directory. `predictions.csv` contains each frame's time, quality
label, optional truth/correctness, scores, bounding box, and processing time.
`--save-frames` additionally writes raw and annotated JPEGs. Omit it for long
runs to avoid filling the disk. The score is an SVM decision value, **not a
probability**. An amber uncertain prediction is not an automatic good brick.

The checkpoint used scikit-learn 1.9.1, transformers 5.0.0, and Python 3.12
when serialized. The bundled encoder is loaded locally; there is no model
download at runtime. For PyTorch installation on another platform, use the
[official instructions](https://pytorch.org/get-started/locally/).

## Check setup without a camera

If a sample image is already on the computer, use:

```bash
python brick_calibration/live_rgb_classifier.py \
  --image /path/to/brick_photo.png --no-window --truth bad
```

This saves a side-by-side `preview.jpg` and `prediction.json` in a new
`live_results/<session timestamp>/` directory. Remove `--no-window` to open
the preview window.

## If the files have not reached the camera computer

Copy the viewer, its six supporting Python modules, `requirements-live.txt`,
and the **entire** `approach_results/baseline_checkpoint/` directory. Do not
copy only the `.joblib` files: the bundled DINOv2 model is also required.
Keep the same `brick_calibration/` directory layout shown above. A transfer
from the lab to a separate Mac folder can be made without copying the 458 MB
`brick_splits/` dataset:

```bash
mkdir -p /Users/varunteja/Downloads/brick_live
cd /Users/varunteja/Downloads/brick_live
ssh s95vkond@YOUR_LAB_HOST "cd '/ipb70/home_domuser2/s95vkond/Desktop/Franks robotics/Hackathon-MarmonxFranks' && tar -cf - \
  brick_calibration/live_rgb_classifier.py \
  brick_calibration/fusion_checkpoint.py \
  brick_calibration/evaluate_approaches.py \
  brick_calibration/evaluate_improvements.py \
  brick_calibration/evaluate_dino_approaches.py \
  brick_calibration/evaluate_canonical_a5.py \
  brick_calibration/canonical_surface.py \
  brick_calibration/requirements-live.txt \
  brick_calibration/LIVE_RGB.md \
  brick_calibration/approach_results/baseline_checkpoint" | tar -xf -
```

Replace `YOUR_LAB_HOST` with the SSH host/IP used for your earlier transfers.
Then install dependencies and run the same `python brick_calibration/live_rgb_classifier.py`
command from `/Users/varunteja/Downloads/brick_live`. The lab host `photolab95`
may resolve only inside the lab network.

## Timing and capture conditions

The camera preview keeps updating while classification runs in a worker
thread. If a prediction takes longer than the three-second interval, the viewer
skips scheduled captures rather than building a backlog. The footer and CSV
show the actual processing time. `--device auto` tries CUDA, then Apple MPS,
then CPU; use `--device cpu` if a device causes errors. The viewer requests
1920 × 1080 capture to match the original photographs and prints the resolution
actually supplied by the camera.

For a meaningful live test, keep the brick centered with consistent lighting
and distance, present both known-good and known-bad physical bricks, and check
the saved raw/annotated frames when predictions disagree with `--truth`.
