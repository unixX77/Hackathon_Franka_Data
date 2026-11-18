# Color-only brick quality classifier

The user confirmed the Brick 1–10 good/bad folder labels are based on color.
This classifier predicts quality from LAB color histograms within a SAM3 brick
mask. It uses no texture descriptors, DINOv2 features, crack geometry, or
deformation features for the quality decision. SAM3 still uses visual and
geometric information to locate the brick and select a mask.

At training time, grouped validation chooses between a full L/a/b histogram
and a chroma-only a/b histogram, and selects linear-SVM regularization. The
final checkpoint selected **a/b chroma histograms**, so its quality score uses
32 color histogram bins. `color_svm.joblib` is separate from Approach 17.
The saved classifier emits good for score <= 0 and bad for score > 0; Approach
17 threshold policies use a different score scale and do not apply here.

## Evaluation

Leave-one-physical-brick-out evaluation was run on 258 photos with the saved
SAM3 masks. Feature choice and regularization were selected within the training
bricks in each outer fold. Every photo received a binary prediction.

| Actual label | Predicted good | Predicted bad |
|---|---:|---:|
| Good | 57 | 52 |
| Bad | 70 | 79 |

Overall accuracy was **136/258 = 52.7%**; good photo recall was 57/109 =
52.3%, and bad photo recall was 79/149 = 53.0%. Aggregating each held-out
brick's photos with a median score correctly classified 5/10 physical bricks.
The ten bricks are a small sample, and this result is not a live-camera
accuracy measurement. In particular, 70/149 bad photos were predicted good.

`held_out_image_predictions.csv` and `held_out_brick_predictions.csv` show
each result, and `selected_parameters.csv` records the training-only parameter
choice per held-out brick. The earlier color-only Approach 4 used different
automatic masks and evaluation settings, so its 172/258 result is not a
controlled comparison of this SAM3-mask model.

Color is affected by illumination, shadows, white balance, the visible face,
and segmentation. A crack or deformation is not explicitly measured, but it
can change which pixels are visible or their color distribution; this model
does not guarantee invariance to such changes. To verify the absence of a
spatial feature, the same masked image and a version with its masked pixels
randomly rearranged produced exactly the same color-only score.

## Mac live viewer

After copying the updated `brick_calibration/remote_rgb_client.py` from
photolab into the Mac runtime, run:

```bash
python brick_calibration/remote_rgb_client.py \
  --host 131.220.238.95 --camera 1 --backend avfoundation \
  --pipeline sam3_color --interval 3 --save-frames
```

The model supports up to ten SAM3-detected bricks per frame. Green means good
and red means bad according to the color-only score. The window saves per-frame
metadata and optional raw/annotated photos to `live_results/<timestamp>/` on
the Mac. An SSH session loads SAM3 and the color-only model once. The saved
offline metrics are not a measurement of live-camera accuracy.

To retrain and reproduce the held-out evaluation on photolab:

```bash
/home_domuser2/s95vkond/miniconda3/envs/sam3/bin/python \
  brick_calibration/color_only_classifier.py
```
