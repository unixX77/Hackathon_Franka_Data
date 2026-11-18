# Approach 13 status: blocked on official model access

The implementation is complete and uses the identical masks, 224x224 crops,
valid-patch rule, nearest-good cosine distance, top-5% aggregation, inner
threshold selection, and leave-one-physical-brick-out folds as Approach 12.

The official model requested is:

```text
facebook/dinov3-vits16-pretrain-lvd1689m
```

On 2026-10-25, the configured Hugging Face token received `403 Forbidden` and
was not in the model's authorized list. Request/accept access at:

https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m

After approval, rerun:

```bash
HF_HOME=/cache/s95vkond/huggingface \
python brick_calibration/evaluate_patch_anomaly.py \
  --data /cache/s95vkond/test_data/rgb_bricks \
  --results brick_calibration/approach_results \
  --cache /cache/s95vkond/brick_calibration_features \
  --approach 13_dinov3_true_patch_anomaly \
  --model facebook/dinov3-vits16-pretrain-lvd1689m \
  --token-file /home_domuser2/s95vkond/.cache/huggingface/token
```

No Approach 13 metrics have been generated. Approach 12 and all earlier
results remain valid and untouched.

