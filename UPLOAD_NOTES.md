# Project folder upload notes

`Hackathon-MarmonxFranks/` is a snapshot of the project workspace, including its generated outputs and datasets. The original repository's internal `.git` directory is excluded. Existing source paths, hardcoded configuration, and results are otherwise retained. This snapshot does not contain the separate raw data and model caches that live elsewhere on the lab computer.

The ZIP `data_collection/captured_images_rgb_stack.zip` and the DINOv2 model weights use Git LFS. Install Git LFS and run `git lfs pull` to retrieve them after cloning. The DINOv2-small weights were restored from the project's pinned public Hugging Face revision and verified against the original recorded SHA-256 `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`.

These four source symlinks were already broken at the time of upload because their cache targets were missing; the links are retained as-is and do not contain the feature data:

- `brick_calibration/approach_results/local_defect_candidate/sam3_mask_features.npz`
- `brick_calibration/approach_results/improvement_feature_cache/dinov2_rotated_patch_features.npz`
- `brick_calibration/approach_results/dino_feature_cache/dinov2_features.npz`
- `brick_calibration/approach_results/23_colour_space_comparison/pixel_samples.npz`

Some project scripts contain lab-specific paths and may need configuration on another computer. Relative symlinks used by the brick dataset splits are preserved; platforms without symlink support may require special Git settings.
