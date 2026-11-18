# Brick-disjoint train / validation / test splits

Created by `brick_calibration/make_brick_splits.py` (seed 2026) from
`/cache/s95vkond/test_data/rgb_bricks` (258 images, 10 physical bricks).

## Families

| Family | Splits | Train | Validation | Test | Set aside |
|---|---:|---|---|---|---|
| `with_brick6` | 24 | 2 good + 4 bad (~154 images) | 1 good + 1 bad (~51) | 1 good + 1 bad (~52) | none |
| `without_brick6` | 20 | 2 good + 3 bad (~130 images) | 1 good + 1 bad (~51) | 1 good + 1 bad (~52) | Brick_6_bad (24) |

- Every (good, bad) test pair is used exactly once in each family.
- The validation pair is drawn by the seed from the remaining bricks, excluding
  Brick 6. The 20 test pairs common to both families have identical validation
  bricks, so they differ only in whether Brick 6 is in training or set aside.
- Brick 6 is never used for validation. In `with_brick6` it is in test in 4
  splits and in training in the other 20.
- Primary split (seeded): test = Brick_2_good + Brick_10_bad
  (`without_brick6/split_09`, `with_brick6/split_11`).

## Layout

`<family>/split_XX/{train,val,test[,set_aside]}/Brick_N_label/*.png`

This is the local copy of the splits in `/cache/s95vkond/brick_splits`, kept here
because the cache disk is full. `images/` is a byte-identical copy of
`/cache/s95vkond/test_data/rgb_bricks` (SHA-256 checked, 458 MB). The split folders
contain relative symbolic links into `images/`, so move or copy
`brick_splits/` as a whole (use `cp -a` or `rsync -a` to keep the links).

The brick assignments and the per-image manifest are identical to the cache
version, and evaluating Approaches 5 and 17 on either copy gives identical
results. Recreate with:

```bash
python make_brick_splits.py --data brick_splits/images --out brick_splits --link-mode symlink
```
`<family>/split_manifest.csv` lists every image per split with its SHA-256;
`<family>/split_assignment.json` lists the bricks per subset.

## Verification (`split_verification.json`)

Passed for all 44 splits:
- no byte-identical images in the dataset (SHA-256 over all 258 files);
- no brick, file path or image content shared between train, val, test and set-aside;
- every image appears in exactly one subset of every split;
- train, val and test each contain a good and a bad brick;
- every file on disk is a link to the listed source image.

A deliberately corrupted split (a training brick added to test) was detected
by all three overlap checks.
