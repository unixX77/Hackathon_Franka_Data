#!/usr/bin/env python3
"""Create brick-disjoint train/validation/test splits and verify them.

Two split families are written:

* ``with_brick6``    - all ten bricks take part; Brick 6 is a normal bad brick.
* ``without_brick6`` - Brick 6 is never used for training, validation or test;
                       it is stored under ``set_aside`` and scored separately.

Every split holds out exactly one good and one bad physical brick as the test
set, and every possible test pair is used once (24 and 20 splits). The
validation pair (one good, one bad) is drawn with a fixed seed from the
remaining bricks, excluding Brick 6, so for the 20 test pairs shared by both
families the validation bricks are identical. The only difference between
those splits is whether Brick 6 is in the training set or set aside.

Images are materialised with hard links by default (no data is copied; the
cache disk is nearly full). ``--link-mode symlink`` uses relative symbolic
links instead, for a split tree that sits next to a local copy of the images on
another filesystem. Each split directory is laid out as
``split_XX/{train,val,test[,set_aside]}/Brick_N_label/*.png``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from evaluate_approaches import discover_images

SET_ASIDE_BRICK = "Brick_6_bad"
SEED = 2026


def brick_number(brick):
    return int(brick.split("_")[1])


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def design(bricks_by_label, seed):
    good = sorted(bricks_by_label["good"], key=brick_number)
    bad = sorted(bricks_by_label["bad"], key=brick_number)
    rng = np.random.default_rng(seed)
    validation = {}
    for test_good in good:
        for test_bad in bad:
            good_pool = [b for b in good if b != test_good]
            bad_pool = [b for b in bad if b not in (test_bad, SET_ASIDE_BRICK)]
            validation[(test_good, test_bad)] = (str(rng.choice(good_pool)),
                                                 str(rng.choice(bad_pool)))
    families = {}
    for family, excluded in (("with_brick6", set()), ("without_brick6", {SET_ASIDE_BRICK})):
        splits = []
        for test_good in good:
            for test_bad in bad:
                if test_bad in excluded:
                    continue
                val = validation[(test_good, test_bad)]
                test = (test_good, test_bad)
                train = tuple(b for b in good + bad
                              if b not in test + val and b not in excluded)
                splits.append({"train": train, "val": val, "test": test,
                               "set_aside": tuple(sorted(excluded))})
        families[family] = splits
    shared = [s["test"] for s in families["without_brick6"]]
    primary_test = shared[int(rng.integers(len(shared)))]
    return families, primary_test


def link(source, destination, mode="hardlink"):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if os.path.samefile(source, destination):
            return
        raise FileExistsError(f"{destination} exists and is not the source image")
    if mode == "symlink":
        os.symlink(os.path.relpath(Path(source).resolve(), destination.parent.resolve()), destination)
    else:
        os.link(source, destination)  # never copy onto the full cache disk


def verify(family_dir, splits, records, hashes):
    """Return a list of failed checks (empty means the family is correct)."""
    failures = []
    by_brick = {}
    for record in records:
        by_brick.setdefault(record["brick_id"], []).append(record)
    for index, split in enumerate(splits):
        name = f"split_{index:02d}"
        files = {part: {str(r["path"]) for b in split[part] for r in by_brick[b]}
                 for part in ("train", "val", "test", "set_aside")}
        content = {part: {hashes[f] for f in paths} for part, paths in files.items()}
        parts = list(files)
        for i, first in enumerate(parts):
            for second in parts[i + 1:]:
                if set(split[first]) & set(split[second]):
                    failures.append(f"{name}: bricks shared by {first} and {second}")
                if files[first] & files[second]:
                    failures.append(f"{name}: image paths shared by {first} and {second}")
                if content[first] & content[second]:
                    failures.append(f"{name}: identical image content in {first} and {second}")
        if sum(len(paths) for paths in files.values()) != len(records):
            failures.append(f"{name}: split does not account for every image exactly once")
        for part in ("train", "val", "test"):
            labels = {r["label_name"] for b in split[part] for r in by_brick[b]}
            if labels != {"good", "bad"}:
                failures.append(f"{name}: {part} lacks a good or bad brick")
        on_disk = {part: sorted((family_dir / name / part).rglob("*.png"))
                   for part in parts}
        for part, paths in on_disk.items():
            if len(paths) != len(files[part]):
                failures.append(f"{name}/{part}: {len(paths)} files on disk, "
                                f"{len(files[part])} in manifest")
            for path in paths:
                source = Path(records[0]["path"]).parents[1] / path.parent.name / path.name
                if not os.path.samefile(path, source):
                    failures.append(f"{path} does not point to {source}")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--link-mode", choices=("hardlink", "symlink"), default="hardlink")
    args = parser.parse_args()
    records = discover_images(args.data)
    hashes = {str(r["path"]): sha256(r["path"]) for r in records}
    duplicates = len(hashes) - len(set(hashes.values()))
    if duplicates:
        raise ValueError(f"{duplicates} byte-identical duplicate images in the dataset")
    bricks_by_label = {}
    for record in records:
        bricks_by_label.setdefault(record["label_name"], set()).add(record["brick_id"])
    families, primary_test = design(bricks_by_label, args.seed)

    report = {"seed": args.seed, "dataset": str(args.data), "images": len(records),
              "byte_identical_duplicates": duplicates,
              "primary_test_pair": list(primary_test), "families": {}}
    for family, splits in families.items():
        family_dir = args.out / family
        rows = []
        for index, split in enumerate(splits):
            for part, bricks in split.items():
                for record in records:
                    if record["brick_id"] not in bricks:
                        continue
                    destination = (family_dir / f"split_{index:02d}" / part /
                                   record["brick_id"] / record["path"].name)
                    link(record["path"], destination, args.link_mode)
                    rows.append({"split": f"split_{index:02d}", "subset": part,
                                 "brick_id": record["brick_id"],
                                 "label": record["label_name"],
                                 "source_file": str(record["path"]),
                                 "split_file": str(destination),
                                 "sha256": hashes[str(record["path"])]})
        with (family_dir / "split_manifest.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        assignment = [{"split": f"split_{i:02d}",
                       **{part: list(bricks) for part, bricks in split.items()},
                       "primary": split["test"] == primary_test}
                      for i, split in enumerate(splits)]
        (family_dir / "split_assignment.json").write_text(json.dumps(assignment, indent=2))
        failures = verify(family_dir, splits, records, hashes)
        report["families"][family] = {
            "splits": len(splits), "failed_checks": failures,
            "verified": not failures,
            "images_per_split": {part: int(np.mean([sum(r["brick_id"] in s[part] for r in records)
                                                    for s in splits]))
                                 for part in ("train", "val", "test", "set_aside")}}
        print(f"{family}: {len(splits)} splits, "
              f"{'all checks passed' if not failures else f'{len(failures)} FAILED checks'}")
    (args.out / "split_verification.json").write_text(json.dumps(report, indent=2))
    if any(not family["verified"] for family in report["families"].values()):
        raise SystemExit("split verification failed; see split_verification.json")


if __name__ == "__main__":
    main()
