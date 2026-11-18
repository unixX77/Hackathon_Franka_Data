#!/usr/bin/env python3
"""Audit labelled live predictions and optionally upload evidence to photolab.

Uses only the Python standard library. It does not retrain or alter Approach 17.
Each predictions.csv should come from one physical brick captured with
--truth good or --truth bad. Run on the Mac after collecting several sessions.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path


REMOTE_ROOT = "/cache/s95vkond/live_score_audits"


def stats(values: list[float]) -> dict:
    return {"min": min(values), "median": statistics.median(values),
            "max": max(values)}


def read_session(path: Path) -> tuple[list[dict], int]:
    rows, skipped = [], 0
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"status", "truth_label", "score_bad", "approach5_score", "dinov2_score"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} is missing columns: {sorted(required - set(reader.fieldnames or []))}")
        for record in reader:
            truth = record["truth_label"].strip().lower()
            if record["status"] != "ok" or truth not in ("good", "bad"):
                skipped += 1
                continue
            try:
                scores = [float(record[key]) for key in
                          ("score_bad", "approach5_score", "dinov2_score")]
            except (TypeError, ValueError):
                skipped += 1
                continue
            if not all(math.isfinite(score) for score in scores):
                skipped += 1
                continue
            rows.append({"session": str(path.parent), "truth": truth,
                         "label": record.get("label", "").strip().lower(),
                         "score_bad": scores[0], "approach5_score": scores[1],
                         "dinov2_score": scores[2],
                         "captured_at": record.get("captured_at", ""),
                         "image_file": record.get("image_file", ""),
                         "box_x": record.get("box_x", ""),
                         "box_y": record.get("box_y", ""),
                         "box_width": record.get("box_width", ""),
                         "box_height": record.get("box_height", ""),
                         "frame_width": record.get("frame_width", ""),
                         "frame_height": record.get("frame_height", "")})
    return rows, skipped


def score_summary(rows: list[dict]) -> dict:
    return {key: stats([row[key] for row in rows]) for key in
            ("score_bad", "approach5_score", "dinov2_score")}


def select_examples(rows: list[dict], limit: int) -> list[dict]:
    if limit == 0:
        return []
    available = [row for row in rows if row["image_file"] and
                 (Path(row["session"]) / row["image_file"]).is_file()]
    if not available:
        return []
    ranked = sorted(available, key=lambda row: row["score_bad"])
    choices = [row for row in available if row["label"] != row["truth"]]
    choices += [ranked[0], ranked[len(ranked) // 2], ranked[-1]]
    choices += available
    selected, seen = [], set()
    for row in choices:
        if row["image_file"] not in seen:
            selected.append(row)
            seen.add(row["image_file"])
        if len(selected) >= limit:
            break
    return selected


def build_report(sessions: list[tuple[Path, list[dict], int]], output: Path,
                 frames_per_session: int) -> dict:
    all_rows = [row for _, rows, _ in sessions for row in rows]
    if not all_rows:
        raise ValueError("No labelled predictions found. Capture each session with --truth good or --truth bad.")
    output.mkdir(parents=True, exist_ok=False)
    csv_dir = output / "source_csv"
    image_dir = output / "sample_frames"
    csv_dir.mkdir()
    image_dir.mkdir()
    per_session, manifest = [], []
    for index, (path, rows, skipped) in enumerate(sessions, 1):
        session_key = f"session_{index:03d}"
        shutil.copy2(path, csv_dir / f"{session_key}_predictions.csv")
        labels = Counter(row["truth"] for row in rows)
        entry = {"session": str(path.parent), "session_key": session_key,
                 "label_counts": dict(labels), "valid_frames": len(rows),
                 "skipped_rows": skipped,
                 "scores": score_summary(rows) if rows else None}
        per_session.append(entry)
        for row in select_examples(rows, frames_per_session):
            source = path.parent / row["image_file"]
            target = image_dir / f"{session_key}_{source.name}"
            shutil.copy2(source, target)
            manifest.append({"sample": str(target.relative_to(output)),
                             "source": str(source), "truth": row["truth"],
                             "score_bad": row["score_bad"]})
    with (output / "all_predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    good = [row["score_bad"] for row in all_rows if row["truth"] == "good"]
    bad = [row["score_bad"] for row in all_rows if row["truth"] == "bad"]
    threshold = (max(good) + min(bad)) / 2 if good and bad and max(good) < min(bad) else None
    report = {
        "note": "Observed live sessions only. A candidate threshold from these same frames is exploratory, not independently validated.",
        "sessions": len(sessions), "frames": len(all_rows),
        "truth_counts": dict(Counter(row["truth"] for row in all_rows)),
        "current_prediction_counts": dict(Counter(row["label"] for row in all_rows)),
        "current_correct_frames": sum(row["label"] == row["truth"] for row in all_rows),
        "good": score_summary([row for row in all_rows if row["truth"] == "good"]) if good else None,
        "bad": score_summary([row for row in all_rows if row["truth"] == "bad"]) if bad else None,
        "highest_good_score": max(good) if good else None,
        "lowest_bad_score": min(bad) if bad else None,
        "ranges_overlap": max(good) >= min(bad) if good and bad else None,
        "candidate_midgap_threshold": threshold,
        "candidate_correct_on_same_frames": (
            sum(("bad" if row["score_bad"] >= threshold else "good") == row["truth"]
                for row in all_rows) if threshold is not None else None),
        "per_session": per_session,
        "sample_manifest": manifest,
    }
    (output / "threshold_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def upload(output: Path, host: str, user: str, remote_root: str) -> str:
    login = f"{user}@{host}"
    subprocess.run(["ssh", login, "mkdir", "-p", remote_root], check=True)
    subprocess.run(["scp", "-r", str(output), f"{login}:{remote_root}/"], check=True)
    return f"{login}:{remote_root}/{output.name}/"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd() / "live_results",
                        help="Folder containing one or more live session folders")
    parser.add_argument("--csv", type=Path, action="append", default=[],
                        help="Explicit predictions.csv; may be repeated")
    parser.add_argument("--output-root", type=Path,
                        default=Path.cwd() / "live_score_audit")
    parser.add_argument("--frames-per-session", type=int, default=3)
    parser.add_argument("--upload", action="store_true",
                        help="Send report, original CSVs, and sample raw frames to photolab")
    parser.add_argument("--host", default="131.220.238.95")
    parser.add_argument("--user", default="s95vkond")
    parser.add_argument("--remote-root", default=REMOTE_ROOT)
    args = parser.parse_args()
    if args.frames_per_session < 0:
        parser.error("--frames-per-session must be nonnegative")
    paths = sorted(set(args.csv or args.root.rglob("predictions.csv")))
    if not paths:
        parser.error(f"No predictions.csv files found under {args.root}")
    sessions = [(path, *read_session(path)) for path in paths]
    output = args.output_root / datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    report = build_report(sessions, output, args.frames_per_session)
    print(f"Local report: {output / 'threshold_report.json'}")
    print(f"Labelled frames: {report['frames']}; skipped rows: "
          f"{sum(item['skipped_rows'] for item in report['per_session'])}; "
          f"sample raw images: {len(report['sample_manifest'])}")
    if not report["sample_manifest"]:
        print("No raw frames found; CSV scores are still included. Use --save-frames in future live sessions.")
    print(f"Good maximum: {report['highest_good_score']}; bad minimum: {report['lowest_bad_score']}")
    if report["ranges_overlap"] is True:
        print("Score ranges overlap: no single threshold separates these frames.")
    elif report["candidate_midgap_threshold"] is not None:
        print(f"Exploratory midpoint threshold: {report['candidate_midgap_threshold']:+.3f}")
    else:
        print("Both good and bad labelled sessions are needed to estimate a threshold.")
    if args.upload:
        print(f"Uploaded to {upload(output, args.host, args.user, args.remote_root)}")


if __name__ == "__main__":
    main()
