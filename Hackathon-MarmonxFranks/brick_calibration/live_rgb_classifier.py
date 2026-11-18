#!/usr/bin/env python3
"""Preview a USB RGB camera and classify one centered brick every few seconds.

The left panel is live camera video. The right panel shows the most recently
classified frame, with a bounding box around the automatic foreground mask.
Use --image PATH --no-window to test the exact processing path without a camera.
"""
from __future__ import annotations

import argparse
import csv
import json
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from evaluate_approaches import material_mask
from fusion_checkpoint import FusionClassifier


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "approach_results" / "baseline_checkpoint"
WINDOW = "Brick inspection | q or Esc to quit"
COLOURS = {"good": (0, 205, 0), "bad": (0, 0, 230),
           "uncertain": (0, 190, 255)}  # OpenCV BGR
FIELDS = ("captured_at", "status", "truth_label", "label", "correct", "score_bad", "approach5_score",
          "dinov2_score", "box_x", "box_y", "box_width", "box_height",
          "inference_seconds", "frame_width", "frame_height", "image_file", "error")


def timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def annotate(frame: np.ndarray, mask: np.ndarray, prediction: dict) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Show the foreground extent; the box is not a top-face segmentation."""
    selected = cv2.findNonZero((mask > 0).astype(np.uint8))
    if selected is None:
        raise ValueError("No brick foreground found")
    x, y, width, height = cv2.boundingRect(selected)
    colour = COLOURS[prediction["label"]]
    annotated = frame.copy()
    cv2.rectangle(annotated, (x, y), (x + width - 1, y + height - 1), colour, 2)
    label = prediction["label"].upper()
    score = prediction["score_bad"]
    text = f"{label}  score {score:+.3f}"
    scale = max(0.55, min(frame.shape[1] / 1200.0, 1.1))
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    text_x = min(x, max(0, frame.shape[1] - text_width - 12))
    text_y = y - 10 if y >= text_height + baseline + 14 else y + text_height + 12
    cv2.rectangle(annotated, (text_x, text_y - text_height - 7),
                  (text_x + text_width + 9, text_y + baseline + 5), (15, 15, 15), -1)
    cv2.putText(annotated, text, (text_x + 4, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness, cv2.LINE_AA)
    return annotated, (x, y, width, height)


def classify_snapshot(frame: np.ndarray, classifier: FusionClassifier,
                      captured_at: str, truth_label: str | None = None) -> dict:
    started = time.monotonic()
    row = {field: "" for field in FIELDS}
    row.update({"captured_at": captured_at, "truth_label": truth_label or "", "frame_width": frame.shape[1],
                "frame_height": frame.shape[0]})
    annotated = frame.copy()
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask = material_mask(rgb)
        prediction = classifier.classify(rgb, mask=mask)
        annotated, (x, y, width, height) = annotate(frame, mask, prediction)
        row.update({"status": "ok", "label": prediction["label"],
                    "correct": int(prediction["label"] == truth_label) if truth_label else "",
                    "score_bad": prediction["score_bad"],
                    "approach5_score": prediction["approach5_score"],
                    "dinov2_score": prediction["dinov2_score"],
                    "box_x": x, "box_y": y, "box_width": width,
                    "box_height": height})
    except (ValueError, cv2.error) as error:
        row.update({"status": "error", "error": str(error)})
        cv2.putText(annotated, "NO RELIABLE BRICK MASK", (25, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 190, 255), 2, cv2.LINE_AA)
    row["inference_seconds"] = round(time.monotonic() - started, 3)
    return {"raw": frame, "annotated": annotated, "record": row}


def panel(frame: np.ndarray, width: int) -> np.ndarray:
    height = max(1, round(frame.shape[0] * width / frame.shape[1]))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def compose(live: np.ndarray, latest: dict | None, panel_width: int,
            interval: float, processing: bool) -> np.ndarray:
    left = panel(live, panel_width)
    right = (panel(latest["annotated"], panel_width) if latest is not None
             else np.zeros_like(left))
    if right.shape[0] != left.shape[0]:
        right = cv2.resize(right, (panel_width, left.shape[0]))
    canvas = np.full((left.shape[0] + 100, panel_width * 2 + 6, 3), 23, np.uint8)
    canvas[46:46 + left.shape[0], :panel_width] = left
    canvas[46:46 + left.shape[0], panel_width + 6:] = right
    cv2.putText(canvas, "LIVE CAMERA", (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(canvas, "LATEST PREDICTION", (panel_width + 18, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (230, 230, 230), 2, cv2.LINE_AA)
    status = (f"Capturing every {interval:g}s | " +
              ("processing..." if processing else "q / Esc: quit"))
    if latest is not None:
        record = latest["record"]
        if record["status"] == "ok":
            status += f" | {record['captured_at']} | {record['label'].upper()} | {record['inference_seconds']:.1f}s"
        else:
            status += f" | {record['captured_at']} | ERROR"
    cv2.putText(canvas, status, (12, canvas.shape[0] - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


def open_camera(index: int, backend: str, width: int, height: int) -> cv2.VideoCapture:
    backends = {"auto": cv2.CAP_ANY, "avfoundation": cv2.CAP_AVFOUNDATION,
                "dshow": cv2.CAP_DSHOW, "v4l2": cv2.CAP_V4L2}
    camera = cv2.VideoCapture(index, backends[backend])
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Cannot open camera {index} with backend {backend}. Try --camera 1 or another --backend.")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    actual = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(f"Camera {index} opened at {actual[0]}x{actual[1]} pixels", flush=True)
    return camera


def worker(classifier: FusionClassifier, jobs: queue.Queue, results: queue.Queue,
           stop: threading.Event, truth_label: str | None) -> None:
    while not stop.is_set():
        try:
            frame, captured_at = jobs.get(timeout=0.1)
        except queue.Empty:
            continue
        try:
            results.put(classify_snapshot(frame, classifier, captured_at, truth_label))
        except Exception as error:
            row = {field: "" for field in FIELDS}
            row.update({"captured_at": captured_at, "truth_label": truth_label or "", "status": "error",
                        "error": f"{type(error).__name__}: {error}",
                        "frame_width": frame.shape[1], "frame_height": frame.shape[0]})
            annotated = frame.copy()
            cv2.putText(annotated, "CLASSIFICATION ERROR", (25, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 190, 255), 2, cv2.LINE_AA)
            results.put({"raw": frame, "annotated": annotated, "record": row})
        finally:
            jobs.task_done()


def log_result(result: dict, writer: csv.DictWriter, stream, destination: Path,
               save_frames: bool) -> None:
    row = result["record"]
    if save_frames:
        basename = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        raw = destination / f"raw_{basename}.jpg"
        annotated = destination / f"prediction_{basename}.jpg"
        if not cv2.imwrite(str(raw), result["raw"]) or not cv2.imwrite(str(annotated), result["annotated"]):
            raise OSError(f"Could not save frames under {destination}")
        row["image_file"] = raw.name
    writer.writerow(row)
    stream.flush()
    if row["status"] == "ok":
        print(f"{row['captured_at']} {row['label'].upper()} score={row['score_bad']:+.3f} "
              f"box=({row['box_x']},{row['box_y']},{row['box_width']},{row['box_height']}) "
              f"time={row['inference_seconds']:.2f}s", flush=True)
    else:
        print(f"{row['captured_at']} error: {row['error']}", flush=True)


def run_image(image_path: Path, classifier: FusionClassifier, destination: Path,
              panel_width: int, no_window: bool, truth_label: str | None) -> None:
    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    result = classify_snapshot(frame, classifier, timestamp(), truth_label)
    composite = compose(frame, result, panel_width, 3, False)
    preview = destination / "preview.jpg"
    if not cv2.imwrite(str(preview), composite):
        raise OSError(f"Could not save {preview}")
    (destination / "prediction.json").write_text(json.dumps(result["record"], indent=2))
    print(json.dumps({"prediction": result["record"], "preview": str(preview)}, indent=2))
    if not no_window:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.imshow(WINDOW, composite)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_camera(args, classifier: FusionClassifier, destination: Path) -> None:
    camera = open_camera(args.camera, args.backend, args.width, args.height)
    jobs: queue.Queue = queue.Queue(maxsize=1)
    results: queue.Queue = queue.Queue()
    stop = threading.Event()
    thread = threading.Thread(target=worker, args=(classifier, jobs, results, stop, args.truth), daemon=True)
    thread.start()
    latest = None
    processing = False
    next_capture = time.monotonic()
    read_failures = 0
    csv_path = destination / "predictions.csv"
    print(f"Logging predictions to {csv_path.resolve()}", flush=True)
    print("Focus the preview window. Press q or Esc to quit.", flush=True)
    try:
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            while True:
                ok, frame = camera.read()
                if not ok:
                    read_failures += 1
                    if read_failures >= 20:
                        raise RuntimeError("Camera stopped delivering frames. Check the camera connection and permissions.")
                    time.sleep(0.05)
                    continue
                read_failures = 0
                now = time.monotonic()
                if now >= next_capture:
                    if not processing:
                        jobs.put_nowait((frame.copy(), timestamp()))
                        processing = True
                    next_capture = now + args.interval
                while True:
                    try:
                        latest = results.get_nowait()
                    except queue.Empty:
                        break
                    processing = False
                    log_result(latest, writer, stream, destination, args.save_frames)
                cv2.imshow(WINDOW, compose(frame, latest, args.display_width,
                                           args.interval, processing))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        stop.set()
        camera.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="USB camera index, usually 0 or 1")
    parser.add_argument("--backend", choices=("auto", "avfoundation", "dshow", "v4l2"),
                        default="auto", help="OpenCV camera backend")
    parser.add_argument("--width", type=int, default=1920, help="Requested capture width")
    parser.add_argument("--height", type=int, default=1080, help="Requested capture height")
    parser.add_argument("--interval", type=float, default=3.0, help="Seconds between classification attempts")
    parser.add_argument("--display-width", type=int, default=640, help="Width of each preview panel")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto",
                        help="Inference device; auto uses CUDA, then Apple MPS, then CPU")
    parser.add_argument("--output", type=Path,
                        default=Path.cwd() / "live_results" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--save-frames", action="store_true", help="Save raw and annotated JPEGs (uses disk space)")
    parser.add_argument("--image", type=Path, help="Classify one saved image instead of opening a camera")
    parser.add_argument("--truth", choices=("good", "bad"),
                        help="Optional ground-truth label for this brick/session")
    parser.add_argument("--no-window", action="store_true", help="Headless mode for --image only")
    args = parser.parse_args()
    if args.interval <= 0 or args.display_width < 200 or args.width < 1 or args.height < 1:
        parser.error("interval must be positive; display width >= 200; capture dimensions >= 1")
    if args.no_window and args.image is None:
        parser.error("--no-window requires --image")
    if not args.checkpoint.is_dir():
        parser.error(f"checkpoint directory not found: {args.checkpoint}")
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Loading Approach 17 checkpoint from {args.checkpoint.resolve()}...", flush=True)
    import torch
    from transformers.utils import logging as transformers_logging
    transformers_logging.disable_progress_bar()
    device = args.device
    if device == "auto":
        device = ("cuda" if torch.cuda.is_available() else
                  "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Inference device: {device}", flush=True)
    classifier = FusionClassifier.load(args.checkpoint, device=device)
    if args.image is not None:
        run_image(args.image, classifier, args.output, args.display_width,
                  args.no_window, args.truth)
    else:
        run_camera(args, classifier, args.output)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, cv2.error) as error:
        raise SystemExit(str(error)) from error
    except KeyboardInterrupt:
        pass
