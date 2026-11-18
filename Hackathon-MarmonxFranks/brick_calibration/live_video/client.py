#!/usr/bin/env python3
"""Mac RGB viewer for the isolated live_video Approach 17 variant.

Only OpenCV and NumPy are needed on the Mac. The checkpoint stays on photolab.
Select one brick's region at startup; each request contains a fresh burst.
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import queue
import shlex
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


WINDOW = "Remote brick inspection | q or Esc to quit"
LAB_REPO = "/ipb70/home_domuser2/s95vkond/Desktop/Franks robotics/Hackathon-MarmonxFranks"
LAB_PYTHON = "/home_domuser2/s95vkond/miniconda3/envs/sam3/bin/python"
REMOTE_SERVER = "brick_calibration/live_video/server.py"
COLORS = {"good": (0, 205, 0), "bad": (0, 0, 230),
          "uncertain": (0, 190, 255)}
CSV_FIELDS = ("captured_at", "status", "truth_label", "label", "correct",
              "score_bad", "approach5_score", "dinov2_score", "box_x", "box_y",
              "box_width", "box_height", "inference_seconds", "roundtrip_seconds",
              "frame_width", "frame_height", "mask_fraction", "box_fraction",
              "clipped_material_fraction", "background_like_fraction", "quality_flags", "frames_used",
              "frames_received", "image_file", "error")


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class RemoteModel:
    def __init__(self, args, command_override=None):
        if command_override is None:
            remote = (f"cd {shlex.quote(args.remote_repo)} && exec "
                      f"{shlex.quote(args.remote_python)} -u "
                      f"{REMOTE_SERVER}")
            command = ["ssh", "-T", "-o", "ConnectTimeout=10",
                       f"{args.user}@{args.host}", remote]
        else:
            command = command_override
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=None,
                                        text=True, bufsize=1)
        try:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("SSH ended before the model was ready. Check the host, VPN, SSH login, and lab Python path.")
            ready = json.loads(line)
            if ready.get("type") != "ready":
                raise RuntimeError(f"Unexpected server response: {ready}")
            print(f"Connected to {ready['model']} on {ready['device']}", flush=True)
        except Exception:
            self.close()
            raise

    def predict(self, frames: list[np.ndarray], frame_id: int, captured_at: str,
                truth_label: str | None, jpeg_quality: int,
                roi: tuple[int, int, int, int] | None) -> dict:
        images = []
        for frame in frames:
            ok, image = cv2.imencode(".jpg", frame,
                                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if not ok:
                raise RuntimeError("Could not JPEG-encode the camera frame")
            images.append(base64.b64encode(image).decode("ascii"))
        request = {"frame_id": frame_id, "captured_at": captured_at,
                   "truth_label": truth_label,
                   "images_b64": images, "roi": roi}
        started = time.monotonic()
        if self.process.poll() is not None:
            raise RuntimeError(f"Remote SSH process exited with code {self.process.returncode}")
        self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Connection to photolab closed while waiting for a prediction")
        reply = json.loads(line)
        if reply.get("type") == "error":
            raise RuntimeError(reply.get("error", "Remote inference failed"))
        if reply.get("type") != "prediction" or reply.get("frame_id") != frame_id:
            raise RuntimeError(f"Unexpected response for frame {frame_id}: {reply}")
        reply["roundtrip_seconds"] = round(time.monotonic() - started, 3)
        return reply

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.communicate()


def annotate(frame: np.ndarray, record: dict) -> np.ndarray:
    annotated = frame.copy()
    if record.get("status") != "ok":
        cv2.putText(annotated, "NO VALID BRICK MASK", (25, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 190, 255), 2, cv2.LINE_AA)
        return annotated
    label = record["label"]
    if label not in COLORS:
        raise ValueError(f"Unexpected label from server: {label}")
    color = COLORS[label]
    if record.get("mask_png_b64"):
        encoded = np.frombuffer(base64.b64decode(record["mask_png_b64"]), np.uint8)
        mask = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        ox, oy = int(record["mask_origin_x"]), int(record["mask_origin_y"])
        if mask is not None and oy + mask.shape[0] <= frame.shape[0] and ox + mask.shape[1] <= frame.shape[1]:
            region = annotated[oy:oy + mask.shape[0], ox:ox + mask.shape[1]]
            tint = np.full_like(region, (255, 180, 0))
            blended = cv2.addWeighted(region, .82, tint, .18, 0)
            region[mask > 0] = blended[mask > 0]
    x, y, width, height = (int(record[key]) for key in
                           ("box_x", "box_y", "box_width", "box_height"))
    if not (0 <= x < frame.shape[1] and 0 <= y < frame.shape[0] and
            1 <= width <= frame.shape[1] - x and 1 <= height <= frame.shape[0] - y):
        raise ValueError("Server returned a bounding box outside the frame")
    cv2.rectangle(annotated, (x, y), (x + width - 1, y + height - 1), color, 2)
    text = f"{label.upper()}  score {float(record['score_bad']):+.3f}"
    scale = max(0.55, min(frame.shape[1] / 1200.0, 1.1))
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                                         scale, 2)
    text_x = min(x, max(0, frame.shape[1] - text_width - 12))
    text_y = y - 10 if y >= text_height + baseline + 14 else y + text_height + 12
    cv2.rectangle(annotated, (text_x, text_y - text_height - 7),
                  (text_x + text_width + 9, text_y + baseline + 5), (15, 15, 15), -1)
    cv2.putText(annotated, text, (text_x + 4, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 2, cv2.LINE_AA)
    return annotated


def compose(live: np.ndarray, latest: dict | None, width: int,
            interval: float, processing: bool) -> np.ndarray:
    height = max(1, round(live.shape[0] * width / live.shape[1]))
    left = cv2.resize(live, (width, height), interpolation=cv2.INTER_AREA)
    right = (cv2.resize(latest["annotated"], (width, height),
                        interpolation=cv2.INTER_AREA) if latest is not None
             else np.zeros_like(left))
    canvas = np.full((height + 100, width * 2 + 6, 3), 23, np.uint8)
    canvas[46:46 + height, :width] = left
    canvas[46:46 + height, width + 6:] = right
    cv2.putText(canvas, "LIVE MAC CAMERA", (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(canvas, "PHOTOLAB PREDICTION", (width + 18, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (230, 230, 230), 2, cv2.LINE_AA)
    status = f"Every {interval:g}s | " + ("waiting for photolab..." if processing else "q / Esc: quit")
    if latest is not None:
        row = latest["record"]
        status += f" | {row['captured_at']} | {str(row.get('label') or 'ERROR').upper()}"
        if row.get("roundtrip_seconds") != "":
            status += f" | round trip {float(row['roundtrip_seconds']):.1f}s"
    cv2.putText(canvas, status, (12, canvas.shape[0] - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


def worker(model: RemoteModel, jobs: queue.Queue, answers: queue.Queue,
           stop: threading.Event, truth: str | None, quality: int,
           roi: tuple[int, int, int, int] | None) -> None:
    while not stop.is_set():
        try:
            frames, frame_id, captured_at = jobs.get(timeout=0.1)
        except queue.Empty:
            continue
        try:
            record = model.predict(frames, frame_id, captured_at, truth, quality, roi)
            frame = frames[int(record.get("frame_index_used", len(frames) - 1))]
            answers.put({"raw": frame, "annotated": annotate(frame, record),
                         "record": record})
        except Exception as error:
            answers.put({"fatal": f"{type(error).__name__}: {error}"})
            stop.set()
        finally:
            jobs.task_done()


def open_camera(args) -> cv2.VideoCapture:
    backends = {"auto": cv2.CAP_ANY, "avfoundation": cv2.CAP_AVFOUNDATION,
                "dshow": cv2.CAP_DSHOW, "v4l2": cv2.CAP_V4L2}
    camera = cv2.VideoCapture(args.camera, backends[args.backend])
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Could not open camera {args.camera}. Try --camera 0 or 1 and check macOS camera permission.")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    print(f"Camera resolution: {int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))}", flush=True)
    return camera


def save_result(result: dict, writer: csv.DictWriter, stream, output: Path,
                save_frames: bool) -> None:
    row = {key: result["record"].get(key, "") for key in CSV_FIELDS}
    if save_frames:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        raw_path = output / f"raw_{stamp}.jpg"
        annotated_path = output / f"prediction_{stamp}.jpg"
        if not cv2.imwrite(str(raw_path), result["raw"]) or not cv2.imwrite(str(annotated_path), result["annotated"]):
            raise OSError(f"Could not save frames under {output}")
        row["image_file"] = raw_path.name
    writer.writerow(row)
    stream.flush()
    print(f"{row['captured_at']} {str(row['label'] or row['status']).upper()} "
          f"score={row['score_bad']} roundtrip={row['roundtrip_seconds']}s", flush=True)


def run_image(args, model: RemoteModel) -> None:
    frame = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Could not read image: {args.image}")
    result = model.predict([frame], 1, timestamp(), args.truth, args.jpeg_quality,
                           args.roi)
    annotated = annotate(frame, result)
    composed = compose(frame, {"annotated": annotated, "record": result},
                       args.display_width, args.interval, False)
    preview = args.output / "preview.jpg"
    if not cv2.imwrite(str(preview), composed):
        raise OSError(f"Could not write {preview}")
    (args.output / "prediction.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"prediction": result, "preview": str(preview)}, indent=2))
    if not args.no_window:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.imshow(WINDOW, composed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_camera(args, model: RemoteModel) -> None:
    camera = open_camera(args)
    roi = args.roi
    if roi is None and not args.full_frame:
        ok, sample = camera.read()
        if not ok:
            raise RuntimeError("Could not read camera frame for ROI selection")
        print("Select a region containing ONE brick plus a little background; press Enter.", flush=True)
        selection = cv2.selectROI("Select one brick", sample, fromCenter=False,
                                  showCrosshair=True)
        cv2.destroyWindow("Select one brick")
        if selection[2] < 64 or selection[3] < 64:
            raise RuntimeError("No valid brick ROI selected")
        roi = tuple(map(int, selection))
    print(f"Brick ROI: {roi or 'full frame'}", flush=True)
    jobs: queue.Queue = queue.Queue(maxsize=1)
    answers: queue.Queue = queue.Queue()
    stop = threading.Event()
    thread = threading.Thread(target=worker, args=(model, jobs, answers, stop,
                                                   args.truth, args.jpeg_quality,
                                                   roi), daemon=True)
    thread.start()
    latest = None
    processing = False
    frame_id = 0
    next_capture = time.monotonic()
    failures = 0
    print(f"Results: {args.output.resolve()}", flush=True)
    try:
        with (args.output / "predictions.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            while True:
                ok, frame = camera.read()
                if not ok:
                    failures += 1
                    if failures >= 20:
                        raise RuntimeError("Camera stopped delivering frames")
                    time.sleep(0.05)
                    continue
                failures = 0
                now = time.monotonic()
                if now >= next_capture:
                    if not processing:
                        frames = [frame.copy()]
                        for _ in range(args.burst - 1):
                            time.sleep(.08)
                            fresh, next_frame = camera.read()
                            if not fresh:
                                raise RuntimeError("Camera stopped during burst capture")
                            frames.append(next_frame.copy())
                        frame_id += 1
                        jobs.put_nowait((frames, frame_id, timestamp()))
                        processing = True
                    next_capture = now + args.interval
                while True:
                    try:
                        answer = answers.get_nowait()
                    except queue.Empty:
                        break
                    if "fatal" in answer:
                        raise RuntimeError(answer["fatal"])
                    latest = answer
                    processing = False
                    save_result(latest, writer, stream, args.output, args.save_frames)
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
    parser.add_argument("--host", default="131.220.238.95", help="Photolab SSH host/IP")
    parser.add_argument("--user", default="s95vkond", help="Photolab SSH username")
    parser.add_argument("--remote-repo", default=LAB_REPO)
    parser.add_argument("--remote-python", default=LAB_PYTHON)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--backend", choices=("auto", "avfoundation", "dshow", "v4l2"),
                        default="auto")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--interval", type=float, default=3.0)
    parser.add_argument("--display-width", type=int, default=640)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--burst", type=int, default=3,
                        help="Fresh frames per prediction, 1 to 5")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                        help="Fixed single-brick region in full-frame pixels")
    parser.add_argument("--full-frame", action="store_true",
                        help="Skip interactive ROI selection (single-brick full frame only)")
    parser.add_argument("--truth", choices=("good", "bad"),
                        help="Optional known label for the current brick")
    parser.add_argument("--output", type=Path,
                        default=Path.cwd() / "live_results" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--image", type=Path, help="Test one saved image through photolab")
    parser.add_argument("--no-window", action="store_true", help="Headless, requires --image")
    args = parser.parse_args()
    if args.interval <= 0 or args.display_width < 200 or args.width < 1 or args.height < 1:
        parser.error("Invalid interval or preview/camera dimensions")
    if not 60 <= args.jpeg_quality <= 100:
        parser.error("JPEG quality must be between 60 and 100")
    if not 1 <= args.burst <= 5:
        parser.error("--burst must be between 1 and 5")
    if args.no_window and args.image is None:
        parser.error("--no-window requires --image")
    args.output.mkdir(parents=True, exist_ok=True)
    model = RemoteModel(args)
    try:
        if args.image is not None:
            run_image(args, model)
        else:
            run_camera(args, model)
    finally:
        model.close()


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, cv2.error) as error:
        raise SystemExit(str(error)) from error
    except KeyboardInterrupt:
        pass
