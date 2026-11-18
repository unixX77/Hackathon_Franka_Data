"""Show a Mac camera stream with periodic remote SAM3 segmentation over SSH.

Frames and results stay in memory. The camera remains smooth while one worker
waits for GPU inference, and the latest segmentation is displayed on the right.
"""

import argparse
import json
import queue
import shlex
import struct
import subprocess
import threading
import time

import cv2
import numpy as np


SSH_OPTIONS = [
    '-o', 'ControlMaster=auto',
    '-o', 'ControlPersist=10m',
    '-o', 'ControlPath=/tmp/sam3-live-%C',
]
SERVER_SCRIPT = (
    '/home_domuser2/s95vkond/Desktop/Franks robotics/Hackathon-MarmonxFranks/'
    'perception/sam3_detection/stream_sam3_server.py'
)
REMOTE_CONDA = '/home_domuser2/s95vkond/miniconda3/bin/conda'


def read_exact(stream, size):
    data = bytearray()
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            raise EOFError('SAM3 server closed the stream')
        data.extend(chunk)
    return bytes(data)


def inference_worker(args, requests, results, busy, stop):
    remote_command = (
        'HF_HUB_CACHE=/tmp/s95vkond_hf_sam3 '
        'HF_HUB_DISABLE_PROGRESS_BARS=1 '
        f'{shlex.quote(args.remote_conda)} run --no-capture-output -n sam3 python '
        f'{shlex.quote(args.server_script)} '
        f'--threshold {args.threshold} --min-rel-area {args.min_rel_area} '
        f'--min-rect {args.min_rect}'
    )
    process = subprocess.Popen(
        ['ssh', *SSH_OPTIONS, '-T', args.remote, remote_command],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        while not stop.is_set():
            try:
                frame = requests.get(timeout=0.1)
            except queue.Empty:
                continue
            busy.set()
            try:
                ok, encoded = cv2.imencode(
                    '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                if not ok:
                    raise RuntimeError('could not encode camera frame')
                payload = encoded.tobytes()
                process.stdin.write(struct.pack('!I', len(payload)))
                process.stdin.write(payload)
                process.stdin.flush()

                header = read_exact(process.stdout, 8)
                metadata_size, image_size = struct.unpack('!II', header)
                metadata = json.loads(read_exact(process.stdout, metadata_size))
                image_bytes = read_exact(process.stdout, image_size)
                annotated = cv2.imdecode(
                    np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                if annotated is None:
                    raise RuntimeError('could not decode SAM3 response')
                while not results.empty():
                    results.get_nowait()
                results.put((annotated, metadata))
            except Exception as error:
                results.put((None, {'error': f'{type(error).__name__}: {error}'}))
                stop.set()
            finally:
                busy.clear()
    finally:
        if process.stdin:
            process.stdin.close()
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()


def status_panel(frame, lines):
    panel = frame.copy()
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, 0), (panel.shape[1], 80), (0, 0, 0), -1)
    panel = cv2.addWeighted(overlay, 0.55, panel, 0.45, 0)
    for index, line in enumerate(lines):
        cv2.putText(panel, line, (12, 27 + index * 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 255, 0), 2, cv2.LINE_AA)
    return panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--interval', type=float, default=7.0)
    parser.add_argument('--remote', default='s95vkond@131.220.238.95')
    parser.add_argument('--server-script', default=SERVER_SCRIPT)
    parser.add_argument('--remote-conda', default=REMOTE_CONDA)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--min-rel-area', type=float, default=0.4)
    parser.add_argument('--min-rect', type=float, default=0.7)
    parser.add_argument('--jpeg-quality', type=int, default=90)
    args = parser.parse_args()

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit(f'Cannot open camera {args.camera}; try --camera 1 or --camera 2')

    requests, results = queue.Queue(maxsize=1), queue.Queue(maxsize=1)
    busy, stop = threading.Event(), threading.Event()
    worker = threading.Thread(
        target=inference_worker, args=(args, requests, results, busy, stop), daemon=True)
    worker.start()

    window = 'Live RGB + SAM3 segmentation - q: quit'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    latest, metadata = None, None
    next_send = time.monotonic()
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError('could not read a camera frame')

            try:
                while True:
                    new_image, new_metadata = results.get_nowait()
                    metadata = new_metadata
                    if new_image is not None:
                        latest = new_image
            except queue.Empty:
                pass

            # Keep the window open after a worker failure so the real error is
            # visible instead of disappearing with the GUI.
            worker_failed = stop.is_set()

            now = time.monotonic()
            if now >= next_send and not busy.is_set() and requests.empty():
                requests.put(frame.copy())
                next_send = now + args.interval

            state = 'SAM3 processing...' if busy.is_set() else 'waiting for next sample'
            if metadata and 'error' in metadata:
                state = f"ERROR: {metadata['error']}"
            live = status_panel(frame, [f'Live camera | {state}', 'q: quit'])
            if latest is None:
                right = np.full_like(frame, 35)
                message = ('SAM3 failed - see error above; press q'
                           if worker_failed else 'Waiting for first SAM3 result...')
                cv2.putText(right, message, (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255), 2)
            else:
                right = cv2.resize(latest, (frame.shape[1], frame.shape[0]))
                right = status_panel(right, [
                    f"SAM3: {metadata['bricks']} bricks | {metadata['good']} good | "
                    f"{metadata['bad']} bad",
                    f"Inference: {metadata['inference_seconds']:.2f}s",
                ])
            cv2.imshow(window, np.hstack([live, right]))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        stop.set()
        camera.release()
        cv2.destroyAllWindows()
        worker.join(timeout=4)


if __name__ == '__main__':
    main()
