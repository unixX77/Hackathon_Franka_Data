"""Capture an RGB frame at a fixed interval for local or remote SAM3 inference."""

import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path

import cv2


SSH_OPTIONS = [
    '-o', 'ControlMaster=auto',
    '-o', 'ControlPersist=10m',
    '-o', 'ControlPath=/tmp/sam3-live-%C',
]


def upload(path, remote, remote_dir):
    """Atomically upload one frame so the remote watcher never sees a partial file."""
    part = f'{remote_dir.rstrip("/")}/.{path.name}.part'
    final = f'{remote_dir.rstrip("/")}/{path.name}'
    subprocess.run(['scp', *SSH_OPTIONS, '-q', str(path), f'{remote}:{part}'], check=True)
    subprocess.run(['ssh', *SSH_OPTIONS, remote, 'mv', '--', part, final], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--interval', type=float, default=5.0)
    parser.add_argument('--output', type=Path, default=Path('live_frames'))
    parser.add_argument('--remote', help='SSH destination, e.g. s95vkond@131.220.238.95')
    parser.add_argument('--remote-dir', default='/tmp/s95vkond_sam3_live/incoming')
    parser.add_argument('--jpeg-quality', type=int, default=95)
    parser.add_argument('--preview', action='store_true',
                        help='show the live camera feed; press q or Escape to stop')
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    if args.remote:
        subprocess.run(['ssh', *SSH_OPTIONS, args.remote, 'mkdir', '-p', args.remote_dir], check=True)

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit(f'Cannot open camera {args.camera}; try --camera 1 or --camera 2')
    print(f'Capturing every {args.interval:g}s; press Ctrl+C to stop.')
    window = 'Periodic RGB capture - q: quit'
    if args.preview:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    count = 0
    next_capture = time.monotonic()
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError('Could not read a frame from the camera')
            now = time.monotonic()
            remaining = max(0.0, next_capture - now)
            if args.preview:
                display = frame.copy()
                cv2.putText(display,
                            f'Every {args.interval:g}s | sent: {count} | next: {remaining:.1f}s | q: quit',
                            (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2,
                            cv2.LINE_AA)
                cv2.imshow(window, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    break
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break

            if now >= next_capture:
                stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                path = args.output / f'rgb_{stamp}.jpg'
                if not cv2.imwrite(str(path), frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]):
                    raise RuntimeError(f'Could not save {path}')
                print(f'Saved {path}', flush=True)
                if args.remote:
                    upload(path, args.remote, args.remote_dir)
                    print(f'Uploaded {path.name}', flush=True)
                count += 1
                next_capture = time.monotonic() + args.interval

            if not args.preview:
                time.sleep(min(0.05, max(0.0, next_capture - time.monotonic())))
    except KeyboardInterrupt:
        print('\nCapture stopped.')
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
