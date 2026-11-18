"""Keep SAM3 loaded and process new camera frames from a watched directory."""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from segment_lego_sam3_single_prompt import BrickSegmenter, add_detection_args, save_result


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png'}


def ready_images(root, seen):
    """Return unseen images whose size is stable, avoiding partial network uploads."""
    candidates = [p for p in root.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and p not in seen]
    sizes = {p: p.stat().st_size for p in candidates}
    time.sleep(0.25)
    return [p for p in candidates if p.exists() and p.stat().st_size == sizes[p] and sizes[p] > 0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='directory receiving camera frames')
    parser.add_argument('--out', type=Path, default=Path('live_results'))
    parser.add_argument('--interval', type=float, default=5.0,
                        help='minimum seconds between inference starts (default: 5)')
    parser.add_argument('--include-existing', action='store_true',
                        help='process images already present at startup')
    parser.add_argument('--max-frames', type=int, default=0,
                        help='stop after N frames; 0 runs until Ctrl+C')
    add_detection_args(parser)
    args = parser.parse_args()

    args.input.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f'[live] loading {args.model}', flush=True)
    segmenter = BrickSegmenter(args)
    print(f'[live] watching {args.input.resolve()} every {args.interval:g}s', flush=True)

    seen = set() if args.include_existing else {
        p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    }
    history = []
    last_start = 0.0
    try:
        while args.max_frames <= 0 or len(history) < args.max_frames:
            paths = sorted(ready_images(args.input, seen), key=lambda p: p.stat().st_mtime)
            if not paths:
                time.sleep(0.25)
                continue

            # If capture outruns inference, process only the newest frame.
            path = paths[-1]
            seen.update(paths)
            delay = args.interval - (time.monotonic() - last_start)
            if delay > 0:
                time.sleep(delay)
            last_start = time.monotonic()
            started = time.perf_counter()
            rgb = np.array(Image.open(path).convert('RGB'))
            semantic, instance, records, rejected, offset = segmenter(rgb)
            out_dir = args.out / path.stem
            entry = save_result(out_dir, path, rgb, semantic, instance, records,
                                rejected, offset, args)
            elapsed = time.perf_counter() - started
            result = {
                'time': datetime.now().isoformat(timespec='seconds'),
                **entry,
                'inference_seconds': round(elapsed, 3),
            }
            history.append(result)
            (args.out / 'live_summary.json').write_text(json.dumps(history, indent=2))
            print(f"[live] {path.name}: {entry['bricks']} bricks, {entry['good']} good, "
                  f"{entry['bad']} bad, {elapsed:.2f}s", flush=True)
    except KeyboardInterrupt:
        print('\n[live] stopped', flush=True)


if __name__ == '__main__':
    main()
