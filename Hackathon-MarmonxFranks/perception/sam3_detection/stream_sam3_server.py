"""Serve SAM3 inference over a length-prefixed binary stream on stdin/stdout.

This process is intended to run through SSH. It never stores input frames or
results: each JPEG request receives JSON metadata and one annotated JPEG.
"""

import argparse
import json
import struct
import sys
import time

import cv2
import numpy as np

from segment_lego_sam3_single_prompt import (
    BrickSegmenter, add_detection_args, preview,
)


MAX_FRAME_BYTES = 20 * 1024 * 1024


def read_exact(stream, size):
    data = bytearray()
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jpeg-quality', type=int, default=90)
    add_detection_args(parser)
    args = parser.parse_args()

    print(f'[stream-server] loading {args.model}', file=sys.stderr, flush=True)
    segmenter = BrickSegmenter(args)
    print(f'[stream-server] ready on {segmenter.device}', file=sys.stderr, flush=True)
    source, sink = sys.stdin.buffer, sys.stdout.buffer

    while True:
        header = read_exact(source, 4)
        if header is None:
            break
        length = struct.unpack('!I', header)[0]
        if not 0 < length <= MAX_FRAME_BYTES:
            raise ValueError(f'invalid frame length: {length}')
        payload = read_exact(source, length)
        if payload is None:
            break

        started = time.perf_counter()
        bgr = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError('could not decode JPEG request')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        semantic, instance, records, rejected, offset = segmenter(rgb)

        # The single-prompt preview is input | instances | semantic. Return the
        # semantic panel so the Mac can display labels directly on the frame.
        rendered = preview(rgb, instance, records)
        annotated_rgb = rendered[:, 2 * rgb.shape[1]:3 * rgb.shape[1]]
        ok, encoded = cv2.imencode(
            '.jpg', cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
        if not ok:
            raise RuntimeError('could not encode annotated frame')

        per_colour = {}
        for record in records:
            per_colour[record['colour']] = per_colour.get(record['colour'], 0) + 1
        good = sum(record['quality'] == 'good' for record in records)
        metadata = json.dumps({
            'bricks': len(records), 'good': good, 'bad': len(records) - good,
            'per_colour': per_colour, 'rejected': len(rejected),
            'colour_offset_lab': np.round(offset, 1).tolist(),
            'inference_seconds': round(time.perf_counter() - started, 3),
        }).encode('utf-8')
        image_bytes = encoded.tobytes()
        sink.write(struct.pack('!II', len(metadata), len(image_bytes)))
        sink.write(metadata)
        sink.write(image_bytes)
        sink.flush()


if __name__ == '__main__':
    main()
