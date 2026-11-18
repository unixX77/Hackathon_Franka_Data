"""Visualize classified boxes, exact top-plane pixels, and grasp poses."""

import argparse
import json
import warnings
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


LEGEND = {
    'cyan_fill': 'measured pixels on the selected physical top surface',
    'green_outline': 'classified good brick',
    'red_outline': 'classified bad brick',
    'yellow_cross': 'exact top-surface grasp centre (ray-plane intersection)',
    'blue_line': 'brick +X / long-axis orientation',
    'magenta_arrow': 'gripper approach direction toward the brick',
    'white_circle': 'pre-grasp point projected into the image',
}


def polygon_mask(shape, corners):
    mask = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(mask, np.rint(corners).astype(np.int32), 1)
    return mask > 0


def median_depth(paths):
    frames = [np.asarray(Image.open(path), dtype=np.float32) for path in paths]
    stack = np.stack(frames)
    stack[stack <= 0] = np.nan
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='All-NaN slice encountered')
        depth = np.nanmedian(stack, axis=0)
    return np.nan_to_num(depth, nan=0.0)


def xyz_image(depth, scale, intrinsics):
    height, width = depth.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    z = depth * scale
    x = (u - intrinsics['ppx']) * z / intrinsics['fx']
    y = (v - intrinsics['ppy']) * z / intrinsics['fy']
    return np.dstack((x, y, z))


def project(point, intrinsics):
    x, y, z = map(float, point)
    if z <= 1e-8:
        return None
    return np.array([intrinsics['fx'] * x / z + intrinsics['ppx'],
                     intrinsics['fy'] * y / z + intrinsics['ppy']])


def draw_cross(image, point, colour, radius=5):
    x, y = np.rint(point).astype(int)
    cv2.line(image, (x - radius, y), (x + radius, y), colour, 1, cv2.LINE_AA)
    cv2.line(image, (x, y - radius), (x, y + radius), colour, 1, cv2.LINE_AA)


def visualize(quality_path, output_root, top_band_mm):
    quality = json.loads(quality_path.read_text())
    pose_path = quality_path.parent / 'brick_poses.json'
    poses = json.loads(pose_path.read_text())
    segmentation_path = Path(poses['segmentation_source'])
    segmentation = json.loads(segmentation_path.read_text())
    rgb = np.asarray(Image.open(quality['rgb']).convert('RGB'))
    camera_info = json.loads(Path(poses['camera_info']).read_text())
    intrinsics = camera_info['aligned_depth_intrinsics']
    depth_paths = [Path(value) for value in segmentation['depth_sources']]
    depth = median_depth(depth_paths)
    xyz = xyz_image(depth, camera_info['depth_scale_m'], intrinsics)
    normal = np.asarray(segmentation['support_plane']['normal_camera'])
    offset = float(segmentation['support_plane']['d'])
    top_height = float(segmentation['top_height_m'])
    heights = xyz @ normal + offset
    top_pixels = ((depth > 0) &
                  (np.abs(heights - top_height) <= top_band_mm / 1000.0))

    overlay = rgb.copy()
    union_top = np.zeros(rgb.shape[:2], bool)
    for box in quality['boxes']:
        union_top |= top_pixels & polygon_mask(rgb.shape[:2], box['corners_xy'])
    cyan = np.array([20, 220, 235], np.uint8)
    overlay[union_top] = (0.58 * overlay[union_top] + 0.42 * cyan).astype(np.uint8)

    pose_by_id = {record['id']: record for record in poses['bricks']}
    for box in quality['boxes']:
        record = pose_by_id[box['id']]
        outline = (20, 220, 20) if box['quality'] == 'good' else (240, 25, 25)
        corners = np.rint(box['corners_xy']).astype(np.int32)
        cv2.polylines(overlay, [corners], True, outline, 1, cv2.LINE_AA)

        transform = np.asarray(record['pose_camera']['T_frame_brick'])
        centre_3d = transform[:3, 3]
        centre = project(centre_3d, intrinsics)
        long_end = project(centre_3d + transform[:3, 0] * .018, intrinsics)
        approach_end = project(centre_3d + np.asarray(
            record['approach_direction_camera']) * .035, intrinsics)
        pregrasp = project(record['pregrasp_translation_camera_m'], intrinsics)
        if centre is None:
            continue
        draw_cross(overlay, centre, (255, 230, 20))
        c = tuple(np.rint(centre).astype(int))
        if long_end is not None:
            cv2.line(overlay, c, tuple(np.rint(long_end).astype(int)),
                     (30, 100, 255), 1, cv2.LINE_AA)
        if approach_end is not None:
            cv2.arrowedLine(overlay, tuple(np.rint(approach_end).astype(int)), c,
                            (235, 40, 235), 1, cv2.LINE_AA, tipLength=.25)
        if pregrasp is not None:
            cv2.circle(overlay, tuple(np.rint(pregrasp).astype(int)),
                       3, (255, 255, 255), 1, cv2.LINE_AA)

    segmentation_image = segmentation_path.parent / 'result.png'
    segmented = (np.asarray(Image.open(segmentation_image).convert('RGB'))
                 if segmentation_image.exists() else rgb.copy())
    comparison = np.hstack((rgb, segmented, overlay))
    dataset, recording, frame = quality['dataset'], quality['recording'], quality['frame']
    destination = output_root / dataset / recording / f'{frame:06d}'
    destination.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(destination / 'pose_visualization.png')
    Image.fromarray(comparison).save(destination / 'pose_comparison.png')
    metadata = {
        'schema': 'brick-pose-visualization-v1',
        'panels': ['original RGB', 'top-layer segmentation', 'quality and pose'],
        'legend': LEGEND,
        'top_surface_band_mm': top_band_mm,
        'quality_source': str(quality_path),
        'pose_source': str(pose_path),
        'segmentation_source': str(segmentation_path),
        'brick_count': len(quality['boxes']),
    }
    (destination / 'visualization.json').write_text(json.dumps(metadata, indent=2))
    return metadata, destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('classified_results', type=Path)
    parser.add_argument('--out', type=Path, default=Path('classifier/viz_box'))
    parser.add_argument('--top-band-mm', type=float, default=2.5)
    args = parser.parse_args()
    inputs = sorted(args.classified_results.rglob('quality_boxes.json'))
    if not inputs:
        raise SystemExit(f'No quality_boxes.json found under {args.classified_results}')
    summary = []
    for index, path in enumerate(inputs, 1):
        print(f'[{index}/{len(inputs)}] {path}', flush=True)
        metadata, destination = visualize(path, args.out, args.top_band_mm)
        summary.append({
            'dataset': path.parents[2].name,
            'recording': path.parents[1].name,
            'frame': path.parent.name,
            'brick_count': metadata['brick_count'],
            'visualization': str(destination / 'pose_comparison.png'),
        })
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'summary.json').write_text(json.dumps({
        'legend': LEGEND, 'results': summary}, indent=2))
    print(f'Wrote {len(summary)} visualizations to {args.out}')


if __name__ == '__main__':
    main()
