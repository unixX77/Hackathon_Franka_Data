"""Canonical brick-surface crops for training and segmented robot detections."""
from __future__ import annotations

import cv2
import numpy as np


def canonicalize_masked_object(rgb, mask, output_size=(256, 128)):
    """Align the long silhouette axis horizontally and return image and mask."""
    points = np.column_stack(np.nonzero(mask))[:, ::-1].astype(np.float32)
    if len(points) < 20:
        raise ValueError("mask contains too few foreground pixels")
    center, (width, height), angle = cv2.minAreaRect(points)
    if width < height:
        angle += 90.0
        width, height = height, width
    rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
    border = np.median(np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1])),
                       axis=0).tolist()
    rotated = cv2.warpAffine(rgb, rotation, (rgb.shape[1], rgb.shape[0]),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=border)
    rotated_mask = cv2.warpAffine(mask.astype(np.uint8), rotation,
                                  (rgb.shape[1], rgb.shape[0]),
                                  flags=cv2.INTER_NEAREST)
    crop = cv2.getRectSubPix(rotated, (max(2, round(width)), max(2, round(height))), center)
    crop_mask = cv2.getRectSubPix(rotated_mask, (max(2, round(width)), max(2, round(height))), center)
    crop = cv2.resize(crop, output_size, interpolation=cv2.INTER_AREA)
    crop_mask = cv2.resize(crop_mask, output_size, interpolation=cv2.INTER_NEAREST)
    return crop, (crop_mask > 0).astype(np.uint8)


def canonicalize_detected_surface(rgb, corners_xy, output_size=(256, 128)):
    """Perspective-warp a segmentation quadrilateral to long-axis horizontal."""
    points = np.asarray(corners_xy, np.float32).reshape(4, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    lengths = [np.linalg.norm(ordered[(i + 1) % 4] - ordered[i]) for i in range(4)]
    if lengths[0] < lengths[1]:
        ordered = np.roll(ordered, -1, axis=0)
    width, height = output_size
    destination = np.asarray([[0, 0], [width - 1, 0],
                              [width - 1, height - 1], [0, height - 1]], np.float32)
    transform = cv2.getPerspectiveTransform(ordered, destination)
    return cv2.warpPerspective(rgb, transform, output_size, flags=cv2.INTER_LINEAR)

