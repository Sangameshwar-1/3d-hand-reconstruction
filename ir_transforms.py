"""
ir_transforms.py — Active Infrared Preprocessing and Augmentation Pipeline.

Addresses:
  1. Active IR Brightness Saturation: Simulates 850nm inverse-square LED flash
     blooming and intensity blowout on palmar surfaces at close range (<150mm).
  2. Contrast Equalization: Optional CLAHE to reveal subtle contour boundaries.
  3. Perspective stereo crops: (2, 128, 128) virtual-camera crops of both views.
"""

import math
import random
from typing import Tuple, Optional
import numpy as np
import torch

from perspective_crop import CropGeometry, hand_center_from_boxes, make_crop_geometry, render_view


class ActiveIRSaturationAugmentor:
    """
    Simulates physical active IR illumination effects:
      - 1/r^2 intense flash saturation (hotspots clipped to 255).
      - Diffuse blooming around bright finger pads and palm.
      - Global exposure drop when hand recedes from IR LEDs.
    """

    def __init__(
        self,
        saturation_prob: float = 0.5,
        max_hotspots: int = 3,
        hotspot_radius_range: Tuple[int, int] = (10, 35),
        bloom_factor_range: Tuple[float, float] = (1.2, 2.5),
    ):
        self.saturation_prob = saturation_prob
        self.max_hotspots = max_hotspots
        self.hotspot_radius_range = hotspot_radius_range
        self.bloom_factor_range = bloom_factor_range

    def __call__(self, img_left: np.ndarray, img_right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() > self.saturation_prob:
            return img_left, img_right

        # Work on copies
        l_aug = img_left.copy().astype(np.float32)
        r_aug = img_right.copy().astype(np.float32)

        h, w = img_left.shape[:2]
        num_spots = random.randint(1, self.max_hotspots)

        for _ in range(num_spots):
            # Pick a center biased toward brighter regions (palmar surfaces)
            cx = random.randint(w // 4, 3 * w // 4)
            cy = random.randint(h // 4, 3 * h // 4)
            radius = random.randint(*self.hotspot_radius_range)
            gain = random.uniform(*self.bloom_factor_range)

            # Generate Gaussian circular kernel
            y, x = np.ogrid[:h, :w]
            dist_sq = (x - cx) ** 2 + (y - cy) ** 2
            mask = np.exp(-dist_sq / (2.0 * (radius ** 2)))

            # Apply same active illumination hotspot to both stereo views
            l_aug += l_aug * mask * gain
            r_aug += r_aug * mask * gain

        # Clip values to 8-bit dynamic range [0, 255]
        l_aug = np.clip(l_aug, 0, 255).astype(np.uint8)
        r_aug = np.clip(r_aug, 0, 255).astype(np.uint8)

        return l_aug, r_aug


def jitter_bbox(
    bbox: Tuple[int, int, int, int],
    img_shape: Tuple[int, int],
    shift: float = 0.08,
    scale: Tuple[float, float] = (0.9, 1.15),
    errors: Optional[np.ndarray] = None,
) -> Tuple[int, int, int, int]:
    """Random shift/scale of a square box (training only), mimicking the
    localisation error of HandLocNet. With `errors` (rows of measured
    (dx/side, dy/side, side ratio), see hand_locator.measure_box_errors) a
    real locator error is replayed instead of the uniform ranges."""
    h, w = img_shape[:2]
    x1, y1, x2, y2 = bbox
    gt_side = x2 - x1
    if errors is not None and len(errors):
        dx, dy, ratio = errors[random.randrange(len(errors))]
    else:
        ratio = random.uniform(*scale)
        dx, dy = random.uniform(-shift, shift) * ratio, random.uniform(-shift, shift) * ratio
    side = int(min(max(gt_side * ratio, 32), h, w))
    cx = (x1 + x2) / 2 + dx * gt_side
    cy = (y1 + y2) / 2 + dy * gt_side
    nx1 = int(np.clip(cx - side / 2, 0, w - side))
    ny1 = int(np.clip(cy - side / 2, 0, h - side))
    return (nx1, ny1, nx1 + side, ny1 + side)


class StereoPerspectivePreprocessor:
    """
    Raw stereo IR views + per-view hand boxes -> perspective stereo crops
    (perspective_crop.py):
      returns (x (2, S, S) in [-1, 1], geo (GEO_DIM,), CropGeometry) or None
      when the boxes cannot be triangulated.
    Training jitters both boxes (locator error) and rolls the virtual camera
    by up to +/- max_roll_deg; labels must be rotated with
    geometry.leap_to_virtual, so the augmentation stays exact.
    """

    def __init__(self, target_size: int = 128, is_training: bool = False, max_roll_deg: float = 45.0):
        self.target_size = target_size
        self.is_training = is_training
        self.max_roll = math.radians(max_roll_deg)
        self.saturator = ActiveIRSaturationAugmentor(saturation_prob=0.5 if is_training else 0.0)
        self.box_errors: Optional[np.ndarray] = None

    def geometry(self, luts, img_shape, bbox_left, bbox_right) -> Optional[CropGeometry]:
        hc = None
        if self.is_training:
            hc = hand_center_from_boxes(luts, jitter_bbox(bbox_left, img_shape, errors=self.box_errors),
                                        jitter_bbox(bbox_right, img_shape, errors=self.box_errors))
        if hc is None:
            hc = hand_center_from_boxes(luts, bbox_left, bbox_right)
        if hc is None:
            return None
        roll = random.uniform(-self.max_roll, self.max_roll) if self.is_training else 0.0
        return make_crop_geometry(luts, hc[0], hc[1], self.target_size, roll)

    def __call__(self, img_left, img_right, luts, bbox_left, bbox_right):
        geom = self.geometry(luts, img_left.shape, bbox_left, bbox_right)
        if geom is None:
            return None
        crop_l = render_view(img_left, luts["stereo_left"], geom, "stereo_left")
        crop_r = render_view(img_right, luts["stereo_right"], geom, "stereo_right")
        if self.is_training:
            crop_l, crop_r = self.saturator(crop_l, crop_r)
        stacked = np.stack([crop_l, crop_r]).astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(stacked), torch.from_numpy(geom.geo_features()), geom
