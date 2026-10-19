"""
perspective_crop.py — Virtual-camera stereo crops of the raw 3Di fisheye views.

A plain box crop of a fisheye image hides where the hand was in the image, and
the same hand pose looks differently rotated at different image positions, so
a network predicting in device axes cannot recover global orientation from it.
Following perspective crop layers (Yu et al., CVPR 2021) and UmeTrack (Han et
al., SIGGRAPH Asia 2022), both views are instead re-rendered through ONE
virtual pinhole rotation aimed at the hand:

  * both virtual cameras share rotation and focal length, so the crops form a
    rectified stereo pair (epipolar lines along the crop x axis at roll 0);
  * joints are predicted in that virtual camera frame, which no longer depends
    on where the hand sits in the fisheye image;
  * rolling the virtual camera about its axis is an exact rotation
    augmentation: image and 3D labels rotate together.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from ir_geometry import CAMERAS, LEAP_TO_CAM, RectilinearLUT

GEO_DIM = 2


@dataclass
class CropGeometry:
    cam_rotation: np.ndarray        # (3, 3) camera axes -> virtual camera axes
    focal: float                    # virtual focal length in crop pixels
    offsets: Dict[str, np.ndarray]  # per view: hand centre in normalised virtual coords
    center_mm: np.ndarray           # triangulated hand centre, LeapC space
    size: int

    @property
    def leap_to_virtual(self) -> np.ndarray:
        """(3, 3) rotation taking LeapC-space vectors into the virtual camera frame."""
        return self.cam_rotation @ LEAP_TO_CAM

    def geo_features(self) -> np.ndarray:
        """Distance and angular size of the hand, which a crop alone cannot reveal."""
        return np.array([np.linalg.norm(self.center_mm) / 300.0,
                         self.size / (2.0 * self.focal)], dtype=np.float32)


def _closest_point(o1, d1, o2, d2) -> Optional[np.ndarray]:
    w0 = o1 - o2
    a, b, c = d1 @ d1, d1 @ d2, d2 @ d2
    d, e = d1 @ w0, d2 @ w0
    denom = a * c - b * b
    if denom < 1e-9:
        return None
    t1 = (b * e - c * d) / denom
    t2 = (a * e - b * d) / denom
    if t1 <= 0 or t2 <= 0:
        return None
    return 0.5 * ((o1 + t1 * d1) + (o2 + t2 * d2))


def hand_center_from_boxes(
    luts: Dict[str, RectilinearLUT],
    bbox_l: Tuple[int, int, int, int],
    bbox_r: Tuple[int, int, int, int],
) -> Optional[Tuple[np.ndarray, float]]:
    """Triangulate the two box centres -> (centre in LeapC mm, angular radius in rad)."""
    origins, dirs, radii = [], [], []
    for cam, (x1, y1, x2, y2) in zip(CAMERAS, (bbox_l, bbox_r)):
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        pts = np.array([[cx, cy], [x1, cy], [x2, cy], [cx, y1], [cx, y2]], dtype=np.float64)
        rays = luts[cam].pixel_to_ray(pts)
        if not np.isfinite(rays[0]).all():
            return None
        rays = rays[np.isfinite(rays).all(axis=1)]
        d = np.c_[rays, np.ones(len(rays))]
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        if len(d) > 1:
            radii.append(float(np.arccos(np.clip(d[1:] @ d[0], -1.0, 1.0)).max()))
        origins.append(luts[cam].camera_center_mm)
        dirs.append(LEAP_TO_CAM.T @ d[0])
    if not radii:
        return None
    center = _closest_point(origins[0], dirs[0], origins[1], dirs[1])
    if center is None or not (30.0 < center[1] < 1000.0):
        return None
    return center, float(np.clip(np.mean(radii), math.radians(5), math.radians(60)))


def make_crop_geometry(
    luts: Dict[str, RectilinearLUT],
    center_mm: np.ndarray,
    ang_radius: float,
    size: int,
    roll: float = 0.0,
) -> CropGeometry:
    z = LEAP_TO_CAM @ center_mm
    z = z / np.linalg.norm(z)
    ex = np.array([1.0, 0.0, 0.0])  # stereo baseline direction in camera axes
    x = ex - (ex @ z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c, s = math.cos(roll), math.sin(roll)
    x, y = c * x + s * y, -s * x + c * y
    rot = np.stack([x, y, z])
    offsets = {}
    for cam in CAMERAS:
        p = rot @ (LEAP_TO_CAM @ (center_mm - luts[cam].camera_center_mm))
        offsets[cam] = p[:2] / p[2]
    focal = (size / 2.0) / math.tan(ang_radius)
    return CropGeometry(rot, focal, offsets, np.asarray(center_mm, dtype=np.float64), size)


def render_view(img: np.ndarray, lut: RectilinearLUT, geom: CropGeometry, cam: str) -> np.ndarray:
    """Resample one raw fisheye view into the virtual pinhole crop."""
    S = geom.size
    a = (np.arange(S) + 0.5 - S / 2.0) / geom.focal
    off = geom.offsets[cam]
    nx, ny = np.meshgrid(a + off[0], a + off[1])
    d_virt = np.stack([nx, ny, np.ones_like(nx)], axis=-1).reshape(-1, 3)
    d_cam = d_virt @ geom.cam_rotation  # rows of R^T d
    with np.errstate(divide="ignore", invalid="ignore"):
        front = d_cam[:, 2] > 1e-6
        rx = np.where(front, d_cam[:, 0] / d_cam[:, 2], np.nan)
        ry = np.where(front, d_cam[:, 1] / d_cam[:, 2], np.nan)
    px = lut.ray_to_pixel(rx, ry).reshape(S, S, 2)
    px = np.where(np.isfinite(px), px, -1.0).astype(np.float32)
    return cv2.remap(img, px[..., 0], px[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def project_to_crop(joints_mm: np.ndarray, lut: RectilinearLUT, geom: CropGeometry, cam: str) -> np.ndarray:
    """(N, 3) LeapC joints -> (N, 2) pixel coordinates in the virtual crop."""
    p = (geom.leap_to_virtual @ (joints_mm - lut.camera_center_mm).T).T
    n = p[:, :2] / p[:, 2:3]
    return geom.focal * (n - geom.offsets[cam]) + geom.size / 2.0 - 0.5
