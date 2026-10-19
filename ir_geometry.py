"""
ir_geometry.py — LeapC tracking space -> 3Di IR pixel projection (no LeapC needed).

Each capture session stores `rect_lut.npz`: a dense table of LeapC's own
LeapRectilinearToPixel() for both cameras, sampled on a grid of rectilinear ray
coordinates. With it, joints can be projected into the raw fisheye images
offline, using the device's factory calibration.

Convention (verified on a 3Di facing the user, by maximising IR brightness
along the projected skeleton over 40 frames in both views):
    p   = joint_mm with x' = -(x + cam_offset_x)   (mirror: device faces user)
    ray = (x' / y, z / y)                          (y = distance from device)
    pixel = LUT_cam(ray)
where cam_offset_x is the x translation of the camera's LeapC extrinsic matrix
(-32.9 mm left, +32.9 mm right). Run check_projection.py to re-verify after
changing the mounting or tracking mode.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

PROJECTION_X_SIGN = -1.0
CAMERAS = ("stereo_left", "stereo_right")

# LeapC axes -> camera axes (x, y = ray directions, z = depth); both cameras are
# parallel, so p_cam = LEAP_TO_CAM @ (p - camera_center_mm). det = +1.
LEAP_TO_CAM = np.array([[PROJECTION_X_SIGN, 0.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [0.0, 1.0, 0.0]])


class RectilinearLUT:
    """Bilinear lookup of rectilinear ray (rx, ry) -> pixel for one camera."""

    def __init__(self, grid: np.ndarray, table: np.ndarray, x_offset_mm: float):
        self.grid = grid
        self.table = table  # (G, G, 2) indexed [ry, rx]
        self.x_offset_mm = x_offset_mm
        self.camera_center_mm = np.array([-x_offset_mm, 0.0, 0.0])
        flat = table.reshape(-1, 2).astype(np.float64)
        self._flat_table = np.where(np.isfinite(flat), flat, 1e9)

    def pixel_to_ray(self, px: np.ndarray, iters: int = 8) -> np.ndarray:
        """(N, 2) pixels -> (N, 2) rays: nearest table entry, then Gauss-Newton on the bilinear LUT."""
        px = np.asarray(px, dtype=np.float64).reshape(-1, 2)
        g = self.grid
        idx = np.argmin(((self._flat_table[None] - px[:, None]) ** 2).sum(-1), axis=1)
        iy, ix = np.divmod(idx, len(g))
        r = np.stack([g[ix], g[iy]], axis=1)
        eps = 1e-3
        for _ in range(iters):
            f0 = self.ray_to_pixel(r[:, 0], r[:, 1]) - px
            if np.nanmax(np.abs(f0), initial=0.0) < 1e-3:
                break
            fx =self.ray_to_pixel(r[:, 0] + eps, r[:, 1]) - px
            fy = self.ray_to_pixel(r[:, 0], r[:, 1] + eps) - px
            J = np.stack([(fx - f0) / eps, (fy - f0) / eps], axis=2)  # (N, 2, 2)
            ok = np.isfinite(J).all(axis=(1, 2)) & np.isfinite(f0).all(axis=1)
            ok &= np.abs(np.linalg.det(np.where(ok[:, None, None], J, np.eye(2)))) > 1e-9
            if not ok.any():
                break
            r[ok] -= np.linalg.solve(J[ok], f0[ok][:, :, None])[:, :, 0]
        err = self.ray_to_pixel(r[:, 0], r[:, 1]) - px
        r[~(np.linalg.norm(err, axis=1) < 0.5)] = np.nan
        return r

    def ray_to_pixel(self, rx: np.ndarray, ry: np.ndarray) -> np.ndarray:
        g = self.grid
        step = g[1] - g[0]
        fx = (rx - g[0]) / step
        fy = (ry - g[0]) / step
        n = len(g) - 1
        ok = np.isfinite(fx) & np.isfinite(fy) & (fx >= 0) & (fx < n) & (fy >= 0) & (fy < n)
        out = np.full((len(rx), 2), np.nan)
        if not ok.any():
            return out
        x0 = np.floor(fx[ok]).astype(int)
        y0 = np.floor(fy[ok]).astype(int)
        ax = (fx[ok] - x0)[:, None]
        ay = (fy[ok] - y0)[:, None]
        T = self.table
        out[ok] = ((1 - ax) * (1 - ay) * T[y0, x0] + ax * (1 - ay) * T[y0, x0 + 1]
                   + (1 - ax) * ay * T[y0 + 1, x0] + ax * ay * T[y0 + 1, x0 + 1])
        return out

    def project(self, joints_mm: np.ndarray) -> np.ndarray:
        """(N, 3) LeapC-space joints in mm -> (N, 2) pixels (NaN if outside)."""
        x = PROJECTION_X_SIGN * (joints_mm[:, 0] + self.x_offset_mm)
        depth = joints_mm[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            rx = np.where(depth > 1.0, x / depth, np.nan)
            ry = np.where(depth > 1.0, joints_mm[:, 2] / depth, np.nan)
        return self.ray_to_pixel(rx, ry)


def save_session_lut(session_dir: Path, grid: np.ndarray, tables: Dict[str, np.ndarray]) -> None:
    np.savez_compressed(Path(session_dir) / "rect_lut.npz", grid=grid,
                        stereo_left=tables["stereo_left"], stereo_right=tables["stereo_right"])


def load_session_luts(session_dir: Path) -> Optional[Dict[str, RectilinearLUT]]:
    """Per-camera LUTs for a capture session, or None for sessions without one."""
    import json
    session_dir = Path(session_dir)
    lut_path = session_dir / "rect_lut.npz"
    calib_path = session_dir / "calibration.json"
    if not (lut_path.exists() and calib_path.exists()):
        return None
    data = np.load(lut_path)
    with open(calib_path) as f:
        calib = json.load(f)
    return {
        cam: RectilinearLUT(data["grid"], data[cam], calib[cam]["extrinsic_matrix"][0][3])
        for cam in CAMERAS
    }


def bbox_from_pixels(
    px: np.ndarray,
    img_shape: Tuple[int, int],
    pad: float = 0.25,
    min_size: int = 48,
) -> Optional[Tuple[int, int, int, int]]:
    """Square (x1, y1, x2, y2) box around projected joints, padded, clamped to the image."""
    h, w = img_shape[:2]
    valid = np.isfinite(px).all(axis=1)
    if valid.sum() < 15:
        return None
    p = px[valid]
    x1, y1 = p.min(axis=0)
    x2, y2 = p.max(axis=0)
    side = int(min(max(max(x2 - x1, y2 - y1) * (1.0 + 2 * pad), min_size), h, w))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if not (0 <= cx < w and 0 <= cy < h):
        return None
    bx1 = int(np.clip(cx - side / 2, 0, w - side))
    by1 = int(np.clip(cy - side / 2, 0, h - side))
    return (bx1, by1, bx1 + side, by1 + side)
