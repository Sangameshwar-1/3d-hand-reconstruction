"""
leap_ir.py — Shared Ultraleap 3Di stereo IR access.

Single source of truth for reading raw stereo IR frames and tracked joints from
LeapC, used by capture_ir.py, live_fist_tracking.py and live_leap_ir.py.

Two details matter for correctness:
  1. LeapC delivers both camera images in ONE buffer. `image[i].data` points to
     the start of that buffer for both cameras; the right image begins at
     `image[1].offset` bytes (= width * height). Ignoring the offset silently
     yields two copies of the left image.
  2. Image and tracking events for the same sensor frame share `info.frame_id`
     (and the same timestamp), so pairs are matched exactly by frame id rather
     than by "latest event wins".
"""

import sys
import threading
import time
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import numpy as np

if sys.platform != "win32":
    raise RuntimeError(
        "The Ultraleap 3Di service and leapc_cffi bindings are installed on Windows only. "
        "Run this script from Windows PowerShell (python capture_ir.py ...), not WSL/Linux. "
        "Training and evaluation (train_*.py, eval_fist.py) do work under WSL."
    )

import leap
from leap.enums import PolicyFlag
from leapc_cffi import ffi, libleapc

import config


# ==============================================================================
# Raw data extraction
# ==============================================================================

def read_stereo_images(event) -> Tuple[np.ndarray, np.ndarray]:
    """Copy the (left, right) uint8 IR images out of a LeapC ImageEvent."""
    out = []
    for i in range(2):
        im = event._data.image[i]
        w, h = im.properties.width, im.properties.height
        base = ffi.cast("uint8_t *", im.data) + im.offset
        out.append(np.frombuffer(ffi.buffer(base, w * h), dtype=np.uint8).reshape(h, w).copy())
    return out[0], out[1]


def extract_joints(hand) -> np.ndarray:
    """
    (21, 3) float32 joints in mm, native LeapC tracking space.

    Joint 0 is the wrist (arm.next_joint); joints 1-20 are next_joint of the
    4 bones of each digit (thumb, index, middle, ring, pinky), matching
    config.JOINT_NAMES.
    """
    pts = [hand.arm.next_joint]
    for d in range(5):
        digit = hand.digits[d]
        for b in range(4):
            pts.append(digit.bones[b].next_joint)
    return np.array([[p.x, p.y, p.z] for p in pts], dtype=np.float32)


def hand_metadata(hand) -> Dict:
    """Per-hand quality and pose descriptors stored alongside every label."""
    return {
        "hand_type": "Left" if hand.type == leap.HandType.Left else "Right",
        "confidence": float(hand.confidence),
        "visible_time_us": int(hand.visible_time),
        "grab_strength": float(hand.grab_strength),
        "pinch_strength": float(hand.pinch_strength),
        "fingers_extended": [bool(hand.digits[d].is_extended) for d in range(5)],
        "palm_position_mm": [float(hand.palm.position.x),
                             float(hand.palm.position.y),
                             float(hand.palm.position.z)],
    }


def joints_to_dict(joints: np.ndarray) -> Dict[str, Dict[str, float]]:
    """(21, 3) array -> {"idx_Name": {"x", "y", "z"}} as used by dataset_utils."""
    return {
        f"{i}_{config.JOINT_NAMES[i]}": {"x": float(x), "y": float(y), "z": float(z)}
        for i, (x, y, z) in enumerate(joints)
    }


def read_stereo_calibration(connection) -> Dict:
    """Factory calibration of both IR cameras as reported by the tracking service."""
    ptr = connection.get_connection_ptr()
    calib = {}
    for name in ("stereo_left", "stereo_right"):
        cam = getattr(libleapc, f"eLeapPerspectiveType_{name}")
        K = ffi.new("float[]", 9)
        D = ffi.new("float[]", 8)
        E = ffi.new("float[]", 16)
        libleapc.LeapCameraMatrix(ptr, cam, K)
        libleapc.LeapDistortionCoeffs(ptr, cam, D)
        libleapc.LeapExtrinsicCameraMatrix(ptr, cam, E)
        calib[name] = {
            "camera_matrix": np.array(list(K)).reshape(3, 3).tolist(),
            "distortion_coeffs": list(D),
            # LeapC returns the 4x4 extrinsic in column-major order
            "extrinsic_matrix": np.array(list(E)).reshape(4, 4).T.tolist(),
        }
    return calib


def read_rectilinear_luts(connection, half_range: float = 4.0, n: int = 161) -> Tuple[np.ndarray, Dict]:
    """
    Sample LeapRectilinearToPixel() on an n x n grid of ray coordinates for both
    IR cameras. Only valid after the service has streamed at least one image.
    Returns (grid, {camera_name: (n, n, 2) pixel table indexed [ry, rx]}).
    """
    ptr = connection.get_connection_ptr()
    grid = np.linspace(-half_range, half_range, n)
    tables = {}
    v = ffi.new("LEAP_VECTOR *")
    for name in ("stereo_left", "stereo_right"):
        cam = getattr(libleapc, f"eLeapPerspectiveType_{name}")
        table = np.empty((n, n, 2), dtype=np.float32)
        for iy, ry in enumerate(grid):
            for ix, rx in enumerate(grid):
                v.x, v.y, v.z = float(rx), float(ry), 1.0
                p = libleapc.LeapRectilinearToPixel(ptr, cam, v[0])
                table[iy, ix] = (p.x, p.y)
        tables[name] = table
    return grid, tables


# ==============================================================================
# Frame-id synchronised receiver
# ==============================================================================

class StereoFrame:
    """One sensor frame: stereo IR images plus the tracked hand (if any)."""

    __slots__ = ("frame_id", "timestamp_us", "left", "right", "joints", "meta", "hands_seen")

    def __init__(self, frame_id, timestamp_us, left, right, joints, meta, hands_seen=()):
        self.frame_id = frame_id
        self.timestamp_us = timestamp_us
        self.left = left
        self.right = right
        self.joints = joints  # (21, 3) mm or None
        self.meta = meta      # hand_metadata() dict or None
        self.hands_seen = hands_seen  # "left"/"right" for every tracked hand, before the hand filter


class StereoIRReceiver(leap.Listener):
    """
    LeapC listener that pairs image and tracking events by frame id.

    `latest()` returns the most recent complete StereoFrame. Frames are only
    published once both halves have arrived, so images and joints always come
    from the same sensor exposure.
    """

    _MAX_PENDING = 64

    def __init__(self, hand_filter: str = "any"):
        super().__init__()
        self.hand_filter = hand_filter.lower()
        self.lock = threading.Lock()
        self._images: "OrderedDict[int, tuple]" = OrderedDict()
        self._tracks: "OrderedDict[int, tuple]" = OrderedDict()
        self._latest: Optional[StereoFrame] = None
        self._new_frame = threading.Condition(self.lock)
        self.device_info = "Searching for Ultraleap 3Di..."
        self.fps = 0.0
        self._fps_count = 0
        self._fps_t0 = time.time()

    # -- LeapC callbacks (poll thread) -----------------------------------------
    def on_connection_event(self, event):
        print("[LEAP-IR] Connected to Ultraleap tracking service.")

    def on_device_event(self, event):
        try:
            with event.device.open():
                info = event.device.get_info()
                self.device_info = f"Ultraleap 3Di ({info.serial})"
                print(f"[LEAP-IR] Device: {self.device_info}")
        except Exception:
            pass

    def on_image_event(self, event):
        try:
            left, right = read_stereo_images(event)
        except Exception as e:
            print(f"[LEAP-IR] Failed to read IR images: {e}")
            return
        fid, ts = event._data.info.frame_id, event._data.info.timestamp
        with self.lock:
            self._images[fid] = (ts, left, right)
            self._try_publish(fid)
            self._trim(self._images)

    def on_tracking_event(self, event):
        chosen = None
        seen = tuple("left" if h.type == leap.HandType.Left else "right" for h in event.hands)
        for h, h_type in zip(event.hands, seen):
            if self.hand_filter in ("any", h_type):
                chosen = h
                break
        if chosen is not None:
            track = (extract_joints(chosen), hand_metadata(chosen), seen)
        else:
            track = (None, None, seen)
        fid = event.info.frame_id
        with self.lock:
            self._tracks[fid] = track
            self._try_publish(fid)
            self._trim(self._tracks)

    # -- internals (lock held) --------------------------------------------------
    def _try_publish(self, fid):
        if fid not in self._images or fid not in self._tracks:
            return
        ts, left, right = self._images.pop(fid)
        joints, meta, seen = self._tracks.pop(fid)
        self._latest = StereoFrame(fid, ts, left, right, joints, meta, seen)
        self._fps_count += 1
        now = time.time()
        if now - self._fps_t0 >= 1.0:
            self.fps = self._fps_count / (now - self._fps_t0)
            self._fps_count = 0
            self._fps_t0 = now
        self._new_frame.notify_all()

    def _trim(self, d):
        while len(d) > self._MAX_PENDING:
            d.popitem(last=False)

    # -- consumer API -----------------------------------------------------------
    def latest(self) -> Optional[StereoFrame]:
        with self.lock:
            return self._latest

    def wait_for_frame(self, after_frame_id: int = -1, timeout: float = 0.05) -> Optional[StereoFrame]:
        """Block until a frame newer than `after_frame_id` is available."""
        with self.lock:
            if self._latest is None or self._latest.frame_id <= after_frame_id:
                self._new_frame.wait(timeout)
            return self._latest


def open_stereo_connection(receiver: StereoIRReceiver) -> "leap.Connection":
    """Create a connection with the receiver attached. Use as `with conn.open():`
    and call `enable_images(conn)` inside the block."""
    conn = leap.Connection()
    conn.add_listener(receiver)
    return conn


def enable_images(conn) -> None:
    conn.set_policy_flags(flags_to_set=[PolicyFlag.Images])
