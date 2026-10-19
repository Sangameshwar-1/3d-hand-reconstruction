"""
live_fist_tracking.py — Real-Time Live Dual-IR 3D Hand Reconstruction Dashboard.

Consumes live active 850nm stereo IR camera frames directly from the Ultraleap 3Di
hardware, feeds the synchronized (2, 128, 128) stereo crops into Stereo3DFistNet,
and renders a unified side-by-side dashboard in real time at 60+ FPS on CPU:
  [ Left Panel: Active 850nm IR Stream ]  |  [ Right Panel: 3D Reconstructed Skeleton ]

Key Features:
  1. Hand-Centric Crop: each IR view is cropped independently by HandLocNet
     (train_handloc.py). Without a locator checkpoint the crop is placed from
     the Ultraleap joints instead and labelled "Leap-assisted" on screen.
  2. Sensor view (default): joints are drawn as the 3Di sees them, so the hand's
     real 3D orientation is visible. 'o' switches to the upright hand view
     (wrist -> middle MCP is up), which hides orientation but makes finger
     bending easy to compare.
  3. 3 Display Modes:
     - [1] Ultraleap 3Di Ground Truth Reference
     - [2] 3DFist Neural Model Prediction (default when a valid checkpoint loads)
     - [3] Dual Comparison Overlay (GT vs 3DFist + live root-relative MPJPE in mm)
  4. Interactive 3D Perspective Rotation (Yaw / Pitch) via keyboard.

Controls:
  - '1'      : Mode 1 — Ground Truth Reference (Ultraleap 3Di)
  - '2'      : Mode 2 — 3DFist Neural Network Prediction
  - '3' / 't': Mode 3 — Dual Comparison (GT vs 3DFist + Live Error)
  - 'a' / 'd': Rotate 3D view horizontally (Yaw)
  - 'w' / 's': Tilt 3D view vertically (Pitch)
  - 'r'      : Reset 3D camera rotation angle
  - 'o'      : Toggle Sensor view (true orientation) / Upright hand view
  - 'c'      : Toggle IR panel between Single (Left) and Stereo (Left + Right)
  - 'q'      : Quit

Usage:
  python live_fist_tracking.py
  python live_fist_tracking.py --weights checkpoints/fist_best.pth
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

import config
import leap_ir
from fist_model import load_fist_checkpoint
from hand_locator import HandLocator
from ir_geometry import LEAP_TO_CAM, RectilinearLUT, bbox_from_pixels
from ir_transforms import StereoPerspectivePreprocessor

# LeapC -> screen (x right, y up) as seen from the 3Di, matching the IR panel
SENSOR_VIEW = np.diag([1.0, -1.0, -1.0]) @ LEAP_TO_CAM


def render_guidance_screen(canvas_size: int = 500) -> np.ndarray:
    """Renders a visual guide when no hand is detected."""
    canvas = np.full((canvas_size, canvas_size, 3), 18, dtype=np.uint8)

    cx, cy = canvas_size // 2, canvas_size // 2 + 10
    color_guide = (70, 75, 85)

    # Stylized hand silhouette
    cv2.rectangle(canvas, (cx - 30, cy + 80), (cx + 30, cy + 130), color_guide, 2)
    cv2.ellipse(canvas, (cx, cy + 30), (55, 60), 0, 0, 360, color_guide, 2)
    finger_tips = [
        (cx - 50, cy - 10),  # Thumb
        (cx - 30, cy - 70),  # Index
        (cx, cy - 85),       # Middle
        (cx + 25, cy - 75),  # Ring
        (cx + 45, cy - 45),  # Pinky
    ]
    finger_bases = [
        (cx - 35, cy + 20),
        (cx - 20, cy - 10),
        (cx, cy - 15),
        (cx + 20, cy - 10),
        (cx + 35, cy + 5),
    ]
    for b, t in zip(finger_bases, finger_tips):
        cv2.line(canvas, b, t, color_guide, 2)
        cv2.circle(canvas, t, 5, (90, 100, 115), -1)

    cv2.putText(canvas, "3DFist: 3D Hand Reconstruction", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(canvas, "[!] WAITING FOR HAND", (cx - 120, cy - 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
    cv2.putText(canvas, "Place your hand 100-300mm above the 3Di camera", (25, canvas_size - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1)
    cv2.putText(canvas, "Palm facing sensor | Curl fingers to test 3DFist", (25, canvas_size - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (140, 220, 140), 1)
    cv2.putText(canvas, "Hotkeys: [1] Ground Truth | [2] 3DFist Model | [3] Dual Overlay", (25, canvas_size - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 140, 140), 1)

    return canvas


def hand_frame_rotation(joints_3d: np.ndarray) -> np.ndarray:
    """
    Rotation (3x3, rows = axes) of an upright frame built from the hand itself:
      - Y (up):    wrist -> middle MCP (finger direction)
      - X (width): index MCP -> pinky MCP, orthogonalised against Y
      - Z (depth): palm normal (X x Y)
    Independent of how the 3Di is mounted or which tracking mode it runs in.
    """
    y = joints_3d[9] - joints_3d[0]
    y = y / (np.linalg.norm(y) + 1e-8)
    x = joints_3d[17] - joints_3d[5]
    x = x - np.dot(x, y) * y
    x = x / (np.linalg.norm(x) + 1e-8)
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=0)


def to_canonical_hand_frame(joints_3d: np.ndarray, rotation: np.ndarray = None) -> np.ndarray:
    """Root-align to the wrist and express joints in an upright hand frame.
    Pass `rotation` to draw several skeletons in one shared frame."""
    pts = joints_3d - joints_3d[0]
    if rotation is None:
        rotation = hand_frame_rotation(joints_3d)
    return pts @ rotation.T


def root_relative_mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    """MPJPE in mm after wrist alignment; both in LeapC tracking space."""
    return float(np.mean(np.linalg.norm((pred - pred[0]) - (gt - gt[0]), axis=1)))


def draw_isometric_skeleton(
    joints_3d: np.ndarray,
    canvas_size: int = 500,
    rot_x_deg: float = 15.0,
    rot_y_deg: float = 20.0,
    frame_rotation: np.ndarray = None,
    color_scheme: str = "depth",  # 'depth', 'gt_green', 'model_orange'
    secondary_joints: np.ndarray = None,
    mode_name: str = "3DFist Prediction",
    error_mm: float = None,
    grab_strength: float = 0.0,
    center_on_hand: bool = False,
    hand_label: str = None,
) -> np.ndarray:
    """Renders 3D hand joints into an isometric canvas with palm mesh. Without
    frame_rotation the hand is drawn upright in its own frame; center_on_hand
    centres the drawing on the joint centroid instead of a fixed wrist spot."""
    canvas = np.full((canvas_size, canvas_size, 3), 18, dtype=np.uint8)

    # Background grid
    grid_color = (30, 34, 40)
    for g in range(0, canvas_size, 50):
        cv2.line(canvas, (g, 0), (g, canvas_size), grid_color, 1)
        cv2.line(canvas, (0, g), (canvas_size, g), grid_color, 1)

    # Canonical upright hand coordinates
    if frame_rotation is None:
        frame_rotation = hand_frame_rotation(joints_3d)
    canon_pts = to_canonical_hand_frame(joints_3d, frame_rotation)
    shift = canon_pts.mean(axis=0) if center_on_hand else np.zeros(3)
    canon_pts = canon_pts - shift

    # 3D Rotation matrices
    rx = np.radians(rot_x_deg)
    ry = np.radians(rot_y_deg)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])

    scale = canvas_size / (260.0 if center_on_hand else 220.0)  # ~hand span (mm) across the canvas
    cx = canvas_size // 2
    cy = canvas_size // 2 + (0 if center_on_hand else 120)  # else wrist near bottom center

    def project_pts(pts):
        rot = pts @ Rx.T @ Ry.T
        pts_2d = []
        for p in rot:
            px = int(np.clip(p[0] * scale + cx, 0, canvas_size - 1))
            py = int(np.clip(-p[1] * scale + cy, 0, canvas_size - 1))  # -Y so positive Y points UP
            pts_2d.append((px, py))
        return rot, pts_2d

    rot_main, pts_main = project_pts(canon_pts)

    # Complete anatomical connections (5 finger chains + palm knuckle arch)
    connections = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20),
        # Palm & Knuckle Arch
        (1, 5), (5, 9), (9, 13), (13, 17),
    ]

    # 1. If secondary skeleton provided (Dual Mode), draw ground truth as dashed/gray
    if secondary_joints is not None:
        canon_sec = to_canonical_hand_frame(secondary_joints, frame_rotation) - shift
        _, pts_sec = project_pts(canon_sec)
        for a, b in connections:
            cv2.line(canvas, pts_sec[a], pts_sec[b], (60, 160, 60), 2, cv2.LINE_AA)
        for i, (px, py) in enumerate(pts_sec):
            cv2.circle(canvas, (px, py), 4, (0, 255, 120), -1, cv2.LINE_AA)

    # 2. Draw Main Skeleton
    z_vals = rot_main[:, 2]
    z_min, z_max = -60.0, 60.0

    def get_color(z):
        if color_scheme == "gt_green":
            return (0, 220, 100)
        elif color_scheme == "model_orange":
            return (0, 140, 255)
        # Depth color gradient: Cyan-Blue (near) to Orange-Red (far)
        t = float(np.clip((z - z_min) / (z_max - z_min), 0.0, 1.0))
        b = int((1.0 - t) * 230 + t * 40)
        g = int((1.0 - t) * 200 + t * 80)
        r = int((1.0 - t) * 30 + t * 240)
        return (b, g, r)

    for a, b in connections:
        is_palm = (a, b) in [(1, 5), (5, 9), (9, 13), (13, 17)]
        thickness = 1 if is_palm else 3
        z_mid = (z_vals[a] + z_vals[b]) / 2.0
        col = (100, 110, 130) if is_palm else get_color(z_mid)
        cv2.line(canvas, pts_main[a], pts_main[b], col, thickness, cv2.LINE_AA)

    for i, (px, py) in enumerate(pts_main):
        is_tip = i in (4, 8, 12, 16, 20)
        is_wrist = (i == 0)
        r = 6 if is_tip else (7 if is_wrist else 4)
        j_col = (0, 255, 130) if is_tip else ((0, 180, 255) if is_wrist else (240, 240, 240))
        cv2.circle(canvas, (px, py), r, j_col, -1, cv2.LINE_AA)
        cv2.circle(canvas, (px, py), r + 1, (0, 0, 0), 1, cv2.LINE_AA)

    # Overlay HUD
    cv2.putText(canvas, f"Mode: {mode_name}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    if hand_label:
        cv2.putText(canvas, hand_label, (15, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

    if error_mm is not None:
        cv2.putText(canvas, f"Live MPJPE: {error_mm:.1f} mm", (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    elif grab_strength > 0.65:
        cv2.putText(canvas, f"Pose: Curled Fist ({int(grab_strength*100)}%)", (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 165, 255), 2)
    else:
        cv2.putText(canvas, "Pose: Open / Flexing", (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 120), 1)

    view_name = "Sensor view" if center_on_hand else "Upright hand view"
    cv2.putText(canvas, f"{view_name} [o] | Yaw {int(rot_y_deg)} | Pitch {int(rot_x_deg)}", (15, canvas_size - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 160, 160), 1)
    cv2.putText(canvas, "[1] GT Reference | [2] 3DFist Model | [3] Dual Overlay", (15, canvas_size - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 180), 1)

    return canvas


def draw_crop_box(panel, bbox, img_shape, x_off, panel_w, panel_h, color, label):
    if bbox is None:
        return
    sx, sy = panel_w / img_shape[1], panel_h / img_shape[0]
    x1, y1, x2, y2 = bbox
    p1 = (int(x1 * sx) + x_off, int(y1 * sy))
    p2 = (int(x2 * sx) + x_off, int(y2 * sy))
    cv2.rectangle(panel, p1, p2, color, 2, cv2.LINE_AA)
    cv2.putText(panel, label, (p1[0], max(18, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def render_no_model_panel(canvas_size: int, reason: str) -> np.ndarray:
    canvas = np.full((canvas_size, canvas_size, 3), 18, dtype=np.uint8)
    cv2.putText(canvas, "No valid 3DFist checkpoint", (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 140, 255), 2)
    y = 80
    for line in reason:
        cv2.putText(canvas, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        y += 22
    return canvas


def main():
    parser = argparse.ArgumentParser(description="3DFist Real-Time Live Tracking Dashboard")
    parser.add_argument("--weights", type=str, default="checkpoints/fist_best.pth",
                        help="Trained 3DFist weights path")
    parser.add_argument("--handloc", type=str, default="checkpoints/handloc_best.pth",
                        help="Trained HandLocNet weights (IR-only hand crop)")
    parser.add_argument("--allow-stale", action="store_true",
                        help="Load checkpoints from an older input pipeline anyway")
    args = parser.parse_args()

    device = torch.device("cpu")
    model = None
    no_model_reason = []
    weights_path = Path(args.weights)
    if not weights_path.exists():
        no_model_reason = [f"{weights_path} not found.", "Record: python capture_ir.py --pose <name>",
                           "Train:  python train_fist.py --dataset dataset_ir"]
    else:
        try:
            model, _ = load_fist_checkpoint(weights_path, device, allow_stale=args.allow_stale)
            print(f"[LIVE-3DFIST] Loaded weights: {weights_path}")
        except ValueError as e:
            print(f"[LIVE-3DFIST] {e}")
            no_model_reason = ["Checkpoint was trained with an old input", "pipeline (not stereo IR).",
                               "Record: python capture_ir.py --pose <name>",
                               "Train:  python train_fist.py --dataset dataset_ir"]

    locator = None
    if Path(args.handloc).exists():
        try:
            locator = HandLocator(args.handloc, device)
            print(f"[LIVE-3DFIST] Hand locator: {args.handloc}")
        except ValueError as e:
            print(f"[LIVE-3DFIST] {e}")
    if locator is None:
        print("[LIVE-3DFIST] No HandLocNet checkpoint: crops will be Leap-assisted "
              "(train one with: python train_handloc.py)")
    crop_source = "HandLocNet" if locator is not None else "Leap-assisted"
    leap_luts = None

    preprocessor = StereoPerspectivePreprocessor(target_size=config.IR_INPUT_SIZE, is_training=False)

    receiver = leap_ir.StereoIRReceiver()
    conn = leap_ir.open_stereo_connection(receiver)

    window_name = "3DFist: Stereo IR 3D Hand Reconstruction Dashboard"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass
    cv2.moveWindow(window_name, 60, 40)

    print("[LIVE-3DFIST] Connecting to Ultraleap 3Di camera...")

    # Visualization state
    sensor_view = True
    rot_x, rot_y = 0.0, 0.0
    show_stereo_ir = False
    pred_hand_label = None
    p_right_avg = None
    display_mode = 2 if model is not None else 1
    last_fid = -1
    pred_joints_3d = None
    bbox_l = bbox_r = None
    infer_ms = 0.0

    with conn.open():
        leap_ir.enable_images(conn)
        print("[LIVE-3DFIST] Dual-IR streaming active. Press 'q' to quit.")
        print("[LIVE-3DFIST] Hotkeys: [1] Ground Truth | [2] 3DFist Model | [3] Dual Overlay | [a/d] Rotate")

        prev_t = time.time()

        while True:
            frame = receiver.wait_for_frame(after_frame_id=last_fid)

            if frame is None:
                splash = np.zeros((500, 1004, 3), dtype=np.uint8)
                cv2.putText(splash, "Connecting to Ultraleap 3Di Stereo IR...", (40, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
                cv2.putText(splash, "Waiting for hardware stream and tracking service...", (40, 270),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.60, (180, 180, 180), 1)
                cv2.imshow(window_name, splash)
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break
                continue

            left_img, right_img = frame.left, frame.right
            gt_joints = frame.joints
            meta = frame.meta or {}
            grab_str = meta.get("grab_strength", 0.0)

            # 1-2. Per-view hand crop (same function as training) + forward pass
            if frame.frame_id != last_fid:
                last_fid = frame.frame_id
                if leap_luts is None:
                    print("[LIVE-3DFIST] Reading the 3Di calibration (one-off, a few seconds)...")
                    calib = leap_ir.read_stereo_calibration(conn)
                    grid, tables = leap_ir.read_rectilinear_luts(conn)
                    leap_luts = {cam: RectilinearLUT(grid, tables[cam], calib[cam]["extrinsic_matrix"][0][3])
                                 for cam in tables}
                if locator is not None:
                    bbox_l, bbox_r, _ = locator.locate_pair(left_img, right_img)
                elif gt_joints is not None:
                    bbox_l = bbox_from_pixels(leap_luts["stereo_left"].project(gt_joints), left_img.shape)
                    bbox_r = bbox_from_pixels(leap_luts["stereo_right"].project(gt_joints), right_img.shape)
                else:
                    bbox_l = bbox_r = None
                pred_joints_3d = None
                pred_hand_label = None
                if model is not None and bbox_l is not None and bbox_r is not None:
                    t_infer_start = time.time()
                    out = preprocessor(left_img, right_img, leap_luts, bbox_l, bbox_r)
                    if out is not None:
                        x, geo, geom = out
                        with torch.no_grad():
                            if getattr(model, "has_hand_head", False):
                                pred_v, logit = model(x[None].to(device), geo[None].to(device), return_hand=True)
                                p_right = float(torch.sigmoid(logit[0]))
                                # A tracked hand keeps its side, so average out single-frame flips
                                p_right_avg = p_right if p_right_avg is None else 0.85 * p_right_avg + 0.15 * p_right
                                pred_hand_label = (f"Model: RIGHT hand ({p_right_avg * 100:.0f}%)" if p_right_avg >= 0.5
                                                   else f"Model: LEFT hand ({(1 - p_right_avg) * 100:.0f}%)")
                            else:
                                pred_v = model(x[None].to(device), geo[None].to(device))
                            pred_virtual = pred_v[0].cpu().numpy()
                        pred_joints_3d = pred_virtual @ geom.leap_to_virtual  # wrist-relative, LeapC axes
                    infer_ms = (time.time() - t_infer_start) * 1000.0
            hand_found = bbox_l is not None and bbox_r is not None
            if not hand_found:
                p_right_avg = None

            # 3. Render Left Panel: IR Camera Feed (500x500)
            box_color = (0, 255, 255) if hand_found else (80, 80, 80)
            if show_stereo_ir:
                ir_panel = cv2.cvtColor(np.hstack([cv2.resize(left_img, (250, 500)),
                                                   cv2.resize(right_img, (250, 500))]), cv2.COLOR_GRAY2BGR)
                cv2.line(ir_panel, (250, 0), (250, 500), (0, 255, 0), 1)
                cv2.putText(ir_panel, "Left IR", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
                cv2.putText(ir_panel, "Right IR", (265, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
                draw_crop_box(ir_panel, bbox_l, left_img.shape, 0, 250, 500, box_color, "L crop")
                draw_crop_box(ir_panel, bbox_r, right_img.shape, 250, 250, 500, box_color, "R crop")
            else:
                ir_panel = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)
                ir_panel = cv2.resize(ir_panel, (500, 500), interpolation=cv2.INTER_LINEAR)
                cv2.putText(ir_panel, "Ultraleap 3Di: 850nm Active IR", (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                draw_crop_box(ir_panel, bbox_l, left_img.shape, 0, 500, 500, box_color, f"Hand crop ({crop_source})")

            # 4. Render Right Panel: 3D Reconstructed Skeleton
            view_rot = SENSOR_VIEW if sensor_view else None
            gt_hand_label = f"Ultraleap: {meta['hand_type'].upper()} hand" if meta.get("hand_type") else None
            if display_mode in (2, 3) and model is None:
                skel_panel = render_no_model_panel(500, no_model_reason)
            elif display_mode == 1:
                if gt_joints is None:
                    skel_panel = render_guidance_screen(canvas_size=500)
                else:
                    skel_panel = draw_isometric_skeleton(
                        gt_joints, canvas_size=500, rot_x_deg=rot_x, rot_y_deg=rot_y,
                        frame_rotation=view_rot, center_on_hand=sensor_view,
                        color_scheme="depth", mode_name="[1] Ultraleap GT Reference",
                        grab_strength=grab_str, hand_label=gt_hand_label,
                    )
            elif pred_joints_3d is None:
                skel_panel = render_guidance_screen(canvas_size=500)
            elif display_mode == 2:
                skel_panel = draw_isometric_skeleton(
                    pred_joints_3d, canvas_size=500, rot_x_deg=rot_x, rot_y_deg=rot_y,
                    frame_rotation=view_rot, center_on_hand=sensor_view,
                    color_scheme="depth", mode_name="[2] 3DFist Neural Model",
                    grab_strength=grab_str, hand_label=pred_hand_label,
                )
            else:
                # Dual overlay: both skeletons in one shared frame so differences stay visible
                live_err = None
                rot = view_rot
                if gt_joints is not None:
                    live_err = root_relative_mpjpe(pred_joints_3d, gt_joints)
                    if rot is None:
                        rot = hand_frame_rotation(gt_joints)
                skel_panel = draw_isometric_skeleton(
                    pred_joints_3d, canvas_size=500, rot_x_deg=rot_x, rot_y_deg=rot_y,
                    frame_rotation=rot, center_on_hand=sensor_view, color_scheme="model_orange",
                    secondary_joints=gt_joints,
                    mode_name="[3] Dual (Green: GT | Orange: 3DFist)",
                    error_mm=live_err, grab_strength=grab_str,
                    hand_label=" | ".join(s for s in (pred_hand_label, gt_hand_label) if s) or None,
                )

            # 5. Assemble Side-by-Side Dashboard (500x1004)
            divider = np.full((500, 4, 3), 45, dtype=np.uint8)
            dashboard_top = np.hstack([ir_panel, divider, skel_panel])

            # 6. Bottom Status Bar
            now = time.time()
            fps = 1.0 / max(1e-4, now - prev_t)
            prev_t = now

            status_bar = np.full((45, dashboard_top.shape[1], 3), 24, dtype=np.uint8)
            active_mode_str = {1: "[1] GT", 2: "[2] 3DFist", 3: "[3] Dual"}[display_mode]
            telemetry = (f"FPS: {int(fps)} | Sensor: {receiver.fps:.0f} | Infer: {infer_ms:.1f}ms (CPU) | "
                         f"Active: {active_mode_str} | Crop: {crop_source}")
            if meta:
                telemetry += f" | {meta['hand_type']} conf {meta['confidence']:.2f}"
            cv2.putText(status_bar, telemetry, (15, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            hints = "[1] GT | [2] 3DFist | [3] Dual | [o] Sensor/Upright view | [a/d] Rotate | [w/s] Tilt | [c] Stereo | [q] Quit"
            cv2.putText(status_bar, hints, (15, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1)

            cv2.imshow(window_name, np.vstack([dashboard_top, status_bar]))

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                display_mode = 1
            elif key == ord('2'):
                display_mode = 2
            elif key in (ord('3'), ord('t')):
                display_mode = 3
            elif key == ord('a'):
                rot_y = (rot_y - 10.0) % 360
            elif key == ord('d'):
                rot_y = (rot_y + 10.0) % 360
            elif key == ord('w'):
                rot_x = np.clip(rot_x + 5.0, -85.0, 85.0)
            elif key == ord('s'):
                rot_x = np.clip(rot_x - 5.0, -85.0, 85.0)
            elif key == ord('r'):
                rot_x, rot_y = (0.0, 0.0) if sensor_view else (15.0, 20.0)
            elif key == ord('o'):
                sensor_view = not sensor_view
                rot_x, rot_y = (0.0, 0.0) if sensor_view else (15.0, 20.0)
            elif key == ord('c'):
                show_stereo_ir = not show_stereo_ir

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
