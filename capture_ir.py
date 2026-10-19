"""
capture_ir.py — Native Ultraleap 3Di Stereo Infrared (IR) Dataset Capture.

Records raw stereo 850nm IR frames together with the Ultraleap 3D hand joints
of the SAME sensor frame (matched by LeapC frame id), without any external RGB
webcam or checkerboard calibration.

Each run creates a new session folder:
  dataset_ir/session_YYYYmmdd_HHMMSS/
  ├── left/            # 384x384 raw Left IR images (.png)
  ├── right/           # 384x384 raw Right IR images (.png)
  ├── leap/            # 21 3D joints in mm (native LeapC space) + hand metadata (.json)
  ├── calibration.json # Factory intrinsics/distortion/extrinsics of both IR cameras
  ├── rect_lut.npz     # LeapC ray->pixel tables (projects joints into the images)
  └── metadata.json

Usage:
  python capture_ir.py --pose open_palm --duration 60
  python capture_ir.py --pose fist --hand right --fps 10
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import config
import leap_ir
from ir_geometry import save_session_lut


def run_capture(output_dir: Path, duration: float, hand_filter: str, fps_cap: float,
                pose: str, min_confidence: float):
    session_dir = Path(output_dir) / datetime.now().strftime("session_%Y%m%d_%H%M%S")
    left_dir, right_dir, leap_dir = (session_dir / d for d in ("left", "right", "leap"))
    for d in (left_dir, right_dir, leap_dir):
        d.mkdir(parents=True, exist_ok=True)

    receiver = leap_ir.StereoIRReceiver(hand_filter=hand_filter)
    conn = leap_ir.open_stereo_connection(receiver)

    min_save_interval_us = 1e6 / fps_cap
    last_saved_ts = -1e18
    saved = 0
    last_fid = -1

    print()
    print("=" * 60)
    print("  Ultraleap 3Di — Stereo IR Dataset Capture")
    print(f"  Session:     {session_dir}")
    print(f"  Duration:    {duration:.1f} s   Save rate: {fps_cap:.1f} FPS")
    print(f"  Pose label:  {pose}   Hand filter: {hand_filter}")
    print("=" * 60)

    with conn.open():
        leap_ir.enable_images(conn)
        window_name = "Ultraleap 3Di IR Capture Preview"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        calib = None
        start_ts = None

        while True:
            frame = receiver.wait_for_frame(after_frame_id=last_fid)
            if frame is None:
                placeholder = np.zeros((384, 768, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Waiting for Ultraleap IR frames...", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                cv2.imshow(window_name, placeholder)
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break
                continue
            if frame.frame_id == last_fid:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            last_fid = frame.frame_id

            if calib is None:
                # Calibration is only valid once the service has streamed images
                calib = leap_ir.read_stereo_calibration(conn)
                with open(session_dir / "calibration.json", "w") as f:
                    json.dump(calib, f, indent=2)
                grid, tables = leap_ir.read_rectilinear_luts(conn)
                save_session_lut(session_dir, grid, tables)
                start_ts = frame.timestamp_us

            elapsed = (frame.timestamp_us - start_ts) / 1e6
            if duration > 0 and elapsed >= duration:
                print(f"\n[CAPTURE-IR] Reached duration limit ({duration:.1f}s). Stopping.")
                break

            has_hand = frame.joints is not None
            good = has_hand and frame.meta["confidence"] >= min_confidence
            if good and (frame.timestamp_us - last_saved_ts) >= min_save_interval_us:
                saved += 1
                last_saved_ts = frame.timestamp_us
                name = f"frame_{saved:05d}"
                cv2.imwrite(str(left_dir / f"{name}.png"), frame.left)
                cv2.imwrite(str(right_dir / f"{name}.png"), frame.right)
                record = {
                    "frame_id": saved,
                    "leap_frame_id": int(frame.frame_id),
                    "timestamp_us": int(frame.timestamp_us),
                    "pose_label": pose,
                    **frame.meta,
                    "joints": leap_ir.joints_to_dict(frame.joints),
                }
                with open(leap_dir / f"{name}.json", "w") as f:
                    json.dump(record, f, indent=2)

            # Preview: both views side by side
            display = cv2.cvtColor(np.hstack([frame.left, frame.right]), cv2.COLOR_GRAY2BGR)
            rem = max(0.0, duration - elapsed) if duration > 0 else elapsed
            cv2.putText(display, f"Saved: {saved} | {rem:.1f}s | pose: {pose}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display, f"Synced FPS: {receiver.fps:.1f}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if has_hand:
                m = frame.meta
                col = (0, 220, 255) if good else (0, 0, 255)
                cv2.putText(display, f"{m['hand_type']} hand  conf {m['confidence']:.2f}  "
                                     f"grab {m['grab_strength']:.2f}", (10, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
            else:
                cv2.putText(display, "Waiting for hand...", (10, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            cv2.imshow(window_name, display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[CAPTURE-IR] Interrupted by user.")
                break

        cv2.destroyAllWindows()

    meta = {
        "modality": "active_infrared_850nm_stereo",
        "device": receiver.device_info,
        "image_resolution": [384, 384],
        "stereo": True,
        "joint_space": "leapc_tracking_mm",
        "sync": "leapc_frame_id",
        "pose_label": pose,
        "hand_filter": hand_filter,
        "save_fps": fps_cap,
        "min_confidence": min_confidence,
        "total_frames": saved,
        "capture_date": datetime.now().isoformat(),
    }
    with open(session_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print()
    print("=" * 60)
    print(f"  Capture Complete: {saved} synchronized stereo IR frames saved")
    print(f"  Session: {session_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Ultraleap 3Di Native Stereo IR Capture")
    parser.add_argument("--output-dir", type=str, default=str(config.DATASET_IR_DIR),
                        help="Dataset root; a new session folder is created inside it")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Capture duration in seconds (0 for indefinite)")
    parser.add_argument("--hand", type=str, default="any", choices=["any", "left", "right"],
                        help="Filter hand type")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="Saved frames per second (the sensor runs at ~90; "
                             "consecutive frames are near-duplicates)")
    parser.add_argument("--pose", type=str, default="free",
                        help="Pose label stored with every frame, e.g. open_palm, fist, pinch")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="Skip frames whose Ultraleap hand confidence is below this")
    args = parser.parse_args()

    run_capture(
        output_dir=Path(args.output_dir),
        duration=args.duration,
        hand_filter=args.hand,
        fps_cap=args.fps,
        pose=args.pose,
        min_confidence=args.min_confidence,
    )


if __name__ == "__main__":
    main()
