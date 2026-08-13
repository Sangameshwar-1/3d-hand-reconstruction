"""
capture.py — Stage 1: Multithreaded Temporal Synchronization.

Producer-Consumer architecture with three threads:
  Thread 1 (RGB Producer)  — reads webcam frames via OpenCV.
  Thread 2 (3Di Producer)  — receives 90 FPS tracking events from LeapC.
  Thread 3 (Consumer)      — pairs frames whose timestamps are ≤ 15 ms apart
                              and writes them to disk.

Usage:
    python capture.py --camera 0 --duration 60 --output-dir dataset
    python capture.py --hand right --threshold 0.010
"""

import argparse
import sys
import time
import threading
from pathlib import Path

import cv2
import numpy as np

try:
    import leap
except ImportError:
    leap = None
    print("[WARNING] leapc-python-bindings not installed. "
          "Leap tracking will be simulated with dummy data.")

import config
from dataset_utils import (
    create_dataset_dirs,
    save_rgb_frame,
    save_leap_json,
    save_session_metadata,
)


# ==============================================================================
# Joint Extraction from LeapC Hand Object
# ==============================================================================

def extract_21_joints(hand) -> dict:
    """
    Extract the 21-joint dictionary from a LeapC hand object.

    Traversal order:
        Joint  0: Wrist         — hand.arm.next_joint
        Joints 1-4: Thumb       — digit 0, bones 0-3 (prev → next → next → next)
        Joints 5-8: Index       — digit 1
        Joints 9-12: Middle     — digit 2
        Joints 13-16: Ring      — digit 3
        Joints 17-20: Pinky     — digit 4

    Returns:
        Dict of "idx_JointName" → {"x": float, "y": float, "z": float}
        Coordinates are in millimetres (native LeapC units).
    """
    joints = {}

    # Joint 0: Wrist
    wrist = hand.arm.next_joint
    joints[f"0_{config.JOINT_NAMES[0]}"] = {
        "x": float(wrist.x),
        "y": float(wrist.y),
        "z": float(wrist.z),
    }

    # Joints 1–20: Five digits × 4 joints each
    joint_idx = 1
    for digit_idx in range(5):
        digit = hand.digits[digit_idx]
        for bone_idx in range(4):
            bone = digit.bones[bone_idx]
            if bone_idx == 0:
                # First bone: use prev_joint (base of metacarpal)
                pos = bone.prev_joint
            else:
                # Subsequent bones: use prev_joint (== prev bone's next_joint)
                pos = bone.prev_joint

            # Actually, the correct 21-keypoint extraction is:
            #   bone 0 prev_joint  → CMC / MCP base
            #   bone 0 next_joint  → MCP
            #   bone 1 next_joint  → PIP
            #   bone 2 next_joint  → DIP
            #   bone 3 next_joint  → Tip
            # But since bone_i.next_joint == bone_(i+1).prev_joint,
            # we can use next_joint for all 4 bones to get joints 1-4 per digit.
            pos = bone.next_joint

            name = config.JOINT_NAMES[joint_idx]
            joints[f"{joint_idx}_{name}"] = {
                "x": float(pos.x),
                "y": float(pos.y),
                "z": float(pos.z),
            }
            joint_idx += 1

    return joints


def get_hand_type_str(hand) -> str:
    """Return 'Left' or 'Right' from a LeapC hand object."""
    return "Left" if str(hand.type) == "HandType.Left" else "Right"


# ==============================================================================
# Dummy / Simulated Leap Data (for testing without hardware)
# ==============================================================================

def generate_dummy_joints() -> dict:
    """Generate fake 21-joint data for testing without a 3Di device."""
    joints = {}
    for idx, name in config.JOINT_NAMES.items():
        joints[f"{idx}_{name}"] = {
            "x": round(float(np.random.randn() * 30 + 50), 2),
            "y": round(float(np.random.randn() * 30 + 200), 2),
            "z": round(float(np.random.randn() * 30 - 100), 2),
        }
    return joints


# ==============================================================================
# Shared State
# ==============================================================================

class SharedState:
    """Thread-safe container for the latest RGB frame and Leap joints."""

    def __init__(self):
        self.lock = threading.Lock()

        # RGB
        self.rgb_frame = None
        self.rgb_timestamp = 0.0

        # Leap
        self.leap_joints = None
        self.leap_timestamp = 0.0
        self.leap_hand_type = "Unknown"

        # Control
        self.running = True
        self.frame_count = 0


# ==============================================================================
# Thread 1: RGB Producer
# ==============================================================================

def rgb_producer(state: SharedState, camera_index: int,
                 width: int, height: int):
    """Continuously read frames from the RGB webcam."""
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {camera_index}")
        state.running = False
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[RGB] Camera opened: {actual_w}×{actual_h}")

    consecutive_failures = 0
    while state.running:
        ret, frame = cap.read()
        if not ret:
            consecutive_failures += 1
            if consecutive_failures % 30 == 0:
                print(f"[RGB WARNING] Failed to read frame from camera {camera_index}! (Is it blocked or used by another app?)")
            time.sleep(0.01)
            continue

        consecutive_failures = 0
        ts = time.time()
        with state.lock:
            state.rgb_frame = frame
            state.rgb_timestamp = ts

    cap.release()
    print("[RGB] Producer stopped.")


# ==============================================================================
# Thread 2: LeapC Producer (Real or Simulated)
# ==============================================================================

class LeapTrackingListener(leap.Listener if leap else object):
    """Listener that receives tracking events from the Ultraleap 3Di."""

    def __init__(self, state: SharedState, hand_filter: str = "any"):
        if leap:
            super().__init__()
        self.state = state
        self.hand_filter = hand_filter  # "any", "left", or "right"

    def on_tracking_event(self, event):
        if not event.hands:
            return

        # Select hand based on filter
        hand = None
        for h in event.hands:
            ht = get_hand_type_str(h)
            if self.hand_filter == "any":
                hand = h
                break
            elif self.hand_filter == "left" and ht == "Left":
                hand = h
                break
            elif self.hand_filter == "right" and ht == "Right":
                hand = h
                break

        if hand is None:
            return

        ts = time.time()
        joints = extract_21_joints(hand)
        hand_type = get_hand_type_str(hand)

        with self.state.lock:
            self.state.leap_joints = joints
            self.state.leap_timestamp = ts
            self.state.leap_hand_type = hand_type


def leap_producer_real(state: SharedState, hand_filter: str):
    """Connect to LeapC and start receiving tracking events."""
    connection = leap.Connection()
    listener = LeapTrackingListener(state, hand_filter)
    connection.add_listener(listener)

    print("[LEAP] Connecting to Ultraleap 3Di…")
    with connection.open():
        print("[LEAP] Connected. Receiving tracking data.")
        while state.running:
            time.sleep(0.005)  # Yield — events arrive via callback
    print("[LEAP] Producer stopped.")


def leap_producer_simulated(state: SharedState, hand_filter: str):
    """Simulated Leap producer for testing without hardware."""
    print("[LEAP] Running in SIMULATED mode (no 3Di connected).")
    while state.running:
        ts = time.time()
        joints = generate_dummy_joints()

        with state.lock:
            state.leap_joints = joints
            state.leap_timestamp = ts
            state.leap_hand_type = "Right"

        time.sleep(1.0 / 90.0)  # Simulate 90 FPS
    print("[LEAP] Simulated producer stopped.")


# ==============================================================================
# Thread 3: Consumer (Synchronizer)
# ==============================================================================

def consumer(state: SharedState, rgb_dir: Path, leap_dir: Path,
             threshold: float, show_preview: bool):
    """
    Pair RGB frames with Leap joints when their timestamps are within threshold.
    Save matched pairs to disk.
    """
    last_save_time = 0.0

    print(f"[SYNC] Consumer running. Threshold: {threshold * 1000:.1f} ms")

    while state.running:
        time.sleep(0.005)  # Poll at ~200 Hz

        with state.lock:
            rgb_frame = state.rgb_frame
            rgb_ts = state.rgb_timestamp
            leap_joints = state.leap_joints
            leap_ts = state.leap_timestamp
            hand_type = state.leap_hand_type

        # Optional live preview (always show RGB if available)
        if show_preview and rgb_frame is not None:
            display = rgb_frame.copy()
            cv2.putText(display, f"Saved: {state.frame_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if leap_joints is None:
                cv2.putText(display, "Waiting for Hand...", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                current_delta = abs(rgb_ts - leap_ts)
                cv2.putText(display, f"dt: {current_delta * 1000:.1f}ms", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
            cv2.imshow("3Di Capture", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                state.running = False
                continue

        # Check both sources have data
        if rgb_frame is None or leap_joints is None:
            continue

        # Check timestamps within threshold
        delta = abs(rgb_ts - leap_ts)
        if delta > threshold:
            continue

        # Debounce — don't save faster than MIN_SAVE_INTERVAL
        now = time.time()
        if now - last_save_time < config.MIN_SAVE_INTERVAL_SEC:
            continue

        # Save the synchronized pair
        state.frame_count += 1
        fid = state.frame_count

        save_rgb_frame(rgb_dir, fid, rgb_frame)
        save_leap_json(leap_dir, fid, leap_joints, hand_type, rgb_ts, leap_ts)

        last_save_time = now

        # Status update
        if fid % 10 == 0 or fid <= 5:
            print(f"  [SYNC] Saved pair #{fid:04d}  |  Δt = {delta * 1000:.1f} ms  |  "
                  f"Hand: {hand_type}")

    if show_preview:
        cv2.destroyAllWindows()
    print(f"[SYNC] Consumer stopped. Total pairs saved: {state.frame_count}")


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Synchronized RGB + LeapC capture."
    )
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX,
                        help="OpenCV camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=config.CAMERA_WIDTH,
                        help="Requested camera width")
    parser.add_argument("--height", type=int, default=config.CAMERA_HEIGHT,
                        help="Requested camera height")
    parser.add_argument("--duration", type=float, default=config.CAPTURE_DURATION_SEC,
                        help="Capture duration in seconds (default: 60)")
    parser.add_argument("--output-dir", type=str, default=str(config.DATASET_DIR),
                        help="Output dataset directory")
    parser.add_argument("--threshold", type=float, default=config.SYNC_THRESHOLD_SEC,
                        help="Max sync delta in seconds (default: 0.015)")
    parser.add_argument("--hand", type=str, default="any",
                        choices=["any", "left", "right"],
                        help="Which hand to track")
    parser.add_argument("--no-preview", action="store_true",
                        help="Disable live preview window")
    parser.add_argument("--simulate-leap", action="store_true",
                        help="Use simulated Leap data (no hardware needed)")
    args = parser.parse_args()

    # Create output directories
    dataset_root, rgb_dir, leap_dir = create_dataset_dirs(Path(args.output_dir))

    # Shared state
    state = SharedState()

    # Decide Leap mode
    use_simulated = args.simulate_leap or (leap is None)

    # Build threads
    t_rgb = threading.Thread(
        target=rgb_producer,
        args=(state, args.camera, args.width, args.height),
        daemon=True,
        name="RGB-Producer",
    )

    if use_simulated:
        t_leap = threading.Thread(
            target=leap_producer_simulated,
            args=(state, args.hand),
            daemon=True,
            name="Leap-Simulated",
        )
    else:
        t_leap = threading.Thread(
            target=leap_producer_real,
            args=(state, args.hand),
            daemon=True,
            name="Leap-Producer",
        )

    t_consumer = threading.Thread(
        target=consumer,
        args=(state, rgb_dir, leap_dir, args.threshold, not args.no_preview),
        daemon=True,
        name="Consumer",
    )

    # Start
    print("=" * 60)
    print("  Ultraleap 3Di + RGB Synchronized Capture")
    print("=" * 60)
    print(f"  Camera:     index {args.camera} ({args.width}×{args.height})")
    print(f"  Leap mode:  {'SIMULATED' if use_simulated else 'REAL (3Di)'}")
    print(f"  Hand:       {args.hand}")
    print(f"  Threshold:  {args.threshold * 1000:.1f} ms")
    print(f"  Duration:   {args.duration} s")
    print(f"  Output:     {dataset_root}")
    print("=" * 60)
    print("Press Ctrl+C or 'q' in preview to stop early.\n")

    t_rgb.start()
    t_leap.start()
    time.sleep(0.5)  # Let producers warm up
    t_consumer.start()

    # Run for the specified duration
    start_time = time.time()
    try:
        while state.running and (time.time() - start_time) < args.duration:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted by user.")

    # Shutdown
    state.running = False
    time.sleep(0.5)  # Let threads wind down

    elapsed = time.time() - start_time

    # Save session metadata
    cap_temp = cv2.VideoCapture(args.camera)
    res = (int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH)),
           int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap_temp.release()

    save_session_metadata(
        dataset_root,
        total_frames=state.frame_count,
        camera_index=args.camera,
        camera_resolution=res,
        duration_sec=elapsed,
        sync_threshold=args.threshold,
        extra={"hand_filter": args.hand,
               "simulated_leap": use_simulated},
    )

    print(f"\n{'=' * 60}")
    print(f"  Capture complete!")
    print(f"  Pairs saved: {state.frame_count}")
    print(f"  Duration:    {elapsed:.1f} s")
    print(f"  Avg rate:    {state.frame_count / max(elapsed, 0.01):.1f} pairs/s")
    print(f"  Dataset:     {dataset_root}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
