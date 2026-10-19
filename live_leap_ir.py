"""
live_leap_ir.py — Live Ultraleap 3Di Infrared (IR) Visualizer.

Captures and displays the raw stereo infrared camera streams directly from
the Ultraleap 3Di hardware at high speed (75-90 FPS).

Features:
  - Real-time Left and Right 850nm active IR camera feeds.
  - Optional side-by-side stereo or single camera view.
  - Real-time 3D hand joint overlay from the Hyperion tracking engine.
  - Interactive snapshot saving ('s' key) for dataset collection.
  - Quit with 'q'.

Usage:
  python live_leap_ir.py
  python live_leap_ir.py --stereo
"""

import argparse
import sys
import time
import threading
from pathlib import Path

import cv2
import numpy as np

import leap
from leap.enums import PolicyFlag

import config
from leap_ir import read_stereo_images


class LeapIRCapture:
    """Thread-safe receiver for Ultraleap IR images and hand tracking events."""

    def __init__(self):
        self.lock = threading.Lock()
        self.latest_left_ir = None
        self.latest_right_ir = None
        self.latest_hands = []
        self.latest_timestamp = 0
        self.fps_counter = 0
        self.fps = 0.0
        self._last_fps_time = time.time()
        self.is_connected = False
        self.device_info = "Searching..."

    def on_connection(self):
        self.is_connected = True

    def on_device(self, info_str):
        self.device_info = info_str

    def update_images(self, left_img, right_img, timestamp):
        with self.lock:
            self.latest_left_ir = left_img
            self.latest_right_ir = right_img
            self.latest_timestamp = timestamp
            self.fps_counter += 1
            now = time.time()
            elapsed = now - self._last_fps_time
            if elapsed >= 1.0:
                self.fps = self.fps_counter / elapsed
                self.fps_counter = 0
                self._last_fps_time = now

    def update_hands(self, hands):
        with self.lock:
            self.latest_hands = hands

    def get_state(self):
        with self.lock:
            left = None if self.latest_left_ir is None else self.latest_left_ir.copy()
            right = None if self.latest_right_ir is None else self.latest_right_ir.copy()
            hands = list(self.latest_hands)
            return left, right, hands, self.fps, self.device_info


class LeapIRListener(leap.Listener):
    def __init__(self, capture_state: LeapIRCapture):
        super().__init__()
        self.state = capture_state

    def on_connection_event(self, event):
        print("[LEAP-IR] Connected to Ultraleap tracking service.")
        self.state.on_connection()

    def on_device_event(self, event):
        try:
            with event.device.open():
                info = event.device.get_info()
                desc = f"Ultraleap 3Di ({info.serial})"
                print(f"[LEAP-IR] Hardware detected: {desc}")
                self.state.on_device(desc)
        except Exception:
            pass

    def on_image_event(self, event):
        try:
            # Camera 0: Left IR, Camera 1: Right IR (shared buffer, offset-aware)
            img_left, img_right = read_stereo_images(event)

            ts = event._data.info.timestamp if hasattr(event, "_data") else int(time.time() * 1e6)
            self.state.update_images(img_left, img_right, ts)
        except Exception as e:
            print(f"[ERROR] on_image_event failed: {e}")

    def on_tracking_event(self, event):
        hands_summary = []
        for hand in event.hands:
            hand_type = "Left" if hand.type == leap.HandType.Left else "Right"
            palm_pos = (hand.palm.position.x, hand.palm.position.y, hand.palm.position.z)
            hands_summary.append({
                "type": hand_type,
                "palm": palm_pos,
                "pinch": hand.pinch_strength,
                "grab": hand.grab_strength,
            })
        self.state.update_hands(hands_summary)


def run_visualizer(stereo: bool = False, scale: int = 2):
    """
    Main loop to render live Ultraleap 3Di IR visuals.
    """
    state = LeapIRCapture()
    if sys.platform != "win32":
        print("[WARNING] You are running under Linux/WSL. The Ultraleap service runs on Windows.")
        print("          Please run this script from Windows PowerShell or Command Prompt:")
        print("          python live_leap_ir.py")

    listener = LeapIRListener(state)

    conn = leap.Connection()
    conn.add_listener(listener)

    print("[LEAP-IR] Opening connection to Ultraleap service...")
    with conn.open():
        # Request raw image stream from 3Di
        print("[LEAP-IR] Enabling PolicyFlag.Images on device...")
        conn.set_policy_flags(flags_to_set=[PolicyFlag.Images])

        window_name = "Ultraleap 3Di IR Visualizer"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        try:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass
        cv2.moveWindow(window_name, 100, 100)

        print(f"[UI] Window '{window_name}' initialized. Bringing to front.")
        print("[LEAP-IR] Streaming started. Press 'q' to quit, 's' to snapshot, 'c' to toggle stereo view.")

        snapshots_dir = config.PROJECT_ROOT / "ir_snapshots"
        snapshots_dir.mkdir(exist_ok=True)
        snapshot_count = 0

        first_frame_shown = False
        while True:
            left_img, right_img, hands, fps, dev_info = state.get_state()

            if left_img is None or right_img is None:
                # Waiting for first frames
                placeholder = np.zeros((384 * scale, 384 * scale, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Connecting to Ultraleap 3Di IR...", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                cv2.imshow(window_name, placeholder)
                cv2.waitKey(20)
                continue

            if not first_frame_shown:
                print(f"[UI] First IR frame received! Displaying {left_img.shape[1]}x{left_img.shape[0]} feed.")
                first_frame_shown = True

            # Convert 8-bit grayscale to BGR for color overlays
            left_bgr = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)
            right_bgr = cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR)

            # Apply scale (e.g. 384x384 -> 768x768)
            if scale != 1:
                left_bgr = cv2.resize(left_bgr, (left_img.shape[1] * scale, left_img.shape[0] * scale),
                                      interpolation=cv2.INTER_LINEAR)
                right_bgr = cv2.resize(right_bgr, (right_img.shape[1] * scale, right_img.shape[0] * scale),
                                       interpolation=cv2.INTER_LINEAR)

            if stereo:
                display = np.hstack([left_bgr, right_bgr])
                h, w = display.shape[:2]
                cv2.putText(display, "Left IR (850nm)", (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display, "Right IR (850nm)", (left_bgr.shape[1] + 15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                display = left_bgr
                cv2.putText(display, "Ultraleap 3Di — Active IR (850nm)", (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            # Render HUD Info
            info_text = f"FPS: {fps:.1f} | Device: {dev_info}"
            cv2.putText(display, info_text, (15, display.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            # Render Tracking Hands HUD
            if hands:
                for idx, h_info in enumerate(hands):
                    h_text = f"{h_info['type']} Hand: Palm Z={h_info['palm'][2]:.0f}mm | Pinch={h_info['pinch']:.2f}"
                    cv2.putText(display, h_text, (15, 60 + idx * 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
            else:
                cv2.putText(display, "Hand: Not in view", (15, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1)

            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                stereo = not stereo
            elif key == ord('s'):
                snapshot_count += 1
                t_str = int(time.time())
                fn_left = snapshots_dir / f"ir_left_{t_str}_{snapshot_count:03d}.png"
                fn_right = snapshots_dir / f"ir_right_{t_str}_{snapshot_count:03d}.png"
                cv2.imwrite(str(fn_left), left_img)
                cv2.imwrite(str(fn_right), right_img)
                print(f"[SNAPSHOT] Saved -> {fn_left.name} and {fn_right.name}")

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Ultraleap 3Di Infrared (IR) Live Stream")
    parser.add_argument("--stereo", action="store_true",
                        help="Show both Left and Right stereo IR streams side by side")
    parser.add_argument("--scale", type=int, default=2,
                        help="Display scale factor (default: 2, scales 384x384 -> 768x768)")
    args = parser.parse_args()

    run_visualizer(stereo=args.stereo, scale=args.scale)


if __name__ == "__main__":
    main()
