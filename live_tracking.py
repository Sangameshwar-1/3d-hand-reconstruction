import argparse
import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import torch
import torchvision.transforms as T
import config
from finetune import FineTunedHandPoseModel

# --- Shared State for Async PyTorch Inference ---
class AsyncInferencer:
    def __init__(self, weights_path):
        # ----------------------------------------------------------------
        # Load in fp16 (half-precision) to halve the model's RAM footprint
        # from ~345 MB to ~172 MB.  PyTorch converts weights in-place with
        # no temporary copy, unlike ONNX Runtime's graph optimizer.
        # torch.inference_mode() further prevents gradient/activation storage.
        # ----------------------------------------------------------------
        print("[MODEL] Loading FineTunedHandPoseModel in fp16...")
        self.device = torch.device('cpu')
        self.model = FineTunedHandPoseModel(freeze_backbone=True)
        state = torch.load(weights_path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state)
        self.model.half()   # fp32 -> fp16  (halves RAM in-place, no temp copy)
        self.model.eval()
        print("[MODEL] Ready.")

        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE), antialias=True),
            T.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ])

        self.latest_frame  = None
        self.latest_joints = None
        self.lock    = threading.Lock()
        self.running = True

        self.thread = threading.Thread(target=self._run_inference, daemon=True)
        self.thread.start()

    def _run_inference(self):
        while self.running:
            frame = None
            with self.lock:
                if self.latest_frame is not None:
                    frame = self.latest_frame
                    self.latest_frame = None  # consume

            if frame is None:
                time.sleep(0.01)
                continue

            # Preprocess: (H,W,3) uint8 RGB -> (1,3,224,224) fp16 tensor
            tensor = self.transform(frame).unsqueeze(0).half().to(self.device)

            with torch.inference_mode():
                out = self.model(tensor)   # (1, 63)

            joints = out.squeeze(0).float().numpy().reshape(config.NUM_JOINTS, 3)

            with self.lock:
                self.latest_joints = joints


    def update_frame(self, frame_rgb):
        with self.lock:
            self.latest_frame = frame_rgb

    def get_joints(self):
        with self.lock:
            return self.latest_joints
            
    def stop(self):
        self.running = False
        self.thread.join()

def draw_3d_skeleton(joints_3d, size=500):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Root align for display stability
    pts = joints_3d - joints_3d[0]
    
    # Tilt 20 deg around X, 35 deg around Y
    tx = np.radians(20)
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(tx), -np.sin(tx)],
                    [0, np.sin(tx), np.cos(tx)]])
    ty = np.radians(35)
    R_y = np.array([[np.cos(ty), 0, np.sin(ty)],
                    [0, 1, 0],
                    [-np.sin(ty), 0, np.cos(ty)]])
    
    pts_rot = pts.dot(R_x.T).dot(R_y.T)
    
    # --- FIXED SCALE INSTEAD OF ADAPTIVE SCALE ---
    # A single outlier joint caused the adaptive bounding box to explode to 5000mm,
    # squashing the hand into a 10px dot. Our joints are already in absolute mm!
    # A typical hand is ~200mm. 300mm will comfortably fit a 500px canvas.
    scale = size / 300.0  
    
    # The wrist (root) is at (0,0,0). We place it slightly below the center.
    cx = size // 2
    cy = size // 2 + 100
    
    z_vals = pts_rot[:, 2]
    # We clip z_min and z_max to reasonable physical bounds (-150mm to +150mm) 
    # so a single outlier doesn't break the depth coloring either!
    z_min, z_max = -150.0, 150.0
    z_range = 300.0
    
    def depth_colour(z):
        # Clip z to bounds for color
        z = max(z_min, min(z_max, z))
        t = float((z - z_min) / z_range)  # 0 to 1
        return (int(t * 220) + 35, 40, int((1 - t) * 220) + 35)
        
    pts_2d = []
    for p in pts_rot:
        px = int(p[0] * scale + cx)
        py = int(-p[1] * scale + cy)
        pts_2d.append((px, py))
        
    for a, b in config.SKELETON_CONNECTIONS:
        z_avg = (z_vals[a] + z_vals[b]) / 2
        cv2.line(img, pts_2d[a], pts_2d[b], depth_colour(z_avg), 2)
        
    for i, (px, py) in enumerate(pts_2d):
        t = float((z_vals[i] - z_min) / z_range)
        radius = int(3 + 4 * t)  # larger radius for near points
        cv2.circle(img, (px, py), radius, (0, 255, 80), -1)
        
    cv2.putText(img, "Physics-Informed ONNX View", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, "blue=near  red=far", (10, size - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
    return img

def open_working_camera(camera_index: int = None):
    """
    Open the requested camera or probe available cameras, testing actual frame reads.
    Windows laptops often have Camera 0 as an IR/Windows Hello sensor that reports
    isOpened()==True but fails cap.read().
    """
    candidates = []
    if camera_index is not None:
        candidates.append(camera_index)
    if config.CAMERA_INDEX not in candidates:
        candidates.append(config.CAMERA_INDEX)
    for idx in [1, 0, 2]:
        if idx not in candidates:
            candidates.append(idx)

    for idx in candidates:
        for backend in [None, cv2.CAP_DSHOW]:
            cap = cv2.VideoCapture(idx) if backend is None else cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                for _ in range(5):
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        backend_str = "Default" if backend is None else "DSHOW"
                        print(f"[LIVE] Connected to Camera {idx} ({backend_str} backend). Native Res: {frame.shape[1]}x{frame.shape[0]}")
                        return cap, idx
                    time.sleep(0.05)
                cap.release()

    return None, None

def main():
    parser = argparse.ArgumentParser(description="Live 3D Hand Tracking")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX,
                        help="Camera device index (default: config.CAMERA_INDEX)")
    args = parser.parse_args()

    print("[LIVE] Initializing Production Edge Model...")

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Use the PyTorch checkpoint directly (fp16, ~172 MB) instead of the
    # ONNX model (~330 MB) which requires an extra ~300 MB optimizer temp copy.
    pth_path = str(config.CHECKPOINTS_DIR / "best_model.pth")
    inferencer = AsyncInferencer(pth_path)

    cap, active_cam = open_working_camera(args.camera)
    if cap is None:
        print("[ERROR] Cannot open any working webcam! Please ensure it is plugged in and not used by another app.")
        inferencer.stop()
        return

    # Lower resolution to reduce per-frame memory usage (~40 % less RAM than 640x480).
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print(f"[LIVE] Starting Live Tracking on Camera {active_cam}... Press 'q' to quit.")
    
    pTime = 0
    fail_count = 0
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            fail_count += 1
            if fail_count % 30 == 0:
                print(f"[WARN] Camera {active_cam} waiting for valid frame ({fail_count} empty reads)...")
            # On Windows, the first few frames are often empty while the
            # camera sensor warms up. Sleep briefly so we don't busy-spin,
            # and pump the OpenCV event queue so the window can appear.
            time.sleep(0.05)
            cv2.waitKey(1)
            continue
        fail_count = 0

        # Keep an unflipped RGB copy — the ONNX model was trained on straight
        # Leap-rig frames, so we must never pass a mirrored image to it.
        image_unflipped_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H_img, W_img = image_unflipped_rgb.shape[:2]

        # Flip image horizontally for the MediaPipe selfie-view display
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # --- Run MediaPipe FIRST so we can compute a tight hand crop ---
        image.flags.writeable = False
        results = hands.process(image_rgb)
        image.flags.writeable = True

        img_3d = np.zeros((500, 500, 3), dtype=np.uint8)

        # Only send to ONNX / draw 3D if MediaPipe confirms a hand is present
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            # ------------------------------------------------------------------
            # Compute a padded bounding box around the first detected hand.
            # MediaPipe landmark x-coords are normalised in the FLIPPED frame.
            # To crop from the UNFLIPPED image we mirror: x_unflip = 1 - x_flip
            # ------------------------------------------------------------------
            PAD = 40  # pixels of padding around the hand
            lm = results.multi_hand_landmarks[0].landmark

            xs_unflipped = [(1.0 - l.x) * W_img for l in lm]
            ys            = [l.y           * H_img for l in lm]

            x1 = max(0,     int(min(xs_unflipped)) - PAD)
            x2 = min(W_img, int(max(xs_unflipped)) + PAD)
            y1 = max(0,     int(min(ys))            - PAD)
            y2 = min(H_img, int(max(ys))            + PAD)

            hand_crop = image_unflipped_rgb[y1:y2, x1:x2].copy()

            # Free the full-res buffer immediately — the crop is all we need.
            del image_unflipped_rgb

            if hand_crop.size > 0:
                # Send only the tight hand crop — the model was trained on
                # hand-filled images, not full-frame webcam views.
                inferencer.update_frame(hand_crop)

            # Draw a yellow box on the selfie display to show the ONNX crop
            # (mirror x-coords back to flipped display space)
            x1_disp, x2_disp = W_img - x2, W_img - x1
            cv2.rectangle(image, (x1_disp, y1), (x2_disp, y2), (0, 220, 255), 2)
            cv2.putText(image, "ONNX crop", (x1_disp + 4, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)

            # Fetch latest async ONNX result and render the 3-D skeleton
            joints_3d = inferencer.get_joints()
            if joints_3d is not None:
                # Mirror the X axis so joints match the selfie-view
                joints_3d_selfie = joints_3d.copy()
                joints_3d_selfie[:, 0] = -joints_3d_selfie[:, 0]
                img_3d = draw_3d_skeleton(joints_3d_selfie)
            else:
                cv2.putText(img_3d, "ONNX Warming Up...", (100, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
        else:
            cv2.putText(img_3d, "No hand detected", (100, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)

        cTime = time.time()
        fps = 1 / (cTime - pTime + 1e-6)
        pTime = cTime
        
        cv2.putText(image, f'FPS: {int(fps)}', (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        # Combine into a unified side-by-side dashboard
        img_2d_resized = cv2.resize(image, (500, 500), interpolation=cv2.INTER_LINEAR)
        divider = np.full((500, 4, 3), 50, dtype=np.uint8)
        dashboard = np.hstack([img_2d_resized, divider, img_3d])

        cv2.imshow("3D Hand Tracking Dashboard (2D Webcam + 3D Reconstruction)", dashboard)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    inferencer.stop()
    hands.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
