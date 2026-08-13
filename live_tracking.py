import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import onnxruntime as ort
import config

# --- Shared State for Async Inference ---
class AsyncInferencer:
    def __init__(self, model_path):
        self.ort_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.latest_frame = None
        self.latest_joints = None
        self.lock = threading.Lock()
        self.running = True
        
        self.thread = threading.Thread(target=self._run_inference, daemon=True)
        self.thread.start()

    def _preprocess(self, image_rgb):
        img = cv2.resize(image_rgb, (config.IMAGE_SIZE, config.IMAGE_SIZE))
        img = img.astype(np.float32) / 255.0
        img = (img - np.array(config.IMAGENET_MEAN, dtype=np.float32)) / np.array(config.IMAGENET_STD, dtype=np.float32)
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0).astype(np.float32)

    def _run_inference(self):
        while self.running:
            frame = None
            with self.lock:
                if self.latest_frame is not None:
                    frame = self.latest_frame.copy()
                    self.latest_frame = None # Consume it
            
            if frame is None:
                time.sleep(0.01)
                continue
                
            input_tensor = self._preprocess(frame)
            outputs = self.ort_session.run(None, {'input_image': input_tensor})
            joints = outputs[0].reshape(config.NUM_JOINTS, 3)
            
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

def main():
    print("[LIVE] Initializing Production Edge Model...")
    
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    onnx_path = str(config.PROJECT_ROOT / "hamer_champion.onnx")
    inferencer = AsyncInferencer(onnx_path)

    cap = cv2.VideoCapture(0)
    print("[LIVE] Starting Live Tracking... Press 'q' to quit.")
    
    pTime = 0
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
            
        # The ViT model was trained on UNFLIPPED frames (straight from the Leap rig).
        # We MUST pass the raw unflipped image to the ONNX engine, otherwise the 
        # mirrored background and hand will cause the model to completely hallucinate.
        image_unflipped_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inferencer.update_frame(image_unflipped_rgb)

        # Flip image horizontally for the MediaPipe selfie-view display
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image.flags.writeable = False
        results = hands.process(image_rgb)
        image.flags.writeable = True
        
        img_3d = np.zeros((500, 500, 3), dtype=np.uint8)
        
        # Only draw ONNX 3D if MediaPipe confirms a hand is in view
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
            
            # Fetch latest async ONNX result (from unflipped perspective)
            joints_3d = inferencer.get_joints()
            if joints_3d is not None:
                # Mirror the X axis of the joints so they match the selfie-view!
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
        
        cv2.putText(image, f'FPS: {int(fps)}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Webcam: 2D Overlay (MediaPipe)", image)
        cv2.imshow("3D Engine: Spatial Geometry (Our ONNX Model)", img_3d)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    inferencer.stop()
    hands.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
