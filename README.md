# 3DFist — Stereo IR 3D Hand Pose Estimation for Front-Facing (Palmar) Views

**3DFist** is a lightweight, real-time deep learning model designed to reconstruct **21 3D hand-joint coordinates $(X, Y, Z)$ in millimetres** from front-facing (palmar) views using synchronized Left and Right active 850nm infrared (IR) streams from an **Ultraleap 3Di stereo camera** (Hyperion 6.2.0, Windows 11, CPU-only Dell Latitude 3440).

---

## 🎯 Key Design & Engineering Highlights

| Requirement | 3DFist Implementation |
| :--- | :--- |
| **Input** | Raw (fisheye) Left & Right 8-bit IR views, each cropped around the hand and resized to $128 \times 128$, stacked to $(2, 128, 128)$ |
| **Output** | 21 3D hand joints $(X, Y, Z)$ in millimetres, LeapC tracking axes, root-relative (wrist at $[0, 0, 0]$) |
| **Model Size** | **1.38M parameters** (Stereo3DFistNet) + a small HandLocNet hand locator |
| **Supervision** | Reference labels from Ultraleap Hyperion tracking of the *same* sensor frame (matched by LeapC `frame_id`) |
| **Evaluation Metrics**| **MPJPE** (Mean Per-Joint Position Error) and **PA-MPJPE** (Procrustes-Aligned MPJPE) in mm |
| **Hardware Target** | Dell Latitude 3440 (Intel Core CPU, Windows 11, No discrete GPU required) |

### Solutions to Core Challenges

1. **Active IR Brightness Saturation at Close Range**:
   - At close proximity to the 3Di's 850nm pulsed LEDs, inverse-square illumination causes intense blooming and pixel clamping at 255, destroying palmar crease gradients.
   - **Solution**: `ActiveIRSaturationAugmentor` in [`ir_transforms.py`](file:///c:/Users/SANGAMESHWAR/OneDrive%20-%20International%20Institute%20of%20Information%20Technology/Documents/COMPUTER-PROJECT/3dfist/ir_transforms.py) dynamically simulates radial overexposure hotspots and sensor blooming during training. Dual-channel CLAHE preprocessing normalizes dynamic range across specular highlights.

2. **Finger Self-Occlusion in Curled Fist Poses**:
   - When fingers curl inward toward the palm, fingertips are partially or fully occluded from the front-facing camera view.
   - **Solution**: `PhysicsInformedFistLoss` in [`fist_loss.py`](file:///c:/Users/SANGAMESHWAR/OneDrive%20-%20International%20Institute%20of%20Information%20Technology/Documents/COMPUTER-PROJECT/3dfist/fist_loss.py) incorporates:
     - **$\mathcal{L}_{\text{bone}}$ (Bone Length Invariance)**: Penalizes deviations from anatomical phalangeal bone lengths, preventing occluded fingertips from collapsing.
     - **$\mathcal{L}_{\text{angle}}$ (Anatomical Flexion Bounds)**: Penalizes unphysical joint hyperextension during curled palmar fist formation.
     - **$\mathcal{L}_{z}$ (Anisotropic Depth Loss)**: Multiplies Z-depth error by $\lambda_z = 3.0$ to enforce stereo parallax triangulation.

---

## 🏗️ Architecture

```
Raw Left IR (384x384) ──> HandLocNet ──> box ──┐                  ┌─> virtual L crop 128x128 ──┐
                                               ├─> triangulate ──>│   (shared rotation, focal)  ├─> (2, 128, 128)
Raw Right IR (384x384) ─> HandLocNet ──> box ──┘   hand centre    └─> virtual R crop 128x128 ──┘
                                                                                                    │
                         Stereo3DFistNet: Conv2d(2 -> 32, s=2) stem                                 │
                         -> Inverted-residual + SE stages (48, 80, 144, 224 ch) -> 1x1 conv 384      │
                         -> global average pool ⊕ [hand distance, angular size] -> MLP -> 63        │
                                                                                                    │
                (21, 3) joints in mm, wrist at the origin, virtual camera frame -> rotated to LeapC axes
```

- **Perspective crops** (`perspective_crop.py`, after PCLs, Yu et al. CVPR 2021 and UmeTrack,
  Han et al. 2022): the two box centres are triangulated, and both raw fisheye views are
  re-rendered through one virtual pinhole camera aimed at the hand. This removes the fisheye
  distortion, gives a near-rectified stereo pair, and makes the target independent of where
  the hand sits in the image.
- **Rotation augmentation**: during training the virtual camera rolls by up to ±45°, and the
  3D labels are rotated with it, so the augmentation is exact.
- **Training boxes** come from projecting the Ultraleap joints into each view with the
  device's factory calibration (`ir_geometry.py`), jittered to mimic locator error.
- **Test-time boxes** come from HandLocNet (IR only). Without a HandLocNet checkpoint the live
  dashboard places the boxes from the Ultraleap joints and labels them *Leap-assisted*.

---

## ⚡ Workflow

### 1. Capture stereo IR sessions
Each run creates `dataset_ir/session_<timestamp>/` with `left/`, `right/`, `leap/`,
`calibration.json`, `rect_lut.npz` and `metadata.json`. Record several sessions covering
different poses (the `--pose` label is stored with every frame):
```bash
python capture_ir.py --pose open_palm --duration 60
python capture_ir.py --pose fist --duration 60
python capture_ir.py --pose pinch --duration 60
```
Train/val splits are made **by session** (frames inside a session are near-duplicates), so record
at least 5 sessions. Captures made before the stereo fix (identical left/right images, no
`rect_lut.npz`) are skipped automatically.

### 2. Train the hand locator and the 3D model
```bash
python train_handloc.py --dataset dataset_ir --epochs 30
python train_fist.py --dataset dataset_ir --epochs 40 --batch-size 8 --lr 1e-3
```

### 3. Evaluate
```bash
python eval_fist.py --dataset dataset_ir --weights checkpoints/fist_best.pth --handloc checkpoints/handloc_best.pth
```
Reports MPJPE, PA-MPJPE, per-axis/finger/joint errors and a **mean-pose baseline** (error of
always predicting the average training pose — a model that does not beat it has learned
nothing). Omit `--handloc` to evaluate with oracle crops. Outputs `results/eval_3dfist.json`
and `results/eval_plot_3dfist.png`.

### 4. Live dashboard
```bash
python live_fist_tracking.py
```
Keys: `1` Ultraleap reference, `2` 3DFist, `3` overlay with live root-relative MPJPE,
`a/d/w/s` rotate, `c` stereo view, `q` quit. Checkpoints from the old pipeline are refused
(override with `--allow-stale`).

---

## 📷 RGB Webcam Fallback Pipeline (Optional)

### Stage 2: Dataset Formatting
Handled automatically during Capture. Output structure:
```
dataset_raw/
├── rgb/
│   ├── frame_0001.jpg
├── leap/
│   ├── frame_0001.json
└── metadata.json
```

### Stage 3: Extrinsic Spatial Calibration
Align the LeapC coordinate space to the RGB camera.

**3a. Intrinsic Calibration:**
```bash
python calibrate_intrinsics.py --camera 0 --num-views 15
```

**3b. Extrinsic Calibration (Interactive):**
Follow the on-screen prompts to touch the checkerboard corners with your index finger while the 3Di tracks it.
```bash
python calibrate_extrinsics.py --camera 0
```

**3c. Transform Dataset:**
Apply the computed matrix to shift the Leap data into the Camera space.
```bash
python transform.py --dataset dataset_raw --output dataset_aligned
```

### Stage 4: Baseline Inference & Evaluation
Evaluate an existing model against the aligned ground truth.

**4a. Inference (MediaPipe default):**
```bash
python inference.py --dataset dataset_aligned --model mediapipe
```

**4b. Error Evaluation (MPJPE):**
Calculates the Mean Per Joint Position Error and outputs a Z-axis hallucination analysis.
```bash
python evaluate.py --dataset dataset_aligned --model mediapipe
```

### Stage 5: Fine-Tuning
Fine-tune a Vision Transformer to correct depth errors.
```bash
python finetune.py --dataset dataset_aligned --epochs 50 --batch-size 8
```

### Visualization
Check synchronization and calibration by projecting 3D joints onto 2D frames.
```bash
python visualize.py --dataset dataset_aligned
```

---

## 📐 Coordinate Systems

- **Raw LeapC**: Right-handed, millimetres, origin at the centre of the 3Di device. 3DFist predicts in these axes.
- **IR pixels**: with the 3Di facing the user, a joint projects as `ray = (-(x + cam_x) / y, z / y)` → LeapC `RectilinearToPixel` (stored per session in `rect_lut.npz`; `cam_x` = ∓32.9 mm).
- **Aligned (Camera Space)**: Transformed via `solvePnP`. Origin at the RGB camera lens.
- **Root-Aligned**: For evaluation and training, Joint 0 (Wrist) is subtracted to remove global translation (0,0,0 origin).
