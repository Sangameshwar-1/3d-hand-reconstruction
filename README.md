# Ultraleap 3Di Monocular RGB Evaluation Framework

A complete Python pipeline that synchronizes an **Ultraleap 3Di stereo-IR tracker** with a **standard RGB webcam** to acquire markerless 3D ground truth. This ground truth is used to benchmark and fine-tune monocular 3D hand pose AI models (like MediaPipe or HaMeR) to evaluate and correct real-world Z-axis depth hallucination.

---

## 🛠 Prerequisites

### Hardware
- Ultraleap 3Di Tracker
- RGB Webcam (e.g. Dell Latitude 3440 integrated camera)
- A printed checkerboard pattern for calibration

### Software
1. **Gemini Ultraleap Hand Tracking Service**: Must be installed and running.
2. **Python Environment**: Python 3.10+ recommended.

```bash
pip install -r requirements.txt
```

> [!IMPORTANT]
> The `leapc-python-bindings` package is not available on PyPI and must be built from source.
> ```bash
> git clone https://github.com/ultraleap/leapc-python-bindings
> cd leapc-python-bindings
> pip install -r requirements.txt
> python -m build leapc-cffi
> pip install leapc-cffi/dist/leapc_cffi-0.0.1.tar.gz
> pip install -e leapc-python-api
> ```

---

## 🚀 Pipeline Workflow

### Stage 1: Multithreaded Synchronized Capture
Capture perfectly timed (`Δt ≤ 15ms`) pairs of RGB frames and 3Di joint events.
```bash
python capture.py --camera 0 --duration 60 --output-dir dataset_raw
```
*(Use `--simulate-leap` if testing without hardware).*

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

- **Raw LeapC**: Right-handed, millimetres, origin at the centre of the 3Di device.
- **Aligned (Camera Space)**: Transformed via `solvePnP`. Origin at the RGB camera lens.
- **Root-Aligned**: For evaluation and training, Joint 0 (Wrist) is subtracted to remove global translation (0,0,0 origin).
