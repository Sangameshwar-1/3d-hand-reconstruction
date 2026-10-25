# Project Status

_Last updated: 2026-10-21_

## Goal

Use **two Ultraleap 3Di units** facing the scene from different sides. Together they see
what one alone cannot, which gives ground truth for the hidden parts.
Train a model that takes **one 3Di's view** and outputs the **complete 3D scene**,
including the hidden parts: the hand's own occluded fingers, and any held object
(e.g. a whole cup when only part of it is visible).

## Team

- Sangameshwar: built everything up to 3DFist. Has `dataset_ir/` and the checkpoints.
- Amitabh: joined 2026-10-21. Working on the two-sensor phase.

## Where we are: single-sensor stage

Everything in the repo uses **one Ultraleap 3Di** (its two built-in IR imagers are a
stereo pair inside one device, not two cameras with different viewpoints).
There is no code to open, calibrate, sync or fuse a second 3D sensor.

| Area | State | Files |
| :--- | :--- | :--- |
| 3Di IR capture (L/R IR + Leap joints, per session) | Done | `capture_ir.py`, `leap_ir.py`, `live_leap_ir.py` |
| 3DFist model (stereo IR → 21 joints, mm) | Trained, evaluated | `fist_model.py`, `train_fist.py`, `fist_loss.py`, `eval_fist.py` |
| Hand locator (HandLocNet) | Trained | `hand_locator.py`, `train_handloc.py` |
| Live demo | Done | `live_fist_tracking.py` |
| Older RGB webcam + 3Di pipeline (MediaPipe, ViT finetune) | Done, superseded | `capture.py`, `calibrate_*.py`, `transform.py`, `finetune.py`, `inference.py`, `evaluate.py` |
| **Second 3D sensor: capture / sync / extrinsic calibration** | **Not started** | — |
| **Fused two-view ground truth (occluded joints)** | **Not started** | — |
| **Single-view → occluded-part model** | **Not started** | — |

### Results so far (`results/`)

| Model | MPJPE (mm) | PA-MPJPE (mm) | Notes |
| :--- | ---: | ---: | :--- |
| Mean-pose baseline | 71.7 | — | |
| 3DFist (1.38M params) | 23.6 | 11.4 | 1600 samples, HandLocNet crops |
| ResNet18 variants | 19.2 – 20.0 | 9.5 – 10.9 | 11.7M params |
| RGB ViT finetune | 14.4 | — | 74 frames only, same-session — not comparable |
| MediaPipe (RGB) | 179.4 | — | |

### Key limitation

Labels come from **Ultraleap's own tracking of the same device**. Joints occluded from
the 3Di are guessed by Ultraleap's hand model, not observed — so current labels cannot
teach "what is behind". This is exactly what the second camera has to fix.

### Repo hygiene notes

- `dataset_ir/` (the data behind the 3DFist results) is gitignored and **not on this machine**. Checkpoints are also absent.
- `leapc-python-bindings` and `third-party/hamer` are empty (submodules not initialised).
- `__pycache__/*.pyc` are tracked despite `.gitignore`.
- README links point to another machine's path (`c:/Users/SANGAMESHWAR/...`).

## Do we need Sangameshwar's `dataset_ir/` and checkpoints?

**No, they aren't blocking.** The new goal needs new two-sensor recordings anyway. The old
labels came from a single device, so they never saw the hidden parts. The **code** is
what we reuse (`leap_ir.py`, `capture_ir.py`, `ir_geometry.py`, `perspective_crop.py`).
Nice to have later: the old 3DFist numbers as a baseline, and the weights as a starting point.

## Technical facts that shape the plan

- **A 3Di does not output depth or a point cloud.** It gives two IR images plus
  Ultraleap's hand-joint tracking. Cup geometry has to come from our own stereo depth
  (disparity from the L/R IR pair, using `rect_lut.npz` / factory calibration).
- **Ultraleap tracks hands only**, not objects. Any cup ground truth has to come from fused depth.
- **Two active-IR units facing each other may interfere.** Each one's 850 nm LEDs light
  up the other's images. Test this first, before building anything else.

## Next steps

1. **Hardware smoke test:** both 3Di units on one PC through LeapC multi-device; record both
   IR streams at once; check how bad the interference is at the planned angles.
2. Two-device capture: extend `capture_ir.py` to save per-device streams plus timestamps.
3. Extrinsic calibration between units (shared target visible in IR, or the same hand
   tracked by both and aligned with Procrustes).
4. Per-view stereo depth → fused point cloud in one frame = ground truth for "full scene".
5. Model: one view's input → completed 3D (representation TBD) + hand joints.

## Open questions

- Output representation: point cloud, voxels, or mesh? (Point cloud is the simplest fit for fused stereo depth.)
- Planned mounting: opposite each other (180°) or about 90° apart? This decides interference and coverage.
- Do we have both 3Di units and a PC with enough USB 3 bandwidth for two streams?

## Log

- 2026-10-21 — Status file created from repo review. Confirmed: single-sensor stage.
- 2026-10-21 — Goal clarified: two 3Di units, single-view → full 3D (hand + held object). Old data not required.
