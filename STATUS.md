# Project Status

_Last updated: 2026-10-25_

## Goal

Use **two Ultraleap 3Di units** facing the scene from different sides. Together they see
what one alone cannot, which gives ground truth for the hidden parts.
Train a model that takes **one 3Di's view** and outputs the **complete 3D scene**,
including the hidden parts: the hand's own occluded fingers, and any held object
(e.g. a whole cup when only part of it is visible).

## Team

- Sangameshwar: built everything up to 3DFist v8. Has `dataset_ir/` and the checkpoints.
- Amitabh: joined 2026-10-21. Working on the two-sensor phase.

## Where we are: single-sensor stage (3DFist v8)

Everything in the repo uses **one Ultraleap 3Di** (its two built-in IR imagers are a
stereo pair inside one device, not two cameras with different viewpoints).
There is no code to open, calibrate, sync or fuse a second 3D sensor.

| Area | State | Files |
| :--- | :--- | :--- |
| 3Di IR capture (L/R IR + Leap joints, per session) | Done | `capture_ir.py`, `leap_ir.py`, `live_leap_ir.py` |
| Fisheye → virtual pinhole stereo crops | Done | `ir_geometry.py`, `perspective_crop.py`, `ir_transforms.py` |
| Hand locator (HandLocNet) | Trained (v6) | `hand_locator.py`, `train_handloc.py` |
| 3DFist pose model (stereo IR → 21 joints + left/right) | Trained, evaluated (v8) | `fist_model.py`, `fist_loss.py`, `fist_dataset.py`, `train_fist.py` |
| Evaluation | Done | `eval_fist.py`, `fist_metrics.py`, `lr_report.py` |
| Live demo | Done | `live_fist_tracking.py` |
| **Kaggle / local training notebook** | **New, tested on synthetic data** | `train_kaggle.ipynb` |
| Older RGB webcam pipeline (MediaPipe, ViT, HaMeR) | Removed from `main` in PR #1 | — |
| **Second 3D sensor: capture / sync / extrinsic calibration** | **Not started** | — |
| **Fused two-view ground truth (occluded joints)** | **Not started** | — |
| **Single-view → occluded-part model** | **Not started** | — |

### Results so far (`results/`)

Full pipeline (HandLocNet crops, `eval_fist.py`):

| Model | Test frames | MPJPE (mm) | PA-MPJPE (mm) | Mean-pose baseline (mm) |
| :--- | ---: | ---: | ---: | ---: |
| 3DFist from scratch (1.4M params) | 1,600 | 23.6 | 11.4 | 71.7 |
| ResNet-18 v1 / v2 / v3 (11.7M) | 1,600 | 19.2 – 20.0 | 9.5 – 10.9 | 71.7 |
| v5 (label fix) | 2,513 | 12.9 | 5.9 | 49.4 |
| v6 | 2,513 | 12.2 | 6.1 | 50.6 |
| v7 (adds low sensor position) | 2,934 | 12.5 | 5.7 | 52.9 |
| **v8 (per-finger curl loss)** | 2,933 | **11.9** | **5.2** | 52.8 |

v8 with labelled crops: 10.3 mm and 99% left/right at the original position, 12.9 mm and 90% at
the low position (`lr_report_v8_*.txt`). The test set changed at v5 and v7, so rows are only
comparable within the same test-frame count.

### Data (on Sangameshwar's machine, per README)

- 53 sessions, 23,685 stereo frames, 3.2 GB; 17,582 frames pass the label filters.
- One person, one 3Di, positions: original, low, under the desk.
- The 9 sessions from 2026-10-20 (desk glare) are not used yet: only 38% of their frames are usable.

### Key limitation

Labels come from **Ultraleap's own tracking of the same device**. Joints occluded from
the 3Di are guessed by Ultraleap's hand model, not observed. Ultraleap also names the wrong hand
on about 25% of frames at the original position and about 60% at the low position; those frames
are dropped. So current labels cannot teach "what is behind". This is exactly what the second
camera has to fix.

### Repo hygiene notes

- `dataset_ir/` and `checkpoints/` are **not in the repo**, on GitHub or on this machine.
  The README says both are included (dataset in git, weights via LFS). That is wrong for the
  current `main` and should be corrected.
- The README links an `old-main` branch for the removed RGB pipeline; GitHub has only `main`.
- `leapc-python-bindings` is still an uninitialised submodule (empty folder).
- Fixed since the last update: `__pycache__` is no longer tracked; `third-party/hamer` and the
  machine-specific README paths are gone.

## Training on Kaggle (`train_kaggle.ipynb`)

- Runs HandLocNet → 3DFist (v8 recipe) → `eval_fist.py` + `lr_report.py`.
- **First predicts the total run time**: it times frame loading, crop rendering and one training
  step of each model on the actual machine, then scales up by the dataset size.
  After the run, it prints actual vs predicted time.
- Dataset and weights are inputs, not part of the repo: set `DATA_DIR` / `CKPT_DIR`, or leave them
  `None` to search `/kaggle/input` (Kaggle) or `./dataset_ir` (local).
- Reuses an attached `handloc*.pth`; resumes from `<RUN_NAME>_last.pth` if one is attached (for
  runs over Kaggle's 12 h limit).
- On Kaggle it clones `main` from GitHub, so the repo changes it needs must be pushed first
  (`lr_report.py` now reads the dataset path from `$DATASET_IR`).
- Tested end-to-end on CPU with a small synthetic dataset (3 sessions, 360 frames): runs cleanly,
  predicted 2:41 vs actual 3:07. Not yet run on Kaggle with the real data.

## Do we need Sangameshwar's `dataset_ir/` and checkpoints?

**Not for the main goal.** The new goal needs new two-sensor recordings anyway. The old
labels came from a single device, so they never saw the hidden parts. The **code** is
what we reuse (`leap_ir.py`, `capture_ir.py`, `ir_geometry.py`, `perspective_crop.py`).
They are needed to retrain or reproduce v8 (e.g. with the notebook), as a baseline, and as
starting weights.

## Technical facts that shape the plan

- **A 3Di does not output depth or a point cloud.** It gives two IR images plus
  Ultraleap's hand-joint tracking. Cup geometry has to come from our own stereo depth
  (disparity from the L/R IR pair, using `rect_lut.npz` / factory calibration).
- **Ultraleap tracks hands only**, not objects. Any cup ground truth has to come from fused depth.
- **Two active-IR units facing each other may interfere.** Each one's 850 nm LEDs light
  up the other's images. Test this first, before building anything else.
- Desk glare already ruins Ultraleap's labels at one sensor position (2026-10-20 sessions);
  the second unit's LEDs could have a similar effect.

## Next steps

1. Get `dataset_ir/` (and ideally the v8 weights) from Sangameshwar, upload to Kaggle, and run
   `train_kaggle.ipynb` once to confirm the estimate and reproduce v8 (~11.9 mm full pipeline).
2. **Hardware smoke test:** both 3Di units on one PC through LeapC multi-device; record both
   IR streams at once; check how bad the interference is at the planned angles.
3. Two-device capture: extend `capture_ir.py` to save per-device streams plus timestamps.
4. Extrinsic calibration between units (shared target visible in IR, or the same hand
   tracked by both and aligned with Procrustes).
5. Per-view stereo depth → fused point cloud in one frame = ground truth for "full scene".
6. Model: one view's input → completed 3D (representation TBD) + hand joints.

## Open questions

- Output representation: point cloud, voxels, or mesh? (Point cloud is the simplest fit for fused stereo depth.)
- Planned mounting: opposite each other (180°) or about 90° apart? This decides interference and coverage.
- Do we have both 3Di units and a PC with enough USB 3 bandwidth for two streams?

## Log

- 2026-10-21 — Status file created from repo review. Confirmed: single-sensor stage.
- 2026-10-21 — Goal clarified: two 3Di units, single-view → full 3D (hand + held object). Old data not required.
- 2026-10-25 — PR #1 merged: 3DFist v8 (11.9 mm full pipeline), RGB pipeline removed, README rewritten.
- 2026-10-25 — Added `train_kaggle.ipynb` (runtime estimate, configurable data/weights paths, resume).
  `lr_report.py`: dataset path from `$DATASET_IR`; no worker processes on Windows.
