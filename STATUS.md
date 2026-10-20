# Project Status

_Last updated: 2026-10-21_

## Goal (as currently understood — see Open Questions)

Use **two 3D sensor cameras** viewing the hand from different sides to build a dataset
where the second camera supplies ground truth for the parts the first camera cannot see.
Train a model that, from **one camera only**, predicts what is hidden behind the object.

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

## Next steps (proposed, pending answers below)

1. Choose/confirm second sensor and mounting (e.g. opposite side of the hand).
2. Multi-device capture with timestamp sync.
3. Extrinsic calibration between the two sensors (shared target).
4. Fuse both views → ground truth for joints/surface hidden from camera A.
5. Train single-view model (camera A input) against fused labels; evaluate on occluded joints specifically.

## Open questions

- What exactly are the two sensors — two Ultraleap 3Di units, or a 3Di plus a depth camera (RealSense/Kinect/etc.)?
- "Behind the object": the hand's own occluded fingers (e.g. fist), or a hand occluded by a held object? Is the target joints only, or a full surface/shape?
- At inference, is the input strictly the single front camera?
- Where is `dataset_ir/`, and do we have the trained checkpoints?

## Log

- 2026-10-21 — Status file created from repo review. Confirmed: single-sensor stage.
