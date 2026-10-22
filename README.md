# 3DFist: 3D hand pose from Ultraleap 3Di stereo IR

3DFist predicts the 3D positions of the 21 hand joints, in millimetres, from the two infrared
cameras of an Ultraleap 3Di. At run time it uses only the IR images. Ultraleap's hand tracking
is used for training labels only: every saved stereo frame is paired with the joints Ultraleap
tracked in that same sensor frame.

The project focuses on what goes wrong most often with an IR hand tracker seen from the front:
curled fingers and fists, the thumb, and telling the left hand from the right.

**Current best model: v8** (`fist_r18_v8`)

| Test set | Joint error | Left/right correct | Finger-curl error |
| :--- | ---: | ---: | ---: |
| Original sensor position (6 sessions, 3,293 frames) | 10.3 mm | 99% | 4% |
| Low sensor position (2 sessions, 1,066 frames) | 12.9 mm | 90% | 7% |
| Full pipeline, own hand detector (8 sessions, 2,934 frames) | 11.9 mm | | |

How these are measured is explained under [Results by version](#results-by-version).

---

## How it works

```
left IR 384x384  -> HandLocNet -> box --+                       +-> virtual left crop 128x128 --+
                                        +-> triangulate hand -->|  (one shared virtual camera)  +-> (2, 128, 128)
right IR 384x384 -> HandLocNet -> box --+                       +-> virtual right crop 128x128 -+
                                                                                                |
   ResNet-18 (ImageNet weights, shared by both views) -> fuse the two 4x4 feature maps  <-------+
   -> pool + [hand distance, angular size] -> MLP -> 21 x 3 joints (mm, wrist at the origin)
                                                  -> left/right logit
```

1. **Find the hand.** HandLocNet ([hand_locator.py](hand_locator.py)) is a small CenterNet-style
   detector that finds the hand in each raw IR view.
2. **Crop through a virtual camera.** The two box centres are triangulated, and both fisheye views
   are re-rendered through one virtual pinhole camera aimed at the hand
   ([perspective_crop.py](perspective_crop.py), [ir_transforms.py](ir_transforms.py)). This removes
   the fisheye distortion and makes the crops independent of where the hand is in the image.
3. **Predict the joints.** A ResNet-18 shared by both views encodes each crop. The two feature maps
   are fused, and an MLP outputs 21 joints in millimetres relative to the wrist
   ([fist_model.py](fist_model.py), 11.7 M parameters). A second small head predicts left/right.
4. **Decide left or right.** The live viewer decides left/right from the predicted skeleton itself:
   which way the fingers and thumb fold relative to the palm. It adds this evidence up over
   consecutive frames. The rule is 100% correct on Ultraleap's own correct skeletons, and it beats
   the classification head (99% vs 90% for v8 at the original position).

**Training** ([train_fist.py](train_fist.py), [fist_loss.py](fist_loss.py))

- Losses:
  - Smooth-L1 joint loss, with depth weighted x3.
  - Bone-length loss.
  - Joint-limit loss against impossible folds.
  - Per-finger curl loss (since v8): how far each finger bends and which way it folds.
- Augmentation:
  - The virtual camera rolls up to ±180°, with the labels rotated exactly.
  - Exact left/right mirroring: flip the images and swap the views.
  - Simulated IR saturation hotspots.
  - Crop jitter that replays HandLocNet's measured errors.
  - Oversampling of half-curled hands and fists.

---

## Results by version

**How results are measured**

- **Joint error:** mean distance between predicted and Ultraleap joints, wrist-aligned (MPJPE).
  - It is measured on test frames where Ultraleap named the correct hand, with crops placed from
    the labels.
- **Left/right correct:** share of all test frames where the skeleton rule gives the hand named
  at capture.
  - This includes frames that Ultraleap itself got wrong.
- **Test sessions:** they are never used for training. They are listed in
  [dataset_ir/val_sessions.txt](dataset_ir/val_sessions.txt).

| Version | What changed | Joint error, original position | L/R, original position | Joint error, low position | L/R, low position |
| :--- | :--- | ---: | ---: | ---: | ---: |
| v0 | From-scratch Stereo3DFistNet (1.4 M parameters) | – | – | – | – |
| v1 `fist_resnet18` | ImageNet ResNet-18 shared by both views replaces the from-scratch network | –¹ | – | – | – |
| v2 `fist_r18_v2` | Left/right head, ±180° roll augmentation, curl-balanced sampling | 23.8 mm | – | – | – |
| v3 `fist_r18_v3` | Exact left/right mirror augmentation | 26.4 mm | – | – | – |
| v4 | Cancelled: one training session was found to be mislabelled | – | – | – | – |
| v5 `fist_r18_v5` | **Label fix** (see [Dataset](#ultraleap-label-errors)): drop frames where Ultraleap names the wrong hand; dedicated fist sessions; fixed named-hand test split | 12.4 mm | 92% | – | – |
| v6 `fist_r18_v6` | Drop 747 disputed frames from older, unnamed sessions; more left-hand and back-of-hand takes | 11.4 mm | 97% | 16.8 mm | 84% |
| v7 `fist_r18_v7` | 7 takes at a new, low sensor position | 11.6 mm | 96% | 13.9 mm | 90% |
| **v8** `fist_r18_v8` | **Per-finger curl loss**; left-fist take from a third position | **10.3 mm** | **99%** | **12.9 mm** | **90%** |

¹ v1 scored 24.7 mm on an earlier 4-session test, where v2 scored 19.4 mm.

**Full pipeline** (HandLocNet crops, [eval_fist.py](eval_fist.py)):

| Version | Test set | Joint error |
| :--- | :--- | ---: |
| v5 | 6 sessions | 12.87 mm |
| v6 | 6 sessions | 12.18 mm |
| v7 | 8 sessions | 12.46 mm |
| v8 | 8 sessions | 11.9 mm |

**What the curl loss fixed in v8 (v7 → v8)**

| | Original position | Low position |
| :--- | ---: | ---: |
| Finger-curl error, as a share of a full fist | 11% → 4% | 10% → 7% |
| Straight ring finger shown as curled | 83% → 0% | 37% → 0% |
| Tucked thumb shown as straight (right hand) | 67–78% → 33–39% | – |

Result files are in [results/](results/): `eval_r18_v*.json`, `eval_plot_r18_v*.png`,
`lr_report_v*.txt` and the training logs.

---

## Dataset

`dataset_ir/` is included in this repository: 3.2 GB, 53 sessions, 23,685 stereo frames, all
recorded by one person with one Ultraleap 3Di.

```
dataset_ir/
├── val_sessions.txt          test sessions (one per line); every other session is training data
├── excluded_frames.txt       747 frames of unnamed sessions dropped from training (see below)
└── session_YYYYMMDD_HHMMSS/
    ├── left/frame_00001.png   left IR view, 384x384, 8-bit
    ├── right/frame_00001.png  right IR view, same sensor frame
    ├── leap/frame_00001.json  Ultraleap label for that frame
    ├── calibration.json       factory stereo calibration of the device
    ├── rect_lut.npz           LeapC pixel <-> ray lookup tables, per camera
    └── metadata.json          pose label, save rate, capture time
```

Each `leap/*.json` holds:

- **joints:** 21 joints in millimetres, LeapC tracking space: wrist, then 4 joints per finger from
  the thumb to the pinky.
- **Ultraleap's hand readings:** `hand_type`, `confidence`, `grab_strength`, `pinch_strength`,
  `fingers_extended` and palm position.
- **IDs:** the capture `pose_label`, plus frame IDs and timestamps.

A few sessions have no `metadata.json` because the capture was closed early. They load
normally.

### Sessions

| Recorded | Sensor position | Sessions | Poses | Frames | Usable¹ | Role |
| :--- | :--- | ---: | :--- | ---: | ---: | :--- |
| 2026-10-17/18 | Original, facing the user | 14 | open_palm, fist, pinch, spread, mixed (hand not named) | 6,654 | 5,907 | Training |
| 2026-10-19 09:58–11:04 | Original | 15 | left/right × palm toward / back toward, fists | 6,487 | 4,751 | 6 test sessions (3,293 frames); the rest training |
| 2026-10-19 13:26–13:34 | Original | 7 | left/right back toward, mixed | 4,606 | 3,636 | Training |
| 2026-10-19 19:02–19:11 | Low, near the table, looking up | 7 | fists, mixed, back toward | 3,616 | 2,253 | 2 test sessions (1,066 frames); the rest training |
| 2026-10-19 21:59 | Under the desk, close to the body | 1 | left_fist | 278 | 256 | Training |
| 2026-10-20 03:00–03:22 | Low, desk in view (strong IR glare) | 9 | single fingers, thumb-in fists, rotations | 2,044 | 779 | Not used yet |
| **Total** | | **53** | | **23,685** | **17,582** | |

¹ **Usable** means the frame passed the label filters described below.

- **v8 training set:** 13,865 frames, which is every usable frame recorded up to 2026-10-19 that
  is not in a test session.
- **Not yet used:** the 2026-10-20 sessions came after v8 was trained. Their glare made
  Ultraleap's labels poor, so only 38% of their frames are usable.

### Ultraleap label errors

Ultraleap often mistakes a right hand with the palm toward the sensor for a left hand with the
back toward it, and the other way round. In IR the two look the same. When this happens, the
hand label is wrong, and so is the skeleton: its fingers fold the wrong way.

| Where | Ultraleap named the wrong hand on |
| :--- | ---: |
| Named sessions, original position | about 25% of frames |
| Low position | about 60% of frames |
| Low position, left hand only | about 75% of frames |

Such skeletons cannot be repaired, so they are removed:

- **Named sessions** (pose label starting `left_` or `right_`): frames where Ultraleap's
  `hand_type` disagrees with the named hand are dropped
  ([fist_dataset.py](fist_dataset.py)).
- **Older unnamed sessions:** frames where a trained model's skeleton disagrees with Ultraleap's
  hand are listed in `excluded_frames.txt` and dropped.
- **Test scoring:** left/right is scored against the hand named at capture, not Ultraleap's
  label.

### Downloading without the dataset

A normal `git clone` downloads all 3.2 GB. To get only the code:

```bash
git clone --filter=blob:none --sparse https://github.com/Sangameshwar-1/3d-hand-reconstruction.git
cd 3d-hand-reconstruction
git sparse-checkout set --no-cone '/*' '!/dataset_ir/'
```

---

## Setup

- **Laptop (capture and live viewer):** Windows 11, Python 3.11, Ultraleap 3Di with the Ultraleap
  Hyperion tracking service.
  - Install the LeapC Python bindings from [leapc-python-bindings/](leapc-python-bindings/); see
    [requirements.txt](requirements.txt).
  - Install the rest with `pip install -r requirements.txt`.
- **Training:** one GPU. The models here were trained on a GTX 1080 Ti on the IIIT ADA SLURM
  cluster: about 2 h 20 min for 60 epochs, including HandLocNet.
- **Trained weights** (`checkpoints/*.pth`, about 140 MB each) are not in the repository because
  they exceed GitHub's 100 MB file limit. Reproduce them with the commands below.

---

## Usage

### 1. Record

```bash
python capture_ir.py --pose right_fist --duration 40
python capture_ir.py --pose left_thumb_in_fist --duration 40
```

- **Pose names:** start them with `right_` or `left_` so frames with the wrong Ultraleap hand can
  be dropped. Don't use `--hand`, because Ultraleap's hand label is the thing that's unreliable.
- **Check while recording:** the window shows `Saved: N (usable M)`. "Usable" frames are the ones
  where Ultraleap names the correct hand. If it stays low, hide your hand for 2 s and bring it
  back in open.
- **Other hand:** frames with two hands in view are refused.
- **Glare:** avoid bright surfaces in view, such as a desk right in front of the sensor. Their
  glare makes Ultraleap's labels (and the model) much worse.

### 2. Train

```bash
python train_handloc.py --dataset dataset_ir --epochs 30 --device cuda --out checkpoints/handloc_v6_best.pth

python train_fist.py --dataset dataset_ir --epochs 60 --batch-size 8 --lr 1e-3 --device cuda --workers 8 \
    --handloc checkpoints/handloc_v6_best.pth --backbone resnet18 --hand-head --max-roll 180 \
    --balance-curl --mirror --curl-weight 10 --run-name fist_r18_v8
```

To continue an interrupted run, add `--resume checkpoints/fist_r18_v8_last.pth`.

### 3. Evaluate

```bash
# full pipeline: MPJPE, PA-MPJPE, per-axis/finger/joint errors, mean-pose baseline
python eval_fist.py --dataset dataset_ir --weights checkpoints/fist_r18_v8_best.pth \
    --handloc checkpoints/handloc_v6_best.pth --out results/eval_r18_v8.json

# per hand and side (right/left x palm/back, fists): joint error, left/right and per-finger curl
python lr_report.py checkpoints/fist_r18_v7_best.pth checkpoints/fist_r18_v8_best.pth
```

### 4. Live viewer

```bash
python live_fist_tracking.py --weights checkpoints/fist_r18_v8_best.pth --handloc checkpoints/handloc_v6_best.pth
```

The left panel shows the IR stream with the hand crop. The right panel shows the 3D skeleton:

- **Hand label:** the model's left/right call, with its confidence.
- **Finger curl:** one bar per finger (T, I, M, R, P), from 0 (straight) to full (folded). In
  mode 3, a thin green line under each bar shows Ultraleap's value.

| Key | Action |
| :--- | :--- |
| `1` / `2` / `3` | Ultraleap skeleton / 3DFist skeleton / both overlaid, with live error |
| `a` `d` / `w` `s` | Rotate the 3D view (yaw / pitch); `r` resets it |
| `o` | Sensor view (true orientation) or upright hand view |
| `c` | IR panel: left view only, or both views |
| `q` | Quit |

---

## Known limitations

- **Labels come from Ultraleap.** Where Ultraleap fails, there are no correct labels to learn from.
  This is why the left hand at the low sensor position is still barely tested.
- **The tucked thumb** is still missed on about a third of right-hand frames.
- **Left hand with the back toward the sensor** is the hardest pose: 22 mm joint error.
- **One person, one room.** The model has not been tested on other people's hands.

---

## Other files

- [config.py](config.py): paths, joint names, skeleton connections.
- [leap_ir.py](leap_ir.py): reads stereo IR images and hand tracking from LeapC.
- [ir_geometry.py](ir_geometry.py): projects 3D joints into the IR views using the device calibration.
- [fist_dataset.py](fist_dataset.py): loads sessions, applies the test split and label filters.
- [fist_metrics.py](fist_metrics.py): MPJPE and PA-MPJPE.
- [dataset_utils.py](dataset_utils.py): reads and writes the joint JSON files.
- [live_leap_ir.py](live_leap_ir.py): raw stereo IR viewer with Ultraleap's skeleton overlaid.
- **Earlier RGB-webcam pipeline (removed):** this compared MediaPipe, a fine-tuned ViT and
  HaMeR against Ultraleap on an RGB webcam. It is not part of 3DFist and was removed from the
  repository on 2026-10-22. Its code and data are kept on the
  [`old-main`](https://github.com/Sangameshwar-1/3d-hand-reconstruction/tree/old-main) branch.

## Coordinate systems

- **LeapC:** right-handed, millimetres, origin at the centre of the 3Di. With the sensor facing
  you, +x points to your right (image left).
- **IR pixels:** a joint projects as `ray = (-(x + cam_x) / y, z / y)`, then through LeapC's
  rectilinear lookup table (`rect_lut.npz`). `cam_x` is -32.9 mm for the left camera and
  +32.9 mm for the right.
- **Model output:** joints relative to the wrist, in the virtual crop camera's frame. They are
  rotated back to LeapC axes for display and evaluation.
