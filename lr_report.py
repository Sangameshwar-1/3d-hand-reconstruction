"""L/R and joint error per (true hand x palm/back side) on the held-out sessions.

Keeps the frames where Ultraleap named the wrong hand (labelled with the hand named at
capture), so L/R is scored on every frame; joint error only on frames where Ultraleap agrees.
Usage: python lr_report.py ckpt1.pth [ckpt2.pth ...]   (dataset: $DATASET_IR, default dataset_ir)
"""
import os, sys
from pathlib import Path
sys.path.insert(0, os.getcwd())
import numpy as np
import torch
from torch.utils.data import DataLoader

import fist_dataset as fd
from fist_loss import finger_curl
from fist_model import load_fist_checkpoint

_load = fd.load_joint_json


def load_with_named_hand(p):
    d = _load(p)
    named = d.get("pose_label", "").split("_")[0].capitalize()
    if named in ("Left", "Right"):
        g = d.get("grab_strength", 0.0)
        d["grab_strength"] = g if d.get("hand_type") == named else -1.0 - g  # smuggle agreement via sign
        d["hand_type"] = named
    return d


fd.load_joint_json = load_with_named_hand
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ds = fd.StereoIRDataset(os.environ.get("DATASET_IR", "dataset_ir"), split="val", val_ratio=0.2)
agree = np.array([m["grab"] >= 0 for m in ds.meta])
fist = np.array([(m["grab"] if m["grab"] >= 0 else -1.0 - m["grab"]) >= 0.8 for m in ds.meta])
batches = list(DataLoader(ds, batch_size=32, num_workers=8 if os.name != "nt" else 0))  # no __main__ guard: spawn would re-run this file
tg = np.concatenate([b[2].numpy() for b in batches])
right = np.concatenate([b[4].numpy() for b in batches]) > 0.5

TIPS, MCPS = [8, 12, 16, 20], [5, 9, 13, 17]


def normal(j):
    n = np.cross(j[:, 9] - j[:, 0], j[:, 5] - j[:, 17])
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def geo_right(j):
    """Right hand <=> palm-side scores negative (curl of fingers + thumb opposition)."""
    n = normal(j)
    mcp = j[:, MCPS].mean(1)
    curl = np.einsum("ij,ij->i", n, j[:, TIPS].mean(1) - mcp)
    thumb = np.einsum("ij,ij->i", n, j[:, 4] - mcp)
    return {"curl": curl < 0, "thumb": thumb < 0, "curl+thumb": curl + thumb < 0}


gf, gs = (t.numpy() for t in finger_curl(torch.from_numpy(tg).float()))


def display_flex(j):
    """Live viewer's per-finger fold bars (live_fist_tracking.finger_flexion), batched."""
    f = finger_curl(torch.from_numpy(j).float())[0].numpy() / 0.65
    pw = np.linalg.norm(j[:, 5] - j[:, 17], axis=1) + 1e-6
    f[:, 0] = (1.6 - np.linalg.norm(j[:, 4] - j[:, [9, 13]].mean(1), axis=1) / pw) / 0.8
    return np.clip(f, 0, 1)


gd = display_flex(tg)
palm = np.sign(normal(tg)[:, 2]) * np.where(right, 1, -1) > 0
groups = {"R palm": right & palm, "R back": right & ~palm, "L palm": ~right & palm, "L back": ~right & ~palm,
          "R fist": right & fist, "L fist": ~right & fist}


def row(name, vals):
    print(f"  {name:22s}" + "".join(f"{v:>10s}" for v in vals))


print(f"\n{len(tg)} test frames; Ultraleap named the wrong hand in {(~agree).sum()}")
row("", list(groups) + ["all"])
row("frames", [str(g.sum()) for g in groups.values()] + [str(len(tg))])
masks = list(groups.values()) + [np.ones(len(tg), bool)]
pct = lambda ok: [f"{ok[m].mean() * 100:.0f}%" if m.any() else "-" for m in masks]
row("Ultraleap L/R", pct(agree))
for k, v in geo_right(tg).items():
    row(f"GT skeleton {k}", pct((v == right) | ~agree))

for w in sys.argv[1:]:
    m, _ = load_fist_checkpoint(Path(w), DEV)
    preds, logits = [], []
    with torch.no_grad():
        for x, geo, *_ in batches:
            p, lg = m(x.to(DEV), geo.to(DEV), return_hand=True)
            preds.append(p.cpu().numpy())
            if lg is not None:
                logits.append(lg.cpu().numpy())
    pr = np.concatenate(preds)
    err = np.linalg.norm(pr - tg, axis=2).mean(1)
    print(f"\n{Path(w).stem}")
    row("MPJPE mm (Ultraleap ok)", [f"{err[mm & agree].mean():.1f}" if (mm & agree).any() else "-" for mm in masks])
    pf, ps = (t.numpy() for t in finger_curl(torch.from_numpy(pr).float()))
    cerr = np.abs(pf - gf).mean(1) / 0.65 * 100
    row("curl err % (Ultraleap ok)", [f"{cerr[mm & agree].mean():.0f}" if (mm & agree).any() else "-" for mm in masks])
    dir_ok = ((ps > 0) == (gs > 0)) | (np.abs(gs) <= 0.3)
    row("fold dir ok (Ultraleap ok)", [f"{dir_ok[mm & agree].mean() * 100:.0f}%" if (mm & agree).any() else "-"
                                       for mm in masks])
    pd = display_flex(pr)
    for i, name in enumerate("TIMRP"):
        st, fo = agree & (gd[:, i] < 0.25), agree & (gd[:, i] > 0.6)
        row(f"{name} straight->curled", [f"{(pd[mm & st, i] > 0.5).mean() * 100:.0f}%" if (mm & st).any() else "-"
                                         for mm in masks])
        row(f"{name} folded->missed", [f"{(pd[mm & fo, i] < 0.3).mean() * 100:.0f}%" if (mm & fo).any() else "-"
                                       for mm in masks])
    if logits:
        row("L/R head", pct((np.concatenate(logits) > 0) == right))
    for k, v in geo_right(pr).items():
        row(f"L/R from skeleton {k}", pct(v == right))
