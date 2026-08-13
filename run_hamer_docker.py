import os
import sys
import json
import argparse
from pathlib import Path

import cv2
import torch
import numpy as np
from tqdm import tqdm

sys.path.append('/app')
from hamer.models import load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset
from vitpose_model import ViTPoseModel

import config

def get_mano_to_leap_indices():
    """
    MANO joints to Leap (MediaPipe) joints mapping.
    MANO order typically:
    0: Wrist
    1: Index MCP, 2: PIP, 3: DIP
    4: Middle MCP, 5: PIP, 6: DIP
    7: Pinky MCP, 8: PIP, 9: DIP
    10: Ring MCP, 11: PIP, 12: DIP
    13: Thumb CMC, 14: MCP, 15: IP
    16: Thumb tip, 17: Index tip, 18: Middle tip, 19: Ring tip, 20: Pinky tip
    
    Leap/MediaPipe order:
    0: Wrist
    1-4: Thumb (CMC, MCP, IP, Tip)
    5-8: Index (MCP, PIP, DIP, Tip)
    9-12: Middle (MCP, PIP, DIP, Tip)
    13-16: Ring (MCP, PIP, DIP, Tip)
    17-20: Pinky (MCP, PIP, DIP, Tip)
    """
    return [
        0,              # 0: Wrist -> Wrist
        13, 14, 15, 16, # 1-4: Thumb CMC, MCP, IP, Tip -> Thumb
        1, 2, 3, 17,    # 5-8: Index MCP, PIP, DIP, Tip -> Index
        4, 5, 6, 18,    # 9-12: Middle MCP, PIP, DIP, Tip -> Middle
        10, 11, 12, 19, # 13-16: Ring MCP, PIP, DIP, Tip -> Ring
        7, 8, 9, 20     # 17-20: Pinky MCP, PIP, DIP, Tip -> Pinky
    ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    rgb_dir = dataset_dir / "rgb"
    out_dir = dataset_dir / "predictions_hamer"
    out_dir.mkdir(exist_ok=True)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Load HaMeR
    print("[DOCKER] Loading HaMeR model...")
    model, model_cfg = load_hamer(DEFAULT_CHECKPOINT)
    model = model.to(device)
    model.eval()
    
    # Load detectors
    print("[DOCKER] Loading Detectors...")
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    from detectron2.config import LazyConfig
    import hamer
    cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)
    
    cpm = ViTPoseModel(device)
    
    # Process images
    img_files = sorted(list(rgb_dir.glob("*.jpg")) + list(rgb_dir.glob("*.png")))
    print(f"[DOCKER] Found {len(img_files)} images to process.")
    
    mano_to_leap = get_mano_to_leap_indices()
    
    for img_path in tqdm(img_files):
        img_cv2 = cv2.imread(str(img_path))
        if img_cv2 is None: continue
        
        # 1. Detect Person
        det_out = detector(img_cv2)
        img = img_cv2.copy()[:, :, ::-1] # RGB
        
        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()
        
        if len(pred_bboxes) == 0: continue
        
        # 2. Detect Hands via ViTPose
        vitposes_out = cpm.predict_pose(img, [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)])
        bboxes, is_right = [], []
        
        for vitposes in vitposes_out:
            right_hand_keyp = vitposes['keypoints'][-21:]
            keyp = right_hand_keyp
            valid = keyp[:,2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                bboxes.append(bbox)
                is_right.append(1) # Right hand only for this dataset
                break # Just take the first valid right hand
                
        if len(bboxes) == 0: continue
        
        # 3. HaMeR Reconstruction
        boxes = np.stack(bboxes)
        right = np.stack(is_right)
        ds = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=2.0)
        loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
        
        for batch in loader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)
            
            # Extract 3D keypoints (in metres) and convert to mm
            # out['pred_keypoints_3d'] shape is (B, 21, 3) for MANO
            joints_3d_mano = out['pred_keypoints_3d'][0].detach().cpu().numpy() * 1000.0 # mm
            
            # Map MANO to Leap order
            joints_3d_leap = joints_3d_mano[mano_to_leap]
            
            # Save JSON
            joints_dict = {}
            for i, name in config.JOINT_NAMES.items():
                joints_dict[f"{i}_{name}"] = {
                    "x": float(joints_3d_leap[i, 0]),
                    "y": float(joints_3d_leap[i, 1]),
                    "z": float(joints_3d_leap[i, 2])
                }
                
            out_file = out_dir / f"{img_path.stem}.json"
            with open(out_file, "w") as f:
                json.dump({"joints": joints_dict}, f, indent=2)
                
            break # Just one hand per image

if __name__ == '__main__':
    main()
