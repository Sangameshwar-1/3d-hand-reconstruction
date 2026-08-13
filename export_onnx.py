import torch
import config
from finetune import FineTunedHandPoseModel

def main():
    print("[EXPORT] Initializing model architecture...")
    model = FineTunedHandPoseModel(freeze_backbone=True)
    
    weights_path = config.CHECKPOINTS_DIR / "best_model.pth"
    print(f"[EXPORT] Loading champion weights from {weights_path}...")
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    
    # Must call eval() for inference mode
    model.eval()
    
    # Dummy input matching ViT-B/16 expected shape: (Batch Size, Channels, Height, Width)
    dummy_input = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE, requires_grad=True)
    
    onnx_file_path = config.PROJECT_ROOT / "hamer_champion.onnx"
    print(f"[EXPORT] Exporting to ONNX at {onnx_file_path}...")
    
    torch.onnx.export(
        model,                      
        dummy_input,                
        str(onnx_file_path),             
        export_params=True,         
        opset_version=18,           # ViT-B/16 uses LayerNorm, added to ONNX at opset 17
        do_constant_folding=True,   
        input_names=['input_image'],
        output_names=['pred_joints']
    )
    print("[EXPORT] Success! ONNX model saved.")

if __name__ == "__main__":
    main()
