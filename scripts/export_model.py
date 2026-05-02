from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from models.hybrid_segformer_unet import HybridSegformerUNet
from utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--onnx-out", default="checkpoints/hybrid_segformer.onnx")
    parser.add_argument("--ts-out", default="checkpoints/hybrid_segformer.ts")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridSegformerUNet(
        cnn_backbone=cfg["model"]["cnn_backbone"],
        transformer_backbone=cfg["model"]["transformer_backbone"],
        classes=cfg["model"]["classes"],
    ).to(device)
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    dummy = torch.randn(1, 3, cfg["data"]["image_size"], cfg["data"]["image_size"]).to(device)
    # Some environments (notably with Transformer layers) can fail
    # trace graph consistency checks even though export itself is valid.
    traced = torch.jit.trace(model, dummy, check_trace=False)
    traced.save(args.ts_out)
    torch.onnx.export(
        model,
        dummy,
        args.onnx_out,
        input_names=["image"],
        output_names=["mask_logits"],
        dynamic_axes={"image": {0: "batch"}, "mask_logits": {0: "batch"}},
        opset_version=18,
    )
    print(f"Exported TorchScript: {args.ts_out}")
    print(f"Exported ONNX: {args.onnx_out}")


if __name__ == "__main__":
    main()
