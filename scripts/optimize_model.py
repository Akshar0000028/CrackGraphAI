"""
Model optimization utilities for production deployment.
Supports ONNX export, TensorRT, and quantization.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.onnx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_optimizer")


def load_model(
    model_path: str,
    cnn_backbone: str = "resnet34",
    transformer_backbone: str = "mit_b0",
    device: str = "cuda",
) -> nn.Module:
    """Load trained model from checkpoint."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from models.hybrid_segformer_unet import HybridSegformerUNet
    
    model = HybridSegformerUNet(
        cnn_backbone=cnn_backbone,
        transformer_backbone=transformer_backbone,
        classes=1,
    ).to(device)
    
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded checkpoint from {model_path}")
    else:
        logger.warning(f"Checkpoint not found: {model_path}, using initialized weights")
    
    model.eval()
    return model


def export_onnx(
    model: nn.Module,
    output_path: str,
    input_size: Tuple[int, int] = (256, 256),
    opset_version: int = 17,
    dynamic_axes: bool = True,
) -> str:
    """
    Export model to ONNX format.
    
    Args:
        model: PyTorch model
        output_path: Output ONNX file path
        input_size: Input image size (H, W)
        opset_version: ONNX opset version
        dynamic_axes: Enable dynamic batch/height/width
        
    Returns:
        Path to exported ONNX file
    """
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(next(model.parameters()).device)
    
    axes = None
    if dynamic_axes:
        axes = {
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=axes,
    )
    
    logger.info(f"Exported ONNX model to {output_path}")
    
    # Verify with ONNX Runtime
    try:
        import onnxruntime as ort
        import numpy as np
        
        ort_session = ort.InferenceSession(output_path)
        
        # Test inference
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        logger.info(f"ONNX verification passed. Output shape: {ort_outputs[0].shape}")
        
    except ImportError:
        logger.warning("onnxruntime not installed, skipping verification")
    except Exception as e:
        logger.error(f"ONNX verification failed: {e}")
    
    return output_path


def quantize_onnx(
    onnx_path: str,
    output_path: str,
    quantization_mode: str = "dynamic",
) -> str:
    """
    Quantize ONNX model for faster inference.
    
    Args:
        onnx_path: Input ONNX model path
        output_path: Output quantized model path
        quantization_mode: 'dynamic' or 'static'
        
    Returns:
        Path to quantized model
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        if quantization_mode == "dynamic":
            quantize_dynamic(
                model_input=onnx_path,
                model_output=output_path,
                weight_type=QuantType.QInt8,
                optimize_model=True,
            )
            logger.info(f"Dynamic quantization complete: {output_path}")
        else:
            logger.warning("Static quantization requires calibration data, using dynamic")
            quantize_dynamic(
                model_input=onnx_path,
                model_output=output_path,
                weight_type=QuantType.QInt8,
            )
        
        # Report size reduction
        original_size = os.path.getsize(onnx_path) / (1024 * 1024)
        quantized_size = os.path.getsize(output_path) / (1024 * 1024)
        reduction = (1 - quantized_size / original_size) * 100
        
        logger.info(f"Model size: {original_size:.2f}MB → {quantized_size:.2f}MB ({reduction:.1f}% reduction)")
        
        return output_path
        
    except ImportError:
        logger.error("onnxruntime not installed, skipping quantization")
        return onnx_path


def export_torchscript(
    model: nn.Module,
    output_path: str,
    method: str = "trace",
    input_size: Tuple[int, int] = (256, 256),
) -> str:
    """
    Export model to TorchScript.
    
    Args:
        model: PyTorch model
        output_path: Output TorchScript file path
        method: 'trace' or 'script'
        input_size: Input image size
        
    Returns:
        Path to exported TorchScript file
    """
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(device)
    
    if method == "trace":
        scripted = torch.jit.trace(model, dummy_input)
    else:
        scripted = torch.jit.script(model)
    
    # Optimize for inference
    scripted = torch.jit.optimize_for_inference(scripted)
    
    scripted.save(output_path)
    logger.info(f"Exported TorchScript model to {output_path}")
    
    # Verify
    try:
        loaded = torch.jit.load(output_path)
        with torch.no_grad():
            output = loaded(dummy_input)
        logger.info(f"TorchScript verification passed. Output shape: {output.shape}")
    except Exception as e:
        logger.error(f"TorchScript verification failed: {e}")
    
    return output_path


def export_tensorrt(
    onnx_path: str,
    output_path: str,
    fp16: bool = True,
    workspace_size: int = 4096,
) -> str:
    """
    Convert ONNX model to TensorRT engine.
    
    Args:
        onnx_path: Input ONNX model path
        output_path: Output TensorRT engine path
        fp16: Enable FP16 precision
        workspace_size: Workspace size in MB
        
    Returns:
        Path to TensorRT engine
    """
    try:
        import tensorrt as trt
        
        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        
        # Parse ONNX
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    logger.error(f"ONNX parse error: {parser.get_error(error)}")
                raise RuntimeError("Failed to parse ONNX")
        
        # Build config
        config = builder.create_builder_config()
        config.max_workspace_size = workspace_size * (1 << 20)  # MB to bytes
        
        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("Enabled FP16 precision")
        
        # Build engine
        engine = builder.build_engine(network, config)
        
        if engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
        
        # Save engine
        with open(output_path, 'wb') as f:
            f.write(engine.serialize())
        
        logger.info(f"Exported TensorRT engine to {output_path}")
        return output_path
        
    except ImportError:
        logger.error("TensorRT not installed, skipping export")
        return onnx_path
    except Exception as e:
        logger.error(f"TensorRT export failed: {e}")
        return onnx_path


def benchmark_model(
    model_path: str,
    model_type: str = "pytorch",
    input_size: Tuple[int, int] = (256, 256),
    num_runs: int = 100,
    warmup: int = 10,
) -> Dict[str, float]:
    """
    Benchmark model inference performance.
    
    Args:
        model_path: Path to model file
        model_type: 'pytorch', 'onnx', 'torchscript', or 'tensorrt'
        input_size: Input image size
        num_runs: Number of inference runs
        warmup: Number of warmup runs
        
    Returns:
        Dict with benchmark results
    """
    import time
    import numpy as np
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(device)
    
    if model_type == "pytorch":
        model = torch.load(model_path, map_location=device)
        model.eval()
        
        infer_fn = lambda: model(dummy_input)
        
    elif model_type == "torchscript":
        model = torch.jit.load(model_path, map_location=device)
        
        infer_fn = lambda: model(dummy_input)
        
    elif model_type == "onnx":
        import onnxruntime as ort
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device.type == 'cuda' else ['CPUExecutionProvider']
        session = ort.InferenceSession(model_path, providers=providers)
        input_name = session.get_inputs()[0].name
        
        infer_fn = lambda: session.run(None, {input_name: dummy_input.cpu().numpy()})
        
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Warmup
    for _ in range(warmup):
        _ = infer_fn()
        if device.type == 'cuda':
            torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start = time.time()
        _ = infer_fn()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        times.append(time.time() - start)
    
    return {
        "mean_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "min_ms": np.min(times) * 1000,
        "max_ms": np.max(times) * 1000,
        "p50_ms": np.percentile(times, 50) * 1000,
        "p95_ms": np.percentile(times, 95) * 1000,
        "p99_ms": np.percentile(times, 99) * 1000,
        "throughput": 1.0 / np.mean(times),
    }


def main():
    parser = argparse.ArgumentParser(description="Optimize CrackGraphAI model for production")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--output-dir", default="optimized_models", help="Output directory")
    parser.add_argument("--export-onnx", action="store_true", help="Export to ONNX")
    parser.add_argument("--quantize", action="store_true", help="Quantize ONNX model")
    parser.add_argument("--export-torchscript", action="store_true", help="Export to TorchScript")
    parser.add_argument("--export-tensorrt", action="store_true", help="Export to TensorRT")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark all formats")
    parser.add_argument("--input-size", type=int, nargs=2, default=[256, 256], help="Input size H W")
    parser.add_argument("--cnn-backbone", default="resnet34", help="CNN backbone name")
    parser.add_argument("--transformer-backbone", default="mit_b0", help="Transformer backbone name")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load model
    model = load_model(
        args.checkpoint,
        cnn_backbone=args.cnn_backbone,
        transformer_backbone=args.transformer_backbone,
        device=device,
    )
    
    results = {}
    
    # Export ONNX
    if args.export_onnx:
        onnx_path = output_dir / "model.onnx"
        export_onnx(
            model,
            str(onnx_path),
            input_size=tuple(args.input_size),
            dynamic_axes=True,
        )
        results['onnx'] = str(onnx_path)
        
        # Quantize
        if args.quantize:
            quantized_path = output_dir / "model_quantized.onnx"
            quantize_onnx(str(onnx_path), str(quantized_path))
            results['onnx_quantized'] = str(quantized_path)
        
        # TensorRT
        if args.export_tensorrt and device == "cuda":
            trt_path = output_dir / "model.trt"
            export_tensorrt(str(onnx_path), str(trt_path))
            results['tensorrt'] = str(trt_path)
    
    # Export TorchScript
    if args.export_torchscript:
        ts_path = output_dir / "model.pt"
        export_torchscript(model, str(ts_path), method="trace", input_size=tuple(args.input_size))
        results['torchscript'] = str(ts_path)
    
    # Benchmark
    if args.benchmark:
        logger.info("\n" + "="*50)
        logger.info("BENCHMARK RESULTS")
        logger.info("="*50)
        
        # Benchmark PyTorch
        pt_results = benchmark_model(
            args.checkpoint,
            "pytorch",
            input_size=tuple(args.input_size),
        )
        logger.info(f"\nPyTorch:")
        for k, v in pt_results.items():
            logger.info(f"  {k}: {v:.2f}")
        
        # Benchmark others if available
        if 'torchscript' in results:
            ts_results = benchmark_model(
                results['torchscript'],
                "torchscript",
                input_size=tuple(args.input_size),
            )
            logger.info(f"\nTorchScript:")
            for k, v in ts_results.items():
                logger.info(f"  {k}: {v:.2f}")
        
        if 'onnx' in results:
            ort_results = benchmark_model(
                results['onnx'],
                "onnx",
                input_size=tuple(args.input_size),
            )
            logger.info(f"\nONNX Runtime:")
            for k, v in ort_results.items():
                logger.info(f"  {k}: {v:.2f}")
    
    logger.info("\nOptimization complete!")
    logger.info(f"Models saved to: {output_dir}")


if __name__ == "__main__":
    main()
