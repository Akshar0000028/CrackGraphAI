from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load and validate YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    if config is None:
        raise ValueError(f"Configuration file {path} is empty or invalid")
    
    # Validate required top-level keys
    required_keys = {"model", "data", "training", "inference"}
    missing_keys = required_keys - set(config.keys())
    if missing_keys:
        raise ValueError(f"Configuration missing required keys: {missing_keys}")
    
    # Validate model config
    model_config = config.get("model", {})
    model_required = {"cnn_backbone", "transformer_backbone", "classes"}
    model_missing = model_required - set(model_config.keys())
    if model_missing:
        raise ValueError(f"Model config missing required keys: {model_missing}")
    
    # Validate data config
    data_config = config.get("data", {})
    data_required = {"image_size"}
    data_missing = data_required - set(data_config.keys())
    if data_missing:
        raise ValueError(f"Data config missing required keys: {data_missing}")
    
    # Validate inference config
    inference_config = config.get("inference", {})
    inference_required = {"threshold"}
    inference_missing = inference_required - set(inference_config.keys())
    if inference_missing:
        raise ValueError(f"Inference config missing required keys: {inference_missing}")
    
    return config


def load_env_config(env_path: str | Path = ".env") -> None:
    """Load environment variables from .env file if it exists."""
    env_file = Path(env_path)
    if env_file.exists():
        load_dotenv(env_file)


def get_env_var(key: str, default: Any = None, cast: type = str) -> Any:
    """Get environment variable with optional type casting."""
    value = os.getenv(key, default)
    if value is None:
        return None
    if cast == bool:
        return value.lower() in ("true", "1", "yes", "on")
    if cast == int:
        return int(value)
    if cast == float:
        return float(value)
    return value
