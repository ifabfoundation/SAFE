"""
Utilities Module
================
Shared utility functions across modules.
"""

import os
import json
import random
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import tensorflow as tf


def set_random_seeds(seed: int = 42):
    """
    Set seeds for reproducibility across all libraries.
    
    Args:
        seed: Seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def configure_gpu():
    """
    Configure GPU for TensorFlow (memory growth).
    
    Returns:
        List of available GPUs
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPUs configured: {len(gpus)} devices")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPU available, using CPU")
    return gpus


def get_device():
    """Return the available PyTorch device."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str) -> str:
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Directory path
        
    Returns:
        Directory path
    """
    os.makedirs(path, exist_ok=True)
    return path


def save_json(data: Dict, filepath: str):
    """
    Save dictionary in JSON format.
    
    Args:
        data: Dictionary to save
        filepath: File path
    """
    ensure_dir(os.path.dirname(filepath) or '.')
    
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, pd.Series):
            return obj.tolist()
        return obj
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=convert)


def load_json(filepath: str) -> Dict:
    """
    Load dictionary from JSON file.
    
    Args:
        filepath: File path
        
    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def print_separator(title: str = "", char: str = "=", length: int = 80):
    """Print formatted separator."""
    if title:
        print(f"\n{char * length}")
        print(title)
        print(char * length)
    else:
        print(char * length)


def format_time(seconds: float) -> str:
    """
    Format seconds into readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string (e.g.: "2h 30m 15s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {int(s)}s"
    else:
        h, remainder = divmod(seconds, 3600)
        m, s = divmod(remainder, 60)
        return f"{int(h)}h {int(m)}m {int(s)}s"


def get_memory_usage() -> Dict[str, float]:
    """
    Return current memory usage.
    
    Returns:
        Dictionary with used/available memory in GB
    """
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            'total_gb': mem.total / (1024**3),
            'available_gb': mem.available / (1024**3),
            'used_gb': mem.used / (1024**3),
            'percent': mem.percent
        }
    except ImportError:
        return {}


def estimate_memory_for_sequences(
    n_samples: int,
    window_size: int,
    num_features: int,
    dtype_bytes: int = 4
) -> float:
    """
    Estimate memory needed for time-series sequences.
    
    Args:
        n_samples: Number of sequences
        window_size: Window size
        num_features: Number of features
        dtype_bytes: Bytes per element (4 for float32)
        
    Returns:
        Estimated memory in GB
    """
    bytes_needed = n_samples * window_size * num_features * dtype_bytes
    return bytes_needed / (1024**3)


def validate_dataframe(
    df: pd.DataFrame,
    required_cols: Optional[List[str]] = None,
    numeric_only: bool = False
) -> bool:
    """
    Validate a DataFrame.
    
    Args:
        df: DataFrame to validate
        required_cols: Required columns
        numeric_only: If True, verify all columns are numeric
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If validation fails
    """
    if df.empty:
        raise ValueError("Empty DataFrame")
    
    if required_cols:
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
    
    if numeric_only:
        non_numeric = df.select_dtypes(exclude=['number']).columns.tolist()
        if non_numeric:
            raise ValueError(f"Non-numeric columns: {non_numeric}")
    
    return True


def chunk_array(arr: np.ndarray, chunk_size: int):
    """
    Split array into chunks.
    
    Args:
        arr: Array to split
        chunk_size: Chunk size
        
    Yields:
        Array chunks
    """
    for i in range(0, len(arr), chunk_size):
        yield arr[i:i + chunk_size]


def moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate moving average.
    
    Args:
        arr: Input array
        window: Window size
        
    Returns:
        Moving average
    """
    return np.convolve(arr, np.ones(window) / window, mode='valid')


def normalize_array(arr: np.ndarray) -> np.ndarray:
    """
    Normalize array to [0, 1].
    
    Args:
        arr: Input array
        
    Returns:
        Normalized array
    """
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val - min_val == 0:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)


def safe_divide(a: np.ndarray, b: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """
    Safe division (avoids division by zero).
    
    Args:
        a: Numerator
        b: Denominator
        fill_value: Value to use when b=0
        
    Returns:
        Division result
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(a, b)
        result[~np.isfinite(result)] = fill_value
    return result


class ProgressTracker:
    """Track progress during long operations."""
    
    def __init__(self, total: int, description: str = "Progress"):
        self.total = total
        self.description = description
        self.current = 0
        self.start_time = None
    
    def start(self):
        """Start tracking."""
        import time
        self.start_time = time.time()
        self.current = 0
    
    def update(self, n: int = 1):
        """Update progress."""
        self.current += n
        self._print_progress()
    
    def _print_progress(self):
        """Print progress bar."""
        import time
        if self.start_time is None:
            return
        
        pct = self.current / self.total * 100
        elapsed = time.time() - self.start_time
        
        if self.current > 0:
            eta = elapsed / self.current * (self.total - self.current)
            eta_str = format_time(eta)
        else:
            eta_str = "???"
        
        bar_len = 30
        filled = int(bar_len * self.current / self.total)
        bar = "█" * filled + "░" * (bar_len - filled)
        
        print(f"\r{self.description}: [{bar}] {pct:.1f}% ETA: {eta_str}", end="", flush=True)
        
        if self.current >= self.total:
            print()  # Final newline
    
    def finish(self):
        """Finish tracking."""
        import time
        if self.start_time:
            elapsed = time.time() - self.start_time
            print(f"\n{self.description} completed in {format_time(elapsed)}")


if __name__ == "__main__":
    print("Utils Module - Test")
    print_separator("Configuration")
    
    set_random_seeds(42)
    print("Random seeds set")
    
    configure_gpu()
    
    mem = get_memory_usage()
    if mem:
        print(f"\nMemoria: {mem['used_gb']:.1f}GB / {mem['total_gb']:.1f}GB ({mem['percent']}%)")
    
    print_separator("Test functions")
    
    arr = np.random.randn(100)
    ma = moving_average(arr, 10)
    print(f"Moving average: input {len(arr)} -> output {len(ma)}")
    
    norm = normalize_array(arr)
    print(f"Normalize: min={norm.min():.2f}, max={norm.max():.2f}")
    
    est = estimate_memory_for_sequences(10000, 100, 16)
    print(f"Memory estimate for 10k sequences: {est:.2f} GB")
    
    print("\nTest completed!")
