from pathlib import Path
from typing import List, Dict, Any, Union
import pandas as pd
import numpy as np


def save_dataset(data: List[Dict[str, Any]], filepath: Union[str, Path]) -> Path:
    """
    Saves a dataset of expressions and point clouds.
    Supports .parquet and .jsonl file extensions.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(data)

    if filepath.suffix == ".parquet":
        # Convert numpy coordinate arrays to lists for parquet serialization
        df["x"] = df["x"].apply(lambda arr: arr.tolist() if isinstance(arr, np.ndarray) else arr)
        df["y"] = df["y"].apply(lambda arr: arr.tolist() if isinstance(arr, np.ndarray) else arr)
        df.to_parquet(filepath, index=False)
    elif filepath.suffix == ".jsonl":
        df["x"] = df["x"].apply(lambda arr: arr.tolist() if isinstance(arr, np.ndarray) else arr)
        df["y"] = df["y"].apply(lambda arr: arr.tolist() if isinstance(arr, np.ndarray) else arr)
        df.to_json(filepath, orient="records", lines=True)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}. Supported: .parquet, .jsonl")

    return filepath


def load_dataset(filepath: Union[str, Path]) -> pd.DataFrame:
    """Loads a dataset from .parquet or .jsonl filepath."""
    filepath = Path(filepath)

    if filepath.suffix == ".parquet":
        return pd.read_parquet(filepath)
    elif filepath.suffix == ".jsonl":
        return pd.read_json(filepath, orient="records", lines=True)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
