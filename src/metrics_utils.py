"""Helpers for saving and comparing experiment metrics.

Example
-------
>>> from pathlib import Path
>>> from metrics_utils import save_metrics_json
>>> save_metrics_json({"a": 1.0}, Path("/tmp/metrics.json"))
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import json
import pandas as pd


def save_metrics_json(metrics: Dict, output_path: Path) -> None:
    """Save a metrics dictionary to JSON.

    Parameters
    ----------
    metrics : Dict
        Metrics dictionary.
    output_path : Path
        Output JSON file path.

    Returns
    -------
    None

    Example
    -------
    >>> from pathlib import Path
    >>> save_metrics_json({"score": 0.5}, Path("/tmp/s.json"))
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)


def load_metrics_json(input_path: Path) -> Dict:
    """Load a metrics dictionary from JSON.

    Parameters
    ----------
    input_path : Path
        Input JSON file path.

    Returns
    -------
    Dict
        Loaded metrics dictionary.

    Example
    -------
    >>> from pathlib import Path
    >>> save_metrics_json({"x": 1}, Path("/tmp/x.json"))
    >>> load_metrics_json(Path("/tmp/x.json"))["x"]
    1
    """
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def comparison_dataframe(rows: List[Dict]) -> pd.DataFrame:
    """Convert a list of metrics rows into a comparison DataFrame.

    Parameters
    ----------
    rows : List[Dict]
        One dictionary per model/run.

    Returns
    -------
    pd.DataFrame
        Comparison table.

    Example
    -------
    >>> df = comparison_dataframe([{"Model": "A", "mAP@0.5": 0.1}])
    >>> list(df.columns)
    ['Model', 'mAP@0.5']
    """
    return pd.DataFrame(rows)
