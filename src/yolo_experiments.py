"""YOLO experiment helpers for the dental pathology project.

This module contains helper classes and functions for configuring, training,
loading, evaluating, and visualizing multiple YOLO experiments on the cleaned
dental X-ray dataset.

It is designed to keep the notebook compact while preserving full
reproducibility and readability.

Example
-------
>>> from pathlib import Path
>>> from yolo_experiments import load_yaml_info
>>> yaml_data = load_yaml_info(Path("/kaggle/working/project_artifacts/data_clean_13.yaml"))
>>> print(yaml_data["nc"])
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import yaml

from ultralytics import YOLO


@dataclass
class YoloExperiment:
    """Container for one YOLO experiment configuration.

    Parameters
    ----------
    key : str
        Short unique key used internally.
    title : str
        Human-readable experiment title.
    model_init : str
        YOLO model initialization checkpoint, e.g. `yolo11s.pt`.
    weights_path : str
        Path to the saved best weights for the experiment.
    epochs : int
        Number of training epochs.
    imgsz : int
        Image size used during training and evaluation.
    batch : int
        Batch size.
    train_enabled : bool
        Whether this experiment should be trained when training is enabled.
    eval_enabled : bool
        Whether this experiment should be evaluated when evaluation is enabled.
    results_png : str | None
        Optional path to the saved YOLO `results.png` file.

    Example
    -------
    >>> exp = YoloExperiment(
    ...     key="baseline",
    ...     title="Baseline",
    ...     model_init="yolo11n.pt",
    ...     weights_path="/tmp/best.pt",
    ...     epochs=20,
    ...     imgsz=640,
    ...     batch=8,
    ... )
    >>> print(exp.title)
    Baseline
    """
    key: str
    title: str
    model_init: str
    weights_path: str
    epochs: int
    imgsz: int
    batch: int
    train_enabled: bool = False
    eval_enabled: bool = True
    results_png: str | None = None


def path_exists(path: Path | str, label: str) -> None:
    """Print whether a path exists.

    Parameters
    ----------
    path : Path | str
        Path to inspect.
    label : str
        Short label printed before the path.

    Returns
    -------
    None

    Example
    -------
    >>> path_exists("/tmp", "Temporary directory")
    """
    path = Path(path)
    print(f"{label}: {path}")
    print("Exists:", path.exists())


def load_yaml_info(yaml_path: Path) -> Optional[dict]:
    """Load a YOLO dataset YAML file and print a short summary.

    Parameters
    ----------
    yaml_path : Path
        Path to the dataset YAML file.

    Returns
    -------
    Optional[dict]
        Parsed YAML dictionary, or None if the file does not exist.

    Example
    -------
    >>> # yaml_data = load_yaml_info(Path("/kaggle/working/project_artifacts/data_clean_13.yaml"))
    """
    if not yaml_path.exists():
        print("YAML file not found:", yaml_path)
        return None

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    print("YAML loaded successfully.")
    print("Keys:", list(data.keys()))
    print("\nClasses:")
    for i, name in enumerate(data["names"]):
        print(f"{i}: {name}")
    return data


def load_model_if_exists(weights_path: str | Path, title: str):
    """Load a YOLO model from saved weights if the file exists.

    Parameters
    ----------
    weights_path : str | Path
        Path to the saved weights file.
    title : str
        Human-readable model title used in printed messages.

    Returns
    -------
    YOLO | None
        Loaded YOLO model if weights exist, otherwise None.

    Example
    -------
    >>> # model = load_model_if_exists("/tmp/best.pt", "Baseline model")
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        print(f"Missing weights for {title}: {weights_path}")
        return None

    print(f"Loading {title} from: {weights_path}")
    return YOLO(str(weights_path))


def train_model(exp: YoloExperiment, data_yaml: Path, project_dir: Path):
    """Train one YOLO experiment.

    Parameters
    ----------
    exp : YoloExperiment
        Experiment configuration.
    data_yaml : Path
        Path to the cleaned YOLO dataset YAML.
    project_dir : Path
        Output directory for YOLO training runs.

    Returns
    -------
    YOLO
        Trained YOLO model object.

    Example
    -------
    >>> # model = train_model(exp, Path("/data.yaml"), Path("/kaggle/working/dental_project"))
    """
    print(f"Training: {exp.title}")
    model = YOLO(exp.model_init)
    model.train(
        data=str(data_yaml),
        epochs=exp.epochs,
        imgsz=exp.imgsz,
        batch=exp.batch,
        project=str(project_dir),
        name=exp.key,
        exist_ok=True,
        plots=True,
    )
    return model


def validate_model(
    model,
    data_yaml: Path,
    imgsz: int,
    batch: int,
    split: str,
    project: Path,
    name: str,
    exist_ok: bool = True,
    plots: bool = True,
):
    """Run YOLO evaluation on a specified dataset split.

    Parameters
    ----------
    model : YOLO
        Loaded YOLO model.
    data_yaml : Path
        Path to the dataset YAML file.
    imgsz : int
        Evaluation image size.
    batch : int
        Evaluation batch size.
    split : str
        Dataset split to evaluate, e.g. `val` or `test`.
    project : Path
        Output project directory.
    name : str
        Run name used by Ultralytics.
    exist_ok : bool
        Whether to allow overwriting an existing run name.
    plots : bool
        Whether to save evaluation plots.

    Returns
    -------
    Any
        Ultralytics metrics object.

    Example
    -------
    >>> # metrics = validate_model(model, Path("/data.yaml"), 640, 8, "val", Path("/runs"), "exp_val")
    """
    return model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        batch=batch,
        split=split,
        project=str(project),
        name=name,
        exist_ok=exist_ok,
        plots=plots,
    )


def metrics_to_row(exp: YoloExperiment, metrics_obj) -> dict:
    """Convert a YOLO metrics object into one comparison-table row.

    Parameters
    ----------
    exp : YoloExperiment
        Experiment configuration.
    metrics_obj : Any
        Ultralytics metrics object returned by `model.val()`.

    Returns
    -------
    dict
        Dictionary containing experiment metadata and summary metrics.

    Example
    -------
    >>> # row = metrics_to_row(exp, metrics_obj)
    """
    return {
        "Model": exp.key,
        "Title": exp.title,
        "Epochs": exp.epochs,
        "Image Size": exp.imgsz,
        "Batch": exp.batch,
        "Precision": float(metrics_obj.box.mp),
        "Recall": float(metrics_obj.box.mr),
        "mAP50": float(metrics_obj.box.map50),
        "mAP50-95": float(metrics_obj.box.map),
    }


def build_comparison_dataframe(
    experiments: Dict[str, YoloExperiment],
    metrics_by_experiment: Dict[str, Any],
) -> pd.DataFrame:
    """Build a comparison DataFrame from multiple YOLO evaluation results.

    Parameters
    ----------
    experiments : Dict[str, YoloExperiment]
        Mapping from experiment key to experiment definition.
    metrics_by_experiment : Dict[str, Any]
        Mapping from experiment key to Ultralytics metrics object.

    Returns
    -------
    pd.DataFrame
        Sorted comparison table.

    Example
    -------
    >>> # df = build_comparison_dataframe(EXPERIMENTS, metrics_by_experiment)
    """
    rows = []
    for exp_key, metrics_obj in metrics_by_experiment.items():
        rows.append(metrics_to_row(experiments[exp_key], metrics_obj))

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    return df.sort_values(
        by=["mAP50", "mAP50-95"],
        ascending=False,
    ).reset_index(drop=True)


def plot_bar_metrics(df: pd.DataFrame, metrics: List[str]) -> None:
    """Plot grouped bar charts for selected YOLO metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Comparison DataFrame containing one row per experiment.
    metrics : List[str]
        Metric column names to plot.

    Returns
    -------
    None

    Example
    -------
    >>> # plot_bar_metrics(df, ["mAP50", "mAP50-95", "Precision", "Recall"])
    """
    plt.figure(figsize=(12, 6))
    x = range(len(df))
    width = 0.18

    for i, metric in enumerate(metrics):
        plt.bar([j + i * width for j in x], df[metric], width=width, label=metric)

    plt.xticks(
        [j + width * (len(metrics) - 1) / 2 for j in x],
        df["Model"],
        rotation=25,
    )
    plt.ylabel("Score")
    plt.title("YOLO Experiment Comparison on Validation Set")
    plt.legend()
    plt.tight_layout()
    plt.show()


def show_results_image(results_png: str | Path | None, title: str) -> None:
    """Display a saved YOLO `results.png` image.

    Parameters
    ----------
    results_png : str | Path | None
        Path to the image file.
    title : str
        Plot title.

    Returns
    -------
    None

    Example
    -------
    >>> # show_results_image("/tmp/results.png", "Experiment curves")
    """
    if results_png is None:
        print(f"No results image configured for {title}")
        return

    results_path = Path(results_png)
    print("Results image:", results_path)

    if not results_path.exists():
        print("File not found.")
        return

    img = Image.open(results_path)
    plt.figure(figsize=(14, 8))
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.show()


def save_metrics_json(metrics: dict, output_path: Path) -> None:
    """Save metrics to a JSON file.

    Parameters
    ----------
    metrics : dict
        Metrics dictionary to save.
    output_path : Path
        Output JSON path.

    Returns
    -------
    None

    Example
    -------
    >>> # save_metrics_json({"mAP50": 0.5}, Path("/tmp/metrics.json"))
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)


def final_test_metrics_to_dict(exp: YoloExperiment, metrics_obj) -> dict:
    """Convert final YOLO test metrics into a serializable dictionary.

    Parameters
    ----------
    exp : YoloExperiment
        Final selected YOLO experiment.
    metrics_obj : Any
        Ultralytics metrics object returned by test evaluation.

    Returns
    -------
    dict
        Final test metrics summary.

    Example
    -------
    >>> # result = final_test_metrics_to_dict(exp, final_test_metrics)
    """
    return {
        "Model": exp.title,
        "Precision": float(metrics_obj.box.mp),
        "Recall": float(metrics_obj.box.mr),
        "mAP50": float(metrics_obj.box.map50),
        "mAP50-95": float(metrics_obj.box.map),
    }


