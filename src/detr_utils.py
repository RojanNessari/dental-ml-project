"""Helpers for preparing and evaluating DETR experiments.

This module handles DETR-specific directory formatting and parsing of DETR
evaluation output.

Example
-------
>>> from pathlib import Path
>>> # prepare_detr_dataset_structure(Path("/coco"), Path("/detr_data"))
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict
import shutil
import subprocess
import re

from io_utils import reset_dir


def prepare_detr_dataset_structure(coco_root: Path, detr_root: Path) -> Dict[str, str]:
    """Create official DETR-style dataset folders from the cleaned COCO dataset.

    Parameters
    ----------
    coco_root : Path
        Root of the cleaned COCO dataset.
    detr_root : Path
        Root of the DETR-formatted dataset.

    Returns
    -------
    Dict[str, str]
        Paths to the train/validation and test DETR roots.

    Example
    -------
    >>> # prepare_detr_dataset_structure(Path("/coco"), Path("/detr_data"))
    """
    trainval_root = detr_root / "coco_trainval"
    test_root = detr_root / "coco_test"

    reset_dir(trainval_root)
    reset_dir(test_root)

    (trainval_root / "annotations").mkdir(parents=True, exist_ok=True)
    (test_root / "annotations").mkdir(parents=True, exist_ok=True)

    shutil.copytree(coco_root / "train" / "images", trainval_root / "train2017")
    shutil.copytree(coco_root / "valid" / "images", trainval_root / "val2017")

    shutil.copy2(coco_root / "annotations" / "instances_train.json",
                 trainval_root / "annotations" / "instances_train2017.json")
    shutil.copy2(coco_root / "annotations" / "instances_valid.json",
                 trainval_root / "annotations" / "instances_val2017.json")

    shutil.copytree(coco_root / "test" / "images", test_root / "val2017")
    shutil.copy2(coco_root / "annotations" / "instances_test.json",
                 test_root / "annotations" / "instances_val2017.json")

    return {
        "trainval_root": str(trainval_root),
        "test_root": str(test_root),
    }


def parse_detr_eval_output(output_text: str) -> Dict[str, float | None]:
    """Extract main AP metrics from DETR evaluation console output.

    Parameters
    ----------
    output_text : str
        Raw DETR evaluation stdout.

    Returns
    -------
    Dict[str, float | None]
        Dictionary containing AP metrics.

    Example
    -------
    >>> txt = "Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.123"
    >>> parse_detr_eval_output(txt)["mAP@0.5:0.95"]
    0.123
    """
    map_5095 = re.search(r"Average Precision.*IoU=0.50:0.95.*= ([0-9.]+)", output_text)
    map_50 = re.search(r"Average Precision.*IoU=0.50\s+\|.*= ([0-9.]+)", output_text)
    map_75 = re.search(r"Average Precision.*IoU=0.75\s+\|.*= ([0-9.]+)", output_text)

    return {
        "mAP@0.5:0.95": float(map_5095.group(1)) if map_5095 else None,
        "mAP@0.5": float(map_50.group(1)) if map_50 else None,
        "mAP@0.75": float(map_75.group(1)) if map_75 else None,
    }


def run_shell_command(command: str) -> subprocess.CompletedProcess:
    """Run a shell command and capture stdout/stderr.

    Parameters
    ----------
    command : str
        Shell command to execute.

    Returns
    -------
    subprocess.CompletedProcess
        Completed process object.

    Example
    -------
    >>> result = run_shell_command("echo hello")
    >>> "hello" in result.stdout
    True
    """
    return subprocess.run(command, shell=True, capture_output=True, text=True)
