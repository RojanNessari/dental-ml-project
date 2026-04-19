"""DETR experiment helpers for the dental pathology project.

This module keeps the DETR pipeline modular and notebook-friendly. It handles
runtime preparation, repository patching, training, evaluation, and metric
extraction so that the notebook can remain focused on reporting and analysis.

Example
-------
>>> from pathlib import Path
>>> # runtime = prepare_detr_runtime(...)
>>> # print(runtime["trainval_root"])
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any
import json
import re
import shutil
import subprocess


@dataclass
class DetrExperiment:
    """Container for one DETR experiment configuration.

    Parameters
    ----------
    key : str
        Short experiment key.
    title : str
        Human-readable experiment title.
    epochs : int
        Number of training epochs.
    batch_size : int
        DETR batch size.
    num_workers : int
        Number of dataloader workers.
    output_dir : str
        Output directory where checkpoints/logs are stored.
    train_enabled : bool
        Whether this run should train when training is enabled.
    eval_enabled : bool
        Whether this run should be evaluated.
    """
    key: str
    title: str
    epochs: int
    batch_size: int
    num_workers: int
    output_dir: str
    train_enabled: bool = False
    eval_enabled: bool = True


def run_shell_command(command: str, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    """Run a shell command and capture stdout/stderr."""
    return subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd is not None else None,
    )


def safe_reset_dir(path: Path) -> Path:
    """Delete and recreate a directory."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_detr_dataset_structure(
    coco_root: Path,
    detr_root: Path,
) -> Dict[str, str]:
    """Create official DETR-style dataset folders from the cleaned COCO dataset.

    Parameters
    ----------
    coco_root : Path
        Root of the cleaned COCO dataset.
    detr_root : Path
        Root directory where DETR-ready folders will be created.

    Returns
    -------
    Dict[str, str]
        Paths to the train/validation and test roots.

    Example
    -------
    >>> # prepare_detr_dataset_structure(Path("/coco"), Path("/detr_data"))
    """
    coco_ann = coco_root / "annotations"
    trainval_root = detr_root / "coco_trainval"
    test_root = detr_root / "coco_test"

    safe_reset_dir(trainval_root)
    safe_reset_dir(test_root)

    (trainval_root / "annotations").mkdir(parents=True, exist_ok=True)
    (test_root / "annotations").mkdir(parents=True, exist_ok=True)

    shutil.copytree(coco_root / "train" / "images", trainval_root / "train2017")
    shutil.copytree(coco_root / "valid" / "images", trainval_root / "val2017")

    shutil.copy2(coco_ann / "instances_train.json",
                 trainval_root / "annotations" / "instances_train2017.json")
    shutil.copy2(coco_ann / "instances_valid.json",
                 trainval_root / "annotations" / "instances_val2017.json")

    shutil.copytree(coco_root / "test" / "images", test_root / "val2017")
    shutil.copy2(coco_ann / "instances_test.json",
                 test_root / "annotations" / "instances_val2017.json")

    return {
        "trainval_root": str(trainval_root),
        "test_root": str(test_root),
    }


def clone_detr_repo(local_detr_repo: Path) -> None:
    """Clone the official DETR repository if it is not already present."""
    if local_detr_repo.exists():
        print("DETR repo already exists.")
        return

    cmd = f'git clone https://github.com/facebookresearch/detr.git "{local_detr_repo}"'
    result = run_shell_command(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to clone DETR repo:\n{result.stderr}")

    print("DETR repo cloned successfully.")


def download_pretrained_checkpoint(local_detr_repo: Path) -> Path:
    """Download the pretrained DETR-R50 checkpoint if needed."""
    ckpt = local_detr_repo / "detr-r50-e632da11.pth"
    if ckpt.exists():
        print("Checkpoint already downloaded.")
        return ckpt

    cmd = (
        f'wget -O "{ckpt}" '
        f'"https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth"'
    )
    result = run_shell_command(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to download checkpoint:\n{result.stderr}")

    print("Checkpoint downloaded successfully.")
    return ckpt


def patch_detr_main(local_detr_repo: Path, custom_num_classes: int = 13) -> None:
    """Patch DETR main.py for custom class count and frequent checkpoint saving."""
    main_py = local_detr_repo / "main.py"
    text = main_py.read_text(encoding="utf-8")

    if "--custom_num_classes" not in text:
        text = text.replace(
            "parser.add_argument('--dataset_file', default='coco')",
            "parser.add_argument('--dataset_file', default='coco')\n"
            f"    parser.add_argument('--custom_num_classes', default={custom_num_classes}, type=int)"
        )

    text = text.replace(
        "num_classes = 91 if args.dataset_file != 'coco_panoptic' else 250",
        "num_classes = args.custom_num_classes if args.dataset_file != 'coco_panoptic' else 250"
    )

    old_load = "checkpoint = torch.load(args.resume, map_location='cpu')"
    new_load = "checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)"
    if old_load in text:
        text = text.replace(old_load, new_load)

    old_block = (
        "if args.output_dir:\n"
        "            checkpoint_paths = [output_dir / 'checkpoint.pth']\n"
        "            # extra checkpoint before LR drop and every 100 epochs\n"
        "            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:\n"
        "                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')"
    )
    new_block = (
        "if args.output_dir:\n"
        "            checkpoint_paths = [output_dir / 'checkpoint.pth']\n"
        "            if (epoch + 1) % 5 == 0:\n"
        "                checkpoint_paths.append(output_dir / f'checkpoint_epoch_{epoch + 1:03}.pth')\n"
        "            # extra checkpoint before LR drop and every 100 epochs\n"
        "            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:\n"
        "                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')"
    )
    if old_block in text:
        text = text.replace(old_block, new_block)

    main_py.write_text(text, encoding="utf-8")
    print("Patched main.py")


def patch_pycocotools_numpy() -> None:
    """Patch pycocotools for NumPy float deprecation."""
    candidates = [
        Path("/usr/local/lib/python3.10/dist-packages/pycocotools/cocoeval.py"),
        Path("/usr/local/lib/python3.11/dist-packages/pycocotools/cocoeval.py"),
        Path("/usr/local/lib/python3.12/dist-packages/pycocotools/cocoeval.py"),
    ]

    cocoeval_path = None
    for candidate in candidates:
        if candidate.exists():
            cocoeval_path = candidate
            break

    if cocoeval_path is None:
        print("Could not find cocoeval.py to patch.")
        return

    text = cocoeval_path.read_text(encoding="utf-8")
    text = text.replace("dtype=np.float", "dtype=float")
    cocoeval_path.write_text(text, encoding="utf-8")
    print("Patched:", cocoeval_path)


def prepare_detr_runtime(
    coco_root: Path,
    detr_data_root: Path,
    local_detr_repo: Path,
    custom_num_classes: int = 13,
) -> Dict[str, str]:
    """Prepare all DETR runtime dependencies in one call.

    This function:
    1. creates DETR-compatible dataset folders
    2. clones the DETR repository
    3. downloads the pretrained checkpoint
    4. patches DETR for custom classes and compatibility issues

    Returns
    -------
    Dict[str, str]
        Dictionary containing the main runtime paths.
    """
    dataset_paths = prepare_detr_dataset_structure(coco_root, detr_data_root)
    clone_detr_repo(local_detr_repo)
    ckpt = download_pretrained_checkpoint(local_detr_repo)
    patch_detr_main(local_detr_repo, custom_num_classes=custom_num_classes)
    patch_pycocotools_numpy()

    return {
        **dataset_paths,
        "local_detr_repo": str(local_detr_repo),
        "pretrained_ckpt": str(ckpt),
    }


def train_detr(
    local_detr_repo: Path,
    trainval_root: Path,
    output_dir: Path,
    pretrained_ckpt: Path,
    epochs: int,
    batch_size: int,
    num_workers: int,
    custom_num_classes: int = 13,
) -> subprocess.CompletedProcess:
    """Train DETR on the train/validation dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = (
        f'python main.py '
        f'--coco_path "{trainval_root}" '
        f'--output_dir "{output_dir}" '
        f'--resume "{pretrained_ckpt}" '
        f'--dataset_file coco '
        f'--custom_num_classes {custom_num_classes} '
        f'--epochs {epochs} '
        f'--batch_size {batch_size} '
        f'--num_workers {num_workers}'
    )
    return run_shell_command(cmd, cwd=local_detr_repo)


def evaluate_detr(
    local_detr_repo: Path,
    coco_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    batch_size: int,
    custom_num_classes: int = 13,
) -> subprocess.CompletedProcess:
    """Evaluate a DETR checkpoint on a specified COCO-style split."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = (
        f'python main.py '
        f'--coco_path "{coco_path}" '
        f'--output_dir "{output_dir}" '
        f'--resume "{checkpoint_path}" '
        f'--dataset_file coco '
        f'--custom_num_classes {custom_num_classes} '
        f'--batch_size {batch_size} '
        f'--eval'
    )
    return run_shell_command(cmd, cwd=local_detr_repo)


def parse_detr_eval_output(output_text: str) -> Dict[str, float | None]:
    """Extract AP metrics from DETR evaluation stdout."""
    map_5095 = re.search(r"Average Precision.*IoU=0.50:0.95.*= ([0-9.]+)", output_text)
    map_50 = re.search(r"Average Precision.*IoU=0.50\s+\|.*= ([0-9.]+)", output_text)
    map_75 = re.search(r"Average Precision.*IoU=0.75\s+\|.*= ([0-9.]+)", output_text)

    return {
        "mAP@0.5:0.95": float(map_5095.group(1)) if map_5095 else None,
        "mAP@0.5": float(map_50.group(1)) if map_50 else None,
        "mAP@0.75": float(map_75.group(1)) if map_75 else None,
    }


def save_metrics_json(metrics: dict, output_path: Path) -> None:
    """Save metrics to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)


def checkpoint_exists(output_dir: Path) -> bool:
    """Check whether the main DETR checkpoint exists."""
    return (output_dir / "checkpoint.pth").exists()


