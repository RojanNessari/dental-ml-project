"""Dataset cleaning helpers for the dental pathology project.

This module builds a cleaned YOLO-format dataset from the original source
dataset. It can remap class IDs, remove unwanted classes, copy images, copy
labels, and write a cleaned data YAML file.

Use this module when you want a clean, reproducible preprocessing stage that
the notebook can call in a few lines.

Example
-------
>>> from pathlib import Path
>>> from config import CLEAN_CLASS_NAMES
>>> class_mapping = {0: 0}
>>> # build_clean_dataset(Path("/src"), Path("/dst"), Path("/dst/data.yaml"), class_mapping, CLEAN_CLASS_NAMES)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional
import shutil
import yaml
from config import VALID_IMAGE_SUFFIXES
from io_utils import ensure_dir
from collections import Counter




def is_image_file(path: Path) -> bool:
    """Check whether a path looks like a supported image file.

    Parameters
    ----------
    path : Path
        File path to inspect.

    Returns
    -------
    bool
        True if the suffix is a supported image extension.

    Example
    -------
    >>> from pathlib import Path
    >>> is_image_file(Path("a.jpg"))
    True
    """
    return path.suffix.lower() in VALID_IMAGE_SUFFIXES


def remap_yolo_label_line(
    line: str,
    class_mapping: Dict[int, int],
) -> Optional[str]:
    """Remap a single YOLO label line from old class IDs to new class IDs.

    Parameters
    ----------
    line : str
        Raw label line from a YOLO `.txt` file. The line may contain either
        bounding-box or polygon-style coordinates.
    class_mapping : Dict[int, int]
        Mapping from original class ID to cleaned class ID.

    Returns
    -------
    Optional[str]
        Remapped label line, or None if the original class should be removed or
        the line is malformed.

    Example
    -------
    >>> remap_yolo_label_line("2 0.5 0.5 0.2 0.2", {2: 1})
    '1 0.5 0.5 0.2 0.2'
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None

    try:
        old_cls = int(float(parts[0]))
    except ValueError:
        return None

    if old_cls not in class_mapping:
        return None

    new_cls = class_mapping[old_cls]
    return " ".join([str(new_cls)] + parts[1:])


def clean_label_file(
    input_path: Path,
    output_path: Path,
    class_mapping: Dict[int, int],
) -> int:
    """Clean and remap one YOLO label file.

    Parameters
    ----------
    input_path : Path
        Input label file.
    output_path : Path
        Output cleaned label file.
    class_mapping : Dict[int, int]
        Mapping from original class IDs to cleaned class IDs.

    Returns
    -------
    int
        Number of kept annotation lines written to the output file.

    Example
    -------
    >>> # clean_label_file(Path("in.txt"), Path("out.txt"), {0: 0})
    """
    kept_lines: List[str] = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            cleaned = remap_yolo_label_line(line, class_mapping)
            if cleaned is not None:
                kept_lines.append(cleaned)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        if kept_lines:
            f.write("\n".join(kept_lines) + "\n")

    return len(kept_lines)


def build_clean_dataset(
    src_root: Path,
    dst_root: Path,
    yaml_path: Path,
    class_mapping: Dict[int, int],
    class_names: List[str],
    split_name_mapping: Optional[Dict[str, str]] = None,
) -> dict:
    """Build a cleaned YOLO dataset and write a YAML file.

    Parameters
    ----------
    src_root : Path
        Root directory of the original YOLO dataset.
    dst_root : Path
        Output root directory for the cleaned dataset.
    yaml_path : Path
        Output path for the cleaned YAML file.
    class_mapping : Dict[int, int]
        Mapping from original class IDs to cleaned class IDs.
    class_names : List[str]
        Final cleaned class names, indexed by cleaned class ID.
    split_name_mapping : Optional[Dict[str, str]]
        Mapping from source split names to destination split names. For example,
        `{"train": "train", "valid": "valid", "test": "test"}`.

    Returns
    -------
    dict
        Summary counts for copied images and kept labels.

    Example
    -------
    >>> # build_clean_dataset(Path("/src"), Path("/dst"), Path("/dst/data.yaml"), {0: 0}, ["A"])
    """
    if split_name_mapping is None:
        split_name_mapping = {"train": "train", "valid": "valid", "test": "test"}

    summary = {
        "images_copied": 0,
        "labels_processed": 0,
        "annotations_kept": 0,
        "split_stats": {},
    }

    for src_split, dst_split in split_name_mapping.items():
        src_img = src_root / src_split / "images"
        src_lbl = src_root / src_split / "labels"
        dst_img = dst_root / dst_split / "images"
        dst_lbl = dst_root / dst_split / "labels"

        ensure_dir(dst_img)
        ensure_dir(dst_lbl)

        split_images = 0
        split_labels = 0
        split_annotations = 0

        for img_file in sorted(src_img.iterdir()):
            if img_file.is_file() and is_image_file(img_file):
                shutil.copy2(img_file, dst_img / img_file.name)
                split_images += 1
                summary["images_copied"] += 1

        for label_file in sorted(src_lbl.glob("*.txt")):
            written = clean_label_file(label_file, dst_lbl / label_file.name, class_mapping)
            split_labels += 1
            split_annotations += written
            summary["labels_processed"] += 1
            summary["annotations_kept"] += written

        summary["split_stats"][dst_split] = {
            "images": split_images,
            "label_files": split_labels,
            "annotations_kept": split_annotations,
        }

    yaml_dict = {
        "train": str((dst_root / "train" / "images").resolve()),
        "val": str((dst_root / "valid" / "images").resolve()),
        "test": str((dst_root / "test" / "images").resolve()),
        "nc": len(class_names),
        "names": class_names,
    }

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_dict, f, sort_keys=False, allow_unicode=True)

    return summary

##--------------------------------------------------------##


def collect_class_id_summary(dataset_root: Path) -> Dict[str, list[int]]:
    """
    Collect the sorted list of class IDs present in each dataset split.

    Parameters
    ----------
    dataset_root : Path
        Root directory of a YOLO-format dataset containing split folders such as
        `train`, `valid`, and `test`.

    Returns
    -------
    Dict[str, list[int]]
        Dictionary mapping each split name to the sorted list of class IDs
        observed in its label files.

    Example
    -------
    >>> summary = collect_class_id_summary(Path("/kaggle/working/project_artifacts/dental-vzrad2-yolo26-clean"))
    >>> print(summary["train"])
    """
    split_summary: Dict[str, list[int]] = {}

    for split in ["train", "valid", "test"]:
        labels_dir = dataset_root / split / "labels"
        class_ids = set()

        if not labels_dir.exists():
            split_summary[split] = []
            continue

        for label_file in labels_dir.glob("*.txt"):
            with open(label_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        cls_id = int(float(parts[0]))
                        class_ids.add(cls_id)
                    except ValueError:
                        continue

        split_summary[split] = sorted(class_ids)

    return split_summary


def print_class_id_summary(dataset_root: Path, title: str = "Class-ID summary") -> None:
    """
    Print the class IDs found in each dataset split.

    Parameters
    ----------
    dataset_root : Path
        Root directory of a YOLO-format dataset.
    title : str
        Title printed before the summary.

    Returns
    -------
    None

    Example
    -------
    >>> print_class_id_summary(Path("/kaggle/working/project_artifacts/dental-vzrad2-yolo26-clean"))
    """
    summary = collect_class_id_summary(dataset_root)

    print(title)
    for split in ["train", "valid", "test"]:
        print(f"{split} class IDs: {summary.get(split, [])}")





def collect_class_frequency(dataset_root: Path) -> Dict[str, Counter]:
    """
    Count the number of annotation instances per class for each dataset split.

    Parameters
    ----------
    dataset_root : Path
        Root directory of a YOLO-format dataset.

    Returns
    -------
    Dict[str, Counter]
        Dictionary mapping each split to a Counter of class frequencies.

    Example
    -------
    >>> freq = collect_class_frequency(Path("/kaggle/working/project_artifacts/dental-vzrad2-yolo26-clean"))
    >>> print(freq["train"])
    """
    split_freq: Dict[str, Counter] = {}

    for split in ["train", "valid", "test"]:
        labels_dir = dataset_root / split / "labels"
        counter = Counter()

        if not labels_dir.exists():
            split_freq[split] = counter
            continue

        for label_file in labels_dir.glob("*.txt"):
            with open(label_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        cls_id = int(float(parts[0]))
                        counter[cls_id] += 1
                    except ValueError:
                        continue

        split_freq[split] = counter

    return split_freq

