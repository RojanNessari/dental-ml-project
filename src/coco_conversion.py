"""Convert cleaned polygon-style YOLO labels into COCO-format annotations.

The dental dataset uses polygon-style YOLO labels rather than plain
`class x_center y_center w h` bounding boxes. This module converts those
polygons into COCO bounding boxes and segmentation fields.

Example
-------
>>> from pathlib import Path
>>> from config import CLEAN_CLASS_NAMES
>>> # convert_polygon_yolo_dataset_to_coco(Path("/yolo"), Path("/coco"), CLEAN_CLASS_NAMES)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence
from PIL import Image
import shutil

from config import VALID_IMAGE_SUFFIXES
from io_utils import ensure_dir, reset_dir, write_json


def polygon_to_coco_bbox(coords: Sequence[float], image_width: int, image_height: int) -> List[float]:
    """Convert normalized polygon points to a COCO bbox in pixels.

    Parameters
    ----------
    coords : Sequence[float]
        Polygon coordinates in normalized form:
        `[x1, y1, x2, y2, ..., xn, yn]`.
    image_width : int
        Image width in pixels.
    image_height : int
        Image height in pixels.

    Returns
    -------
    List[float]
        Bounding box in COCO format: `[x_min, y_min, width, height]`.

    Example
    -------
    >>> bbox = polygon_to_coco_bbox([0.1, 0.1, 0.2, 0.2, 0.2, 0.1], 100, 100)
    >>> len(bbox)
    4
    """
    xs = coords[0::2]
    ys = coords[1::2]

    x_min = max(0.0, min(xs) * image_width)
    y_min = max(0.0, min(ys) * image_height)
    x_max = min(float(image_width), max(xs) * image_width)
    y_max = min(float(image_height), max(ys) * image_height)

    return [x_min, y_min, max(0.0, x_max - x_min), max(0.0, y_max - y_min)]


def polygon_to_coco_segmentation(
    coords: Sequence[float],
    image_width: int,
    image_height: int,
) -> List[List[float]]:
    """Convert normalized polygon points to COCO segmentation coordinates.

    Parameters
    ----------
    coords : Sequence[float]
        Polygon coordinates in normalized form.
    image_width : int
        Image width in pixels.
    image_height : int
        Image height in pixels.

    Returns
    -------
    List[List[float]]
        COCO segmentation list.

    Example
    -------
    >>> seg = polygon_to_coco_segmentation([0.1, 0.1, 0.2, 0.2], 100, 100)
    >>> isinstance(seg, list)
    True
    """
    seg = []
    for i, value in enumerate(coords):
        seg.append(value * image_width if i % 2 == 0 else value * image_height)
    return [seg]


def copy_images_to_coco_structure(clean_root: Path, coco_root: Path, splits: List[str]) -> None:
    """Copy images into a COCO-style folder structure.

    Parameters
    ----------
    clean_root : Path
        Root of the cleaned YOLO dataset.
    coco_root : Path
        Output COCO root directory.
    splits : List[str]
        Dataset split names, for example `["train", "valid", "test"]`.

    Returns
    -------
    None

    Example
    -------
    >>> # copy_images_to_coco_structure(Path("/clean"), Path("/coco"), ["train"])
    """
    reset_dir(coco_root)
    for split in splits:
        ensure_dir(coco_root / split / "images")
    ensure_dir(coco_root / "annotations")

    for split in splits:
        src_img_dir = clean_root / split / "images"
        dst_img_dir = coco_root / split / "images"

        for img_file in src_img_dir.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in VALID_IMAGE_SUFFIXES:
                shutil.copy2(img_file, dst_img_dir / img_file.name)


def convert_polygon_yolo_split_to_coco(
    yolo_root: Path,
    coco_root: Path,
    split: str,
    class_names: List[str],
) -> dict:
    """Convert one cleaned YOLO split into a COCO annotation file.

    Parameters
    ----------
    yolo_root : Path
        Root of the cleaned YOLO dataset.
    coco_root : Path
        Root of the output COCO dataset.
    split : str
        Split name such as `"train"`, `"valid"`, or `"test"`.
    class_names : List[str]
        Final class names.

    Returns
    -------
    dict
        Summary information including number of images and annotations.

    Example
    -------
    >>> # convert_polygon_yolo_split_to_coco(Path("/yolo"), Path("/coco"), "train", ["A"])
    """
    images_dir = yolo_root / split / "images"
    labels_dir = yolo_root / split / "labels"
    output_json_path = coco_root / "annotations" / f"instances_{split}.json"

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(class_names)],
    }

    image_id = 1
    annotation_id = 1
    skipped_lines = 0
    missing_label_files = 0

    image_files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in VALID_IMAGE_SUFFIXES])

    for image_path in image_files:
        with Image.open(image_path) as img:
            width, height = img.size

        coco["images"].append({
            "id": image_id,
            "file_name": image_path.name,
            "width": width,
            "height": height,
        })

        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            missing_label_files += 1
            image_id += 1
            continue

        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 7:
                    skipped_lines += 1
                    continue

                try:
                    class_id = int(float(parts[0]))
                    coords = list(map(float, parts[1:]))
                except ValueError:
                    skipped_lines += 1
                    continue

                if len(coords) % 2 != 0:
                    skipped_lines += 1
                    continue

                if not all(0.0 <= c <= 1.0 for c in coords):
                    skipped_lines += 1
                    continue

                bbox = polygon_to_coco_bbox(coords, width, height)
                area = bbox[2] * bbox[3]
                if area <= 0:
                    skipped_lines += 1
                    continue

                segmentation = polygon_to_coco_segmentation(coords, width, height)

                coco["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": bbox,
                    "area": area,
                    "segmentation": segmentation,
                    "iscrowd": 0,
                })
                annotation_id += 1

        image_id += 1

    write_json(coco, output_json_path, indent=2)

    return {
        "split": split,
        "images": len(coco["images"]),
        "annotations": len(coco["annotations"]),
        "categories": len(coco["categories"]),
        "missing_label_files": missing_label_files,
        "skipped_lines": skipped_lines,
        "json_path": str(output_json_path),
    }


def convert_polygon_yolo_dataset_to_coco(
    yolo_root: Path,
    coco_root: Path,
    class_names: List[str],
    splits: List[str] | None = None,
) -> List[dict]:
    """Convert the full cleaned YOLO dataset into a COCO-style dataset.

    Parameters
    ----------
    yolo_root : Path
        Root of the cleaned YOLO dataset.
    coco_root : Path
        Output COCO root directory.
    class_names : List[str]
        Final class names.
    splits : List[str] | None
        Splits to convert. If None, defaults to `["train", "valid", "test"]`.

    Returns
    -------
    List[dict]
        Per-split conversion summaries.

    Example
    -------
    >>> # convert_polygon_yolo_dataset_to_coco(Path("/yolo"), Path("/coco"), ["A"])
    """
    if splits is None:
        splits = ["train", "valid", "test"]

    copy_images_to_coco_structure(yolo_root, coco_root, splits)
    summaries = []
    for split in splits:
        summaries.append(convert_polygon_yolo_split_to_coco(yolo_root, coco_root, split, class_names))
    return summaries
