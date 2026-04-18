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
from typing import Dict, List, Optional
from config import VALID_IMAGE_SUFFIXES
from collections import Counter




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


##-----------------VALIDATE & VISUALIZATION HELPERS-----------------##
"""Dataset validation and visualization helpers.


This module contains functions for checking the structural correctness of a
cleaned YOLO-format dataset and for visualizing polygon annotations directly
on panoramic dental X-ray images.


These utilities are useful for verifying that preprocessing was applied
correctly before training detection models.


Example
-------
>>> from pathlib import Path
>>> summary = validate_image_label_pairs(
...     Path("/kaggle/working/project_artifacts/dental-vzrad2-yolo26-clean")
... )
>>> print(summary["train"]["matched_pairs"])
"""




def validate_image_label_pairs(dataset_root: Path) -> Dict[str, dict]:
    """
    Validate that each image has a matching label file and each label file has
    a matching image file for every dataset split.


    Parameters
    ----------
    dataset_root : Path
        Root directory of the cleaned YOLO dataset. It should contain
        `train`, `valid`, and `test` subfolders, each with `images/` and
        `labels/` directories.


    Returns
    -------
    Dict[str, dict]
        Dictionary containing validation statistics for each split.


    Example
    -------
    >>> from pathlib import Path
    >>> summary = validate_image_label_pairs(
    ...     Path("/kaggle/working/project_artifacts/dental-vzrad2-yolo26-clean")
    ... )
    >>> print(summary["train"]["matched_pairs"])
    """
    summary: Dict[str, dict] = {}


    for split in ["train", "valid", "test"]:
        images_dir = dataset_root / split / "images"
        labels_dir = dataset_root / split / "labels"


        image_stems = {
            p.stem for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_IMAGE_SUFFIXES
        }
        label_stems = {
            p.stem for p in labels_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".txt"
        }


        matched_pairs = image_stems & label_stems
        missing_labels = image_stems - label_stems
        missing_images = label_stems - image_stems


        summary[split] = {
            "num_images": len(image_stems),
            "num_labels": len(label_stems),
            "matched_pairs": len(matched_pairs),
            "missing_labels": sorted(missing_labels),
            "missing_images": sorted(missing_images),
        }


    return summary




def print_pair_validation_summary(summary: Dict[str, dict]) -> None:
    """
    Print a readable summary of image-label pair validation.


    Parameters
    ----------
    summary : Dict[str, dict]
        Output of `validate_image_label_pairs`.


    Returns
    -------
    None


    Example
    -------
    >>> # summary = validate_image_label_pairs(Path("/data"))
    >>> # print_pair_validation_summary(summary)
    """
    for split, info in summary.items():
        print(f"--- {split.upper()} ---")
        print("Images:", info["num_images"])
        print("Labels:", info["num_labels"])
        print("Matched pairs:", info["matched_pairs"])
        print("Missing labels:", len(info["missing_labels"]))
        print("Missing images:", len(info["missing_images"]))
        print()




def get_image_label_pairs(images_dir: Path, labels_dir: Optional[Path] = None) -> List[Tuple[Path, Path]]:
    """
    Match image files with their corresponding YOLO label files.


    Parameters
    ----------
    images_dir : Path
        Directory containing images.
    labels_dir : Optional[Path]
        Directory containing labels. If None, assumes the standard YOLO
        structure and uses the sibling `labels/` directory of `images_dir`.


    Returns
    -------
    List[Tuple[Path, Path]]
        List of `(image_path, label_path)` pairs.


    Example
    -------
    >>> # pairs = get_image_label_pairs(Path("/data/train/images"))
    >>> # print(len(pairs))
    """
    if labels_dir is None:
        labels_dir = images_dir.parent / "labels"


    image_map = {
        p.stem: p
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_SUFFIXES
    }


    pairs: List[Tuple[Path, Path]] = []
    for label_file in labels_dir.glob("*.txt"):
        stem = label_file.stem
        if stem in image_map:
            pairs.append((image_map[stem], label_file))


    return pairs




def find_images_with_class(
    images_dir: Path,
    target_class_id: int,
    max_images: int = 6,
    random_seed: int = 42,
) -> List[Tuple[Path, Path]]:
    """
    Find a sample of images that contain at least one instance of a target class.


    Parameters
    ----------
    images_dir : Path
        Directory containing images.
    target_class_id : int
        Class ID to search for.
    max_images : int
        Maximum number of image-label pairs to return.
    random_seed : int
        Seed used to shuffle matched results reproducibly.


    Returns
    -------
    List[Tuple[Path, Path]]
        A list of `(image_path, label_path)` pairs containing the target class.


    Example
    -------
    >>> # samples = find_images_with_class(Path("/data/train/images"), 0, max_images=3)
    >>> # print(len(samples))
    """
    pairs = get_image_label_pairs(images_dir)
    matched: List[Tuple[Path, Path]] = []


    for image_path, label_path in pairs:
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    cls_id = int(float(parts[0]))
                except ValueError:
                    continue


                if cls_id == target_class_id:
                    matched.append((image_path, label_path))
                    break


    rng = random.Random(random_seed)
    rng.shuffle(matched)
    return matched[:max_images]




def load_image_and_polygon_points(
    image_path: Path,
    label_path: Path,
) -> Tuple[np.ndarray, int, int, List[Tuple[int, List[Tuple[float, float]]]]]:
    """
    Load an image and parse its polygon annotations.


    Parameters
    ----------
    image_path : Path
        Path to the image file.
    label_path : Path
        Path to the YOLO polygon label file.


    Returns
    -------
    Tuple[np.ndarray, int, int, List[Tuple[int, List[Tuple[float, float]]]]]
        Image array, width, height, and a list of annotations. Each annotation
        is represented as `(class_id, points)`.


    Example
    -------
    >>> # img, w, h, anns = load_image_and_polygon_points(Path("img.jpg"), Path("img.txt"))
    """
    image = Image.open(image_path)
    image_np = np.array(image)
    width, height = image.size


    annotations: List[Tuple[int, List[Tuple[float, float]]]] = []


    if not label_path.exists():
        return image_np, width, height, annotations


    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue


            try:
                class_id = int(float(parts[0]))
                coords = list(map(float, parts[1:]))
            except ValueError:
                continue


            if len(coords) % 2 != 0:
                continue


            points: List[Tuple[float, float]] = []
            for i in range(0, len(coords), 2):
                x = coords[i] * width
                y = coords[i + 1] * height
                points.append((x, y))


            annotations.append((class_id, points))


    return image_np, width, height, annotations




def plot_polygon_labels(
    image_path: Path,
    label_path: Path,
    class_names: List[str],
    title: Optional[str] = None,
    show_labels: bool = True,
    figsize: Tuple[int, int] = (12, 6),
) -> None:
    """
    Plot polygon annotations on top of an X-ray image.


    Parameters
    ----------
    image_path : Path
        Path to the image file.
    label_path : Path
        Path to the corresponding YOLO polygon label file.
    class_names : List[str]
        List of class names indexed by class ID.
    title : Optional[str]
        Plot title. If None, a default title is used.
    show_labels : bool
        Whether to draw text labels near the first point of each polygon.
    figsize : Tuple[int, int]
        Figure size.


    Returns
    -------
    None


    Example
    -------
    >>> # plot_polygon_labels(Path("img.jpg"), Path("img.txt"), ["A", "B"])
    """
    image_np, _, _, annotations = load_image_and_polygon_points(image_path, label_path)


    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image_np, cmap="gray")


    for class_id, points in annotations:
        if len(points) < 2:
            continue


        poly = Polygon(points, fill=False, linewidth=1.5)
        ax.add_patch(poly)


        if show_labels and points and 0 <= class_id < len(class_names):
            ax.text(points[0][0], points[0][1], class_names[class_id], fontsize=8)


    if title is None:
        title = f"Annotated image: {image_path.name}"


    ax.set_title(title)
    ax.axis("off")
    plt.show()




def visualize_class_samples(
    images_dir: Path,
    class_names: List[str],
    target_class_id: int,
    max_images: int = 3,
    split_name: str = "TRAIN",
) -> None:
    """
    Visualize a few images containing a specific class.


    Parameters
    ----------
    images_dir : Path
        Directory containing images.
    class_names : List[str]
        Dataset class names.
    target_class_id : int
        Class ID to highlight through sample selection.
    max_images : int
        Maximum number of examples to display.
    split_name : str
        Split name used in plot titles.


    Returns
    -------
    None


    Example
    -------
    >>> # visualize_class_samples(Path("/data/train/images"), class_names, 0, max_images=2)
    """
    samples = find_images_with_class(images_dir, target_class_id=target_class_id, max_images=max_images)


    for image_path, label_path in samples:
        plot_polygon_labels(
            image_path=image_path,
            label_path=label_path,
            class_names=class_names,
            title=f"{split_name} - polygon labels\n{image_path.name}",
        )



