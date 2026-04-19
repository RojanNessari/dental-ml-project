"""Clinical note generation helpers for dental object detection.

This module converts detection outputs into simple structured quadrant-based
clinical summaries. It is designed to support reproducible reporting without
depending on an external language-model API.

The goal is to assist interpretation and presentation, not to provide medical
diagnosis.

Example
-------
>>> detections = [
...     {"class_name": "Caries", "bbox": [100, 200, 30, 40], "score": 0.82},
...     {"class_name": "Crown", "bbox": [400, 180, 50, 60], "score": 0.91},
... ]
>>> note = generate_structured_clinical_note(detections, image_width=800, image_height=400)
>>> print(note["full_note"])
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Tuple


def bbox_center_xy(bbox: List[float]) -> Tuple[float, float]:
    """
    Compute the center point of a bounding box.

    Parameters
    ----------
    bbox : List[float]
        Bounding box in COCO/XYWH format: [x, y, width, height].

    Returns
    -------
    Tuple[float, float]
        Bounding box center coordinates (cx, cy).

    Example
    -------
    >>> bbox_center_xy([10, 20, 30, 40])
    (25.0, 40.0)
    """
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0


def assign_quadrant_from_bbox(
    bbox: List[float],
    image_width: int,
    image_height: int,
) -> str:
    """
    Assign a detection to one of four image quadrants based on bbox center.

    Parameters
    ----------
    bbox : List[float]
        Bounding box in [x, y, width, height] format.
    image_width : int
        Image width in pixels.
    image_height : int
        Image height in pixels.

    Returns
    -------
    str
        One of:
        - "upper_right"
        - "upper_left"
        - "lower_right"
        - "lower_left"

    Example
    -------
    >>> assign_quadrant_from_bbox([100, 50, 20, 20], 800, 400)
    'upper_left'
    """
    cx, cy = bbox_center_xy(bbox)

    left_half = cx < image_width / 2
    upper_half = cy < image_height / 2

    if upper_half and not left_half:
        return "upper_right"
    if upper_half and left_half:
        return "upper_left"
    if not upper_half and not left_half:
        return "lower_right"
    return "lower_left"


def normalize_detection_record(det: Dict) -> Dict:
    """
    Normalize a detection dictionary into a consistent format.

    Expected keys include:
    - class_name
    - bbox
    - score (optional)

    Parameters
    ----------
    det : Dict
        Raw detection dictionary.

    Returns
    -------
    Dict
        Normalized detection dictionary.

    Example
    -------
    >>> normalize_detection_record({"class_name": "Caries", "bbox": [1,2,3,4]})["class_name"]
    'Caries'
    """
    return {
        "class_name": det["class_name"],
        "bbox": list(det["bbox"]),
        "score": float(det.get("score", 0.0)),
    }


def group_detections_by_quadrant(
    detections: List[Dict],
    image_width: int,
    image_height: int,
) -> Dict[str, List[Dict]]:
    """
    Group detections by anatomical image quadrant.

    Parameters
    ----------
    detections : List[Dict]
        List of detection dictionaries with at least `class_name` and `bbox`.
    image_width : int
        Image width in pixels.
    image_height : int
        Image height in pixels.

    Returns
    -------
    Dict[str, List[Dict]]
        Dictionary mapping quadrant name to detections.

    Example
    -------
    >>> group_detections_by_quadrant(
    ...     [{"class_name": "Caries", "bbox": [10, 10, 20, 20]}],
    ...     800, 400
    ... )["upper_left"][0]["class_name"]
    'Caries'
    """
    grouped = {
        "upper_right": [],
        "upper_left": [],
        "lower_right": [],
        "lower_left": [],
    }

    for raw_det in detections:
        det = normalize_detection_record(raw_det)
        quadrant = assign_quadrant_from_bbox(det["bbox"], image_width, image_height)
        grouped[quadrant].append(det)

    return grouped


def summarize_quadrant_detections(detections: List[Dict]) -> str:
    """
    Create a short summary sentence for one quadrant.

    Parameters
    ----------
    detections : List[Dict]
        Detections in a single quadrant.

    Returns
    -------
    str
        Human-readable summary sentence.

    Example
    -------
    >>> summarize_quadrant_detections([{"class_name": "Caries", "bbox": [1,2,3,4], "score": 0.8}])
    '1 caries detected.'
    """
    if not detections:
        return "No major findings detected."

    counts = Counter(det["class_name"] for det in detections)

    parts = []
    for class_name, count in sorted(counts.items()):
        label = class_name.lower()
        parts.append(f"{count} {label}" + ("" if count == 1 else "s"))

    return ", ".join(parts).capitalize() + " detected."


def generate_structured_clinical_note(
    detections: List[Dict],
    image_width: int,
    image_height: int,
) -> Dict[str, str]:
    """
    Generate quadrant-based and overall note text from detections.

    Parameters
    ----------
    detections : List[Dict]
        Detection dictionaries with fields like `class_name`, `bbox`, and
        optionally `score`.
    image_width : int
        Image width in pixels.
    image_height : int
        Image height in pixels.

    Returns
    -------
    Dict[str, str]
        Dictionary containing:
        - `upper_right`
        - `upper_left`
        - `lower_right`
        - `lower_left`
        - `full_note`

    Example
    -------
    >>> note = generate_structured_clinical_note([], 800, 400)
    >>> "decision support" in note["full_note"].lower()
    True
    """
    grouped = group_detections_by_quadrant(detections, image_width, image_height)

    qmap = {
        "upper_right": summarize_quadrant_detections(grouped["upper_right"]),
        "upper_left": summarize_quadrant_detections(grouped["upper_left"]),
        "lower_right": summarize_quadrant_detections(grouped["lower_right"]),
        "lower_left": summarize_quadrant_detections(grouped["lower_left"]),
    }

    total_counts = Counter(det["class_name"] for det in detections)

    if total_counts:
        most_common = ", ".join(
            f"{count} {name.lower()}" + ("" if count == 1 else "s")
            for name, count in total_counts.most_common(3)
        )
        overview = (
            f"Automated radiographic summary: findings were detected across the image. "
            f"The most frequent detected findings were {most_common}. "
            f"This summary is intended as decision support only and does not replace professional diagnosis."
        )
    else:
        overview = (
            "Automated radiographic summary: no major findings were detected by the model. "
            "This summary is intended as decision support only and does not replace professional diagnosis."
        )

    full_note = (
        f"Upper Right: {qmap['upper_right']} "
        f"Upper Left: {qmap['upper_left']} "
        f"Lower Right: {qmap['lower_right']} "
        f"Lower Left: {qmap['lower_left']} "
        f"{overview}"
    )

    return {
        **qmap,
        "full_note": full_note,
    }


def yolo_result_to_detection_dicts(result, class_names: List[str], score_threshold: float = 0.25) -> List[Dict]:
    """
    Convert one Ultralytics YOLO result object into note-generation detections.

    Parameters
    ----------
    result : Any
        One item from YOLO inference output.
    class_names : List[str]
        Class names indexed by class ID.
    score_threshold : float
        Minimum confidence score to keep a detection.

    Returns
    -------
    List[Dict]
        List of detection dictionaries.

    Example
    -------
    >>> # detections = yolo_result_to_detection_dicts(results[0], class_names)
    """
    detections = []

    if result.boxes is None:
        return detections

    xywh = result.boxes.xywh.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()

    for box, class_id, score in zip(xywh, cls, conf):
        if float(score) < score_threshold:
            continue

        cx, cy, w, h = map(float, box)
        x = cx - w / 2.0
        y = cy - h / 2.0
        class_id = int(class_id)

        detections.append({
            "class_name": class_names[class_id],
            "bbox": [x, y, w, h],
            "score": float(score),
        })

    return detections


