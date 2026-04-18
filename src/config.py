"""Project configuration constants.

This module stores reusable class names and default settings shared across the
project. Keeping these values in one place makes the code easier to maintain
and reduces the chance of using inconsistent mappings across notebooks.

Example
-------
>>> from config import CLEAN_CLASS_NAMES
>>> len(CLEAN_CLASS_NAMES)
13
"""

from __future__ import annotations

CLEAN_CLASS_NAMES = [
    "Caries",
    "Crown",
    "Filling",
    "Implant",
    "Malaligned",
    "Mandibular Canal",
    "Missing teeth",
    "Periapical lesion",
    "Retained root",
    "Root Canal Treatment",
    "Root Piece",
    "Impacted tooth",
    "Maxillary sinus",
]

VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEFAULT_YOLO_NAMES = {i: name for i, name in enumerate(CLEAN_CLASS_NAMES)}
