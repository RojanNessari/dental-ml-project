"""Notebook-level convenience helpers.

These functions are intentionally lightweight. They are useful when the notebook
needs short summary blocks or quick checks without duplicating code.

Example
-------
>>> print_experiment_header("YOLO baseline")
"""

from __future__ import annotations

from pathlib import Path


def print_experiment_header(title: str) -> None:
    """Print a simple visual header for notebook sections.

    Parameters
    ----------
    title : str
        Header title to display.

    Returns
    -------
    None

    Example
    -------
    >>> print_experiment_header("Example")
    """
    print("=" * 60)
    print(title)
    print("=" * 60)


def describe_folder(path: Path) -> None:
    """Print a short directory tree summary.

    Parameters
    ----------
    path : Path
        Folder to inspect.

    Returns
    -------
    None

    Example
    -------
    >>> # describe_folder(Path("/tmp"))
    """
    print(f"Folder: {path}")
    if not path.exists():
        print("Does not exist.")
        return

    for item in sorted(path.iterdir()):
        kind = "DIR " if item.is_dir() else "FILE"
        print(f"{kind:4} {item.name}")
