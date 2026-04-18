"""Utility functions for file-system operations.

These functions keep repetitive path-management code out of the main notebook.

Example
-------
>>> from pathlib import Path
>>> from io_utils import ensure_dir
>>> p = ensure_dir(Path("/tmp/example_folder"))
>>> p.exists()
True
"""

from __future__ import annotations

from pathlib import Path
import json
import shutil
from typing import Any
import subprocess
import zipfile
import gdown


def ensure_dir(path: Path) -> Path:
    """Create a directory if it does not exist.

    Parameters
    ----------
    path : Path
        Directory path to create.

    Returns
    -------
    Path
        The same directory path.

    Example
    -------
    >>> from pathlib import Path
    >>> p = ensure_dir(Path("/tmp/my_dir"))
    >>> p.exists()
    True
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def reset_dir(path: Path) -> Path:
    """Delete and recreate a directory.

    Parameters
    ----------
    path : Path
        Directory path to reset.

    Returns
    -------
    Path
        The recreated directory.

    Example
    -------
    >>> from pathlib import Path
    >>> p = reset_dir(Path("/tmp/reset_me"))
    >>> p.exists()
    True
    """
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(data: Any, output_path: Path, indent: int = 2) -> None:
    """Write Python data to a JSON file.

    Parameters
    ----------
    data : Any
        Serializable Python object.
    output_path : Path
        Output JSON path.
    indent : int
        Number of spaces for pretty printing.

    Returns
    -------
    None

    Example
    -------
    >>> from pathlib import Path
    >>> write_json({"a": 1}, Path("/tmp/example.json"))
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def read_json(input_path: Path) -> Any:
    """Read a JSON file into Python data.

    Parameters
    ----------
    input_path : Path
        Path to a JSON file.

    Returns
    -------
    Any
        Parsed Python object.

    Example
    -------
    >>> from pathlib import Path
    >>> _ = write_json({"b": 2}, Path("/tmp/example_read.json"))
    >>> read_json(Path("/tmp/example_read.json"))["b"]
    2
    """
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_project_root(extract_parent: Path) -> Path:
    """
    Detect the most likely project root directory after extracting the zip file.

    The extracted archive may contain slightly different folder structures
    depending on how the zip was created. This function searches for the
    expected dataset folder and returns the directory that should be used
    as the main project root in later notebook cells.

    Parameters
    ----------
    extract_parent : Path
        Parent directory where the zip file has been extracted.

    Returns
    -------
    Path
        The detected project root directory. If no exact match is found,
        the function returns `extract_parent` as a fallback so the user
        can inspect the folder structure manually.

    Raises
    ------
    FileNotFoundError
        If the extraction parent directory does not exist.

    Example
    -------
    >>> root = detect_project_root(Path("/kaggle/working/project_extracted"))
    >>> print(root)
    """
    if not extract_parent.exists():
        raise FileNotFoundError(f"Extraction folder not found: {extract_parent}")

    direct_candidates = [
        extract_parent / "dental-panoramic-xrays",
        extract_parent / "dental-panoramic-xrays ",
        extract_parent / "project" / "dental-panoramic-xrays",
        extract_parent / "project" / "dental-panoramic-xrays ",
    ]

    for candidate in direct_candidates:
        if candidate.exists():
            return candidate.parent

    for item in sorted(extract_parent.iterdir()):
        if item.is_dir():
            if (item / "dental-panoramic-xrays").exists() or (item / "dental-panoramic-xrays ").exists():
                return item

    return extract_parent


def download_zip_from_gdrive(file_id: str, output_path: Path) -> None:
    """
    Download a zip file from Google Drive using gdown if it does not already exist.

    Parameters
    ----------
    file_id : str
        Google Drive file ID.
    output_path : Path
        Local path where the downloaded zip file should be saved.

    Returns
    -------
    None

    Example
    -------
    >>> download_zip_from_gdrive("abc123", Path("/kaggle/working/data.zip"))
    """
    if output_path.exists():
        print("ZIP file already exists. Skipping download.")
        return

    print("Downloading project zip from Google Drive...")
    cmd = f'gdown "https://drive.google.com/uc?id={file_id}" -O "{output_path}"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Download failed:\\n{result.stderr}")

    print("Download completed.")


def extract_zip_if_needed(zip_path: Path, extract_parent: Path) -> None:
    """
    Extract a zip file into a target directory if that directory does not already exist.

    Parameters
    ----------
    zip_path : Path
        Path to the zip file.
    extract_parent : Path
        Directory where the zip file will be extracted.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the zip file does not exist.

    Example
    -------
    >>> extract_zip_if_needed(Path("/kaggle/working/project.zip"), Path("/kaggle/working/project_extracted"))
    """
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    if extract_parent.exists():
        print("Project already extracted.")
        return

    extract_parent.mkdir(parents=True, exist_ok=True)
    print("Extracting project files...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_parent)

    print("Extraction completed.")


def print_directory_contents(path: Path) -> None:
    """
    Print the top-level contents of a directory for debugging and verification.

    Parameters
    ----------
    path : Path
        Directory to inspect.

    Returns
    -------
    None

    Example
    -------
    >>> print_directory_contents(Path("/kaggle/working"))
    """
    print(f"\\nTop-level contents of: {path}")
    for item in sorted(path.iterdir()):
        print(" -", item.name)



def count_images(folder: Path) -> int:
    """
    Count the number of image files in a given folder.

    Parameters
    ----------
    folder : Path
        Path to the directory containing image files.

    Returns
    -------
    int
        Number of files found in the directory.

    Raises
    ------
    FileNotFoundError
        If the folder does not exist.

    Example
    -------
    >>> from pathlib import Path
    >>> count_images(Path("/kaggle/working/train/images"))
    """
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    return len([p for p in folder.iterdir() if p.is_file()])





