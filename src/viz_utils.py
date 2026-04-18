"""Visualization helpers for plots used in the project report.

Example
-------
>>> # plot_metric_curve([0, 1, 2], [0.1, 0.2, 0.3], "Title", "x", "y")
"""

from __future__ import annotations

from typing import Iterable, Sequence
import matplotlib.pyplot as plt


def plot_metric_curve(
    x_values: Sequence[float],
    y_values: Sequence[float],
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    """Plot a single metric curve with a simple, report-friendly style.

    Parameters
    ----------
    x_values : Sequence[float]
        Values for the x-axis.
    y_values : Sequence[float]
        Values for the y-axis.
    title : str
        Plot title.
    x_label : str
        X-axis label.
    y_label : str
        Y-axis label.

    Returns
    -------
    None

    Example
    -------
    >>> plot_metric_curve([1, 2], [0.1, 0.2], "Example", "Epoch", "Score")
    """
    plt.figure(figsize=(10, 4))
    plt.plot(x_values, y_values, marker="o")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.show()


def plot_two_metric_curves(
    x_values: Sequence[float],
    y1_values: Sequence[float],
    y2_values: Sequence[float],
    label1: str,
    label2: str,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    """Plot two curves on the same axes.

    Parameters
    ----------
    x_values : Sequence[float]
        Values for the x-axis.
    y1_values : Sequence[float]
        First metric values.
    y2_values : Sequence[float]
        Second metric values.
    label1 : str
        Label for the first curve.
    label2 : str
        Label for the second curve.
    title : str
        Plot title.
    x_label : str
        X-axis label.
    y_label : str
        Y-axis label.

    Returns
    -------
    None

    Example
    -------
    >>> plot_two_metric_curves([1, 2], [0.1, 0.2], [0.2, 0.3], "A", "B", "Title", "x", "y")
    """
    plt.figure(figsize=(10, 4))
    plt.plot(x_values, y1_values, marker="o", label=label1)
    plt.plot(x_values, y2_values, marker="o", label=label2)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.show()
