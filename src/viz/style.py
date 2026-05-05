from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


PRIMARY = "#264653"
ACCENT = "#e76f51"
SUPPORT = ["#2a9d8f", "#e9c46a", "#f4a261", "#8ab17d", "#264653"]


def set_style() -> None:
    plt.rcParams.update({
        "figure.figsize": (8, 4.5),
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.edgecolor": "#333",
        "axes.linewidth": 0.8,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "font.family": "DejaVu Sans",
        "grid.color": "#e0e0e0",
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
    })


def save_figure(fig, path: str | Path, formats: tuple[str, ...] = ("png",)) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(path.with_suffix(f".{fmt}"))
