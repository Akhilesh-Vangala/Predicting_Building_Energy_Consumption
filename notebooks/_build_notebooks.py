from __future__ import annotations

import json
import re
import uuid
from pathlib import Path


def _cell_id() -> str:
    return uuid.uuid4().hex[:8]


def cells_from_source(src: str) -> list[dict]:
    parts = re.split(r"^# %%(?:\s*\[(\w+)\])?\s*\n", src, flags=re.MULTILINE)
    cells: list[dict] = []
    if not parts[0].strip():
        parts = parts[1:]
    i = 0
    while i < len(parts):
        kind = parts[i] or "code"
        body = parts[i + 1] if i + 1 < len(parts) else ""
        body = body.rstrip() + "\n"
        if kind == "markdown":
            text = "\n".join(line.lstrip("# ").rstrip() for line in body.splitlines())
            cells.append({
                "cell_type": "markdown",
                "id": _cell_id(),
                "metadata": {},
                "source": text.splitlines(keepends=True),
            })
        else:
            cells.append({
                "cell_type": "code",
                "id": _cell_id(),
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": body.splitlines(keepends=True),
            })
        i += 2
    return cells


def write_notebook(py_path: Path, ipynb_path: Path) -> None:
    src = py_path.read_text()
    cells = cells_from_source(src)
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.13",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    ipynb_path.write_text(json.dumps(nb, indent=1))


def main() -> None:
    root = Path(__file__).parent
    pairs = [
        ("01_eda.py", "01_eda.ipynb"),
        ("02_feature_engineering.py", "02_feature_engineering.ipynb"),
        ("03_linear_models.py", "03_linear_models.ipynb"),
        ("04_tree_models.py", "04_tree_models.ipynb"),
        ("05_time_series.py", "05_time_series.ipynb"),
        ("06_mlp.py", "06_mlp.ipynb"),
        ("07_clustering.py", "07_clustering.ipynb"),
        ("08_analysis.py", "08_analysis.ipynb"),
    ]
    for py, ipynb in pairs:
        py_path = root / py
        ipynb_path = root / ipynb
        if py_path.exists():
            write_notebook(py_path, ipynb_path)
            print(f"wrote {ipynb_path}")


if __name__ == "__main__":
    main()
