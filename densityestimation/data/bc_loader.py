from __future__ import annotations

from pathlib import Path

import numpy as np


def load_bc_data(path: str | Path) -> np.ndarray:
    """
    MATLAB: BCdata = loadBCdata('Data/BCdata.txt') に相当。
    期待フォーマット: 2列 [NORAD_ID, BC(m^2/kg)]。ヘッダ行があってもOK。
    """
    p = Path(path)
    rows = []
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            parts = ln.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                norad = int(float(parts[0]))
                bc = float(parts[1])
            except ValueError:
                continue  # ヘッダ等はスキップ
            rows.append((norad, bc))
    if not rows:
        raise ValueError(f"BCdata appears empty or unreadable: {p}")
    return np.asarray(rows, dtype=float)
