from __future__ import annotations

from pathlib import Path

import numpy as np

# convert_teme_to_j2000 内で参照する EOPMat をセットするための import
from densityestimation.tle.sgp4_wrapper import set_eop_matrix


def load_eop_celestrak(path: str | Path) -> np.ndarray:
    """
    Celestrak配布の EOP ファイルを読み込んで (N,6) の行列を返す。
    列: [xp[arcsec], yp[arcsec], dut1[s], lod[s], ddpsi[rad], ddeps[rad]]
    先頭のヘッダ行は自動スキップ。区切りは空白/カンマどちらでもOK。
    """
    p = Path(path)
    # 数値が6列以上ある行だけ採用
    rows = []
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            try:
                vals = [float(x) for x in ln.replace(",", " ").split()]
            except Exception:
                continue
            if len(vals) >= 6:
                rows.append(vals[:6])
    if not rows:
        raise ValueError(f"EOP file looks empty/not numeric: {p}")
    EOPMat = np.asarray(rows, dtype=float)
    set_eop_matrix(EOPMat)  # ここでグローバルに反映
    return EOPMat
