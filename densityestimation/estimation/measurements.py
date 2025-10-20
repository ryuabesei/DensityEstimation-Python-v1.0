# Equivalent to MATLAB: fullmee2mee(xx,nop,svs)
from __future__ import annotations

import numpy as np


def fullmee2mee(Xp: np.ndarray, nop: int, svs: int) -> np.ndarray:
    """
    入力: Xp (n, 2n+1) with stacked [ (6 MEE + 1 BC)*nop + r ]
    出力: 観測空間（MEEのみ） (6*nop, 2n+1)
    """
    ncols = Xp.shape[1]
    out = []
    for i in range(nop):
        out.append(Xp[i*svs:(i*svs+6), :])
    return np.vstack(out)
