# Origin: Unscented_Transform.m (Mehta 2018 base; Gondelach 2020; Li 2022)
# License: GNU GPL v3
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def unscented_transform(x0f: np.ndarray, kappa: Optional[float] = None
                        ) -> Tuple[np.ndarray, np.ndarray, int, float]:
    """
    Compute UT weights (MATLAB Unscented_Transform と同一仕様).

    Parameters
    ----------
    x0f : array-like
        状態ベクトル（長さ L のみ使用）
    kappa : float, optional
        省略時は 3-L

    Returns
    -------
    Wm : (2L+1,) ndarray
        重み（平均）
    Wc : (2L+1,) ndarray
        重み（共分散）
    L : int
        次元
    lam : float
        λ = α^2 (L+κ) − L
    """
    L = int(np.size(x0f))
    alpha = 1.0
    beta = 2.0
    if kappa is None:
        kappa = 3.0 - L

    lam = alpha**2 * (L + kappa) - L
    denom = (L + lam)

    W0m = lam / denom
    W0c = lam / denom + (1.0 - alpha**2 + beta)
    Wi  = 1.0 / (2.0 * denom)

    Wm = np.empty(2*L + 1, dtype=float)
    Wc = np.empty(2*L + 1, dtype=float)
    Wm[0], Wc[0] = W0m, W0c
    Wm[1:], Wc[1:] = Wi, Wi
    return Wm, Wc, L, lam
