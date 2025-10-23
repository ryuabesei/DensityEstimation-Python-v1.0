# License: GNU GPL v3
#地心座標/測地座標からの高度計算
#地心直交座標（ECI または ECEF）ベクトルから、WGS-84 に近い扁平楕円体（f=1/298.257, req=6378.14 km）を仮定して高度[km]を求める
from __future__ import annotations

import numpy as np


#地心座標ベクトルから高度を算出
def altitude(r: np.ndarray) -> np.ndarray:
    """
    Oblate Earth を仮定した高度を計算（MATLAB altitude.m のポート）

    Parameters
    ----------
    r : ndarray, shape (N, 3)
        ECI/ECEF いずれでもよいが、地心直交座標 [km]

    Returns
    -------
    alt : ndarray, shape (N,)
        楕円体（f=1/298.257, req=6378.14 km）からの高度 [km]
    """
    r = np.asarray(r, dtype=float)
    if r.ndim != 2 or r.shape[1] != 3:
        raise ValueError("r must have shape (N, 3) in kilometers")

    # Earth parameters
    f = 1.0 / 298.257
    req = 6378.14  # km

    rmag = np.sqrt(np.sum(r * r, axis=1))
    # avoid divide-by-zero
    with np.errstate(invalid="ignore", divide="ignore"):
        delta = np.arcsin(r[:, 2] / rmag)

    alt = (
        rmag
        - req
        * (
            1.0
            - f * np.sin(delta) ** 2
            - (f**2 / 2.0) * (np.sin(2.0 * delta) ** 2) * (req / rmag - 0.25)
        )
    )
    return alt
