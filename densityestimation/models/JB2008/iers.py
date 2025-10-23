# densityestimation/models/jb2008/iers.py
from __future__ import annotations

from typing import Tuple

import numpy as np

from densityestimation.models.JB2008.constants import CONST


def iers(eop: np.ndarray, mjd_utc: float, interp: str = "n") -> Tuple[
    float, float, float, float, float, float, float, float, float
]:
    """
    JB2008 版 IERS: IERS テーブルから、指定 MJD(UTC) に対する地球姿勢/時間パラメータを取得。

    Parameters
    ----------
    eop : ndarray, shape (13, N)
        行の意味（MATLAB IERS.m と同じ）:
          1: Year
          2: Month
          3: Day
          4: MJD
          5: x_pole [arcsec]
          6: y_pole [arcsec]
          7: UT1-UTC [s]
          8: LOD [s]
          9: dpsi [arcsec]
         10: deps [arcsec]
         11: dx_pole [arcsec]
         12: dy_pole [arcsec]
         13: TAI-UTC [s]
    mjd_utc : float
        参照する MJD(UTC)
    interp : {'n','l'}
        'n'…同日データをそのまま使用 / 'l'…翌日のデータと線形補間

    Returns
    -------
    (x_pole, y_pole, UT1_UTC, LOD, dpsi, deps, dx_pole, dy_pole, TAI_UTC)
      x_pole, y_pole, dpsi, deps, dx_pole, dy_pole は [rad] に変換済
      UT1_UTC, LOD, TAI_UTC は [s]
    """
    if eop.ndim != 2 or eop.shape[0] < 13:
        raise ValueError("eop must be a (13, N) array (rows as in MATLAB IERS.m).")

    # arcsec → rad
    AS2R = CONST.AS2R  # 4.848...e-6

    mjd_floor = np.floor(mjd_utc)
    mjds = eop[3, :]  # MJD 行

    idxs = np.where(mjds == mjd_floor)[0]
    if idxs.size == 0:
        raise ValueError(f"MJD={mjd_floor} not found in EOP table.")

    i = int(idxs[0])

    if interp == "l":
        # 線形補間（翌日の行が必要）
        if i + 1 >= eop.shape[1]:
            # 末尾なら補間できないので、そのまま使う
            row = eop[:, i]
            fixf = 0.0
            nextrow = row
        else:
            row = eop[:, i]
            nextrow = eop[:, i + 1]
            # 分単位の補間係数
            mfme = 1440.0 * (mjd_utc - np.floor(mjd_utc))  # minutes from 0h
            fixf = mfme / 1440.0

        # 線形補間
        def lerp(a, b): return a + (b - a) * fixf

        x_pole  = lerp(row[4],  nextrow[4])  * AS2R
        y_pole  = lerp(row[5],  nextrow[5])  * AS2R
        UT1_UTC = lerp(row[6],  nextrow[6])
        LOD     = lerp(row[7],  nextrow[7])
        dpsi    = lerp(row[8],  nextrow[8])  * AS2R
        deps    = lerp(row[9],  nextrow[9])  * AS2R
        dx_pole = lerp(row[10], nextrow[10]) * AS2R
        dy_pole = lerp(row[11], nextrow[11]) * AS2R
        TAI_UTC = row[12]  # MATLAB 実装と同様：当日値を採用
    else:
        # 最近傍（同日）をそのまま
        row = eop[:, i]
        x_pole  = float(row[4])  * AS2R
        y_pole  = float(row[5])  * AS2R
        UT1_UTC = float(row[6])
        LOD     = float(row[7])
        dpsi    = float(row[8])  * AS2R
        deps    = float(row[9])  * AS2R
        dx_pole = float(row[10]) * AS2R
        dy_pole = float(row[11]) * AS2R
        TAI_UTC = float(row[12])

    return (x_pole, y_pole, UT1_UTC, LOD, dpsi, deps, dx_pole, dy_pole, TAI_UTC)
