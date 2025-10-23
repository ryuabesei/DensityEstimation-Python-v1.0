# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np

from densityestimation.models.jb2008.jb2008model import JB2008

# 太陽/衛星角などはテスト用の適当値（正規化された入力）
# MJD(UTC) = 2020-12-09 00:00:00 近傍
MJD = 59192.0

def test_jb2008_returns_positive_density():
    # SUN = [RA, Dec] [rad]（それっぽい値）
    SUN = np.array([1.0, 0.1], dtype=float)
    # SAT = [RA, geocentric lat, height(km)]
    SAT = np.array([2.0, 0.5, 400.0], dtype=float)  # 400km 周辺

    # JB2008 入力（“代表値”）：81日平均と当日値のセット
    F10, F10B = 100.0, 100.0
    S10, S10B = 100.0, 100.0
    XM10, XM10B = 100.0, 100.0
    Y10, Y10B = 100.0, 100.0
    DSTDTC = 0.0  # 磁気嵐補正なし

    TEMP, RHO = JB2008(
        MJD, SUN, SAT, F10, F10B, S10, S10B, XM10, XM10B, Y10, Y10B, DSTDTC
    )
    # 返り値の健全性チェック
    assert TEMP.shape == (2,)
    assert np.isfinite(TEMP[0]) and np.isfinite(TEMP[1])
    assert np.isfinite(RHO) and (RHO > 0.0) and (RHO < 1e-8)  # 400kmでの常識的範囲より十分小さい上限
