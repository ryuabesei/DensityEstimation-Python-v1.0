# Port of computeSWinputs_JB2008.m
# License: GNU GPL v3
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

# 型: compute_jb2000_swinputs(yy, doy, hh, mn, ss, SOLdata, DTCdata, eopdata)
# → (GWRAS, SUN, F10, F10B, S10, S10B, XM10, XM10B, Y10, Y10B, DSTDTC)
ComputeSWFunc = Callable[[int, int, int, int, float, np.ndarray, np.ndarray, np.ndarray],
                         Tuple[float, np.ndarray, float, float, float, float, float, float, float, float, float]]

def _jd_to_ymdhms(jd: float) -> Tuple[int, int, int, int, int, float]:
    """ユリウス日 → UTCの年月日時分秒（MATLABの datevec(jdate-1721058.5) 相当）"""
    # 実装簡略（誤差 << 1秒でOK）。高精度にしたければ天文ユーティリティへ切替。
    J = jd + 0.5
    Z = int(J)
    F = J - Z
    if Z >= 2299161:
        a = int((Z - 1867216.25) / 36524.25)
        A = Z + 1 + a - int(a / 4)
    else:
        A = Z
    B = A + 1524
    C = int((B - 122.1) / 365.25)
    D = int(365.25 * C)
    E = int((B - D) / 30.6001)
    day = B - D - int(30.6001 * E) + F
    mon = E - 1 if E < 14 else E - 13
    year = C - 4716 if mon > 2 else C - 4715
    # 日の小数→時分秒
    d_int = int(day)
    frac = day - d_int
    hh = int(frac * 24.0)
    mm = int((frac * 24.0 - hh) * 60.0)
    ss = ((frac * 24.0 - hh) * 60.0 - mm) * 60.0
    return year, mon, d_int, hh, mm, ss

def _day_of_year(year: int, month: int, day: int) -> int:
    import datetime as _dt
    d = _dt.date(year, month, day)
    return int(d.strftime("%j"))

def compute_swinputs_jb2008(
    jd0: float,
    jdf: float,
    *,
    eopdata: np.ndarray,
    SOLdata: np.ndarray,
    DTCdata: np.ndarray,
    compute_jb2000_swinputs: ComputeSWFunc,
) -> np.ndarray:
    """
    Returns
    -------
    Inputs : shape (24, N)  ※MATLAB 実装では 1..24 行を使用
      rows:
        1: jdate
        2: DOY
        3: UThrs
        4-5:  F10, F10B
        6-7:  S10, S10B
        8-9:  XM10, XM10B
        10-11: Y10, Y10B
        12: DSTDTC (smoothed 12h)
        13: GWRAS
        14-15: SUN(1:2)
        16-20: future (+1h) of [DSTDTC,F10,S10,XM10,Y10]
        21-22: [DSTDTC^2, future DSTDTC^2]
        23-24: [DSTDTC*F10, future(DSTDTC*F10)]
    """
    # 1時間刻み（両端含む）で jd0..jdf
    tt = np.arange(jd0, jdf + 1e-12, 1.0 / 24.0, dtype=float)
    N = tt.size
    Inputs = np.zeros((24, N), dtype=float)

    for i, jdate in enumerate(tt):
        yy, mm, dd, hh, mn, ss = _jd_to_ymdhms(jdate)
        doy = _day_of_year(yy, mm, dd)
        UThrs = hh + mn / 60.0 + ss / 3600.0

        # JB2008の宇宙天気値（外部関数で取得）
        GWRAS, SUN, F10, F10B, S10, S10B, XM10, XM10B, Y10, Y10B, DSTDTC = compute_jb2000_swinputs(
            yy, doy, hh, mn, ss, SOLdata, DTCdata, eopdata
        )
        Inputs[0, i] = jdate
        Inputs[1, i] = doy
        Inputs[2, i] = UThrs
        Inputs[3, i] = F10
        Inputs[4, i] = F10B
        Inputs[5, i] = S10
        Inputs[6, i] = S10B
        Inputs[7, i] = XM10
        Inputs[8, i] = XM10B
        Inputs[9, i] = Y10
        Inputs[10, i] = Y10B
        Inputs[11, i] = DSTDTC
        Inputs[12, i] = GWRAS
        Inputs[13, i] = float(SUN[0])
        Inputs[14, i] = float(SUN[1])

    # 12時間移動平均（1時間ステップなのでウィンドウ=12）
    # 端の処理は単純移動平均（validでない）。numpyの畳み込みで近似。
    k = np.ones(12) / 12.0
    dstdtc = Inputs[11, :]
    sm = np.convolve(dstdtc, k, mode="same")
    Inputs[11, :] = sm  # 行12に相当

    # 未来(+1h)を 16..20 行へ
    Inputs[15, :-1] = Inputs[11, 1:]  # DSTDTC
    Inputs[16, :-1] = Inputs[3, 1:]   # F10
    Inputs[17, :-1] = Inputs[5, 1:]   # S10
    Inputs[18, :-1] = Inputs[7, 1:]   # XM10
    Inputs[19, :-1] = Inputs[9, 1:]   # Y10

    # 最後の +1h 分は終端で再計算
    jd_lastp1 = jdf + 1.0 / 24.0
    yy, mm, dd, hh, mn, ss = _jd_to_ymdhms(jd_lastp1)
    doy = _day_of_year(yy, mm, dd)
    _, _, F10, _, S10, _, XM10, _, Y10, _, DSTDTC = compute_jb2000_swinputs(
        yy, doy, hh, mn, ss, SOLdata, DTCdata, eopdata
    )
    Inputs[15, -1] = DSTDTC
    Inputs[16, -1] = F10
    Inputs[17, -1] = S10
    Inputs[18, -1] = XM10
    Inputs[19, -1] = Y10

    # 2乗・混合項
    Inputs[20, :] = Inputs[11, :] ** 2
    Inputs[21, :] = Inputs[15, :] ** 2
    Inputs[22, :] = Inputs[11, :] * Inputs[3, :]
    Inputs[23, :] = Inputs[15, :] * Inputs[16, :]

    return Inputs
