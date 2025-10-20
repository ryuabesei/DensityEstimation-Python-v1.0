# License: GNU GPL v3
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


def _jd_to_ymdhms(jd: float) -> Tuple[int,int,int,int,int,float]:
    J = jd + 0.5
    Z = int(J); F = J - Z
    if Z >= 2299161:
        a = int((Z - 1867216.25) / 36524.25)
        A = Z + 1 + a - int(a/4)
    else:
        A = Z
    B = A + 1524
    C = int((B - 122.1) / 365.25)
    D = int(365.25 * C)
    E = int((B - D) / 30.6001)
    day = B - D - int(30.6001 * E) + F
    mon = E - 1 if E < 14 else E - 13
    year = C - 4716 if mon > 2 else C - 4715
    d_int = int(day); frac = day - d_int
    hh = int(frac*24.0)
    mm = int((frac*24.0 - hh)*60.0)
    ss = ((frac*24.0 - hh)*60.0 - mm)*60.0
    return year, mon, d_int, hh, mm, ss

def _doy(y: int, m: int, d: int) -> int:
    import datetime as _dt
    return int(_dt.date(y, m, d).strftime("%j"))

def _movmean_centered(x: np.ndarray, win: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    k = np.ones(win, dtype=float)
    num = np.convolve(x, k, mode='same')
    den = np.convolve(np.ones_like(x), k, mode='same')
    return num / np.clip(den, 1.0, None)

# MATLAB computeSWnrlmsise(...) 相当のコールバック
#   NRLMSISEモード: compute_swnrlmsise(SWmatDaily, SWmatMonthlyPred, jdate) -> (f107a, f107d, ap7)
ComputeSWFuncNRL = Callable[[np.ndarray, np.ndarray, float], tuple[float, float, np.ndarray]]

def compute_swinputs_nrlmsise(
    jd0: float,
    jdf: float,
    SWmatDaily: np.ndarray,
    SWmatMonthlyPred: np.ndarray,
    *,
    compute_swnrlmsise: ComputeSWFuncNRL,
) -> np.ndarray:
    """
    MATLAB computeSWinputs_NRLMSISE.m のポート。
    Returns
    -------
    Inputs : (41, N) ndarray
      1:  jdate
      2:  DOY
      3:  UThrs
      4:  F10a (smoothed later)
      5:  F10  (smoothed later)
      6..12: Ap(7成分) （daily/3hの平滑を後段で実施）
      13..21: [4..12] の +1h 先（future）
      22..30: [4..12]^2
      31..39: [13..21]^2
      40: F10(now)*Ap2(now)   （MATLABでは 5 × 7 行＝ index 7 は Apの2番手）
      41: F10(next)*Ap2(next)
    """
    tt = np.arange(jd0, jdf + 1e-12, 1.0/24.0)
    N = tt.size
    Inputs = np.zeros((41, N), dtype=float)

    F10a = np.zeros(N); F10 = np.zeros(N)
    AP7  = np.zeros((7, N))  # ap[0..6] を列に

    for i, jd in enumerate(tt):
        yy, mm, dd, hh, mn, ss = _jd_to_ymdhms(jd)
        doy = _doy(yy, mm, dd)
        UThrs = hh + mn/60.0 + ss/3600.0

        f107a, f107d, ap7 = compute_swnrlmsise(SWmatDaily, SWmatMonthlyPred, jd)
        ap7 = np.asarray(ap7, dtype=float).reshape(-1)
        if ap7.size < 7:
            raise ValueError("compute_swnrlmsise must return 7-component Ap vector for NRLMSISE.")

        Inputs[0, i] = jd
        Inputs[1, i] = doy
        Inputs[2, i] = UThrs
        F10a[i] = f107a
        F10[i]  = f107d
        AP7[:, i] = ap7[:7]

    # 平滑（MATLAB互換）
    Inputs[3, :] = _movmean_centered(F10a, 24)   # F10a 24h
    Inputs[4, :] = _movmean_centered(F10 , 24)   # F10  24h
    Inputs[5,  :] = _movmean_centered(AP7[0, :], 24)  # Ap daily 24h
    for k in range(1, 7):  # 残り6つは3h平均
        Inputs[5 + k, :] = _movmean_centered(AP7[k, :], 3)

    # +1h（future）
    Inputs[12:21, :-1] = Inputs[3:12, 1:]
    f107a_last, f107d_last, ap7_last = compute_swnrlmsise(SWmatDaily, SWmatMonthlyPred, jdf + 1.0/24.0)
    ap7_last = np.asarray(ap7_last, dtype=float).reshape(-1)
    Inputs[12, -1] = f107a_last
    Inputs[13, -1] = f107d_last
    Inputs[14:21, -1] = ap7_last[:7]

    # 2乗
    Inputs[21:30, :] = Inputs[3:12, :] ** 2
    Inputs[30:39, :] = Inputs[12:21, :] ** 2

    # 混合項（MATLAB: 40 = F10*Ap(2), 41 = F10_next*Ap_next(2)）
    Inputs[39, :] = Inputs[4, :] * Inputs[6, :]      # now: F10 × Ap2
    Inputs[40, :] = Inputs[13, :] * Inputs[15, :]    # next: F10 × Ap2

    return Inputs
