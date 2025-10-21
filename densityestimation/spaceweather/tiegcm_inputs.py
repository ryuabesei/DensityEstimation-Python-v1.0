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

# 型: compute_swnrlmsise(SWmatDaily, SWmatMonthlyPred, jdate, tie_mode=True)
#   -> (f107Average, f107Daily, kp_array(>=2 entries; index 1が3h-Kp))
ComputeSWFunc = Callable[[np.ndarray, np.ndarray, float, bool], tuple[float, float, np.ndarray]]

def _movmean_centered(x: np.ndarray, win: int) -> np.ndarray:
    """
    中央寄せ移動平均。端は利用可能サンプル数で割る（MATLAB movmean の端部互換）。
    """
    x = np.asarray(x, dtype=float)
    k = np.ones(win, dtype=float)
    num = np.convolve(x, k, mode='same')
    den = np.convolve(np.ones_like(x), k, mode='same')
    return num / np.clip(den, 1.0, None)

def compute_swinputs_tiegcm(
    jd0: float,
    jdf: float,
    SWmatDailyTIEGCM: np.ndarray,
    SWmatMonthlyPredTIEGCM: np.ndarray,
    *,
    compute_swnrlmsise: ComputeSWFunc,
) -> np.ndarray:
    """
    Returns
    -------
    Inputs : (12, N) ndarray
      1: jdate
      2: DOY
      3: UThrs
      4: F107 (daily)
      5: F107a (avg)
      6: Kp (3-hourly; smoothed to hourly series)
      7: F107 (now+1h)
      8: Kp (now+1h)
      9:  Kp^2 (now)
      10: Kp^2 (now+1h)
      11: Kp*F10 (now)
      12: Kp*F10 (now+1h)
    """
    tt = np.arange(jd0, jdf + 1e-12, 1.0 / 24.0)
    N = tt.size
    Inputs = np.zeros((12, N), dtype=float)

    f10 = np.zeros(N); f10a = np.zeros(N); kp3h = np.zeros(N)

    for i, jd in enumerate(tt):
        yy, mm, dd, hh, mn, ss = _jd_to_ymdhms(jd)
        doy = _doy(yy, mm, dd)
        UThrs = hh + mn/60.0 + ss/3600.0

        f107a, f107d, kp_arr = compute_swnrlmsise(SWmatDailyTIEGCM, SWmatMonthlyPredTIEGCM, jd, True)

        Inputs[0, i] = jd
        Inputs[1, i] = doy
        Inputs[2, i] = UThrs
        f10[i]  = f107d
        f10a[i] = f107a
        # MATLAB: kp(2) が3時間 Kp
        kp3h[i] = float(kp_arr[1]) if kp_arr is not None and len(kp_arr) > 1 else np.nan

    # 平滑化（MATLAB互換）
    Inputs[3, :] = _movmean_centered(f10, 24)   # F10 24h
    Inputs[4, :] = _movmean_centered(f10a, 24)  # F10a 24h
    Inputs[5, :] = _movmean_centered(kp3h, 3)   # Kp 3h

    # 未来(+1h)
    Inputs[6, :-1] = Inputs[3, 1:]  # F10 next
    Inputs[7, :-1] = Inputs[5, 1:]  # Kp  next

    f107a_last, f107d_last, kp_last = compute_swnrlmsise(
        SWmatDailyTIEGCM, SWmatMonthlyPredTIEGCM, jdf + 1.0/24.0, True
    )
    Inputs[6, -1] = f107d_last
    Inputs[7, -1] = float(kp_last[1]) if kp_last is not None and len(kp_last) > 1 else np.nan

    # 2乗と混合項
    Inputs[8,  :] = Inputs[5, :] ** 2             # Kp^2 now
    Inputs[9,  :] = Inputs[7, :] ** 2             # Kp^2 next
    Inputs[10, :] = Inputs[5, :] * Inputs[3, :]   # Kp*F10 now
    Inputs[11, :] = Inputs[7, :] * Inputs[6, :]   # Kp*F10 next

    return Inputs
