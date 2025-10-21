# densityestimation/models/jb2008/invjday.py
from __future__ import annotations

from typing import Tuple

import numpy as np


def invjday(jd: float) -> Tuple[int, int, int, int, int, int]:
    """
    Julian Date -> Gregorian calendar (year, month, day, hour, minute, second)
    MATLAB JB2008 の invjday.m と同等の結果（秒は整数丸め）。

    Parameters
    ----------
    jd : float
        Julian Date

    Returns
    -------
    (year, month, day, hr, min, sec) : all int
    """
    JD = float(jd)
    z = int(np.floor(JD + 0.5))
    fday = JD + 0.5 - z
    if fday < 0:
        fday += 1.0
        z -= 1

    if z < 2299161:
        a = z
    else:
        alpha = int(np.floor((z - 1867216.25) / 36524.25))
        a = z + 1 + alpha - int(np.floor(alpha / 4))

    b = a + 1524
    c = int(np.floor((b - 122.1) / 365.25))
    d = int(np.floor(365.25 * c))
    e = int(np.floor((b - d) / 30.6001))

    day_float = b - d - int(np.floor(30.6001 * e)) + fday
    if e < 14:
        month = e - 1
    else:
        month = e - 13

    if month > 2:
        year = c - 4716
    else:
        year = c - 4715

    day = int(np.floor(day_float))
    # 時分秒（整数化：MATLAB 実装は floor）
    hrs  = (day_float - day) * 24.0
    hr   = int(np.floor(hrs))
    mins = (hrs - hr) * 60.0
    minute = int(np.floor(mins))
    secs = (mins - minute) * 60.0
    sec = int(np.floor(secs))

    return int(year), int(month), int(day), int(hr), int(minute), int(sec)
