# GNU GPL v3
from __future__ import annotations

import math


def mjday(year: int, month: int, day: int, hour: int = 0, minute: int = 0, sec: float = 0.0) -> float:
    """
    MJD (Modified Julian Date) を返す。
    元: MATLAB Mjday.m

    Parameters
    ----------
    year, month, day, hour, minute, sec

    Returns
    -------
    float
        MJD = JD - 2400000.5
    """
    y = int(year)
    m = int(month)
    b = 0
    c = 0.0

    # オプションの時分秒は既にデフォルト値を持つ

    if m <= 2:
        y -= 1
        m += 12

    if y < 0:
        c = -0.75  # グレゴリオ補正（負の年の補正項）

    # グレゴリオ歴移行の条件分岐は MATLAB と同じロジック
    if year < 1582:
        pass
    elif year > 1582:
        a = y // 100
        b = 2 - a + (a // 4)
    elif month < 10:
        pass
    elif month > 10:
        a = y // 100
        b = 2 - a + (a // 4)
    elif day <= 4:
        pass
    elif day > 14:
        a = y // 100
        b = 2 - a + (a // 4)
    else:
        raise ValueError("Invalid calendar date for Gregorian reform window.")

    jd = math.floor(365.25 * y + c) + math.floor(30.6001 * (m + 1))
    jd = jd + day + b + 1720994.5
    jd = jd + (hour + minute / 60.0 + sec / 3600.0) / 24.0
    mjd = jd - 2400000.5
    return float(mjd)

__all__ = ["mjday"]
