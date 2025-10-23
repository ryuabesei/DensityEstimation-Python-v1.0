# License: GNU GPL v3
#通日（DOY）・月日変換、UTC→JD/JED、JD/JED→年月日時分秒

from __future__ import annotations

import datetime as _dt
import math
from typing import Tuple


def day_of_year(year: int, month: int, day: int) -> int:
    """
    MATLAB dayofyear.m のポート。
    1月1日を 1 とする通日の整数を返す。

    Parameters
    ----------
    year, month, day : int

    Returns
    -------
    doy : int
    """
    d = _dt.date(year, month, day)
    return int(d.strftime("%j"))  # 001..366 を返す


def days2mdh(year: int, days: float) -> Tuple[int, int, int, int, float]:
    """
    MATLAB days2mdh.m のポート。
    年内通日 (1..366, 小数可) → (month, day, hour, minute, second)。

    Parameters
    ----------
    year : int
    days : float
        年内通日（1.0..366.0）。小数部は日の中の時刻。

    Returns
    -------
    mon : int    # 1..12
    day : int    # 1..31
    hr  : int    # 0..23
    minute : int # 0..59
    sec : float  # 0..59.999...
    """
    if days < 1.0:
        raise ValueError("days must be >= 1.0")

    # 通日の整数部・小数部
    dayofyr = int(days)  # floor
    frac = float(days - dayofyr)

    # 正しいグレゴリオ暦のうるう年判定
    def _is_leap(y: int) -> bool:
        return (y % 4 == 0) and (y % 100 != 0 or y % 400 == 0)

    # 各月の日数
    lmonth = [31, 29 if _is_leap(year) else 28, 31, 30, 31, 30,
              31, 31, 30, 31, 30, 31]

    # 月を決める
    acc = 0
    mon = 1
    while mon <= 12 and dayofyr > acc + lmonth[mon - 1]:
        acc += lmonth[mon - 1]
        mon += 1
    if mon > 12:
        # 366.0 を少し越えるなどの丸め誤差対策：最終日に丸める
        mon = 12
        day = lmonth[-1]
    else:
        day = dayofyr - acc

    # 小数日 → 時分秒
    hours = frac * 24.0
    hr = int(hours)
    minutes = (hours - hr) * 60.0
    minute = int(minutes)
    sec = (minutes - minute) * 60.0

    # 端の丸め（59.999...→次分繰り上げ等）を軽く処理
    if sec >= 59.999999:
        sec = 0.0
        minute += 1
    if minute >= 60:
        minute = 0
        hr += 1
    if hr >= 24:
        hr = 0
        # 翌日に進める（必要なら月末繰り上げも処理）
        day += 1
        if day > lmonth[mon - 1]:
            day = 1
            mon += 1

    return mon, day, hr, minute, sec



def hms2sec(hr: float, minute: float, sec: float) -> float:
    """
    Convert hours, minutes, seconds → total seconds of the day.

    Parameters
    ----------
    hr : float
        Hours [0–24)
    minute : float
        Minutes [0–59]
    sec : float
        Seconds [0–60)

    Returns
    -------
    utsec : float
        Seconds since beginning of the day.
    """
    return hr * 3600.0 + minute * 60.0 + sec


def gstime(jdut1: float) -> float:
    """
    Compute Greenwich Sidereal Time (IAU-82) from Julian Date of UT1.

    Parameters
    ----------
    jdut1 : float
        Julian date (UT1) [days since 4713 BC]

    Returns
    -------
    gst : float
        Greenwich Sidereal Time [radians, 0–2π)

    Notes
    -----
    Reference: Vallado (2007), Eq. 3-43.
    """
    twopi = 2.0 * math.pi
    deg2rad = math.pi / 180.0

    # Julian centuries from J2000.0
    tut1 = (jdut1 - 2451545.0) / 36525.0

    temp = (
        -6.2e-6 * tut1**3
        + 0.093104 * tut1**2
        + (876600.0 * 3600.0 + 8640184.812866) * tut1
        + 67310.54841
    )

    # Convert to radians (360/86400 = 1/240)
    temp = (temp * deg2rad / 240.0) % twopi

    if temp < 0.0:
        temp += twopi

    return temp


def julian(yr: int, mth: int, day: int, hr: int, minute: int, sec: float) -> float:
    """
    MATLAB julian.m と同等。
    暦日 (年/月/日 時:分:秒) からユリウス日 (JD) を求める。

    Parameters
    ----------
    yr, mth, day, hr, minute, sec : int or float

    Returns
    -------
    jd : float
        ユリウス日（UT基準）

    Notes
    -----
    Vallado / MATLAB版 julian.m と同じ式を使用。
    """
    # day-of-year の計算 (MATLAB と同じ条件分岐)
    doy = (
        day
        + 31 * (mth - 1)
        - int(2.2 + 0.4 * mth) * (mth > 2)
        + (1 if (mth > 2 and yr % 4 == 0) else 0)
    )
    jd = (
        2415020.0
        + (yr - 1900) * 365.0
        + int((yr - 1901) / 4.0)
        + (doy - 1)
        + 0.5
        + (3600.0 * hr + 60.0 * minute + sec) / 86400.0
    )
    return float(jd)


def jed2date(jed: float) -> tuple[int, int, int, int, int, float]:
    """
    MATLAB jed2date.m と同等。
    ユリウス日 JED → (year, month, day, hour, minute, second)

    Parameters
    ----------
    jed : float
        Julian Ephemeris Date (ユリウス日)

    Returns
    -------
    (year, month, day, hour, minute, second)
    """
    # MATLAB: datevec(jed - 1721058.5)
    jd = jed - 1721058.5
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
    month = E - 1 if E < 14 else E - 13
    year = C - 4716 if month > 2 else C - 4715

    # 日の小数部分 → 時分秒
    d_int = int(day)
    frac = day - d_int
    hour = int(frac * 24.0)
    minute = int((frac * 24.0 - hour) * 60.0)
    second = ((frac * 24.0 - hour) * 60.0 - minute) * 60.0

    return year, month, d_int, hour, minute, second