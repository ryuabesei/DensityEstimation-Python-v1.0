# densityestimation/timeutils/finddays.py
from __future__ import annotations


def find_days(year: int, month: int, day: int, hr: int, minute: int, sec: float) -> float:
    """
    MATLAB finddays.m のポート。
    年月日時分秒から「年初からの経過日（小数日）」を返す。

    Parameters
    ----------
    year : 1900..2100
    month : 1..12
    day : 1..31
    hr : 0..23
    minute : 0..59
    sec : 0..59.999

    Returns
    -------
    float
        1月1日 0:00:00 を day=1.0 とする年内通算日（小数日）
        例: 1月1日 12:00:00 → 1.5
    """
    # 月ごとの日数（平年ベース）
    lmonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    # うるう年補正（グレゴリオ暦）：400の倍数はうるう年、100の倍数は平年、4の倍数はうるう年
    is_leap = (year % 4 == 0)
    if (year % 100 == 0) and (year % 400 != 0):
        is_leap = False
    if is_leap:
        lmonth[1] = 29

    # 前月までの日数を合計（MATLABの while と同じ結果）
    if not (1 <= month <= 12):
        raise ValueError("month must be in 1..12")
    days = sum(lmonth[: month - 1])

    # 当日 + 時分秒の小数日
    days += day + hr / 24.0 + minute / 1440.0 + float(sec) / 86400.0
    return float(days)
