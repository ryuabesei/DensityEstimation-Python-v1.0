# jb2008_inputs.py
# Port of computeSWinputs_JB2008.m + computeJB2000SWinputs.m
# License: GNU GPL v3
from __future__ import annotations

import math
from typing import Callable, Tuple

import numpy as np

# 型: compute_jb2000_swinputs(yy, doy, hh, mn, ss, SOLdata, DTCdata, eopdata)
# → (GWRAS [rad], SUN(np.array([ra, dec]) [rad]), F10, F10B, S10, S10B, XM10, XM10B, Y10, Y10B, DSTDTC)
ComputeSWFunc = Callable[
    [int, int, int, int, float, np.ndarray, np.ndarray, np.ndarray],
    Tuple[float, np.ndarray, float, float, float, float, float, float, float, float, float],
]


# ---------------------------------------------------------------------
# Helpers (暦変換・天文近似。依存を最小化するため簡易実装)
# ---------------------------------------------------------------------

def _jd_to_ymdhms(jd: float) -> Tuple[int, int, int, int, int, float]:
    """ユリウス日 → UTCの年月日時分秒（MATLABの datevec(jdate-1721058.5) 相当）"""
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


def _days2mdh(year: int, doy: int) -> Tuple[int, int]:
    """年+通日 → 月日"""
    import datetime as _dt
    d0 = _dt.date(year, 1, 1)
    d = d0 + _dt.timedelta(days=doy - 1)
    return d.month, d.day


def _jd_from_calendar(year: int, month: int, day: int, hour: int, minute: int, sec: float) -> float:
    """UTCカレンダー → ユリウス日"""
    A = int((14 - month) / 12)
    y = year + 4800 - A
    m = month + 12 * A - 3
    JDN = day + int((153 * m + 2) / 5) + 365 * y + int(y / 4) - int(y / 100) + int(y / 400) - 32045
    frac = (hour - 12) / 24.0 + minute / 1440.0 + sec / 86400.0
    return JDN + frac


def _mjd(year: int, month: int, day: int, hour: int, minute: int, sec: float) -> float:
    """Modified Julian Date（UTC）"""
    jd = _jd_from_calendar(year, month, day, hour, minute, sec)
    return jd - 2400000.5


def _gmst_from_jd(jd_ut1: float) -> float:
    """グリニッジ恒星時（GMST, rad）。Meeus の近似式（UT1≈UTC と仮定）"""
    T = (jd_ut1 - 2451545.0) / 36525.0
    gmst_deg = 280.46061837 + 360.98564736629 * (jd_ut1 - 2451545.0) + 0.000387933 * T * T - (T ** 3) / 38710000.0
    gmst_rad = math.radians(gmst_deg % 360.0)
    return gmst_rad


def _sun_ra_dec_from_jd(jd: float) -> Tuple[float, float]:
    """太陽 赤経RA/赤緯Dec（rad）。簡易近似（誤差~数分角）"""
    T = (jd - 2451545.0) / 36525.0
    # 太陽の平均黄経 / 平均近点離角
    L0 = 280.46646 + 36000.76983 * T + 0.0003032 * T * T  # deg
    M = 357.52911 + 35999.05029 * T - 0.0001537 * T * T   # deg
    L0 = math.radians(L0 % 360.0)
    M = math.radians(M % 360.0)
    # 近似的な黄経補正
    C = math.radians((1.914602 - 0.004817 * T - 0.000014 * T * T) * math.sin(M)
                     + (0.019993 - 0.000101 * T) * math.sin(2 * M)
                     + 0.000289 * math.sin(3 * M))
    lam = L0 + C  # 真黄経
    # 地球の平均黄道傾斜角（近似）
    eps = math.radians(23.439291 - 0.0130042 * T)
    # 赤経・赤緯
    ra = math.atan2(math.cos(eps) * math.sin(lam), math.cos(lam))
    dec = math.asin(math.sin(eps) * math.sin(lam))
    # [0,2π) に整形
    if ra < 0:
        ra += 2.0 * math.pi
    return ra, dec


# ---------------------------------------------------------------------
# 上位API: computeSWinputs_JB2008（MATLAB互換）
# ---------------------------------------------------------------------

def compute_swinputs_jb2008(
    jd0: float,
    jdf: float,
    *,
    eopdata: np.ndarray,
    SOLdata: np.ndarray,
    DTCdata: np.ndarray,
    compute_jb2000_swinputs: ComputeSWFunc | None = None,
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
    if compute_jb2000_swinputs is None:
        compute_jb2000_swinputs = compute_jb2000_sw_inputs  # 本ファイル内の実装を既定に

    # 1時間刻み（両端含む）で jd0..jdf
    tt = np.arange(jd0, jdf + 1e-12, 1.0 / 24.0, dtype=float)
    N = tt.size
    Inputs = np.zeros((24, N), dtype=float)

    for i, jdate in enumerate(tt):
        yy, mm, dd, hh, mn, ss = _jd_to_ymdhms(jdate)
        doy = _day_of_year(yy, mm, dd)
        UThrs = hh + mn / 60.0 + ss / 3600.0

        # JB2008の宇宙天気値（下の compute_jb2000_sw_inputs を使用）
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


# ---------------------------------------------------------------------
# 下位API: computeJB2000SWinputs（MATLAB互換）
#   ※ CSPICE/IAUを使わず、同等の値を簡易近似で算出（GMST, Sun RA/Dec）
#   ※ SOLdata, DTCdata は MATLAB と同じ並びを想定
#       - SOLdata rows (1-based in MATLAB):
#           3: JD, 4:F10, 5:F10B, 6:S10, 7:S10B, 8:XM10, 9:XM10B, 10:Y10, 11:Y10B
#       - DTCdata rows:
#           1: year, 2: DOY, 3..26: hourly storm DTC values (0..23h → row index = floor(hour)+3)
# ---------------------------------------------------------------------

def compute_jb2000_sw_inputs(
    year: int,
    doy: int,
    hour: int,
    minute: int,
    sec: float,
    SOLdata: np.ndarray,
    DTCdata: np.ndarray,
    eopdata: np.ndarray,  # 互換用（本実装では未使用）
) -> Tuple[float, np.ndarray, float, float, float, float, float, float, float, float, float]:
    """
    MATLAB computeJB2000SWinputs.m の簡易ポート
    Returns
    -------
    GWRAS : float
        Greenwich Apparent Sidereal Angle ≈ GMST [rad] の近似
    SUN : np.ndarray([ra, dec])
        太陽の赤経・赤緯 [rad]（簡易近似）
    F10, F10B, S10, S10B, XM10, XM10B, Y10, Y10B : float
        JB2008用の遅延付き代理量（SOLdataから取得）
    DSTDTC : float
        ジオマグ・ストーム DTC 値（時刻内補間＋0.5 オフセット）
    """
    # --- カレンダー → MJD/JD
    month, day = _days2mdh(year, doy)
    MJD = _mjd(year, month, day, hour, minute, sec)
    JD_current = math.floor(MJD - 1.0 + 2400000.5)  # MATLAB の JD 同等

    # --- SOLdata からラグ付き代理量を抽出
    # 行インデックス: Pythonは0始まりなので -1 する
    jd_row = 2   # MATLAB 3行目
    i_candidates = np.where(np.floor(SOLdata[jd_row, :]) == JD_current)[0]
    if i_candidates.size == 0:
        # 近いインデックス（安全用フォールバック）
        i = int(np.argmin(np.abs(SOLdata[jd_row, :] - JD_current)))
    else:
        i = int(i_candidates[0])

    def srow(k: int) -> float:
        return float(SOLdata[k - 1, i])  # k is 1-based in MATLAB

    F10  = srow(4)
    F10B = srow(5)
    S10  = srow(6)
    S10B = srow(7)

    # XM10 は 2日前のラグ、Y10 は 5日前のラグ（MATLABの i-1, i-4 のオフセット）
    # SOLdata の並び次第で境界をクリップ
    i_m2 = max(i - 1, 0)    # XM10 (コメント上は2日ラグ, 元コードは i-1)
    i_m5 = max(i - 4, 0)    # Y10  (5日ラグ)
    XM10  = float(SOLdata[8 - 1, i_m2])   # row 8
    XM10B = float(SOLdata[9 - 1, i_m2])   # row 9
    Y10   = float(SOLdata[10 - 1, i_m5])  # row 10
    Y10B  = float(SOLdata[11 - 1, i_m5])  # row 11

    # --- DTCdata から DSTDTC（時間線形補間＋0.5）
    # 列選択： year & floor(doy) に一致
    doy_floor = int(math.floor(doy))
    col_candidates = np.where((DTCdata[0, :] == year) & (DTCdata[1, :] == doy_floor))[0]
    if col_candidates.size == 0:
        # フォールバック：近い列
        j = int(np.argmin(np.hypot(DTCdata[0, :] - year, DTCdata[1, :] - doy_floor)))
    else:
        j = int(col_candidates[0])

    ii = int(math.floor(hour)) + 3  # 0..23h → 3..26 row index
    ii = max(3, min(ii, DTCdata.shape[0] - 1))  # 領域内へ
    DSTDTC1 = float(DTCdata[ii - 1, j])  # Pythonは0始まり

    # 次の時間の値
    if ii >= 26:
        # 次の日の最初の時間（row=3）
        if j + 1 < DTCdata.shape[1]:
            DSTDTC2 = float(DTCdata[3 - 1, j + 1])
        else:
            DSTDTC2 = DSTDTC1
    else:
        DSTDTC2 = float(DTCdata[ii, j])

    sec_in_hour = minute * 60.0 + sec
    alpha = np.clip(sec_in_hour / 3600.0, 0.0, 1.0)
    DSTDTC = (1.0 - alpha) * DSTDTC1 + alpha * DSTDTC2
    DSTDTC += 0.5

    # --- GMST (GWRAS 近似) & SUN RA/Dec 近似
    jd_utc = _jd_from_calendar(year, month, day, hour, minute, sec)
    GWRAS = _gmst_from_jd(jd_utc)
    ra_sun, dec_sun = _sun_ra_dec_from_jd(jd_utc)
    SUN = np.array([ra_sun, dec_sun], dtype=float)

    return GWRAS, SUN, F10, F10B, S10, S10B, XM10, XM10B, Y10, Y10B, DSTDTC
