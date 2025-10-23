# License: GNU GPL v3
#経度・緯度・高度（任意形状）と JD(UTC) を入力に、外部から渡された宇宙天気入力生成関数（compute_jb2000_swinputs）JB2008 本体の評価関数（jb2008_model）を使って JB2008 の密度を計算
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

# 依存（あなたの既存ファイル）:
# - jb2008_inputs.compute_jb2000_swinputs(...) を渡す
# - JB2008 モデル本体を呼び出す関数を渡す


def get_density_jb2008_llajd(
    lon_deg: float | np.ndarray,
    lat_deg: float | np.ndarray,
    alt_km: float | np.ndarray,
    jdate: float,
    *,
    compute_jb2000_swinputs: Callable[
        [int, int, int, int, float, np.ndarray, np.ndarray, np.ndarray],
        Tuple[float, np.ndarray, float, float, float, float, float, float, float, float, float],
    ],
    jb2008_model: Callable[
        [float, np.ndarray, np.ndarray, float, float, float, float, float, float, float, float, float],
        Tuple[float, float]
    ],
    eopdata: np.ndarray,
    SOLdata: np.ndarray,
    DTCdata: np.ndarray,
) -> np.ndarray:
    """
    MATLAB getDensityJB2008llajd.m のポート。
    JB2008 本体は外部関数 jb2008_model(...) として注入します。
    Parameters
    ----------
    lon_deg, lat_deg, alt_km : scalar or array (同じ形)
    jdate : JD（UTC）
    compute_jb2000_swinputs : MATLAB computeJB2000SWinputs 相当の関数
    jb2008_model : JB2008(MJD,SUN,SAT, F10,F10B,S10,S10B,XM10,XM10B,Y10,Y10B,DSTDTC)->(T,rho)
                   rho は [kg/m^3] を想定
    Returns
    -------
    rho_kg_km3 : 密度 [kg/km^3] （MATLAB と同じスケーリング）
    """
    lon = np.asarray(lon_deg, dtype=float)
    lat = np.asarray(lat_deg, dtype=float)
    alt = np.asarray(alt_km, dtype=float)
    # ブロードキャスト整形
    lon, lat, alt = np.broadcast_arrays(lon, lat, alt)
    out_shape = lon.shape

    # 日時
    yy, mm, dd, hh, mn, ss = _jd_to_ymdhms(jdate)
    doy = _day_of_year(yy, mm, dd)

    # 宇宙天気（JB2008）
    MJD, GWRAS, SUN, F10, F10B, S10, S10B, XM10, XM10B, Y10, Y10B, DSTDTC = _compute_jb2000_swinputs_py(
        yy, doy, hh, mn, ss, SOLdata, DTCdata, eopdata, compute_jb2000_swinputs
    )

    # SAT = [RA (GWRAS+lon), dec(lat), alt]
    XLON = np.deg2rad(lon)
    SAT1 = (GWRAS + XLON) % (2.0 * np.pi)
    SAT2 = np.deg2rad(lat)
    SAT3 = alt
    SAT = np.stack([SAT1, SAT2, SAT3], axis=-1).reshape(-1, 3)

    # JB2008 本体
    rho = np.empty(SAT.shape[0], dtype=float)
    for i in range(SAT.shape[0]):
        _, rho_i = jb2008_model(
            MJD,
            np.asarray(SUN, dtype=float),
            SAT[i, :],
            F10, F10B, S10, S10B, XM10, XM10B, Y10, Y10B, DSTDTC
        )
        rho[i] = float(rho_i)

    rho = rho.reshape(out_shape)
    # MATLAB は「* 1e9」で [kg/km^3] にしている
    return rho * 1e9


# ---- ヘルパ ----

def _jd_to_ymdhms(jd: float):
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
    return int((_dt.date(year, month, day) - _dt.date(year, 1, 1)).days) + 1


def _compute_jb2000_swinputs_py(
    yy: int, doy: int, hh: int, mn: int, ss: float,
    SOLdata: np.ndarray, DTCdata: np.ndarray, eopdata: np.ndarray,
    fn: Callable[..., tuple]
):
    # 直接 Python 版 compute_jb2000_swinputs を使えるように、関数を渡してもらう設計
    # 戻り順は MATLAB と一致させる
    GWRAS, SUN, F10, F10B, S10, S10B, XM10, XM10B, Y10, Y10B, DSTDTC = fn(
        yy, doy, hh, mn, ss, SOLdata, DTCdata, eopdata
    )
    # MJD を復元（MATLAB の computeJB2000SWinputs は Mjday() を内部で返す）
    # ここでは JD(UTC) から MJD(UTC) を作れば十分（JB2008 実装に依存）
    # MJD = JD - 2400000.5
    # JD は (yy,mm,dd,hh,mn,ss) から再生成してもよいが、上位で jdate を持っているなら差し替えてOK。
    jd_utc = _ymdhms_to_jd(yy, *_mdh_from_doy(yy, doy), hh, mn, ss)
    MJD = jd_utc - 2400000.5
    return MJD, GWRAS, SUN, F10, F10B, S10, S10B, XM10, XM10B, Y10, Y10B, DSTDTC


def _mdh_from_doy(yy: int, doy: int) -> Tuple[int, int]:
    import datetime as _dt
    d0 = _dt.date(yy, 1, 1) + _dt.timedelta(days=doy - 1)
    return d0.month, d0.day


def _ymdhms_to_jd(yy: int, mm: int, dd: int, h: int, m: int, s: float) -> float:
    import math as _m
    if mm <= 2:
        yy -= 1
        mm += 12
    A = yy // 100
    B = 2 - A + A // 5
    day = dd + (h + (m + s / 60.0) / 60.0) / 24.0
    return _m.floor(365.25 * (yy + 4716)) + _m.floor(30.6001 * (mm + 1)) + day + B - 1524.5
