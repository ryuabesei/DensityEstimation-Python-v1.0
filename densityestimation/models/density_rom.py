# Origin: Density_ROM.m
# License: GNU GPL v3
from __future__ import annotations

from datetime import datetime
from typing import Callable, List, Sequence

import numpy as np

# 型ヒント
BasisFcn = Callable[[float, float, float], float]  # (slt[hr], lat[deg], alt[km]) -> scalar


def _matlab_datenum(y: int, m: int, d: int, hh: int, mm: int, ss: float) -> float:
    """
    MATLAB datenum 互換の「相対値」計算。
    実際の絶対オフセットは不要で、差分が一致すればOK。
    → Pythonの toordinal() + 時分秒/86400 を使う。
    """
    dt = datetime(y, m, d, hh, mm, int(np.floor(ss)))
    frac = (ss - np.floor(ss)) / 86400.0
    return dt.toordinal() + hh/24.0 + mm/1440.0 + ss/86400.0  # fracも含める


def _datenum_vec(ymd_array: np.ndarray) -> np.ndarray:
    """
    ymd_array: shape (N, 6) → MATLAB datenum 相当の連続値 (N,)
    """
    ymd_array = np.asarray(ymd_array, dtype=float)
    if ymd_array.ndim == 1:
        ymd_array = ymd_array.reshape(1, -1)
    out = []
    for y, mo, d, hh, mi, ss in ymd_array:
        out.append(_matlab_datenum(int(y), int(mo), int(d), int(hh), int(mi), float(ss)))
    return np.asarray(out, dtype=float)


def _interp1_with_nan(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """
    MATLAB interp1 の既定に合わせ、範囲外は NaN を返す線形補間。
    x: (T,), y: (T,), xq: (N,)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xq = np.asarray(xq, dtype=float)
    yq = np.interp(xq, x, y, left=np.nan, right=np.nan)
    # np.interp は left/right を値で埋めるので、範囲外をNaNにするためにマスク
    mask = (xq < x.min()) | (xq > x.max())
    yq[mask] = np.nan
    return yq


def density_rom(
    ymd: Sequence[Sequence[float]] | Sequence[float],
    lla: Sequence[Sequence[float]] | Sequence[float],
    *,
    X_est: np.ndarray,
    F_U: List[BasisFcn],      # 長さ r の基底関数リスト（F_U[j](slt,lat,alt)）
    M_U: BasisFcn,            # 平均項の関数 M_U(slt,lat,alt)
    TXest_st: Sequence[int],  # 推定開始日時 [Y,M,D,hh,mm,ss]（少なくとも Y,M,D を使用）
    r: int = 10,
) -> np.ndarray:
    """
    ROM で推定した状態 X_est から密度を再構成。

    Parameters
    ----------
    ymd : array-like (N,6) または (6,)
        観測時刻ベクトル [year, month, day, hour, minute, sec]
        単一時刻でも配列でもOK。
    lla : array-like (N,3) or (3,)
        [lat[deg], lon[deg], alt[m]]（緯度・経度・高度）
        ※高度は [m] 入力（MATLAB準拠）→ 内部で [km] へ変換。
    X_est : (nx, T) ndarray
        ROM 状態の時系列。列が時間、行は各状態成分。
    F_U : list of callables, len=r
        変動項の空間基底（SLT, Lat, Alt を引数にとる）。
    M_U : callable
        平均項（SLT, Lat, Alt → 対数密度の平均）。
    TXest_st : sequence
        推定シリーズの開始日時（MATLABの TXest_st と同義）。
    r : int
        使用する末端基底の数（X_est の末尾 r 行を使う）。

    Returns
    -------
    Den_ref : (N,) ndarray
        再構成した密度 [kg/m^3]（10^(rhoVar + MI)）。
        （高度 < 120 km の要素は 0）
    """
    # ---- 入力の整形 ----
    ymd = np.asarray(ymd, dtype=float)
    if ymd.ndim == 1:
        ymd = ymd.reshape(1, -1)  # (1,6)
    assert ymd.shape[1] >= 6, "ymd must have 6 columns [Y M D h m s]"

    lla = np.asarray(lla, dtype=float)
    if lla.ndim == 1:
        lla = lla.reshape(1, -1)  # (1,3)
    assert lla.shape[1] >= 3, "lla must have 3 columns [lat lon alt_m]"

    N = max(ymd.shape[0], lla.shape[0])
    if ymd.shape[0] == 1 and N > 1:
        ymd = np.repeat(ymd, N, axis=0)
    if lla.shape[0] == 1 and N > 1:
        lla = np.repeat(lla, N, axis=0)

    lat_deg = lla[:, 0]
    lon_deg = lla[:, 1]
    alt_km = lla[:, 2] * 1e-3  # m → km

    # ---- MATLAB datenum 互換の時間基準 ----
    # year_st = datenum([TXest_st(1) 1 1])
    year0 = int(TXest_st[0])
    year_st = _matlab_datenum(year0, 1, 1, 0, 0, 0.0)
    # day_st = datenum(TXest_st) - year_st + 1
    if len(TXest_st) >= 6:
        dn_start = _matlab_datenum(int(TXest_st[0]), int(TXest_st[1]), int(TXest_st[2]),
                                   int(TXest_st[3]), int(TXest_st[4]), float(TXest_st[5]))
    else:
        dn_start = _matlab_datenum(int(TXest_st[0]), int(TXest_st[1]), int(TXest_st[2]), 0, 0, 0.0)
    day_st = dn_start - year_st + 1.0

    # day_end = day_st + (len_X-1)/24
    _, len_X = X_est.shape
    day_end = day_st + (len_X - 1) / 24.0

    # T_wh = datenum(ymd) - year_st + 1
    T_wh = _datenum_vec(ymd) - year_st + 1.0  # (N,)

    # Local Solar Time
    # Slt_wh = mod((T_wh - floor(T_wh))*24 + Lon/15, 24)
    Slt_wh = np.mod((T_wh - np.floor(T_wh)) * 24.0 + lon_deg / 15.0, 24.0)

    # ---- X_est の時間補間（MATLAB interp1, 範囲外は NaN）----
    T_in = np.arange(day_st, day_end + 1e-12, 1.0 / 24.0)  # 1時間刻み
    X_est_in = np.empty((X_est.shape[0], N), dtype=float)
    for i in range(X_est.shape[0]):
        X_est_in[i, :] = _interp1_with_nan(T_in, X_est[i, :], T_wh)

    # ---- 密度再構成 ----
    Den_ref = np.zeros(N, dtype=float)

    # 高度 < 120 km → 0
    below120 = alt_km < 120.0

    for i_time in range(N):
        if below120[i_time] or np.any(np.isnan(X_est_in[:, i_time])):
            Den_ref[i_time] = 0.0
            continue

        slt = float(Slt_wh[i_time])
        lat = float(lat_deg[i_time])
        alt = float(alt_km[i_time])

        # rhoVar = Σ_j U_j(slt,lat,alt) * X_est_in[end-r+j, i]
        rhoVar = 0.0
        # X_est の「末尾 r 行」を使用（MATLAB: end-r+j, j=1..r）
        for j in range(1, r + 1):
            basis = F_U[j - 1]  # Pythonは0始まり
            UhI = float(basis(slt, lat, alt))
            idx = -r + (j - 1)
            rhoVar += UhI * float(X_est_in[idx, i_time])

        MI = float(M_U(slt, lat, alt))
        Den_ref[i_time] = 10.0 ** (rhoVar + MI)

    return Den_ref
