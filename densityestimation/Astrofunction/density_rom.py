# License: GNU GPL v3
from __future__ import annotations

import math
from typing import Callable, List, Tuple

import numpy as np

# ---- 基本定数（WGS-84）----
_WGS84_A = 6378.137            # km
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)


def _R3(theta: float) -> np.ndarray:
    """Z軸回り回転行列（右手系, ECI→ECEFに使う：r_ecef = R3(theta) @ r_eci）"""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[ c,  s, 0.0],
                     [-s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=float)


def _gmst_iau2006(jd: float) -> float:
    """
    簡易 GMST (rad) ：IAU 2006/2000A に近い近似式
      theta = 280.46061837 + 360.98564736629*(JD-2451545) + ...
    """
    T = (jd - 2451545.0) / 36525.0
    theta_deg = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * T * T
        - (T ** 3) / 38710000.0
    )
    theta = math.radians(theta_deg % 360.0)
    return theta


def _eci_to_ecef(r_eci: np.ndarray, jd: float) -> np.ndarray:
    """ECI(J2000) 位置ベクトル → ECEF 位置ベクトル（極移動等は無視した簡易版）"""
    th = _gmst_iau2006(jd)
    return _R3(th) @ r_eci


def _ecef_to_geodetic(r_ecef_km: np.ndarray) -> Tuple[float, float, float]:
    """
    ECEF(km) → (lon[deg], lat[deg], alt[km])  WGS-84 準拠の反復法。
    1点用。ベクトル化する場合はループで呼び出してください。
    """
    x, y, z = float(r_ecef_km[0]), float(r_ecef_km[1]), float(r_ecef_km[2])
    lon = math.degrees(math.atan2(y, x))

    a = _WGS84_A
    e2 = _WGS84_E2
    b = a * math.sqrt(1.0 - e2)

    r = math.hypot(x, y)
    if r == 0.0:  # 極付近の安定化
        lat = 90.0 if z >= 0 else -90.0
        alt = abs(z) - b
        return lon, lat, alt

    # Bowring の近似＋数回反復で十分
    E2 = a * a - b * b
    F = 54.0 * b * b * z * z
    G = r * r + (1.0 - e2) * z * z - e2 * E2
    c = (e2 * e2 * F * r * r) / (G * G * G)
    s = (1.0 + c + math.sqrt(c * c + 2.0 * c)) ** (1.0 / 3.0)
    P = F / (3.0 * (s + 1.0 / s + 1.0) ** 2 * G * G)
    Q = math.sqrt(1.0 + 2.0 * e2 * e2 * P)
    r0 = -(P * e2 * r) / (1.0 + Q) + math.sqrt(
        0.5 * a * a * (1.0 + 1.0 / Q)
        - (P * (1.0 - e2) * z * z) / (Q * (1.0 + Q))
        - 0.5 * P * r * r
    )
    U = math.sqrt((r - e2 * r0) ** 2 + z * z)
    V = math.sqrt((r - e2 * r0) ** 2 + (1.0 - e2) * z * z)
    z0 = (b * b * z) / (a * V)

    lat = math.degrees(math.atan2(z + z0 * e2, r))
    alt = U * (1.0 - (b * b) / (a * V))

    return lon, lat, alt


def _jd_to_ymdhms(jd: float) -> Tuple[int, int, int, int, int, float]:
    """ユリウス日 → UTC年月日時分秒（簡易）"""
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


def _ut_hours(jd: float) -> float:
    yy, mm, dd, hh, mn, ss = _jd_to_ymdhms(jd)
    return hh + mn / 60.0 + ss / 3600.0


def get_density_rom(
    pos_eci_km: np.ndarray,
    jdate: float | np.ndarray,
    rom_state: np.ndarray,
    r: int,
    F_U: List[Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]],
    M_U: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    max_atm_alt_km: float,
) -> np.ndarray:
    """
    MATLAB getDensityROM.m のポート。
    Parameters
    ----------
    pos_eci_km : shape (3, N)
    jdate      : JD (float or shape (N,))
    rom_state  : shape (r, N) または (r,)（シグマ点毎の列でもOK）
    F_U        : POD空間モードの補間関数リスト（各f(SLT,lat,alt)->値）
    M_U        : 平均対数密度の補間関数
    Returns
    -------
    rho : shape (N,)
        単位は kg/m^3 を想定（ROM側のスケーリングに依存）
    """
    pos = np.asarray(pos_eci_km, dtype=float)
    if pos.ndim != 2 or pos.shape[0] != 3:
        raise ValueError("pos_eci_km must be (3, N) array")

    N = pos.shape[1]
    jd = np.full(N, float(jdate), dtype=float) if np.isscalar(jdate) else np.asarray(jdate, dtype=float).reshape(-1)
    if jd.size != N:
        raise ValueError("jdate must be scalar or length-N array")

    # rom_state を (r, N) 形状にそろえる
    z = np.asarray(rom_state, dtype=float)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    if z.shape[0] != r:
        raise ValueError(f"rom_state first dimension must be r={r}")
    if z.shape[1] not in (1, N):
        raise ValueError("rom_state must have shape (r,) or (r,N)")
    if z.shape[1] == 1:
        z = np.repeat(z, N, axis=1)

    # ECI→ECEF→(lon,lat,alt)
    lon = np.empty(N, dtype=float)
    lat = np.empty(N, dtype=float)
    alt = np.empty(N, dtype=float)
    for k in range(N):
        r_ecef = _eci_to_ecef(pos[:, k], jd[k])
        lon[k], lat[k], alt[k] = _ecef_to_geodetic(r_ecef)

    # LST（local solar time）
    UT_hrs = np.array([_ut_hours(j) for j in jd])
    lst = UT_hrs + lon / 15.0
    lst = (lst + 24.0) % 24.0

    # 空間モード値
    UhI = np.zeros((N, r), dtype=float)
    for j in range(r):
        UhI[:, j] = F_U[j](lst, lat, alt)

    # 平均
    MI = M_U(lst, lat, alt)  # 対数密度の平均

    # 密度
    # 10^(sum_j UhI_j * z_j + MI)  （log10密度 → 密度）
    rho_log10 = (UhI @ z).diagonal() if z.shape[1] == N else np.sum(UhI * z.T, axis=1)
    rho = 10.0 ** (rho_log10 + MI)

    # 上限高度
    rho = np.where(alt > max_atm_alt_km, 0.0, rho)
    return rho
