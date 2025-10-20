# densityestimation/rom/den_cal.py
# License: GNU GPL v3
from __future__ import annotations

from typing import Sequence

import numpy as np
from sgp4.api import jday


def _wrap_slt(jd: float, lon_deg: float) -> float:
    """
    太陽地方時 SLT = mod( (JD - floor(JD))*24 + lon/15, 24 )
    """
    return ((jd - np.floor(jd)) * 24.0 + lon_deg / 15.0) % 24.0


def _ymd_to_jd(ymd: Sequence[float | int]) -> float:
    """
    (year, month, day, hour, minute, second) → Julian Date
    sgp4.api.jday を使って高信頼に変換する。
    """
    if len(ymd) != 6:
        raise ValueError("ymd must be (year, month, day, hour, minute, second)")
    y, m, d, hh, mm, ss = ymd
    jd0, fr = jday(int(y), int(m), int(d), int(hh), int(mm), float(ss))
    return jd0 + fr


def density_from_bundle(
    ymd: Sequence[float | int],
    lla: Sequence[float],
    *,
    rom_bundle,
    X_est_hist: np.ndarray,
    jd0: float,
    dt_hours: float = 1.0,
) -> float:
    """
    推定済み ROM と状態時系列から、指定時刻・地点の熱圏密度を 1 点計算。

    Parameters
    ----------
    ymd : (year, month, day, hour, minute, second)
        UTC 時刻（推定期間内を想定）。
    lla : (lat_deg, lon_deg, alt_m)
        観測点の緯度[deg], 経度[deg], 高度[m]。高度は海抜。
    rom_bundle :
        run_density_estimation_tle(...) の戻り dict["rom"]（ROMBundle 相当）。
        必須フィールド:
          - F_U : List[Callable(slt, lat_deg, alt_km) -> float] （POD 空間モードの補間器）
          - M_U : Callable(slt, lat_deg, alt_km) -> float       （log10(密度)平均の補間器）
          - maxAtmAlt : float （ROM の上限高度 [km]）
    X_est_hist : ndarray, shape (n_state, m)
        状態の時系列（列が時間）。末尾 r 行が ROM モード z(t)。
    jd0 : float
        推定開始のユリウス日（X_est_hist の 0 列がこの時刻）。
    dt_hours : float, default 1.0
        X_est_hist の時間刻み（時間）。

    Returns
    -------
    rho : float
        密度 [kg/m^3]
    """
    if len(lla) != 3:
        raise ValueError("lla must be (lat_deg, lon_deg, alt_m)")
    lat_deg, lon_deg, alt_m = float(lla[0]), float(lla[1]), float(lla[2])

    alt_km = alt_m * 1e-3
    # MATLAB Density_ROM.m と同じ閾値処理（<120km は 0 扱い）
    if alt_km < 120.0 or alt_km > float(rom_bundle.maxAtmAlt):
        return 0.0

    # 時刻換算
    jd = _ymd_to_jd(ymd)
    slt = _wrap_slt(jd, lon_deg)

    # ROM モード次元 r
    r = len(rom_bundle.F_U)
    if r <= 0:
        raise ValueError("rom_bundle.F_U must contain at least one mode")

    # z(t) の補間（X_est_hist の末尾 r 行が z）
    T_grid = np.arange(X_est_hist.shape[1], dtype=float) * float(dt_hours)
    t_hours = (jd - float(jd0)) * 24.0
    z_hist = X_est_hist[-r:, :]  # shape: (r, m)

    # 端点外は最近傍保持（np.interp は端で値を保持）
    z = np.array([np.interp(t_hours, T_grid, z_hist[k, :]) for k in range(r)])

    # 空間モードの評価 + 平均場（log10 密度）
    rho_var = 0.0
    for j in range(r):
        # 各補間器はスカラ値を返すことを想定
        rho_var += float(rom_bundle.F_U[j](slt, lat_deg, alt_km)) * float(z[j])
    mean_log10 = float(rom_bundle.M_U(slt, lat_deg, alt_km))

    # 密度に変換：rho = 10^(rho_var + mean)
    rho = 10.0 ** (rho_var + mean_log10)
    return float(rho)
