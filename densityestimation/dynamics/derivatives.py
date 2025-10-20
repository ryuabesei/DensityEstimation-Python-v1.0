# Port of computeDerivative_PosVelBcRom.m
# License: GNU GPL v3
from __future__ import annotations

from typing import Callable

import numpy as np


def build_derivative_posvel_bc_rom(
    *,
    AC: np.ndarray,
    BC: np.ndarray,
    SWinputs: np.ndarray,        # shape (>=12, N) with first row jdate
    r: int,
    noo: int,
    svs: int,
    # 依存を注入
    gravity_accel_ecef_fn: Callable[[np.ndarray], np.ndarray],
    get_density_rom_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, int], np.ndarray],
    j2000_to_ecef_xform6_fn: Callable[[float], np.ndarray],
    et0: float,
    jdate0: float,
) -> Callable[[float, np.ndarray], np.ndarray]:
    """
    Returns derivative function f(t, x_flat) -> dx_flat
    - gravity_accel_ecef_fn(rr_ecef_m) : (N,3)m → (N,3)m  [m/s^2]
    - get_density_rom_fn(rr_j2000_km, jdate_arr, rom_state, r) → rho (kg/m^3) [shape: ncols]
      * rr_j2000_km: shape (3, ncols)
      * jdate_arr:   shape (ncols,)
      * rom_state:   shape (r, ncols)
    - j2000_to_ecef_xform6_fn(et) -> 6x6 state transformation matrix (like SPICE sxform)
    """
    # SWinputs: rows [1..]= [jdate, ..., DSTDTC etc.]  MATLAB では interp1 で列を補間している
    jd_sw = SWinputs[0, :]
    sw_rest = SWinputs[1:, :]  # shape (K, N)

    def interp_sw(jdate: np.ndarray) -> np.ndarray:
        # 列方向線形補間。各成分独立に補間。
        # jdateは (ncols,) を想定 → 出力 shape (K, ncols)
        K, N = sw_rest.shape
        out = np.empty((K, jdate.size), dtype=float)
        for k in range(K):
            out[k, :] = np.interp(jdate, jd_sw, sw_rest[k, :], left=sw_rest[k, 0], right=sw_rest[k, -1])
        return out

    def f(t: float, xp: np.ndarray) -> np.ndarray:
        x = xp.reshape(svs * noo + r, -1, order="F")  # (m, ncols)
        m, ncols = x.shape

        # 時刻
        et = et0 + t
        jdate = jdate0 + t / 86400.0
        jdates = np.full((ncols,), jdate, dtype=float)

        # 宇宙天気（現在時刻の列を補間）
        SWt = interp_sw(jdates)  # shape (K, ncols)

        # 出力微分
        dx = np.zeros_like(x)

        # 各オブジェクト
        xform6 = j2000_to_ecef_xform6_fn(et)  # 6x6
        R_ecef_j2000 = xform6[:3, :3]         # J2000→ECEF の回転
        R_j2000_ecef = R_ecef_j2000.T         # ECEF→J2000

        for i in range(noo):
            base = i * svs
            # J2000 pos/vel
            rr_eci = x[base + 0: base + 3, :]  # (3, ncols) [km]
            vv_eci = x[base + 3: base + 6, :]  # (3, ncols) [km/s]

            # ECEF pos/vel（6x6変換を使ってまとめて）
            xe = xform6 @ np.vstack([rr_eci, vv_eci])  # (6, ncols)
            rr_ecef = xe[0:3, :]                        # [km]
            vv_ecef = xe[3:6, :]                        # [km/s]
            speed_ecef = np.linalg.norm(vv_ecef, axis=0)  # (ncols,)

            # 重力加速度 in ECEF [m/s^2]（入力は m）
            aa_grav_ecef = gravity_accel_ecef_fn((rr_ecef.T * 1000.0))  # (ncols, 3) m/s^2
            aa_grav_ecef = aa_grav_ecef.T / 1000.0  # → [km/s^2], shape (3, ncols)

            # 大気密度 [kg/m^3]（ROM）
            rom_state = x[-r:, :]  # (r, ncols)
            rho = get_density_rom_fn(rr_eci, jdates, rom_state, r)  # (ncols,)

            # BC [m^2/(1000kg)]
            b_star = x[base + 6, :]  # (ncols,)

            # 抗力加速度 in ECEF [km/s^2]: -0.5 * b* rho * |v| * v
            aa_drag_ecef = -0.5 * b_star * rho * speed_ecef * vv_ecef  # (3,ncols)

            # 合成加速度 ECEF → J2000（回転のみ）
            aa_ecef = aa_grav_ecef + aa_drag_ecef
            aa_eci = R_j2000_ecef @ aa_ecef  # (3,ncols)

            # 速度の微分 = 加速度
            dx[base + 0, :] = vv_eci[0, :]
            dx[base + 1, :] = vv_eci[1, :]
            dx[base + 2, :] = vv_eci[2, :]
            dx[base + 3: base + 6, :] = aa_eci
            # BC の時間微分は 0
            dx[base + 6, :] = 0.0

        # ROM: dz/dt = AC z + BC u
        # SWt をそのまま u とみなす（列数を合わせる）
        dx[-r:, :] = AC @ x[-r:, :] + BC @ SWt

        return dx.reshape(-1, order="F")

    return f
