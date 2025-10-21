# Origin: propagateState_MeeBcRom.m
# License: GNU GPL v3
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

try:
    from scipy.integrate import solve_ivp
except Exception as e:
    raise RuntimeError("scipy が必要です（scipy.integrate.solve_ivp）。requirements.txt に追加してください。") from e


def _wrap_to_2pi(a: np.ndarray) -> np.ndarray:
    return np.mod(a, 2.0 * np.pi)


def propagate_state_mee_bc_rom(
    x0_mee: np.ndarray,
    t0: float,
    tf: float,
    *,
    AC: np.ndarray,
    BC: np.ndarray,
    SWinputs: dict,
    r: int,
    nop: int,
    svs: int,
    F_U,
    M_U,
    maxAtmAlt: float,
    et0: float,
    jdate0: float,
    # 依存関数を注入（MATLAB: ep2pv, pv2ep, computeDerivative_PosVelBcRom, isdecayed）
    ep2pv_fn: Callable[[np.ndarray, float], Tuple[np.ndarray, np.ndarray]],
    pv2ep_fn: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
    derivative_fn: Callable[[float, np.ndarray], np.ndarray],
    events_fn: Optional[Callable[[float, np.ndarray], float]] = None,
    mu_km3s2: float = 398600.4415,
) -> np.ndarray:
    """
    x0_mee: (svs*nop + r, 2n+1) のシグマ点行列（列ベース）を想定（MATLAB互換）
    返り値: 同次元で tf 時刻へ伝播後の MEE(+BC+ROM) シグマ点行列
    """
    ncols = x0_mee.shape[1]

    # --- MEE → (pos, vel) へ
    xx_pv = x0_mee.copy()
    for k in range(nop):
        for j in range(ncols):
            mee = x0_mee[(k*svs):(k*svs+6), j]
            pos, vel = ep2pv_fn(mee, mu_km3s2)  # (3,), (3,)
            xx_pv[(k*svs):(k*svs+3), j] = pos
            xx_pv[(k*svs+3):(k*svs+6), j] = vel

    # --- ODE 伝播（全シグマ点を 1 ベクトルに詰めて一括伝播）
    x_init = xx_pv.reshape(-1, order="F")  # 列順に詰める（MATLABの reshape と整合）
    def rhs(t, x_flat):
        return derivative_fn(t, x_flat)  # ユーザ提供（MATLAB: computeDerivative_PosVelBcRom）

    if events_fn is not None:
        sol = solve_ivp(rhs, (t0, tf), x_init, rtol=1e-10, atol=1e-10, events=events_fn)
    else:
        sol = solve_ivp(rhs, (t0, tf), x_init, rtol=1e-10, atol=1e-10)

    x_final = sol.y[:, -1]
    xf_pv = x_final.reshape(xx_pv.shape, order="F")

    # --- (pos, vel) → MEE へ戻す + 角度のラップ処理
    xf_mee = xf_pv.copy()
    for k in range(nop):
        for j in range(ncols):
            pos = xf_pv[(k*svs):(k*svs+3), j]
            vel = xf_pv[(k*svs+3):(k*svs+6), j]
            mee = pv2ep_fn(pos, vel, mu_km3s2)  # (6,)
            xf_mee[(k*svs):(k*svs+6), j] = mee

        # L のラップ（名目列＝1列目に対して）
        L0 = xf_mee[k*svs + 5, 0]  # 0始まり→6番目は index 5
        if (L0 > np.pi/2) or (L0 < -np.pi/2):
            xf_mee[k*svs + 5, :] = _wrap_to_2pi(xf_mee[k*svs + 5, :])

    return xf_mee
