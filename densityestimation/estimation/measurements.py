from __future__ import annotations

from typing import Callable, Iterable, Tuple

import numpy as np

from densityestimation.estimation.observations import EstimationStateSpec, unpack_state
from densityestimation.orbit.mee import cartesian_to_mee  # 既存想定


def fullmee2mee(Xp: np.ndarray, nop: int, svs: int) -> np.ndarray:
    """
    Extract only the MEE components from full state trajectories.

    Parameters
    ----------
    Xp : ndarray
        Full state history with shape (svs*nop + r, N) or a single column (svs*nop + r,).
    nop : int
        Number of objects.
    svs : int
        State vector size per object (typically 7 = 6 MEE + 1 BC).

    Returns
    -------
    mee : ndarray
        Stacked MEE states with shape (6*nop, N).
    """
    Xp = np.asarray(Xp, dtype=float)
    if Xp.ndim == 1:
        Xp = Xp[:, None]  # (D,) -> (D,1)

    if Xp.shape[0] < svs * nop:
        raise ValueError(f"Xp has insufficient rows: got {Xp.shape[0]}, need >= {svs*nop}")

    N = Xp.shape[1]
    mee = np.zeros((6 * nop, N), dtype=float)
    for k in range(nop):
        start = svs * k
        mee[6 * k : 6 * (k + 1), :] = Xp[start : start + 6, :]
    return mee


def make_measurement_model(spec: EstimationStateSpec,
                           tle_provider: Callable[[int, float], Tuple[np.ndarray, np.ndarray, float]]):
    """
    Build the measurement function h(x, t_epoch) that returns the observed MEE
    derived from TLE (TEME->ECIは tle_provider 側で済ませる/またはここでMEE変換).

    Parameters
    ----------
    spec : EstimationStateSpec
        n_obj, nz を保持。
    tle_provider : callable
        tle_provider(i_obj, t_epoch) -> (r, v, mu)
        inertial frame position/velocity [km, km/s] と重力係数 mu [km^3/s^2] を返す。

    Returns
    -------
    h : callable
        h(x, t_epoch) -> z_meas (shape=(6*n_obj,))
        物体ごとに (p,f,g,h,k,L) を縦に積んだ観測ベクトル。
        ※ x は本関数では参照しない（観測はTLE起因のため）。
    """
    def h(x: np.ndarray, t_epoch: float) -> np.ndarray:
        zs: list[float] = []
        for i in range(spec.n_obj):
            r_meas, v_meas, mu = tle_provider(i, t_epoch)
            mee_meas = cartesian_to_mee(np.asarray(r_meas, float),
                                        np.asarray(v_meas, float),
                                        float(mu))
            zs.extend([float(mee_meas[0]), float(mee_meas[1]), float(mee_meas[2]),
                       float(mee_meas[3]), float(mee_meas[4]), float(mee_meas[5])])
        return np.array(zs, dtype=float)
    return h


def make_measurement_noise_builder(spec: EstimationStateSpec):
    """
    Build a function R_builder(x_pred) that returns the block-diagonal
    measurement covariance R based on each object's current eccentricity
    e = sqrt(f^2 + g^2).  (対角のみ・物体間独立)

    係数は論文でのスケーリングに倣い、p と (f,g) を e で強めにスケール。
      c1 = 1.5 * max(4*e, 0.0023)
      c2 = 3.0 * max(e/0.004, 1.0)
      R = diag([c1*1e-8, c2*1e-10, c2*1e-10, 1e-9, 1e-9, 1e-8])  for each object
    """
    def _single_R_from_e(e: float) -> np.ndarray:
        e = float(abs(e))
        c1 = 1.5 * max(4.0 * e, 0.0023)
        c2 = 3.0 * max(e / 0.004, 1.0)
        Rp, Rf, Rg, Rh, Rk, RL = (
            c1 * 1e-8,
            c2 * 1e-10,
            c2 * 1e-10,
            1e-9,
            1e-9,
            1e-8,
        )
        return np.diag([Rp, Rf, Rg, Rh, Rk, RL])

    def R_builder(x_pred: np.ndarray) -> np.ndarray:
        # 予測状態から MEE を取り出して e を算出
        mee_list, _, _ = unpack_state(np.asarray(x_pred, float), spec)
        blocks = []
        for (p, f, g, h, k, L) in mee_list:
            e = np.hypot(f, g)
            blocks.append(_single_R_from_e(e))
        # ブロック対角へ
        return _block_diag(*blocks)

    return R_builder


def _block_diag(*arrs: Iterable[np.ndarray]) -> np.ndarray:
    """軽量なブロック対角作成（scipy 非依存）。"""
    arrs = [np.atleast_2d(np.asarray(a, float)) for a in arrs]
    shapes = np.array([a.shape for a in arrs])
    out = np.zeros((shapes[:,0].sum(), shapes[:,1].sum()), dtype=float)
    r, c = 0, 0
    for a in arrs:
        rr, cc = a.shape
        out[r:r+rr, c:c+cc] = a
        r += rr
        c += cc
    return out
