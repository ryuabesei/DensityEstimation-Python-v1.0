# Equivalent to MATLAB: fullmee2mee(xx,nop,svs)
# MEE生成
from __future__ import annotations

import numpy as np

from densityestimation.estimation.observations import EstimationStateSpec, unpack_state
from densityestimation.orbit.mee import cartesian_to_mee  # 既存想定


def fullmee2mee(Xp: np.ndarray, nop: int, svs: int) -> np.ndarray:
    """
    Extracts only the Modified Equinoctial Elements (MEE)
    from the full state vector that also includes BC and ROM.

    Parameters
    ----------
    Xp : ndarray
        Full state vector [svs*nop + r, N]
    nop : int
        Number of objects
    svs : int
        State vector size per object (typically 7 = 6 MEE + 1 BC)

    Returns
    -------
    mee : ndarray
        Stacked MEE states [6*nop, N]
    """
    mee = np.zeros((6 * nop, Xp.shape[1]), dtype=float)
    for k in range(nop):
        mee[6 * k : 6 * (k + 1), :] = Xp[svs * k : svs * k + 6, :]
    return mee


def make_measurement_model(spec: EstimationStateSpec, tle_provider):
    """
    tle_provider: callable(i_obj, t_epoch) -> (r,v) in inertial frame
    """
    def h(x, t_epoch):
        # 物体ごとに「TLEから得たMEE」を観測として返す
        mee_list, bc_list, z = unpack_state(x, spec)
        zdim = len(z)
        z_dummy = np.zeros((zdim,))  # 観測にはROM状態は含めない

        zs = []
        for i in range(spec.n_obj):
            r_meas, v_meas, mu = tle_provider(i, t_epoch)
            mee_meas = cartesian_to_mee(r_meas, v_meas, mu)
            zs.extend(list(mee_meas))  # (p,f,g,h,k,L) の6要素
        # 物体ごと6要素の縦持ち配列を返す
        return np.array(zs, dtype=float)
    return h