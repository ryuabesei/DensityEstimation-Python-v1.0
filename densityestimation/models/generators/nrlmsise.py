from __future__ import annotations

import numpy as np
from numpy.linalg import pinv
from scipy.linalg import logm


def _get(attr_or_dict, key):
    return getattr(attr_or_dict, key) if hasattr(attr_or_dict, key) else attr_or_dict[key]


def generateROM_NRLMSISE(TA, r: int):
    """
    Python port of generateROM_NRLMSISE.m

    Space weather columns (MATLAB コメント準拠):
      [doy, UThrs, F10a, F10, ap1, ap2, ap3, ap4, ap5, ap6, ap7, ap8, ...]
      ここでは F10（col=3→idx=3）と ap のいずれか（col=6→idx=5）を混合に使用。
    """
    U100 = _get(TA, "U100")
    Xrom = _get(TA, "densityDataLogVarROM100")
    SW   = _get(TA, "SWdataFull")

    Uh = U100[:, :r]

    X1 = Xrom[:r, :-1]
    X2 = Xrom[:r, 1:]

    U_base = SW[:-1, :].T  # (m, T-1)

    # 未来 F10 と ap 群（MATLAB 3:11 → Python 2:11）
    fut_F10_ap = SW[1:, 2:11].T           # (9, T-1)
    cur_ap_sq  = (SW[:-1, 2:11]**2).T     # (9, T-1)
    fut_ap_sq  = (SW[1:,  2:11]**2).T     # (9, T-1)

    # 混合項 F10*ap （F10: col=4→idx=3, ap: col=6→idx=5）
    cur_mix = (SW[:-1, 3] * SW[:-1, 5])[None, :]
    fut_mix = (SW[1:,  3] * SW[1:,  5])[None, :]

    U1 = np.vstack([U_base, fut_F10_ap, cur_ap_sq, fut_ap_sq, cur_mix, fut_mix])
    q = U1.shape[0]

    Om  = np.vstack([X1, U1])
    Phi = (X2 @ pinv(Om))
    A   = Phi[:, :r]
    B   = Phi[:, r:]

    dth = 1.0
    Phi_aug = np.block([[A, B], [np.zeros((q, r)), np.eye(q)]])
    PhiC = logm(Phi_aug) / dth

    X2pred = A @ X1 + B @ U1
    err    = X2pred - X2
    Qrom   = np.cov(err)

    return PhiC, Uh, Qrom
