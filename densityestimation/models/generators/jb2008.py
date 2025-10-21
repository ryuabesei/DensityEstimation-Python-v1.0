from __future__ import annotations

import numpy as np
from numpy.linalg import pinv
from scipy.linalg import logm


def _get(attr_or_dict, key):
    return getattr(attr_or_dict, key) if hasattr(attr_or_dict, key) else attr_or_dict[key]


def generateROM_JB2008(TA, r: int):
    """
    Python port of generateROM_JB2008.m

    Space weather columns (MATLAB コメント準拠):
      [doy, UThrs, F10, F10B, S10, S10B, XM10, XM10B, Y10, Y10B, DSTDTC, GWRAS, SUN(1), SUN(2)]
      ※列数は実データに依存。以下の演算で参照する列が存在すること。
    """
    U100 = _get(TA, "U100")
    Xrom = _get(TA, "densityDataLogVarROM100")
    SW   = _get(TA, "SWdataFull")

    Uh = U100[:, :r]

    X1 = Xrom[:r, :-1]
    X2 = Xrom[:r, 1:]

    U_base = SW[:-1, :].T  # (m, T-1)

    # 未来値（col: 11→10 DSTDTC, 3→2 F10, 5→4 S10, 7→6 XM10, 9→8 Y10）
    fut_DSTDTC = SW[1:, 10]
    fut_F10    = SW[1:,  2]
    fut_S10    = SW[1:,  4]
    fut_XM10   = SW[1:,  6]
    fut_Y10    = SW[1:,  8]

    cur_DSTDTC = SW[:-1, 10]
    cur_F10    = SW[:-1,  2]

    U_extra = np.vstack([
        fut_DSTDTC[None, :],   # row 15
        fut_F10[None, :],      # 16
        fut_S10[None, :],      # 17
        fut_XM10[None, :],     # 18
        fut_Y10[None, :],      # 19
        (cur_DSTDTC**2)[None, :],   # 20 (current DSTDTC^2)
        (fut_DSTDTC**2)[None, :],   # 21 (future DSTDTC^2)
        (cur_DSTDTC*cur_F10)[None, :],  # 22
        (fut_DSTDTC*fut_F10)[None, :],  # 23
    ])
    U1 = np.vstack([U_base, U_extra])
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
