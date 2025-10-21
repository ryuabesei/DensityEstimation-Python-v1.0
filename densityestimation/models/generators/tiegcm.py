from __future__ import annotations

import numpy as np
from numpy.linalg import pinv
from scipy.linalg import logm


def _get(attr_or_dict, key):
    """TA が dict でもオブジェクトでも取れるように。"""
    return getattr(attr_or_dict, key) if hasattr(attr_or_dict, key) else attr_or_dict[key]


def generateROM_TIEGCM(TA, r: int):
    """
    Python port of generateROM_TIEGCM.m

    Inputs
    ------
    TA : object or dict
      必要キー:
        - U100 : (n_grid, 100) 左特異ベクトル
        - densityDataLogVarROM100 : (100, T) log10密度のROM係数（時間スナップ）
        - SWdataFull : (T, m) 宇宙天気系列（[doy, UThrs, F10, F10a, Kp, ...]）
    r : int
      ランク（使用モード数）

    Returns
    -------
    PhiC : (r+q, r+q) ndarray  連続時間ROM（1時間刻みで離散化した行列の logm / 1h）
    Uh   : (n_grid, r)         空間モード
    Qrom : (r, r)              1時間予測誤差の共分散
    """
    U100 = _get(TA, "U100")
    Xrom = _get(TA, "densityDataLogVarROM100")
    SW   = _get(TA, "SWdataFull")

    Uh = U100[:, :r]

    # ROM timesnaps
    X1 = Xrom[:r, :-1]   # (r, T-1)
    X2 = Xrom[:r, 1:]    # (r, T-1)

    # Space weather inputs（列インデックスは MATLAB(1-based)→Python(0-based)）
    # base: [doy; UThrs; F10; F10a; Kp]  => SW[0:5]
    U_base = SW[:-1, :].T  # (m, T-1)

    F10_fut = SW[1:, 2]    # col 3
    Kp_fut  = SW[1:, 4]    # col 5
    Kp_cur  = SW[:-1, 4]

    U_extra = np.vstack([
        F10_fut[None, :],                                # row 6
        Kp_fut[None, :],                                 # row 7
        (Kp_cur**2)[None, :],                            # row 8
        (Kp_fut**2)[None, :],                            # row 9
        (SW[:-1, 2]*Kp_cur)[None, :],                    # row 10 (F10*Kp current)
        (SW[1:,  2]*Kp_fut)[None, :],                    # row 11 (F10*Kp future)
    ])
    U1 = np.vstack([U_base, U_extra])  # (q, T-1)
    q = U1.shape[0]

    # DMDc: X2 = [A B] @ [X1; U1]
    Om  = np.vstack([X1, U1])          # (r+q, T-1)
    Phi = (X2 @ pinv(Om))              # (r, r+q)
    A   = Phi[:, :r]
    B   = Phi[:, r:]

    dth = 1.0  # hour
    Phi_aug = np.block([[A, B], [np.zeros((q, r)), np.eye(q)]])
    PhiC = logm(Phi_aug) / dth

    # 誤差共分散
    X2pred = A @ X1 + B @ U1
    err    = X2pred - X2            # (r, T-1)
    Qrom   = np.cov(err)            # rows as variables → (r,r)

    return PhiC, Uh, Qrom
