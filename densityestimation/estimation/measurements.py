# Equivalent to MATLAB: fullmee2mee(xx,nop,svs)
from __future__ import annotations

import numpy as np


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
