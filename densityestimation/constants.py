# densityestimation/constants.py の末尾などに追加
from __future__ import annotations

import math
from typing import Tuple


def get_gravc(whichconst: int) -> Tuple[float, float, float, float, float, float, float, float]:
    """
    NORAD/Vallado 系の重力定数セットを返す。

    Parameters
    ----------
    whichconst : int
        721, 72, or 84

    Returns
    -------
    (tumin, mu, radiusearthkm, xke, j2, j3, j4, j3oj2)
    """
    if whichconst == 721:
        mu = 398600.79964
        radiusearthkm = 6378.135
        xke = 0.0743669161
        tumin = 1.0 / xke
        j2 = 0.001082616
        j3 = -0.00000253881
        j4 = -0.00000165597
        j3oj2 = j3 / j2
    elif whichconst == 72:
        mu = 398600.8
        radiusearthkm = 6378.135
        xke = 60.0 / math.sqrt(radiusearthkm**3 / mu)
        tumin = 1.0 / xke
        j2 = 0.001082616
        j3 = -0.00000253881
        j4 = -0.00000165597
        j3oj2 = j3 / j2
    elif whichconst == 84:
        mu = 398600.5
        radiusearthkm = 6378.137
        xke = 60.0 / math.sqrt(radiusearthkm**3 / mu)
        tumin = 1.0 / xke
        j2 = 0.00108262998905
        j3 = -0.00000253215306
        j4 = -0.00000161098761
        j3oj2 = j3 / j2
    else:
        raise ValueError(f"unknown gravity option ({whichconst})")
    return tumin, mu, radiusearthkm, xke, j2, j3, j4, j3oj2
