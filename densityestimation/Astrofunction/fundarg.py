# License: GNU GPL v3
#天体力学・潮汐・摂動の基本角（fundamental arguments）の計算
#ユリウス世紀 ttt（TT/TDB 基準）を入力として、Delaunay 引数と主要惑星の黄経などの基本角を計算し、ラジアンで返す。
from __future__ import annotations

from typing import Tuple

import numpy as np


def _deg2rad(x: float) -> float:
    return x * np.pi / 180.0

def _wrap_deg(x: float) -> float:
    """wrap to [0,360) degrees"""
    return x % 360.0

def fundarg(
    ttt: float,
    opt: str = "10",
) -> Tuple[float, float, float, float, float, float, float, float, float, float, float]:
    """
    Fundamental arguments (IAU '10/'02/'96/'80).

    Parameters
    ----------
    ttt : float
        Julian centuries of TT (TDB) from J2000.0
    opt : {'10','02','96','80'}
        Theory selection.

    Returns
    -------
    l, l1, f, d, omega,
    lonmer, lonven, lonear, lonmar, lonjup, lonsat, lonurn, lonnep, precrate : radians
    """
    t2 = ttt * ttt
    t3 = t2 * ttt
    t4 = t2 * t2

    # Initialize (deg)
    l = l1 = f = d = omega = 0.0
    lonmer = lonven = lonear = lonmar = lonjup = lonsat = lonurn = lonnep = precrate = 0.0

    if opt == "10":
        # Delaunay (deg)
        l     = 134.96340251 + (1717915923.2178 * ttt + 31.8792 * t2 + 0.051635 * t3 - 0.00024470 * t4) / 3600.0
        l1    = 357.52910918 + (129596581.0481  * ttt - 0.5532   * t2 - 0.000136 * t3 - 0.00001149 * t4) / 3600.0
        f     =  93.27209062 + (1739527262.8478 * ttt - 12.7512  * t2 + 0.001037 * t3 + 0.00000417 * t4) / 3600.0
        d     = 297.85019547 + (1602961601.2090 * ttt - 6.3706   * t2 + 0.006593 * t3 - 0.00003169 * t4) / 3600.0
        omega = 125.04455501 + (  -6962890.5431 * ttt + 7.4722   * t2 + 0.007702 * t3 - 0.00005939 * t4) / 3600.0
        # Planetary (deg)
        lonmer   = 252.250905494  + 149472.6746358   * ttt
        lonven   = 181.979800853  +  58517.8156748   * ttt
        lonear   = 100.466448494  +  35999.3728521   * ttt
        lonmar   = 355.433274605  +  19140.299314    * ttt
        lonjup   =  34.351483900  +   3034.90567464  * ttt
        lonsat   =  50.0774713998 +   1222.11379404  * ttt
        lonurn   = 314.055005137  +    428.466998313 * ttt
        lonnep   = 304.348665499  +    218.486200208 * ttt
        precrate = 1.39697137214 * ttt + 0.0003086 * t2

    elif opt == "02":
        l     = 134.96340251 + 1717915923.2178 * ttt / 3600.0
        l1    = 357.52910918 +  129596581.0481 * ttt / 3600.0
        f     =  93.27209062 + 1739527262.8478 * ttt / 3600.0
        d     = 297.85019547 + 1602961601.2090 * ttt / 3600.0
        omega = 125.04455501 +   -6962890.5431 * ttt / 3600.0
        # Planetary zero; precrate zero

    elif opt == "96":
        l     = 134.96340251 + (1717915923.2178 * ttt + 31.8792 * t2 + 0.051635 * t3 - 0.00024470 * t4) / 3600.0
        l1    = 357.52910918 + ( 129596581.0481 * ttt - 0.5532   * t2 - 0.000136 * t3 - 0.00001149 * t4) / 3600.0
        f     =  93.27209062 + (1739527262.8478 * ttt - 12.7512  * t2 + 0.001037 * t3 + 0.00000417 * t4) / 3600.0
        d     = 297.85019547 + (1602961601.2090 * ttt - 6.3706   * t2 + 0.006593 * t3 - 0.00003169 * t4) / 3600.0
        omega = 125.04455501 + (  -6962890.2665 * ttt + 7.4722   * t2 + 0.007702 * t3 - 0.00005939 * t4) / 3600.0
        lonven   = 181.979800853 +  58517.8156748  * ttt
        lonear   = 100.466448494 +  35999.3728521  * ttt
        lonmar   = 355.433274605 +  19140.299314   * ttt
        lonjup   =  34.351483900 +   3034.90567464 * ttt
        lonsat   =  50.0774713998+   1222.11379404 * ttt
        precrate = 1.39697137214 * ttt + 0.0003086 * t2

    elif opt == "80":
        l     = ((((0.064) * ttt + 31.310) * ttt + 1717915922.6330) * ttt) / 3600.0 + 134.96298139
        l1    = ((((-0.012) * ttt - 0.577) * ttt + 129596581.2240) * ttt) / 3600.0 + 357.52772333
        f     = ((((0.011) * ttt - 13.257) * ttt + 1739527263.1370) * ttt) / 3600.0 + 93.27191028
        d     = ((((0.019) * ttt - 6.891) * ttt + 1602961601.3280) * ttt) / 3600.0 + 297.85036306
        omega = ((((0.008) * ttt + 7.455) * ttt - 6962890.5390) * ttt) / 3600.0 + 125.04452222
        lonmer   = 252.3 + 149472.0 * ttt
        lonven   = 179.9 +  58517.8 * ttt
        lonear   =  98.4 +  35999.4 * ttt
        lonmar   = 353.3 +  19140.3 * ttt
        lonjup   =  32.3 +   3034.9 * ttt
        lonsat   =  48.0 +   1222.1 * ttt
        # lonurn, lonnep, precrate remain 0
    else:
        raise ValueError("opt must be one of '10','02','96','80'")

    # wrap to [0,360) and convert to radians
    vals_deg = [
        _wrap_deg(l), _wrap_deg(l1), _wrap_deg(f), _wrap_deg(d), _wrap_deg(omega),
        _wrap_deg(lonmer), _wrap_deg(lonven), _wrap_deg(lonear), _wrap_deg(lonmar),
        _wrap_deg(lonjup), _wrap_deg(lonsat), _wrap_deg(lonurn), _wrap_deg(lonnep),
        _wrap_deg(precrate),
    ]
    vals_rad = tuple(_deg2rad(v) for v in vals_deg)
    return vals_rad  # type: ignore[return-value]
