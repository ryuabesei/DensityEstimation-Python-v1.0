# GNU GPL v3
from __future__ import annotations

import math

# SAT_Const.m の「global const」を Python で表現
# 既存の constants.py と重複する値が多いので、ここでは同じ名前を提供しつつ
# 他所で参照されがちな `Arcs` などのエイリアスも入れておく。
from dataclasses import dataclass


@dataclass(frozen=True)
class _SatConst:
    # 数学定数
    pi2: float = 2.0 * math.pi
    Rad: float = math.pi / 180.0             # rad/deg
    Deg: float = 180.0 / math.pi             # deg/rad
    Arcs: float = 3600.0 * 180.0 / math.pi   # arcsec/rad

    # 一般
    MJD_J2000: float = 51544.5
    T_B1950: float = -0.500002108
    c_light: float = 299792458.0
    AU: float = 149597870700.0

    # 天体半径・扁平率
    R_Earth: float = 6378.137e3
    f_Earth: float = 1.0 / 298.257223563
    R_Sun: float = 696000e3
    R_Moon: float = 1738e3

    # 地球自転（J2000 の GMST 微分）
    omega_Earth: float = (15.04106717866910 / 3600.0) * (math.pi / 180.0)

    # 重力定数
    GM_Earth: float = 398600.4418e9
    GM_Sun: float = 132712440041.939400e9
    GM_Moon: float = 398600.4418e9 / 81.30056907419062
    GM_Mercury: float = 22031.780000e9
    GM_Venus: float = 324858.592000e9
    GM_Mars: float = 42828.375214e9
    GM_Jupiter: float = 126712764.800000e9
    GM_Saturn: float = 37940585.200000e9
    GM_Uranus: float = 5794548.600000e9
    GM_Neptune: float = 6836527.100580e9
    GM_Pluto: float = 977.0000000000009e9

    # 太陽放射圧（1 AU）
    P_Sol: float = 1367.0 / 299792458.0

const = _SatConst()

__all__ = ["const", "_SatConst"]
