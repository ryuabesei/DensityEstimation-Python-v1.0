# densityestimation/Astrofunction/sgp4_constants.py
# GNU GPL v3
# Vallado 準拠の SGP4 物理定数を読み込む

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SGP4Constants:
    # minutes in one time unit
    tumin: float
    # earth gravitational parameter [km^3/s^2]
    mu: float
    # earth radius [km]
    radiusearthkm: float
    # reciprocal of tumin
    xke: float
    # un-normalized zonal harmonics
    j2: float
    j3: float
    j4: float
    j3oj2: float
    # extra flags from MATLAB code
    opsmode: str
    whichconst: int


def _getgravc(whichconst: int) -> SGP4Constants:
    """
    Port of Vallado's getgravc.m
    whichconst: 721 (WGS-72 low), 72 (WGS-72), 84 (WGS-84)
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
        xke = 60.0 / ((radiusearthkm**3 / mu) ** 0.5)
        tumin = 1.0 / xke
        j2 = 0.001082616
        j3 = -0.00000253881
        j4 = -0.00000165597
        j3oj2 = j3 / j2
    elif whichconst == 84:
        mu = 398600.5
        radiusearthkm = 6378.137
        xke = 60.0 / ((radiusearthkm**3 / mu) ** 0.5)
        tumin = 1.0 / xke
        j2 = 0.00108262998905
        j3 = -0.00000253215306
        j4 = -0.00000161098761
        j3oj2 = j3 / j2
    else:
        raise ValueError(f"Unknown whichconst={whichconst}. Use 721, 72, or 84.")
    # MATLAB loadSGP4: opsmode = 'i' (improved), whichconst = 72
    return SGP4Constants(
        tumin=tumin,
        mu=mu,
        radiusearthkm=radiusearthkm,
        xke=xke,
        j2=j2,
        j3=j3,
        j4=j4,
        j3oj2=j3oj2,
        opsmode="i",
        whichconst=whichconst,
    )


def load_sgp4(whichconst: int = 72, opsmode: str = "i") -> SGP4Constants:
    """
    Python equivalent of MATLAB loadSGP4.m.
    Returns an immutable SGP4Constants dataclass instead of globals.
    """
    consts = _getgravc(whichconst)
    # opsmode は MATLAB で 'i' を想定。必要なら上書きして返す。
    return SGP4Constants(**{**consts.__dict__, "opsmode": opsmode})
