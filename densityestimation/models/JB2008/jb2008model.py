# densityestimation/models/jb2008/jb2008.py
from __future__ import annotations

from typing import Tuple

import numpy as np

from .finddays import find_days
from .invjday import invjday


def JB2008(
    MJD: float,
    SUN: np.ndarray,
    SAT: np.ndarray,
    F10: float,
    F10B: float,
    S10: float,
    S10B: float,
    XM10: float,
    XM10B: float,
    Y10: float,
    Y10B: float,
    DSTDTC: float,
) -> Tuple[np.ndarray, float]:
    """
    Jacchia–Bowman 2008 モデル（CIRA “integration form”）
    Parameters
    ----------
    MJD : float
        Modified Julian Date (JD - 2400000.5)
    SUN : [RA, Dec]  (radians)
    SAT : [RA, geocentric latitude, height_km]  (radians, radians, km)
    F10, F10B, S10, S10B, XM10, XM10B, Y10, Y10B : float
        JB2008 が要求する各インデックス（元 MATLAB の期待どおり）
    DSTDTC : float
        Dst 由来の温度補正 dTc

    Returns
    -------
    TEMP : np.ndarray shape (2,)
        TEMP[0] = exospheric temperature above position [K]
        TEMP[1] = local temperature at position [K]
    RHO : float
        total mass density at position [kg/m^3]
    """
    # ---- constants / coefficients ----
    ALPHA = np.array([0.0, 0.0, 0.0, 0.0, -0.38], dtype=float)  # Eq.(6)
    AL10 = 2.3025851  # log(10)
    AMW = np.array([28.0134, 31.9988, 15.9994, 39.9480, 4.0026, 1.00797], dtype=float)
    AVOGAD = 6.02257e26  # molecules/kmol
    TWOPI = 2.0 * np.pi
    PIOV2 = np.pi / 2.0
    FRAC = np.array([0.78110, 0.20955, 9.3400e-3, 1.2890e-5], dtype=float)  # sea-level N2, O2, Ar, He
    RSTAR = 8314.32
    R1, R2, R3 = 0.010, 0.025, 0.075
    WT = np.array(
        [0.311111111111111, 1.422222222222222, 0.533333333333333, 1.422222222222222, 0.311111111111111],
        dtype=float,
    )  # Newton–Cotes 5-pt
    CHT = np.array([0.22, -0.20e-2, 0.115e-2, -0.211e-5], dtype=float)
    DEGRAD = np.pi / 180.0

    # ---- Eq.(14): Tsbc ----
    FN = (F10B / 240.0) ** 0.25
    if FN > 1.0:
        FN = 1.0
    FSB = F10B * FN + S10B * (1.0 - FN)
    TSUBC = (
        392.4
        + 3.227 * FSB
        + 0.298 * (F10 - F10B)
        + 2.259 * (S10 - S10B)
        + 0.312 * (XM10 - XM10B)
        + 0.178 * (Y10 - Y10B)
    )

    # ---- Eq.(15) ----
    ETA = 0.5 * abs(SAT[1] - SUN[1])
    THETA = 0.5 * abs(SAT[1] + SUN[1])

    # ---- Eq.(16) ----
    H = SAT[0] - SUN[0]
    TAU = H - 0.64577182 + 0.10471976 * np.sin(H + 0.75049158)
    GLAT = SAT[1]
    ZHT = float(SAT[2])
    GLST = H + np.pi
    GLSTHR = (GLST / DEGRAD) * (24.0 / 360.0)
    if GLSTHR >= 24.0:
        GLSTHR -= 24.0
    if GLSTHR < 0.0:
        GLSTHR += 24.0

    # ---- Eq.(17) ----
    C = (np.cos(ETA)) ** 2.5
    S = (np.sin(THETA)) ** 2.5
    DF = S + (C - S) * abs(np.cos(0.5 * TAU)) ** 3
    TSUBL = TSUBC * (1.0 + 0.31 * DF)

    # Local-time/lat correction & Dst correction
    DTCLST = _DTSUB(F10, GLSTHR, GLAT, ZHT)
    TEMP = np.zeros(2, dtype=float)
    TEMP[0] = TSUBL + DSTDTC  # exospheric above position
    TINF = TSUBL + DSTDTC + DTCLST

    # ---- Eq.(9) ----
    TSUBX = 444.3807 + 0.02385 * TINF - 392.8292 * np.exp(-0.0021357 * TINF)
    # ---- Eq.(11) ----
    GSUBX = 0.054285714 * (TSUBX - 183.0)

    # TC (for XLOCAL: Eq.(10)/(13))
    TC0 = TSUBX
    TC1 = GSUBX
    # Eq.(13): A = TC2, (Gx/A) = TC3
    TC2 = (TINF - TSUBX) / PIOV2
    TC3 = GSUBX / TC2
    TC = np.array([TC0, TC1, TC2, TC3], dtype=float)

    # ---- Barometric integration 90→105 km (Eq.5) ----
    Z1 = 90.0
    Z2 = min(ZHT, 105.0)
    AL = np.log(Z2 / Z1)
    N = int(np.floor(AL / R1) + 1)
    ZR = np.exp(AL / N) if N > 0 else 1.0
    ZEND = Z1
    SUM2 = 0.0
    AMBAR1 = _XAMBAR(Z1)
    TLOC1 = _XLOCAL(Z1, TC)
    AIN = AMBAR1 * _XGRAV(Z1) / TLOC1
    AMBAR2 = AMBAR1  # for scope
    TLOC2 = TLOC1

    for _ in range(max(N, 0)):
        Z = ZEND
        ZEND = ZR * Z
        DZ = 0.25 * (ZEND - Z)
        SUM1 = WT[0] * AIN
        for j in range(1, 5):
            Z = Z + DZ
            AMBAR2 = _XAMBAR(Z)
            TLOC2 = _XLOCAL(Z, TC)
            GRAVL = _XGRAV(Z)
            AIN = AMBAR2 * GRAVL / TLOC2
            SUM1 += WT[j] * AIN
        SUM2 += DZ * SUM1

    FACT1 = 1000.0 / RSTAR
    # ρ at ~Z2 (from band 90–105)
    RHO = 3.46e-6 * AMBAR2 * TLOC1 * np.exp(-FACT1 * SUM2) / (AMBAR1 * TLOC2)

    # Eq.(2) number densities (for composition below 105 km)
    ANM = AVOGAD * RHO  # molecules/m^3
    AN = ANM / AMBAR2

    # Eq.(3) initial ln number densities at ~Z2
    FACT2 = ANM / 28.960
    ALN = np.zeros(6, dtype=float)
    ALN[0] = np.log(FRAC[0] * FACT2)  # N2
    ALN[3] = np.log(FRAC[2] * FACT2)  # Ar
    ALN[4] = np.log(FRAC[3] * FACT2)  # He
    # Eq.(4)
    ALN[1] = np.log(FACT2 * (1.0 + FRAC[1]) - AN)  # O2
    ALN[2] = np.log(2.0 * (AN - FACT2))            # O

    if ZHT <= 105.0:
        TEMP[1] = TLOC2
        # negligible H
        ALN[5] = ALN[4] - 25.0

        # Eq.(24): J70 Seasonal–Latitudinal Variation
        TRASH = (MJD - 36204.0) / 365.2422
        CAPPHI = TRASH - np.floor(TRASH)
        DLRSL = (
            0.02 * (ZHT - 90.0) * np.exp(-0.045 * (ZHT - 90.0))
            * _sign_mag(1.0, GLAT) * (np.sin(TWOPI * CAPPHI + 1.72)) * (np.sin(GLAT) ** 2)
        )

        # Eq.(23): Semiannual variation
        DLRSA = 0.0
        if ZHT < 2000.0:
            YRDAY = _TMOUTD(MJD)
            FZZ, GTZ, DLRSA = _SEMIAN08(YRDAY, ZHT, F10B, S10B, XM10B)
            if FZZ < 0.0:
                DLRSA = 0.0

        DLR = AL10 * (DLRSL + DLRSA)
        ALN += DLR

        # mixture → density
        SUMN = 0.0
        SUMNM = 0.0
        for i in range(6):
            Ni = np.exp(ALN[i])
            SUMN += Ni
            SUMNM += Ni * AMW[i]
        RHO = SUMNM / AVOGAD

        # Exospheric high-altitude correction
        FEX = 1.0
        if 1000.0 <= ZHT < 1500.0:
            ZETA = (ZHT - 1000.0) * 0.002
            ZETA2 = ZETA * ZETA
            ZETA3 = ZETA2 * ZETA
            F15C = CHT[0] + CHT[1] * F10B + CHT[2] * 1500.0 + CHT[3] * F10B * 1500.0
            F15C_ZETA = (CHT[2] + CHT[3] * F10B) * 500.0
            FEX2 = 3.0 * F15C - F15C_ZETA - 3.0
            FEX3 = F15C_ZETA - 2.0 * F15C + 2.0
            FEX = 1.0 + FEX2 * ZETA2 + FEX3 * ZETA3
        if ZHT >= 1500.0:
            FEX = CHT[0] + CHT[1] * F10B + CHT[2] * ZHT + CHT[3] * F10B * ZHT

        RHO *= FEX
        return TEMP, float(RHO)

    # ---- If ZHT > 105: continue integration 105→500 and 500→ZHT ----
    # integrate 105..500 with R2
    Z = ZEND
    Z3 = min(ZHT, 500.0)
    AL = np.log(Z3 / Z)
    N = int(np.floor(AL / R2) + 1)
    ZR = np.exp(AL / N) if N > 0 else 1.0
    SUM2 = 0.0
    # note: we come in with AIN from last loop (= GRAVL/TLOC2 at end of band)
    # but recompute safely:
    AIN = _XGRAV(ZEND) / _XLOCAL(ZEND, TC)
    for _ in range(max(N, 0)):
        Z = ZEND
        ZEND = ZR * Z
        DZ = 0.25 * (ZEND - Z)
        SUM1 = WT[0] * AIN
        for j in range(1, 5):
            Z = Z + DZ
            TLOC3 = _XLOCAL(Z, TC)
            GRAVL = _XGRAV(Z)
            AIN = GRAVL / TLOC3
            SUM1 += WT[j] * AIN
        SUM2 += DZ * SUM1
    # integrate 500..ZHT with R2 or R3
    Z4 = max(ZHT, 500.0)
    AL = np.log(Z4 / Z)
    R = R3 if ZHT > 500.0 else R2
    N = int(np.floor(AL / R) + 1)
    ZR = np.exp(AL / N) if N > 0 else 1.0
    SUM3 = 0.0
    for _ in range(max(N, 0)):
        Z = ZEND
        ZEND = ZR * Z
        DZ = 0.25 * (ZEND - Z)
        SUM1 = WT[0] * AIN
        for j in range(1, 5):
            Z = Z + DZ
            TLOC4 = _XLOCAL(Z, TC)
            GRAVL = _XGRAV(Z)
            AIN = GRAVL / TLOC4
            SUM1 += WT[j] * AIN
        SUM3 += DZ * SUM1

    if ZHT > 500.0:
        T500 = TLOC3
        TEMP[1] = TLOC4
        ALTR = np.log(TLOC4 / TLOC2)
        FACT2_all = FACT1 * (SUM2 + SUM3)
        HSIGN = -1.0
    else:
        T500 = TLOC4
        TEMP[1] = TLOC3
        ALTR = np.log(TLOC3 / TLOC2)
        FACT2_all = FACT1 * SUM2
        HSIGN = 1.0

    # Eq.(6) applying diffusion and integration to composition
    for i in range(5):
        ALN[i] = ALN[i] - (1.0 + ALPHA[i]) * ALTR - FACT2_all * AMW[i]

    # Eq.(7): Hydrogen high-altitude
    AL10T5 = np.log10(TINF)
    ALNH5 = (5.5 * AL10T5 - 39.40) * AL10T5 + 73.13
    ALN[5] = AL10 * (ALNH5 + 6.0) + HSIGN * (np.log(TLOC4 / TLOC3) + FACT1 * SUM3 * AMW[5])

    # Seasonal–latitudinal (Eq.24) and semiannual (Eq.23)
    TRASH = (MJD - 36204.0) / 365.2422
    CAPPHI = TRASH - np.floor(TRASH)
    DLRSL = (
        0.02 * (ZHT - 90.0) * np.exp(-0.045 * (ZHT - 90.0))
        * _sign_mag(1.0, GLAT) * (np.sin(TWOPI * CAPPHI + 1.72)) * (np.sin(GLAT) ** 2)
    )
    DLRSA = 0.0
    if ZHT < 2000.0:
        YRDAY = _TMOUTD(MJD)
        FZZ, GTZ, DLRSA = _SEMIAN08(YRDAY, ZHT, F10B, S10B, XM10B)
        if FZZ < 0.0:
            DLRSA = 0.0
    DLR = AL10 * (DLRSL + DLRSA)
    ALN += DLR

    # mixture → density
    SUMN = 0.0
    SUMNM = 0.0
    for i in range(6):
        Ni = np.exp(ALN[i])
        SUMN += Ni
        SUMNM += Ni * AMW[i]
    RHO = SUMNM / AVOGAD

    # Exospheric correction
    FEX = 1.0
    if 1000.0 <= ZHT < 1500.0:
        ZETA = (ZHT - 1000.0) * 0.002
        ZETA2 = ZETA * ZETA
        ZETA3 = ZETA2 * ZETA
        F15C = CHT[0] + CHT[1] * F10B + CHT[2] * 1500.0 + CHT[3] * F10B * 1500.0
        F15C_ZETA = (CHT[2] + CHT[3] * F10B) * 500.0
        FEX2 = 3.0 * F15C - F15C_ZETA - 3.0
        FEX3 = F15C_ZETA - 2.0 * F15C + 2.0
        FEX = 1.0 + FEX2 * ZETA2 + FEX3 * ZETA3
    if ZHT >= 1500.0:
        FEX = CHT[0] + CHT[1] * F10B + CHT[2] * ZHT + CHT[3] * F10B * ZHT

    RHO *= FEX
    return TEMP, float(RHO)


# ----------------- helpers (faithful MATLAB→Python) -----------------

def _XAMBAR(Z: float) -> float:
    """Eq.(1): mean molecular weight at altitude Z [km]."""
    C = np.array([28.15204, -8.5586e-2, +1.2840e-4, -1.0056e-5, -1.0210e-5, +1.5044e-6, +9.9826e-8], dtype=float)
    DZ = Z - 100.0
    AMB = C[6]
    # Horner
    for j in range(5, -1, -1):
        AMB = DZ * AMB + C[j]
    return float(AMB)


def _XGRAV(Z: float) -> float:
    """Eq.(8): local gravity [m/s^2] at altitude Z [km]."""
    return float(9.80665 / (1.0 + Z / 6356.766) ** 2)


def _XLOCAL(Z: float, TC: np.ndarray) -> float:
    """
    Eq.(10) or Eq.(13): local temperature at altitude Z.
    TC = [TSUBX, GSUBX, A(=TC2), GSUBX/A(=TC3)]
    """
    DZ = Z - 125.0
    if DZ > 0.0:
        # Eq.(13)
        return float(TC[0] + TC[2] * np.arctan(TC[3] * DZ * (1.0 + 4.5e-6 * DZ ** 2.5)))
    # Eq.(10)
    return float(((-9.8204695e-6 * DZ - 7.3039742e-4) * DZ ** 2 + 1.0) * DZ * TC[1] + TC[0])


def _DTSUB(F10: float, XLST: float, XLAT: float, ZHT: float) -> float:
    """
    JB2008 DTSUB.m: dTc correction for local solar time and latitude.
    Returns dTc [K].
    """
    B = np.array(
        [
            -0.457512297e1, -0.512114909e1, -0.693003609e2, 0.203716701e3, 0.703316291e3, -0.194349234e4,
            0.110651308e4, -0.174378996e3, 0.188594601e4, -0.709371517e4, 0.922454523e4, -0.384508073e4,
            -0.645841789e1, 0.409703319e2, -0.482006560e3, 0.181870931e4, -0.237389204e4, 0.996703815e3,
            0.361416936e2,
        ],
        dtype=float,
    )
    C = np.array(
        [
            -0.155986211e2, -0.512114909e1, -0.693003609e2, 0.203716701e3, 0.703316291e3, -0.194349234e4,
            0.110651308e4, -0.220835117e3, 0.143256989e4, -0.318481844e4, 0.328981513e4, -0.135332119e4,
            0.199956489e2, -0.127093998e2, 0.212825156e2, -0.275555432e1, 0.110234982e2, 0.148881951e3,
            -0.751640284e3, 0.637876542e3, 0.127093998e2, -0.212825156e2, 0.275555432e1,
        ],
        dtype=float,
    )

    DTC = 0.0
    tx = XLST / 24.0
    ycs = np.cos(XLAT)
    F = (F10 - 100.0) / 100.0

    if 120.0 <= ZHT <= 200.0:
        H = (ZHT - 200.0) / 50.0
        DTC200 = (
            C[16]
            + C[17] * tx * ycs
            + C[18] * tx ** 2 * ycs
            + C[19] * tx ** 3 * ycs
            + C[20] * F * ycs
            + C[21] * tx * F * ycs
            + C[22] * tx ** 2 * F * ycs
        )
        ssum = (
            C[0]
            + B[1] * F
            + C[2] * tx * F
            + C[3] * tx ** 2 * F
            + C[4] * tx ** 3 * F
            + C[5] * tx ** 4 * F
            + C[6] * tx ** 5 * F
            + C[7] * tx * ycs
            + C[8] * tx ** 2 * ycs
            + C[9] * tx ** 3 * ycs
            + C[10] * tx ** 4 * ycs
            + C[11] * tx ** 5 * ycs
            + C[12] * ycs
            + C[13] * F * ycs
            + C[14] * tx * F * ycs
            + C[15] * tx ** 2 * F * ycs
        )
        DTC200DZ = ssum
        CC = 3.0 * DTC200 - DTC200DZ
        DD = DTC200 - CC
        ZP = (ZHT - 120.0) / 80.0
        DTC = CC * ZP ** 2 + DD * ZP ** 3

    if 200.0 < ZHT <= 240.0:
        H = (ZHT - 200.0) / 50.0
        DTC = (
            C[0] * H
            + B[1] * F * H
            + C[2] * tx * F * H
            + C[3] * tx ** 2 * F * H
            + C[4] * tx ** 3 * F * H
            + C[5] * tx ** 4 * F * H
            + C[6] * tx ** 5 * F * H
            + C[7] * tx * ycs * H
            + C[8] * tx ** 2 * ycs * H
            + C[9] * tx ** 3 * ycs * H
            + C[10] * tx ** 4 * ycs * H
            + C[11] * tx ** 5 * ycs * H
            + C[12] * ycs * H
            + C[13] * F * ycs * H
            + C[14] * tx * F * ycs * H
            + C[15] * tx ** 2 * F * ycs * H
            + C[16]
            + C[17] * tx * ycs
            + C[18] * tx ** 2 * ycs
            + C[19] * tx ** 3 * ycs
            + C[20] * F * ycs
            + C[21] * tx * F * ycs
            + C[22] * tx ** 2 * F * ycs
        )

    if 240.0 < ZHT <= 300.0:
        H = 40.0 / 50.0
        AA = (
            C[0] * H
            + B[1] * F * H
            + C[2] * tx * F * H
            + C[3] * tx ** 2 * F * H
            + C[4] * tx ** 3 * F * H
            + C[5] * tx ** 4 * F * H
            + C[6] * tx ** 5 * F * H
            + C[7] * tx * ycs * H
            + C[8] * tx ** 2 * ycs * H
            + C[9] * tx ** 3 * ycs * H
            + C[10] * tx ** 4 * ycs * H
            + C[11] * tx ** 5 * ycs * H
            + C[12] * ycs * H
            + C[13] * F * ycs * H
            + C[14] * tx * F * ycs * H
            + C[15] * tx ** 2 * F * ycs * H
            + C[16]
            + C[17] * tx * ycs
            + C[18] * tx ** 2 * ycs
            + C[19] * tx ** 3 * ycs
            + C[20] * F * ycs
            + C[21] * tx * F * ycs
            + C[22] * tx ** 2 * F * ycs
        )
        BB = (
            C[0]
            + B[1] * F
            + C[2] * tx * F
            + C[3] * tx ** 2 * F
            + C[4] * tx ** 3 * F
            + C[5] * tx ** 4 * F
            + C[6] * tx ** 5 * F
            + C[7] * tx * ycs
            + C[8] * tx ** 2 * ycs
            + C[9] * tx ** 3 * ycs
            + C[10] * tx ** 4 * ycs
            + C[11] * tx ** 5 * ycs
            + C[12] * ycs
            + C[13] * F * ycs
            + C[14] * tx * F * ycs
            + C[15] * tx ** 2 * F * ycs
        )
        HP = 300.0 / 100.0
        DTC300 = (
            B[0]
            + B[1] * F
            + B[2] * tx * F
            + B[3] * tx ** 2 * F
            + B[4] * tx ** 3 * F
            + B[5] * tx ** 4 * F
            + B[6] * tx ** 5 * F
            + B[7] * tx * ycs
            + B[8] * tx ** 2 * ycs
            + B[9] * tx ** 3 * ycs
            + B[10] * tx ** 4 * ycs
            + B[11] * tx ** 5 * ycs
            + B[12] * HP * ycs
            + B[13] * tx * HP * ycs
            + B[14] * tx ** 2 * HP * ycs
            + B[15] * tx ** 3 * HP * ycs
            + B[16] * tx ** 4 * HP * ycs
            + B[17] * tx ** 5 * HP * ycs
            + B[18] * ycs
        )
        DTC300DZ = B[12] * ycs + B[13] * tx * ycs + B[14] * tx ** 2 * ycs + B[15] * tx ** 3 * ycs + B[16] * tx ** 4 * ycs + B[17] * tx ** 5 * ycs
        CC = 3.0 * DTC300 - DTC300DZ - 3.0 * AA - 2.0 * BB
        DD = DTC300 - AA - BB - CC
        ZP = (ZHT - 240.0) / 60.0
        DTC = AA + BB * ZP + CC * ZP ** 2 + DD * ZP ** 3

    if 300.0 < ZHT <= 600.0:
        H = ZHT / 100.0
        DTC = (
            B[0]
            + B[1] * F
            + B[2] * tx * F
            + B[3] * tx ** 2 * F
            + B[4] * tx ** 3 * F
            + B[5] * tx ** 4 * F
            + B[6] * tx ** 5 * F
            + B[7] * tx * ycs
            + B[8] * tx ** 2 * ycs
            + B[9] * tx ** 3 * ycs
            + B[10] * tx ** 4 * ycs
            + B[11] * tx ** 5 * ycs
            + B[12] * H * ycs
            + B[13] * tx * H * ycs
            + B[14] * tx ** 2 * H * ycs
            + B[15] * tx ** 3 * H * ycs
            + B[16] * tx ** 4 * H * ycs
            + B[17] * tx ** 5 * H * ycs
            + B[18] * ycs
        )

    if 600.0 < ZHT <= 800.0:
        ZP = (ZHT - 600.0) / 100.0
        HP = 600.0 / 100.0
        AA = (
            B[0]
            + B[1] * F
            + B[2] * tx * F
            + B[3] * tx ** 2 * F
            + B[4] * tx ** 3 * F
            + B[5] * tx ** 4 * F
            + B[6] * tx ** 5 * F
            + B[7] * tx * ycs
            + B[8] * tx ** 2 * ycs
            + B[9] * tx ** 3 * ycs
            + B[10] * tx ** 4 * ycs
            + B[11] * tx ** 5 * ycs
            + B[12] * HP * ycs
            + B[13] * tx * HP * ycs
            + B[14] * tx ** 2 * HP * ycs
            + B[15] * tx ** 3 * HP * ycs
            + B[16] * tx ** 4 * HP * ycs
            + B[17] * tx ** 5 * HP * ycs
            + B[18] * ycs
        )
        BB = B[12] * ycs + B[13] * tx * ycs + B[14] * tx ** 2 * ycs + B[15] * tx ** 3 * ycs + B[16] * tx ** 4 * ycs + B[17] * tx ** 5 * ycs
        CC = -(3.0 * AA + 4.0 * BB) / 4.0
        DD = (AA + BB) / 4.0
        DTC = AA + BB * ZP + CC * ZP ** 2 + DD * ZP ** 3

    return float(DTC)


def _SEMIAN08(DAY: float, HT: float, F10B: float, S10B: float, XM10B: float) -> Tuple[float, float, float]:
    """
    SEMIAN08: Semiannual variation model (returns FZZ, GTZ, DRLOG).
    """
    TWOPI = 2.0 * np.pi
    # FZ global (1997–2006 fit)
    FZM = np.array([0.2689, -0.01176, 0.02782, -0.02782, 0.0003470], dtype=float)
    # GT global (1997–2006 fit)
    GTM = np.array([-0.3633, 0.08506, 0.2401, -0.1897, -0.2554, -0.01790, 0.0005650, -0.0006407, -0.003418, -0.001252], dtype=float)

    # new centered index for FZ
    FSMB = F10B - 0.70 * S10B - 0.04 * XM10B
    HTZ = HT / 1000.0
    FZZ = FZM[0] + FZM[1] * FSMB + FZM[2] * FSMB * HTZ + FZM[3] * FSMB * HTZ ** 2 + FZM[4] * (FSMB ** 2) * HTZ

    # centered index for GT
    FSMB = F10B - 0.75 * S10B - 0.37 * XM10B
    TAU = (DAY - 1.0) / 365.0
    SIN1P = np.sin(TWOPI * TAU)
    COS1P = np.cos(TWOPI * TAU)
    SIN2P = np.sin(2.0 * TWOPI * TAU)
    COS2P = np.cos(2.0 * TWOPI * TAU)

    GTZ = (
        GTM[0]
        + GTM[1] * SIN1P
        + GTM[2] * COS1P
        + GTM[3] * SIN2P
        + GTM[4] * COS2P
        + GTM[5] * FSMB
        + GTM[6] * FSMB * SIN1P
        + GTM[7] * FSMB * COS1P
        + GTM[8] * FSMB * SIN2P
        + GTM[9] * FSMB * COS2P
    )

    if FZZ < 1e-6:
        FZZ = 1e-6

    DRLOG = FZZ * GTZ
    return float(FZZ), float(GTZ), float(DRLOG)


def _TMOUTD(MJD: float) -> float:
    """TMOUTD: MJD → day-of-year (fractional)."""
    year, month, day, hr, minute, sec = invjday(MJD + 2400000.5)
    return float(find_days(year, month, day, hr, minute, sec))


def _sign_mag(x: float, y: float) -> float:
    """MATLAB の sign_ 相当: 戻り値は |x| に y の符号をつけたもの。"""
    return abs(x) * (1.0 if y >= 0 else -1.0)
