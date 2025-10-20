# Origin: pv2ep.m (Orbital Mechanics with MATLAB by David Eagle, 2013)
from __future__ import annotations

import numpy as np


def pv2ep(rr: np.ndarray, vv: np.ndarray, mu: float) -> np.ndarray:
    """
    r,v (ECI/J2000, km, km/s) → 修正赤道離心要素 MEE [p, f, g, h, k, L]

    Parameters
    ----------
    rr : array-like, shape (3,)
        位置ベクトル [km]
    vv : array-like, shape (3,)
        速度ベクトル [km/s]
    mu : float
        重力定数 [km^3/s^2]

    Returns
    -------
    EP : ndarray, shape (6,)
        MEE = [p, f, g, h, k, L(rad)]
    """
    rr = np.asarray(rr, dtype=float).reshape(3)
    vv = np.asarray(vv, dtype=float).reshape(3)

    radius = np.linalg.norm(rr)

    hv = np.cross(rr, vv)
    hmag = np.linalg.norm(hv)

    p = hmag**2 / mu

    rdotv = float(np.dot(rr, vv))
    rzerod = rdotv / radius

    eccen = np.cross(vv, hv) / mu - rr / radius  # Laplace-Runge-Lenzベクトルから単位化

    # unit angular momentum vector
    hhat = hv / hmag

    # k, h（注意: MATLABの符号・式に厳密準拠）
    denom = 1.0 + hhat[2]
    kmee = hhat[0] / denom
    hmee = -hhat[1] / denom

    # equinoctial frame unit vectors
    fhat = np.array([1.0 - kmee**2 + hmee**2,
                     2.0 * kmee * hmee,
                    -2.0 * kmee])
    ghat = np.array([fhat[1],
                     1.0 + kmee**2 - hmee**2,
                     2.0 * hmee])

    ssqrd = 1.0 + kmee**2 + hmee**2
    fhat /= ssqrd
    ghat /= ssqrd

    # f, g
    f = float(np.dot(eccen, fhat))
    g = float(np.dot(eccen, ghat))

    # true longitude L
    uhat = rr / radius
    vhat = (radius * vv - rzerod * rr) / hmag
    cosl = uhat[0] + vhat[1]
    sinl = uhat[1] - vhat[0]
    L = np.arctan2(sinl, cosl)

    EP = np.array([p, f, g, hmee, kmee, L], dtype=float)
    return EP



def ep2pv(EP: np.ndarray, mu: float) -> tuple[np.ndarray, np.ndarray]:
    """
    MATLAB ep2pv.m のポート
    Equinoctial Parameters (modified equinoctial elements; MEE)
    -> ECI position/velocity

    Parameters
    ----------
    EP : array-like shape (6,)
        [p, f, g, h, k, l]
        p: semilatus rectum [km]
        l: true longitude [rad]
    mu : float
        gravitational parameter [km^3/s^2]

    Returns
    -------
    rr : ndarray shape (3,)
        ECI position [km]
    vv : ndarray shape (3,)
        ECI velocity [km/s]
    """
    EP = np.asarray(EP, dtype=float).reshape(6,)
    p, f, g, h, k, l = EP

    sqrtmup = np.sqrt(mu / p)
    cosl = np.cos(l)
    sinl = np.sin(l)

    q = 1.0 + f * cosl + g * sinl
    r = p / q

    alphasqrd = h**2 - k**2
    ssqrd = 1.0 + h**2 + k**2

    # position
    x = cosl + alphasqrd * cosl + 2.0 * h * k * sinl
    y = sinl - alphasqrd * sinl + 2.0 * h * k * cosl
    z = 2.0 * (h * sinl - k * cosl)
    rr = (r / ssqrd) * np.array([x, y, z], dtype=float)

    # velocity
    vx = (
        -sinl
        - alphasqrd * sinl
        + 2.0 * h * k * cosl
        - g
        + 2.0 * f * h * k
        - alphasqrd * g
    )
    vy = (
        cosl
        - alphasqrd * cosl
        - 2.0 * h * k * sinl
        + f
        - 2.0 * g * h * k
        - alphasqrd * f
    )
    vz = 2.0 * (h * cosl + k * sinl + f * h + g * k)
    vv = (sqrtmup / ssqrd) * np.array([vx, vy, vz], dtype=float)

    return rr, vv