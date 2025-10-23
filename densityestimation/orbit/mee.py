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

MU_EARTH = 3.986004418e14  # m^3/s^2

def _sv_to_coe(r, v, mu=MU_EARTH):
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    R = np.linalg.norm(r)
    V = np.linalg.norm(v)

    h_vec = np.cross(r, v)
    h = np.linalg.norm(h_vec)

    k_hat = np.array([0.0, 0.0, 1.0])
    n_vec = np.cross(k_hat, h_vec)
    n = np.linalg.norm(n_vec)

    e_vec = (np.cross(v, h_vec) / mu) - (r / R)
    e = np.linalg.norm(e_vec)

    # エネルギーから半長軸
    eps = V**2 / 2 - mu / R
    a = np.inf if abs(e - 1.0) < 1e-12 else - mu / (2 * eps)

    i = np.arccos(np.clip(h_vec[2] / (h + 1e-18), -1.0, 1.0))

    # RAAN
    if n > 1e-18:
        Omega = np.arctan2(n_vec[1], n_vec[0])
    else:
        Omega = 0.0

    # 近点引数
    if n > 1e-18 and e > 1e-18:
        argp = np.arccos(np.clip(np.dot(n_vec, e_vec) / (n * e), -1.0, 1.0))
        if e_vec[2] < 0:
            argp = 2*np.pi - argp
    else:
        # 円/赤道軌道の退化ケース
        argp = 0.0

    # 真近点離角
    if e > 1e-18:
        nu = np.arccos(np.clip(np.dot(e_vec, r) / (e * R), -1.0, 1.0))
        if np.dot(r, v) < 0:
            nu = 2*np.pi - nu
    else:
        # 円軌道のときは引数を位置ベクトルと昇交点で定義
        if n > 1e-18:
            cos_u = np.dot(n_vec / n, r / R)
            sin_u = np.dot(np.cross(h_vec / h, n_vec / (n + 1e-18)), r / R)
            u = np.arctan2(sin_u, cos_u)
        else:
            u = np.arctan2(r[1], r[0])
        nu = u  # 近似扱い（円の場合は任意原点）

    return a, e, i, Omega, argp, nu

def cartesian_to_mee(r, v, mu=MU_EARTH):
    """
    (r,v) -> MEE (p,f,g,h,k,L)
    p = a(1-e^2)
    f = e cos(Ω+ω)
    g = e sin(Ω+ω)
    h = tan(i/2) cos Ω
    k = tan(i/2) sin Ω
    L = Ω + ω + ν
    """
    a, e, inc, Omega, argp, nu = _sv_to_coe(r, v, mu)
    p = a * (1 - e**2) if np.isfinite(a) else (np.linalg.norm(np.cross(r, v))**2 / mu)

    th = np.tan(inc / 2.0)
    h_m = th * np.cos(Omega)
    k_m = th * np.sin(Omega)

    Phi = Omega + argp
    f_m = e * np.cos(Phi)
    g_m = e * np.sin(Phi)

    L = Omega + argp + nu
    # 2πに正規化
    L = (L + 2*np.pi) % (2*np.pi)

    return (p, f_m, g_m, h_m, k_m, L)

def mee_to_cartesian(mee, mu=MU_EARTH):
    """
    MEE (p,f,g,h,k,L) -> (r,v)
    """
    p, f, g, h, k, L = mee
    # 復元
    e = np.hypot(f, g)
    th = np.hypot(h, k)
    inc = 2.0 * np.arctan(th)
    Omega = np.arctan2(k, h)
    Phi = np.arctan2(g, f)          # = Ω + ω
    argp = (Phi - Omega + 2*np.pi) % (2*np.pi)
    nu = (L - Phi + 2*np.pi) % (2*np.pi)

    # 位置速度（perifocal）
    r_mag = p / (1.0 + e * np.cos(nu))
    r_pf = np.array([r_mag * np.cos(nu), r_mag * np.sin(nu), 0.0])
    v_pf = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])

    # 回転行列 Q_pX (perifocal -> ECI)
    cO, sO = np.cos(Omega), np.sin(Omega)
    ci, si = np.cos(inc),   np.sin(inc)
    cw, sw = np.cos(argp),  np.sin(argp)

    Q = np.array([
        [ cO*cw - sO*sw*ci,  -cO*sw - sO*cw*ci,  sO*si],
        [ sO*cw + cO*sw*ci,  -sO*sw + cO*cw*ci, -cO*si],
        [ sw*si,              cw*si,             ci   ],
    ])

    r_eci = Q @ r_pf
    v_eci = Q @ v_pf
    return r_eci, v_eci, mu