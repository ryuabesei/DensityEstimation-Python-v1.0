
# densityestimation/tle/sgp4_wrapper.py
# Origin: convertTEMEtoJ2000.m
from __future__ import annotations

from typing import Tuple

import numpy as np
from sgp4.api import WGS72, WGS72OLD, WGS84, Satrec

# EOP をモジュール内にキャッシュ（MATLAB の global 代替）
_EOP_MAT: np.ndarray | None = None

def set_eop_matrix(eop: np.ndarray) -> None:
    """外部で読み込んだ EOP 行列 (N×6, xp/yp/ddpsi/ddeps は [rad]) を設定。"""
    global _EOP_MAT
    _EOP_MAT = np.asarray(eop)

def get_eop_matrix() -> np.ndarray | None:
    return _EOP_MAT


# eop_loader の公式実装を使用（重複実装を避ける）
from densityestimation.data.eop_loader import compute_eop_celestrak as _compute_eop


def convert_teme_to_j2000(rteme: np.ndarray, vteme: np.ndarray, jdate: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert Cartesian r,v from TEME to J2000 (GCRF).

    Parameters
    ----------
    rteme : (3,) km
    vteme : (3,) km/s
    jdate : float (JD UTC)

    Returns
    -------
    reci : (3,) km
    veci : (3,) km/s
    """
    rteme = np.asarray(rteme, dtype=float).reshape(3)
    vteme = np.asarray(vteme, dtype=float).reshape(3)

    eop_mat = get_eop_matrix()
    if eop_mat is None:
        raise ValueError("EOPMat is None. set_eop_matrix() で EOP を設定してください。")

    # EOP 値（ddpsi, ddeps は [rad]）
    eop = _compute_eop(eop_mat, jdate)
    dut1, ddpsi, ddeps, dat = eop.dut1, eop.ddpsi, eop.ddeps, eop.dat

    # JD → [Y,M,D,h,m,s]
    year, mon, day, hr, minute, sec = jed2date(jdate)

    # 時刻変換（TT 世紀 ttt を取得）
    timezone = 0
    (_, _, _, _, _, _, ttt,
     _, _, _, _, _, _, _, _) = convtime(year, mon, day, hr, minute, sec, timezone, dut1, dat)

    # TEME → ECI (J2000/GCRF)
    reci, veci, _ = teme2eciNew(rteme, vteme, np.zeros(3), ttt, ddpsi, ddeps)
    return reci, veci


# --- jed2date: JD → [Y,M,D,h,m,s] ---
def jed2date(jed: float):
    JD = float(jed)
    Z = int(np.floor(JD + 0.5))
    F = (JD + 0.5) - Z
    if Z < 2299161:
        A = Z
    else:
        alpha = int(np.floor((Z - 1867216.25) / 36524.25))
        A = Z + 1 + alpha - int(np.floor(alpha / 4))
    B = A + 1524
    C = int(np.floor((B - 122.1) / 365.25))
    D = int(np.floor(365.25 * C))
    E = int(np.floor((B - D) / 30.6001))
    day = B - D - int(np.floor(30.6001 * E)) + F
    month = E - 1 if E < 14 else E - 13
    year = C - 4716 if month > 2 else C - 4715
    frac_day = day - int(np.floor(day))
    day_int = int(np.floor(day))
    hours = frac_day * 24.0
    hour = int(np.floor(hours))
    minutes = (hours - hour) * 60.0
    minute = int(np.floor(minutes))
    seconds = (minutes - minute) * 60.0
    return int(year), int(month), int(day_int), int(hour), int(minute), float(seconds)


# --- Vallado convtime 周り ---
def jday(year, mon, day, hr, minute, sec):
    y, m = int(year), int(mon)
    D = float(day)
    if m <= 2:
        y -= 1
        m += 12
    A = int(y / 100)
    B = 2 - A + int(A / 4)
    JD = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + D + B - 1524.5
    JD += (hr + (minute + sec / 60.0) / 60.0) / 24.0
    return float(JD)

def hms2sec(h, m, s):
    return float(h) * 3600.0 + float(m) * 60.0 + float(s)

def sec2hms(seconds):
    s = float(seconds)
    h = int(np.floor(s / 3600.0)); s -= h * 3600.0
    m = int(np.floor(s / 60.0));   s -= m * 60.0
    return h, m, s

def convtime(year, mon, day, hr, minute, sec, timezone, dut1, dat):
    jd = jday(year, mon, day, 0, 0, 0.0)
    localhr = timezone + hr
    utc = hms2sec(localhr, minute, sec)
    ut1 = utc + dut1
    hrtemp, mintemp, sectemp = sec2hms(ut1)
    jdut1 = jday(year, mon, day, hrtemp, mintemp, sectemp)
    tut1 = (jdut1 - 2451545.0) / 36525.0
    tai = utc + dat
    hrtemp, mintemp, sectemp = sec2hms(tai)
    jdtai = jday(year, mon, day, hrtemp, mintemp, sectemp)
    tt = tai + 32.184
    hrtemp, mintemp, sectemp = sec2hms(tt)
    jdtt = jday(year, mon, day, hrtemp, mintemp, sectemp)
    ttt = (jdtt - 2451545.0) / 36525.0
    tdb = (tt
           + 0.001657*np.sin(628.3076*ttt + 6.2401)
           + 0.000022*np.sin(575.3385*ttt + 4.2970)
           + 0.000014*np.sin(1256.6152*ttt + 6.1969)
           + 0.000005*np.sin(606.9777*ttt + 4.0212)
           + 0.000005*np.sin(52.9691*ttt + 0.4444)
           + 0.000002*np.sin(21.3299*ttt + 5.5431)
           + 0.000010*ttt*np.sin(628.3076*ttt + 4.2490))
    hrtemp, mintemp, sectemp = sec2hms(tdb)
    jdtdb = jday(year, mon, day, hrtemp, mintemp, sectemp)
    ttdb = (jdtdb - 2451545.0) / 36525.0
    tcg = tt + 6.969290134e-10 * (jdtai - 2443144.5) * 86400.0
    hrtemp, mintemp, sectemp = sec2hms(tcg)
    jdtcg = jday(year, mon, day, hrtemp, mintemp, sectemp)
    tcbmtdb = 1.55051976772e-8 * (jdtai - 2443144.5) * 86400.0
    tcb = tdb + tcbmtdb
    hrtemp, mintemp, sectemp = sec2hms(tcb)
    jdtcb = jday(year, mon, day, hrtemp, mintemp, sectemp)
    return (ut1, tut1, jdut1, utc, tai, tt, ttt, jdtt,
            tdb, ttdb, jdtdb, tcg, jdtcg, tcb, jdtcb)


# --- TEME → ECI (J2000) ---
def teme2eciNew(rteme, vteme, ateme, ttt, ddpsi, ddeps):
    prec, psia, wa, ea, xa = precess(ttt, '80')
    deltapsi, trueeps, meaneps, omega, nut = nutation(ttt, ddpsi, ddeps)
    # equation of equinoxes（geometric terms only）
    eqeg = np.remainder(deltapsi * np.cos(meaneps), 2.0*np.pi)
    eqe = np.array([[ np.cos(eqeg),  np.sin(eqeg), 0.0],
                    [-np.sin(eqeg),  np.cos(eqeg), 0.0],
                    [ 0.0,           0.0,          1.0]], dtype=float)
    tm = prec @ nut @ eqe.T
    reci = tm @ np.asarray(rteme, float).reshape(3)
    veci = tm @ np.asarray(vteme, float).reshape(3)
    aeci = tm @ np.asarray(ateme, float).reshape(3)
    return reci, veci, aeci


def precess(ttt: float, opt: str = '80'):
    import numpy as _np
    convrt = _np.pi / (180.0 * 3600.0)
    ttt2 = ttt * ttt
    ttt3 = ttt2 * ttt
    if opt != '80':
        raise NotImplementedError("precess: 現在 '80' モードのみ実装しています。")
    # IAU 76 precession angles [arcsec]
    psia  = 5038.7784*ttt - 1.07259*ttt2 - 0.001147*ttt3
    wa    = 84381.448                 + 0.05127*ttt2 - 0.007726*ttt3
    ea    = 84381.448 - 46.8150*ttt - 0.00059*ttt2 + 0.001813*ttt3
    xa    =              10.5526*ttt - 2.38064*ttt2 - 0.001125*ttt3
    zeta  = 2306.2181*ttt + 0.30188*ttt2 + 0.017998*ttt3
    theta = 2004.3109*ttt - 0.42665*ttt2 - 0.041833*ttt3
    z     = 2306.2181*ttt + 1.09468*ttt2 + 0.018203*ttt3
    # arcsec → rad
    psia  *= convrt; wa *= convrt; ea *= convrt; xa *= convrt
    zeta  *= convrt; theta *= convrt; z *= convrt
    coszeta  = _np.cos(zeta);  sinzeta  = _np.sin(zeta)
    costheta = _np.cos(theta); sintheta = _np.sin(theta)
    cosz     = _np.cos(z);     sinz     = _np.sin(z)
    prec = _np.array([
        [ coszeta*costheta*cosz - sinzeta*sinz,  coszeta*costheta*sinz + sinzeta*cosz,  coszeta*sintheta],
        [-sinzeta*costheta*cosz - coszeta*sinz, -sinzeta*costheta*sinz + coszeta*cosz, -sinzeta*sintheta],
        [-sintheta*cosz,                          -sintheta*sinz,                         costheta        ],
    ], dtype=float)
    return prec, psia, wa, ea, xa


def nutation(ttt: float, ddpsi: float, ddeps: float):
    import numpy as _np
    iar80, rar80 = iau80in()
    (l, l1, f, d, omega,
     lonmer, lonven, lonear, lonmar, lonjup, lonsat, lonurn, lonnep, precrate) = fundarg(ttt, '80')
    ttt2 = ttt * ttt
    ttt3 = ttt2 * ttt
    # mean obliquity [arcsec → rad]
    meaneps_arcsec = -46.8150*ttt - 0.00059*ttt2 + 0.001813*ttt3 + 84381.448
    meaneps = ((meaneps_arcsec / 3600.0) % 360.0) * (_np.pi/180.0)
    # Σ(係数)
    deltapsi = 0.0
    deltaeps = 0.0
    for i in range(iar80.shape[0]):
        tempval = (iar80[i,0]*l + iar80[i,1]*l1 + iar80[i,2]*f +
                   iar80[i,3]*d + iar80[i,4]*omega)
        deltapsi += (rar80[i,0] + rar80[i,1]*ttt) * _np.sin(tempval)
        deltaeps += (rar80[i,2] + rar80[i,3]*ttt) * _np.cos(tempval)
    asec2rad = _np.pi / (180.0*3600.0)
    deltapsi = (deltapsi * asec2rad + ddpsi) % (2.0*_np.pi)
    deltaeps = (deltaeps * asec2rad + ddeps) % (2.0*_np.pi)
    trueeps = meaneps + deltaeps
    cospsi   = _np.cos(deltapsi);  sinpsi   = _np.sin(deltapsi)
    coseps   = _np.cos(meaneps);   sineps   = _np.sin(meaneps)
    costeps  = _np.cos(trueeps);   sinteps  = _np.sin(trueeps)
    nut = _np.array([
        [  cospsi,                     costeps*sinpsi,                 sinteps*sinpsi],
        [ -coseps*sinpsi,  costeps*coseps*cospsi + sinteps*sineps,  sinteps*coseps*cospsi - sineps*costeps],
        [ -sineps*sinpsi,  costeps*sineps*cospsi - sinteps*coseps,  sinteps*sineps*cospsi + costeps*coseps],
    ], dtype=float)
    return deltapsi, trueeps, meaneps, omega, nut


def iau80in():
    """
    Origin: iau80in.m
    Returns
    -------
    iar80 : (106,5) int ndarray
    rar80 : (106,4) float ndarray  # [rad]（0.0001"→rad に変換済）
    Notes
    -----
    実ファイルの差異に対応：
      - 106×9 形式（標準：5整数 + 4係数）
      - 106×10 形式（先頭に通し番号の列が付与：idx + 5整数 + 4係数）
      - それ以外は「末尾9列」を解釈するフォールバック
    推奨パス: densityestimation/data/nut80.dat
    """
    import os

    import numpy as np

    candidates = [
        "nut80.dat",
        "densityestimation/data/nut80.dat",
        "/code/densityestimation/data/nut80.dat",
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError(
            "nut80.dat が見つかりません。repository の "
            "Python/densityestimation/data/ 配下に置いてください。"
        )

    data = np.loadtxt(path)

    # 形式分岐
    if data.shape == (106, 9):
        # [5 ints | 4 floats]
        core = data
    elif data.shape == (106, 10):
        # 先頭列が通し番号（1..106）なら落とす
        first_col = data[:, 0]
        if np.allclose(first_col, np.arange(1, 107)):
            core = data[:, 1:]  # 9列に整形
        else:
            # 先頭列が index でない場合は末尾9列を採用
            core = data[:, -9:]
    elif data.shape[1] > 9:
        # 列が多い未知フォーマット → 末尾9列を採用（[5 ints | 4 floats]想定）
        core = data[:, -9:]
    else:
        raise ValueError(
            f"nut80.dat の形状が想定外です: {data.shape} "
            "(期待 106×9、または先頭に通し番号が付いた 106×10)"
        )

    if core.shape != (106, 9):
        raise ValueError(f"nut80.dat コアの形状が 106×9 になりませんでした: {core.shape}")

    iar80 = core[:, 0:5].astype(int).copy()
    rar80 = core[:, 5:9].astype(float).copy()

    # 0.0001" → rad
    convrt = 0.0001 * np.pi / (180.0 * 3600.0)
    rar80 *= convrt
    return iar80, rar80



def fundarg(ttt: float, opt: str = '80'):
    import numpy as np
    deg2rad = np.pi / 180.0
    if opt != '80':
        raise NotImplementedError("fundarg: 現在 '80' モードのみ実装しています。")
    l     = ((((0.064)*ttt + 31.310)*ttt + 1717915922.6330)*ttt)/3600.0 + 134.96298139
    l1    = ((((-0.012)*ttt - 0.577)*ttt + 129596581.2240)*ttt)/3600.0 + 357.52772333
    f     = ((((0.011)*ttt - 13.257)*ttt + 1739527263.1370)*ttt)/3600.0 + 93.27191028
    d     = ((((0.019)*ttt - 6.891)*ttt + 1602961601.3280)*ttt)/3600.0 + 297.85036306
    omega = ((((0.008)*ttt + 7.455)*ttt - 6962890.5390)*ttt)/3600.0 + 125.04452222
    lonmer  = 252.3 + 149472.0 * ttt
    lonven  = 179.9 +  58517.8 * ttt
    lonear  =  98.4 +  35999.4 * ttt
    lonmar  = 353.3 +  19140.3 * ttt
    lonjup  =  32.3 +   3034.9 * ttt
    lonsat  =  48.0 +   1222.1 * ttt
    lonurn  =   0.0
    lonnep  =   0.0
    precrate=   0.0
    def wrap_deg_to_rad(x): return ((x % 360.0) * deg2rad)
    l      = wrap_deg_to_rad(l)
    l1     = wrap_deg_to_rad(l1)
    f      = wrap_deg_to_rad(f)
    d      = wrap_deg_to_rad(d)
    omega  = wrap_deg_to_rad(omega)
    lonmer = wrap_deg_to_rad(lonmer)
    lonven = wrap_deg_to_rad(lonven)
    lonear = wrap_deg_to_rad(lonear)
    lonmar = wrap_deg_to_rad(lonmar)
    lonjup = wrap_deg_to_rad(lonjup)
    lonsat = wrap_deg_to_rad(lonsat)
    lonurn = wrap_deg_to_rad(lonurn)
    lonnep = wrap_deg_to_rad(lonnep)
    precrate = wrap_deg_to_rad(precrate)
    return (l, l1, f, d, omega,
            lonmer, lonven, lonear, lonmar, lonjup, lonsat, lonurn, lonnep, precrate)


# SGP4の μ は km^3/s^2（MATLAB getgravc(72) と整合）
MU_WGS72_KM3_S2 = 398600.8
MU_WGS84_KM3_S2 = 398600.5
MU_WGS72OLD_KM3_S2 = 398600.79964

def twoline2rv_edit_py(
    line1: str,
    line2: str,
    whichconst: str = "WGS72",  # "WGS72" | "WGS84" | "WGS72OLD"
) -> Tuple[Satrec, float]:
    """
    MATLAB twoline2rv_edit.m のPython版互換。
    - 戻り値: (satrec, a_TLE[km])  ※a_TLE は TLE の no から算出した半長径
    """
    if whichconst.upper() == "WGS72":
        wc = WGS72;     mu = MU_WGS72_KM3_S2
    elif whichconst.upper() == "WGS84":
        wc = WGS84;     mu = MU_WGS84_KM3_S2
    elif whichconst.upper() == "WGS72OLD":
        wc = WGS72OLD;  mu = MU_WGS72OLD_KM3_S2
    else:
        raise ValueError("whichconst must be one of: 'WGS72', 'WGS84', 'WGS72OLD'")

    satrec = Satrec.twoline2rv(line1.strip(), line2.strip(), wc)
    nk = satrec.no / 60.0  # [rad/s]
    a_TLE = (mu / (nk ** 2)) ** (1.0 / 3.0)  # [km]
    return satrec, float(a_TLE)

