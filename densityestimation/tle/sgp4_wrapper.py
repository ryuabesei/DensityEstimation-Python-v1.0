# Origin: convertTEMEtoJ2000.m
from __future__ import annotations

from typing import Tuple

import numpy as np
from sgp4.api import WGS72, WGS72OLD, WGS84, Satrec

# --- グローバル相当（MATLAB: global EOPMat） ---
EOPMat = None  # 後で .mat から読み込むローダを実装予定

def convert_teme_to_j2000(rteme: np.ndarray, vteme: np.ndarray, jdate: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert Cartesian r,v from TEME to J2000 (GCRF).
    Parameters
    ----------
    rteme : (3,) km
    vteme : (3,) km/s
    jdate : float (JD UTC)  # MATLAB実装に準拠

    Returns
    -------
    reci : (3,) km
    veci : (3,) km/s
    """
    global EOPMat
    rteme = np.asarray(rteme, dtype=float).reshape(3)
    vteme = np.asarray(vteme, dtype=float).reshape(3)

    # [~, ~, dut1, ~, ddpsi, ddeps, dat] = computeEOP_Celestrak( EOPMat, jdate );
    _, _, dut1, _, ddpsi, ddeps, dat = computeEOP_Celestrak(EOPMat, jdate)

    # date = jed2date(jdate) → [year mon day hr min sec]
    year, mon, day, hr, minute, sec = jed2date(jdate)

    timezone = 0
    # convtime(...): ttt を取得（世紀単位のユリウス世紀等、MATLABの戻り値に合わせる）
    (_, _, _, _, _, _, ttt,
     _, _, _, _, _, _, _, _) = convtime(year, mon, day, hr, minute, sec, timezone, dut1, dat)

    # [reci, veci, ~] = teme2eciNew(rteme, vteme, zeros(3,1), ttt, ddpsi, ddeps);
    reci, veci, _ = teme2eciNew(rteme, vteme, np.zeros(3), ttt, ddpsi, ddeps)
    return reci, veci


# --- ここから: MATLAB computeEOP_Celestrak.m の 1:1 Python 移植 ---

def computeEOP_Celestrak(EOPMat: np.ndarray, jdate: float):
    """
    Origin: computeEOP_Celestrak.m
    Inputs
    ------
    EOPMat : ndarray shape (N, 6)
        列1..6 = [xp[arcsec], yp[arcsec], dut1[s], lod[s], ddpsi[rad], ddeps[rad]]
        1962-01-01 00:00:00 の JD を起点に 1日1行で並ぶ想定
    jdate : float
        Julian Date (UTC)
    Returns
    -------
    xp, yp, dut1, lod, ddpsi, ddeps, dat
      dat は leap seconds [s]
    """
    xp = yp = dut1 = lod = ddpsi = ddeps = 0.0
    dat = 26  # s, 1992-07-01 以前の既定値

    # JD of 1962-01-01 00:00:00
    jdate0 = 2437665.5

    if EOPMat is None:
        raise ValueError("EOPMat is None. 地球姿勢パラメータ行列を読み込んでください。")

    # MATLAB: row = floor(jdate - jdate0) + 1;
    row = int(np.floor(jdate - jdate0) + 1)

    if 1 <= row <= EOPMat.shape[0]:
        # MATLABは1始まり, Pythonは0始まり
        r = row - 1
        xp    = float(EOPMat[r, 0])
        yp    = float(EOPMat[r, 1])
        dut1  = float(EOPMat[r, 2])
        lod   = float(EOPMat[r, 3])
        ddpsi = float(EOPMat[r, 4])
        ddeps = float(EOPMat[r, 5])

    # Leap seconds table (UTC-TAI.history 相当)
    DAT = np.array([
        # dat values
        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
         26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37],
        # start JD
        [2441317.5,2441499.5,2441683.5,2442048.5,2442413.5,2442778.5,2443144.5,2443509.5,
         2443874.5,2444239.5,2444786.5,2445151.5,2445516.5,2446247.5,2447161.5,2447892.5,
         2448257.5,2448804.5,2449169.5,2449534.5,2450083.5,2450630.5,2451179.5,2453736.5,
         2454832.5,2456109.5,2457204.5,2457754.5],
        # end JD
        [2441499.5,2441683.5,2442048.5,2442413.5,2442778.5,2443144.5,2443509.5,2443874.5,
         2444239.5,2444786.5,2445151.5,2445516.5,2446247.5,2447161.5,2447892.5,2448257.5,
         2448804.5,2449169.5,2449534.5,2450083.5,2450630.5,2451179.5,2453736.5,2454832.5,
         2456109.5,2457204.5,2457754.5, np.inf]
    ], dtype=float)

    i = 0
    # while ~((jdate<DAT(3,i))&&(jdate>=DAT(2,i)))  → 該当区間を探す
    while not ((jdate < DAT[2, i]) and (jdate >= DAT[1, i])):
        i += 1
        if i >= DAT.shape[1]:
            # 安全側：範囲外なら最後の値を維持
            i = DAT.shape[1] - 1
            break

    dat = int(DAT[0, i])
    return xp, yp, dut1, lod, ddpsi, ddeps, dat
# --- ここまで computeEOP_Celestrak ---


# --- ここから: MATLAB jed2date.m の 1:1 Python 移植（JD→[Y,M,D,h,m,s]） ---
def jed2date(jed: float):
    """
    Origin: jed2date.m
    MATLABの datevec(jed - 1721058.5) 同等:
      Julian Date → [year, month, day, hour, minute, second] (UTC)

    Returns
    -------
    (year, month, day, hour, minute, second_float)
    """
    # JD to Gregorian (UTC) - Fliegel–Van Flandern 系アルゴリズム
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
    if E < 14:
        month = E - 1
    else:
        month = E - 13
    if month > 2:
        year = C - 4716
    else:
        year = C - 4715

    # 小数日の時間成分へ分解
    frac_day = day - int(np.floor(day))
    day_int = int(np.floor(day))
    hours = frac_day * 24.0
    hour = int(np.floor(hours))
    minutes = (hours - hour) * 60.0
    minute = int(np.floor(minutes))
    seconds = (minutes - minute) * 60.0  # float秒（ミリ秒以下も保持）

    return int(year), int(month), int(day_int), int(hour), int(minute), float(seconds)
# --- ここまで jed2date ---


# --- convtime.m の 1:1 Python 実装 ---

def jday(year, mon, day, hr, minute, sec):
    """ValladoのJDN/JD。"""
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
    """秒→(時,分,秒) 整数h,m と float秒 を返す。"""
    s = float(seconds)
    h = int(np.floor(s / 3600.0))
    s -= h * 3600.0
    m = int(np.floor(s / 60.0))
    s -= m * 60.0
    return h, m, s

def convtime(year, mon, day, hr, minute, sec, timezone, dut1, dat):
    """
    Origin: convtime.m (Vallado)
    戻り: (ut1, tut1, jdut1, utc, tai, tt, ttt, jdtt, tdb, ttdb, jdtdb, tcg, jdtcg, tcb, jdtcb)
    """
    deg2rad = np.pi / 180.0

    jd = jday(year, mon, day, 0, 0, 0.0)
    # mjd = jd - 2400000.5  # （未使用）
    # mfme = hr*60 + minute + sec/60.0  # （未使用）

    localhr = timezone + hr
    utc = hms2sec(localhr, minute, sec)

    ut1 = utc + dut1
    hrtemp, mintemp, sectemp = sec2hms(ut1)
    jdut1 = jday(year, mon, day, hrtemp, mintemp, sectemp)
    tut1 = (jdut1 - 2451545.0) / 36525.0

    tai = utc + dat
    hrtemp, mintemp, sectemp = sec2hms(tai)
    jdtai = jday(year, mon, day, hrtemp, mintemp, sectemp)

    tt = tai + 32.184  # sec
    hrtemp, mintemp, sectemp = sec2hms(tt)
    jdtt = jday(year, mon, day, hrtemp, mintemp, sectemp)
    ttt = (jdtt - 2451545.0) / 36525.0

    # USNO circular (14) の近似（rad引数に注意：元式は係数×ttt + 位相）
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

    # TCG（近似）
    tcg = tt + 6.969290134e-10 * (jdtai - 2443144.5) * 86400.0
    hrtemp, mintemp, sectemp = sec2hms(tcg)
    jdtcg = jday(year, mon, day, hrtemp, mintemp, sectemp)

    # TCB（近似）
    tcbmtdb = 1.55051976772e-8 * (jdtai - 2443144.5) * 86400.0
    tcb = tdb + tcbmtdb
    hrtemp, mintemp, sectemp = sec2hms(tcb)
    jdtcb = jday(year, mon, day, hrtemp, mintemp, sectemp)

    return (ut1, tut1, jdut1, utc, tai, tt, ttt, jdtt,
            tdb, ttdb, jdtdb, tcg, jdtcg, tcb, jdtcb)

def teme2eciNew(rteme, vteme, ateme, ttt, ddpsi, ddeps):
    """
    Origin: teme2eciNew.m（本文は関数名 teme2eci）
    Inputs:
        rteme [km], vteme [km/s], ateme [km/s^2], ttt[Julian centuries of TT],
        ddpsi[rad], ddeps[rad]
    Returns:
        reci [km], veci [km/s], aeci [km/s^2]
    """
    prec, psia, wa, ea, xa = precess(ttt, '80')
    deltapsi, trueeps, meaneps, omega, nut = nutation(ttt, ddpsi, ddeps)

    # equation of equinoxes（geometric terms only）
    eqeg = deltapsi * np.cos(meaneps)
    eqeg = np.remainder(eqeg, 2.0*np.pi)

    eqe = np.array([[ np.cos(eqeg),  np.sin(eqeg), 0.0],
                    [-np.sin(eqeg),  np.cos(eqeg), 0.0],
                    [ 0.0,           0.0,          1.0]], dtype=float)

    tm = prec @ nut @ eqe.T

    reci = tm @ np.asarray(rteme, float).reshape(3)
    veci = tm @ np.asarray(vteme, float).reshape(3)
    aeci = tm @ np.asarray(ateme, float).reshape(3)
    return reci, veci, aeci


def precess(ttt: float, opt: str = '80'):
    """
    Origin: precess.m (Vallado)
    Parameters
    ----------
    ttt : float
        Julian centuries of TT since J2000.0
    opt : str
        '80' を想定（teme2eciNew からは '80' で呼ばれる）

    Returns
    -------
    prec : (3,3) ndarray
        MOD → J2000 変換行列（IAU-76/80）
    psia, wa, ea, xa : float
        規範的歳差角 [rad]
    """
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
    psia  *= convrt
    wa    *= convrt
    ea    *= convrt
    xa    *= convrt
    zeta  *= convrt
    theta *= convrt
    z     *= convrt

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
    """
    Origin: nutation.m (Vallado)
    Inputs
    ------
    ttt   : Julian centuries of TT
    ddpsi : delta psi correction to GCRF [rad]
    ddeps : delta eps correction to GCRF [rad]

    Returns
    -------
    deltapsi : float [rad]
    trueeps  : float [rad]
    meaneps  : float [rad]
    omega    : float [rad]
    nut      : (3,3) ndarray  TOD→MOD
    """
    import numpy as _np

    # --- IAU1980 係数と基本引数を取得（要: iau80in, fundarg） ---
    iar80, rar80 = iau80in()   # iar80: (106,5) int, rar80: (106,4) float [arcsec系]
    (l, l1, f, d, omega,
     lonmer, lonven, lonear, lonmar, lonjup, lonsat, lonurn, lonnep, precrate) = fundarg(ttt, '80')

    ttt2 = ttt * ttt
    ttt3 = ttt2 * ttt

    # mean obliquity [arcsec → rad]
    meaneps_arcsec = -46.8150*ttt - 0.00059*ttt2 + 0.001813*ttt3 + 84381.448
    meaneps = ( (meaneps_arcsec / 3600.0) % 360.0 ) * (_np.pi/180.0)

    # Σ(係数)（Valladoの式そのまま）
    deltapsi = 0.0
    deltaeps = 0.0
    # MATLAB は i=106:-1:1 だが加算なので順序は任意
    for i in range(iar80.shape[0]):
        tempval = (iar80[i,0]*l + iar80[i,1]*l1 + iar80[i,2]*f +
                   iar80[i,3]*d + iar80[i,4]*omega)
        deltapsi += (rar80[i,0] + rar80[i,1]*ttt) * _np.sin(tempval)
        deltaeps += (rar80[i,2] + rar80[i,3]*ttt) * _np.cos(tempval)

    # 1980は rar80 が arcsec 単位の係数で与えられる実装が多い
    # Vallado実装準拠の係数に合わせて arcsec→rad 変換（iau80in に依存）
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
    MATLAB同様、作業ディレクトリの 'nut80.dat' から読み込みます。
    推奨パス: densityestimation/data/nut80.dat
    """
    import os

    import numpy as np

    # 探索パス（順に試す）
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

    nut80 = np.loadtxt(path)
    if nut80.shape != (106, 9):
        raise ValueError(f"nut80.dat の形状が想定外です: {nut80.shape} (期待 106×9)")

    iar80 = nut80[:, 0:5].astype(int).copy()
    rar80 = nut80[:, 5:9].astype(float).copy()

    # 0.0001" → rad
    convrt = 0.0001 * np.pi / (180.0 * 3600.0)
    rar80 *= convrt
    return iar80, rar80

def fundarg(ttt: float, opt: str = '80'):
    """
    Origin: fundarg.m
    Parameters
    ----------
    ttt : Julian centuries of TT
    opt : '80'（IAU 1980）を使用
    Returns
    -------
    l, l1, f, d, omega,
    lonmer, lonven, lonear, lonmar, lonjup, lonsat, lonurn, lonnep, precrate [rad]
    """
    import numpy as np
    deg2rad = np.pi / 180.0

    if opt != '80':
        raise NotImplementedError("fundarg: 現在 '80' モードのみ実装しています。")

    # ---- iau 1980 theory（MATLAB式そのまま）----
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

    # deg → rad（0–360 wrap）
    def wrap_deg_to_rad(x):
        return ((x % 360.0) * deg2rad)

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

# densityestimation/tle/sgp4_wrapper.py の先頭あたりに追記
_EOPMat_global = None
def set_eop_matrix(eop: np.ndarray):
    global _EOPMat_global
    _EOPMat_global = np.asarray(eop)
def get_eop_matrix():
    return _EOPMat_global
# computeEOP_Celestrak 呼び出し側で get_eop_matrix() を使うよう変更
# 例: computeEOP_Celestrak(get_eop_matrix(), jdate)


# SGP4の μ は「km^3/s^2」で扱う（MATLAB の getgravc(72) と整合）
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
    - 戻り値: (satrec, a_TLE[km])
      a_TLE は TLE 平均運動 no から計算した半長径（km）。
    """
    if whichconst.upper() == "WGS72":
        wc = WGS72
        mu = MU_WGS72_KM3_S2
    elif whichconst.upper() == "WGS84":
        wc = WGS84
        mu = MU_WGS84_KM3_S2
    elif whichconst.upper() == "WGS72OLD":
        wc = WGS72OLD
        mu = MU_WGS72OLD_KM3_S2
    else:
        raise ValueError("whichconst must be one of: 'WGS72', 'WGS84', 'WGS72OLD'")

    # python-sgp4 で TLE から Satrec を作成（内部で epoch 変換など完了）
    satrec = Satrec.twoline2rv(line1.strip(), line2.strip(), wc)

    # MATLAB と同じ計算で a_TLE を出す:
    # satrec.no は [rad/min]（Kozai mean motion）。nk=[rad/s]
    nk = satrec.no / 60.0
    a_TLE = (mu / (nk ** 2)) ** (1.0 / 3.0)  # km

    return satrec, float(a_TLE)