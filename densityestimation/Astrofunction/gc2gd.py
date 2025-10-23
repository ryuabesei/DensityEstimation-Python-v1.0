# densityestimation/Astrofunction/gc2gd.py
# License: GNU GPL v3
#地心直交座標 → 測地量（経度・緯度・高度）
from __future__ import annotations

import numpy as np
from sgp4.api import jday


def gc2gd(
    r: np.ndarray,
    yr: int,
    mth: int,
    day: int,
    hr: int,
    minute: int,
    sec: float,
    dt: float,
    tf: float,
    flag: int,
):
    """
    Geocentric (ECI/ECEF Cartesian) -> Geodetic quantities.

    Parameters
    ----------
    r : (N,3) ndarray
        Cartesian position(s) [km].
    yr,mth,day,hr,minute,sec : int,float
        UTC timestamp.
    dt : float
        Sampling interval [s].
    tf : float
        Run time [s].
    flag : int
        1なら経度を [-360, 360] に正規化。

    Returns
    -------
    long : (N,) ndarray
        Longitude [deg].
    lat : (N,) ndarray
        Geodetic latitude [deg].
    alt : (N,) ndarray
        Geodetic altitude [km].
    alp : (N,) ndarray
        Right ascension of position vector [deg]（連続化済み）。
    gst : (M,) ndarray
        Greenwich Sidereal Time [deg], M=len(t)（通常は1: dt=0）。
    """
    r = np.asarray(r, dtype=float)
    if r.ndim != 2 or r.shape[1] != 3:
        raise ValueError("r must be shape (N,3) in km")

    # --- time vector t ---
    if dt == 0:
        t = np.array([0.0])
    else:
        # include endpoint like MATLAB 0:dt:tf
        nstep = int(np.floor(tf / dt + 1e-12))
        t = np.arange(nstep + 1, dtype=float) * dt
        # If tf is exactly multiple of dt, this matches MATLAB
        if t.size == 0:
            t = np.array([0.0])

    # --- Earth constants ---
    f = 1.0 / 298.257  # flattening
    req = 6378.14      # km

    # --- position magnitudes ---
    rmag = np.sqrt(np.sum(r * r, axis=1))
    # avoid divide by zero
    rmag = np.where(rmag == 0.0, np.finfo(float).eps, rmag)

    # --- altitude (km) ---
    delta = np.arcsin(np.clip(r[:, 2] / rmag, -1.0, 1.0))
    alt = rmag - req * (
        1.0
        - f * np.sin(delta) ** 2
        - (f ** 2 / 2.0) * (np.sin(2.0 * delta) ** 2) * (req / rmag - 0.25)
    )

    # --- geodetic latitude (deg) ---
    sinfd = (req / rmag) * (
        f * np.sin(2.0 * delta) + (f * f) * np.sin(4.0 * delta) * (req / rmag - 0.25)
    )
    # 数値安全のためクリップ
    sinfd = np.clip(sinfd, -1.0, 1.0)
    lat = (delta + np.arcsin(sinfd)) * 180.0 / np.pi

    # --- Greenwich Sidereal Time (deg) ---
    # jdate = JD(yr,mth,day,hr,minute,sec) + t/86400
    jd0, fr0 = jday(yr, mth, day, hr, minute, float(sec))
    jdate = (jd0 + fr0) + (t / 86400.0)

    # tdays, jcent
    tdays = jdate - 2415020.0
    jcent = tdays / 36525.0

    # ut in degrees (= 360 * UT_hours / 24)
    ut_hours = (sec + t) / 3600.0 + minute / 60.0 + hr
    ut_deg = ut_hours * (360.0 / 24.0)

    # same polynomial as MATLAB code
    gst = 99.6910 + 36000.7689 * jcent + 0.0004 * (jcent ** 2) + ut_deg
    # 不要だが念のため 0..360 正規化はしない（MATLABもしていない）

    # --- Right ascension (deg), then unwrap like MATLAB ---
    alp_rad = np.arctan2(r[:, 1], r[:, 0])  # [-pi,pi]
    alp_rad_unwrapped = np.unwrap(alp_rad)  # 連続化
    alp = alp_rad_unwrapped * 180.0 / np.pi

    # --- Longitude = RA - GST ---
    # 通常は gst がスカラー（dt=0）なのでブロードキャストされる
    long = alp - gst[0] if gst.size == 1 else alp[:, None] - gst[None, :]
    # 返却は (N,) を基本とするため、gstがベクトルのケースは
    # 呼び出し側の想定外なので、ここでは dt=0 前提で 1D に整形
    if long.ndim > 1:
        # 各行同じ長さのベクトル差になってしまうため、
        # 代表として最初の時間成分を返す
        long = long[:, 0]

    # --- make longitude within ±360 around the first element sign, like MATLAB ---
    if long.size > 0:
        ll0 = long[0]
        if ll0 < 0.0:
            # add 360 until first > 0
            while long[0] <= 0.0:
                long = long + 360.0
        else:
            # subtract 360 until first < 360
            while long[0] >= 360.0:
                long = long - 360.0

    # --- if flag==1, clamp all longitudes to [-360,360] via ±360 shifts ---
    if int(flag) == 1:
        # reduce values >360 by 360 repeatedly
        # and values <-360 by +360 repeatedly
        # （有限回で収束）
        changed = True
        # 安全ブレーク
        it = 0
        while changed and it < 1000:
            it += 1
            changed = False
            idx = np.where(long > 360.0)[0]
            if idx.size:
                long[idx] = long[idx] - 360.0
                changed = True
            idx = np.where(long < -360.0)[0]
            if idx.size:
                long[idx] = long[idx] + 360.0
                changed = True

    return long.astype(float), lat.astype(float), alt.astype(float), alp.astype(float), gst.astype(float)
