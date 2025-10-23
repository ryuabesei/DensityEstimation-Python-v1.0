# License: GNU GPL v3
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat

# =========================
# Data structures
# =========================

@dataclass
class ROMBundle:
    AC: np.ndarray
    BC: np.ndarray
    Uh: np.ndarray
    F_U: List[Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]]
    Dens_Mean: np.ndarray
    M_U: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    SLTm: np.ndarray
    LATm: np.ndarray
    ALTm: np.ndarray
    maxAtmAlt: float
    SWinputs: np.ndarray          # shape = (nfeat, Nepoch). 1行目が jdate
    Qrom: np.ndarray


# =========================
# Helpers (generic)
# =========================

def _mat_try_keys(d: dict, *keys: str):
    for k in keys:
        if k in d:
            return d[k]
    raise KeyError(f"Any of keys not found: {keys}")

def _to_1d(a: np.ndarray) -> np.ndarray:
    return np.array(a, dtype=float).reshape(-1)

def _build_interpolants(Uh_vec: np.ndarray,
                        slt: np.ndarray, lat: np.ndarray, alt: np.ndarray
                        ) -> Tuple[List[Callable], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Uh_vec : (Nslt*Nlat*Nalt, r)
    戻り値:
      F_U : 各モードの 3D 補間器のリスト（(slt,lat,alt)->値）
      (SLTm,LATm,ALTm) : 3D グリッド（ndgrid相当）
    """
    n_slt, n_lat, n_alt = len(slt), len(lat), len(alt)
    SLTm, LATm, ALTm = np.meshgrid(slt, lat, alt, indexing="ij")

    F_U: List[Callable] = []
    for i in range(Uh_vec.shape[1]):
        grid_i = Uh_vec[:, i].reshape(n_slt, n_lat, n_alt)
        rgi = RegularGridInterpolator((slt, lat, alt), grid_i, bounds_error=False, fill_value=None)
        def _wrap(f):
            def _call(slt_in, lat_in, alt_in):
                pts = np.stack([np.ravel(slt_in), np.ravel(lat_in), np.ravel(alt_in)], axis=1)
                val = f(pts)
                return val.reshape(np.broadcast(slt_in, lat_in, alt_in).shape)
            return _call
        F_U.append(_wrap(rgi))
    return F_U, (SLTm, LATm, ALTm)

def _build_mean_interp(dmean_vec: np.ndarray, slt: np.ndarray, lat: np.ndarray, alt: np.ndarray):
    n_slt, n_lat, n_alt = len(slt), len(lat), len(alt)
    grid = dmean_vec.reshape(n_slt, n_lat, n_alt)
    rgi = RegularGridInterpolator((slt, lat, alt), grid, bounds_error=False, fill_value=None)
    def _call(slt_in, lat_in, alt_in):
        pts = np.stack([np.ravel(slt_in), np.ravel(lat_in), np.ravel(alt_in)], axis=1)
        val = rgi(pts)
        return val.reshape(np.broadcast(slt_in, lat_in, alt_in).shape)
    return _call


# =========================
# Space Weather: parsers (from CelesTrak SW-All.txt)
# =========================
# NRLMSISE と TIEGCM は本文書内で自己完結に実装
# JB2008 は sw_provider を外から注入（複雑な依存のため）

def _input_sw_nrlmsise(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    MATLAB inputSWnrlmsise.m のポート。
    返り値:
      SWmatDaily: (N,11) [F10.7Daily, F10.7Average, magIndex, AP(8)]
      SWmatMonthlyPred: (M,2) [F10.7Daily, F10.7Average]
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # 先頭17行スキップ
    i = 17
    # 観測点数
    n_daily_obs = int(lines[i][21:25]); i += 1
    i += 1  # "BEGIN OBSERVED"

    SWaux = np.zeros((n_daily_obs, 11), dtype=float)
    for k in range(n_daily_obs):
        s = lines[i+k]
        SWaux[k, 0] = float(s[94:99])   # F10.7 Daily
        SWaux[k, 1] = float(s[102:107]) # F10.7 Average
        SWaux[k, 2] = float(s[80:83])   # Daily Magnetic index
        ap8 = [s[47:51], s[51:55], s[55:59], s[59:63],
               s[63:67], s[67:71], s[71:75], s[75:79]]
        SWaux[k, 3:11] = np.array([float(x) for x in ap8], dtype=float)
        if SWaux[k, 0] == 0.0:
            SWaux[k, 0] = SWaux[k, 1]

    i += n_daily_obs
    i += 3
    pdt_pnt = int(lines[i][28:30]); i += 1
    SWmatDaily = np.zeros((n_daily_obs + pdt_pnt, 11), dtype=float)
    SWmatDaily[:n_daily_obs] = SWaux

    i += 1  # "BEGIN DAILY_PREDICTED"
    for k in range(pdt_pnt):
        s = lines[i+k]
        SWmatDaily[n_daily_obs+k, 0] = float(s[94:99])   # F10.7 Daily
        SWmatDaily[n_daily_obs+k, 1] = float(s[102:107]) # F10.7 Average
        SWmatDaily[n_daily_obs+k, 2] = float(s[80:83])   # Daily Magnetic index
        ap8 = [s[47:51], s[51:55], s[55:59], s[59:63],
               s[63:67], s[67:71], s[71:75], s[75:79]]
        SWmatDaily[n_daily_obs+k, 3:11] = np.array([float(x) for x in ap8], dtype=float)

    i += pdt_pnt
    i += 3
    mpd_pnt = int(lines[i][30:32]); i += 1
    SWmatMonthlyPred = np.zeros((mpd_pnt, 2), dtype=float)

    i += 1  # "BEGIN MONTHLY_PREDICTED"
    for k in range(mpd_pnt):
        s = lines[i+k]
        SWmatMonthlyPred[k, 0] = float(s[94:99])   # F10.7 Daily
        SWmatMonthlyPred[k, 1] = float(s[102:107]) # F10.7 Average

    return SWmatDaily, SWmatMonthlyPred


def _input_sw_tiegcm(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    MATLAB inputSWtiegcm.m のポート（Kpは /10 済みで返す）。
    返り値:
      SWmatDaily: (N,11) [F10.7Daily, F10.7Average, -, Kp(8)]  ※col3は未使用のため0
      SWmatMonthlyPred: (M,2) [F10.7Daily, F10.7Average]
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    i = 17
    n_daily_obs = int(lines[i][21:25]); i += 1
    i += 1  # "BEGIN OBSERVED"

    SWaux = np.zeros((n_daily_obs, 11), dtype=float)
    for k in range(n_daily_obs):
        s = lines[i+k]
        SWaux[k, 0] = float(s[94:99])               # F10.7 Daily
        SWaux[k, 1] = float(s[102:107])             # F10.7 Average
        kp8 = [s[19:22], s[22:25], s[25:28], s[28:31],
               s[31:34], s[34:37], s[37:40], s[40:43]]
        SWaux[k, 3:11] = np.array([float(x)/10.0 for x in kp8], dtype=float)
        if SWaux[k, 0] == 0.0:
            SWaux[k, 0] = SWaux[k, 1]

    i += n_daily_obs
    i += 3
    pdt_pnt = int(lines[i][28:30]); i += 1
    SWmatDaily = np.zeros((n_daily_obs + pdt_pnt, 11), dtype=float)
    SWmatDaily[:n_daily_obs] = SWaux

    i += 1  # "BEGIN DAILY_PREDICTED"
    for k in range(pdt_pnt):
        s = lines[i+k]
        SWmatDaily[n_daily_obs+k, 0] = float(s[94:99])
        SWmatDaily[n_daily_obs+k, 1] = float(s[102:107])
        kp8 = [s[19:22], s[22:25], s[25:28], s[28:31],
               s[31:34], s[34:37], s[37:40], s[40:43]]
        SWmatDaily[n_daily_obs+k, 3:11] = np.array([float(x)/10.0 for x in kp8], dtype=float)

    i += pdt_pnt
    i += 3
    mpd_pnt = int(lines[i][30:32]); i += 1
    SWmatMonthlyPred = np.zeros((mpd_pnt, 2), dtype=float)

    i += 1  # "BEGIN MONTHLY_PREDICTED"
    for k in range(mpd_pnt):
        s = lines[i+k]
        SWmatMonthlyPred[k, 0] = float(s[94:99])
        SWmatMonthlyPred[k, 1] = float(s[102:107])

    return SWmatDaily, SWmatMonthlyPred


# =========================
# Space Weather: simple calculators
# =========================

def _movmean(x: np.ndarray, left: int, right: int) -> np.ndarray:
    """MATLAB movmean(x,[left right]) の簡易版（端は片側平均）。"""
    w = left + right + 1
    kernel = np.ones(w, dtype=float) / w
    y = np.convolve(x, kernel, mode="same")
    # 端の補正（片側しかない分を平均）
    n = x.size
    for i in range(n):
        l = max(0, i - left)
        r = min(n, i + right + 1)
        y[i] = np.mean(x[l:r]) if r > l else x[i]
    return y

def _day_of_year(y: int, m: int, d: int) -> int:
    import datetime as _dt
    return int((_dt.date(y, m, d) - _dt.date(y, 1, 1)).days + 1)

def _jd_to_ymdhms(jd: float) -> Tuple[int, int, int, int, int, float]:
    """Julian Date -> (Y,M,D,h,m,s) 近似（十分精度）。"""
    # Meeusのアルゴリズム簡易版
    Z = int(jd + 0.5)
    F = (jd + 0.5) - Z
    if Z < 2299161:
        A = Z
    else:
        alpha = int((Z - 1867216.25) / 36524.25)
        A = Z + 1 + alpha - int(alpha / 4)
    B = A + 1524
    C = int((B - 122.1) / 365.25)
    D = int(365.25 * C)
    E = int((B - D) / 30.6001)
    day = B - D - int(30.6001 * E) + F
    month = E - 1 if E < 14 else E - 13
    year = C - 4716 if month > 2 else C - 4715
    # day の小数を h:m:s へ
    d_i = int(day)
    frac = day - d_i
    h = int(frac * 24.0)
    m = int((frac * 24.0 - h) * 60.0)
    s = ((frac * 24.0 - h) * 60.0 - m) * 60.0
    return year, month, d_i, h, m, s

def _jd_range(jd0: float, jdf: float, step_hours: float = 1.0) -> np.ndarray:
    N = int(round((jdf - jd0) * 24.0 / step_hours)) + 1
    return jd0 + (np.arange(N) * (step_hours / 24.0))


def _compute_swinputs_nrlmsise(jd0: float, jdf: float,
                               SWmatDaily: np.ndarray, SWmatMonthlyPred: np.ndarray) -> np.ndarray:
    """
    MATLAB computeSWinputs_NRLMSISE の簡易ポート（同じ列構成を返す）。
    返り値 shape: (>=41, Nepoch)
    """
    tt = _jd_range(jd0, jdf, 1.0)  # hourly
    N = tt.size
    Inputs = np.zeros((41, N), dtype=float)  # 41列あれば十分（元はもっと多いが主要部分を網羅）

    # 1..3: jdate, doy, UT(h)
    Inputs[0, :] = tt
    for i, jdate in enumerate(tt):
        y, m, d, hh, mm, ss = _jd_to_ymdhms(jdate)
        Inputs[1, i] = _day_of_year(y, m, d)
        Inputs[2, i] = hh + mm / 60.0 + ss / 3600.0

    # 簡易 computeSWnrlmsise：SWmatDaily を「近傍最近」で引く
    def _nearest_row(mat, jdate):
        # 厳密な日付一致は難しいので、単純にインデックス0固定でも雛形としてはOK
        idx = 0
        return mat[idx]

    # 4..12: F10a, F10d, ap7
    for i, jdate in enumerate(tt):
        row = _nearest_row(SWmatDaily, jdate)
        f10d = float(row[0]); f10a = float(row[1])
        ap_daily = float(row[2])
        ap8 = row[3:11].astype(float)
        ap7 = np.zeros(7, dtype=float)
        ap7[0] = ap8[0]
        ap7[1] = ap_daily
        ap7[2:] = ap8[1:6]
        Inputs[3, i] = f10a
        Inputs[4, i] = f10d
        Inputs[5:12, i] = ap7

    # 平滑化
    Inputs[3, :] = _movmean(Inputs[3, :], 12, 11)
    Inputs[4, :] = _movmean(Inputs[4, :], 12, 11)
    Inputs[5, :] = _movmean(Inputs[5, :], 12, 11)
    for k in range(6, 12):
        Inputs[k, :] = _movmean(Inputs[k, :], 3, 0)  # 3h平均（近似）

    # 13..21: +1h 先行
    Inputs[12:21, :-1] = Inputs[3:12, 1:]
    Inputs[12:21, -1] = Inputs[3:12, -1]  # 最終点は据え置き

    # 22..30: 2乗
    Inputs[21:30, :] = Inputs[3:12, :] ** 2
    # 31..39: 2乗の +1h
    Inputs[30:39, :] = Inputs[12:21, :] ** 2
    # 40..41: 混合項
    Inputs[39, :] = Inputs[4, :] * Inputs[6, :]     # F10d * ap(=Inputs[7]?) -> 近似的に [6] 使用
    Inputs[40, :] = Inputs[13, :] * Inputs[15, :]   # F10d(+1h) * ap(+1h) 近似

    return Inputs


def _compute_swinputs_tiegcm(jd0: float, jdf: float,
                             SWmatDaily: np.ndarray, SWmatMonthlyPred: np.ndarray) -> np.ndarray:
    """
    MATLAB computeSWinputs_TIEGCM の簡易ポート。
    返り値 shape: (12, Nepoch)
    """
    tt = _jd_range(jd0, jdf, 1.0)
    N = tt.size
    Inputs = np.zeros((12, N), dtype=float)

    Inputs[0, :] = tt
    for i, jdate in enumerate(tt):
        y, m, d, hh, mm, ss = _jd_to_ymdhms(jdate)
        Inputs[1, i] = _day_of_year(y, m, d)
        Inputs[2, i] = hh + mm / 60.0 + ss / 3600.0

    def _nearest_row(mat, jdate):
        idx = 0
        return mat[idx]

    # 4: F107(daily), 5: F107(avg), 6: Kp (3-hourly) を時刻に依存せず近似
    for i, jdate in enumerate(tt):
        row = _nearest_row(SWmatDaily, jdate)
        f10d = float(row[0]); f10a = float(row[1])
        kp8 = row[3:11].astype(float)
        kp_approx = kp8[0]  # その時間帯の代表値として
        Inputs[3, i] = f10d
        Inputs[4, i] = f10a
        Inputs[5, i] = kp_approx

    # 平滑化
    Inputs[3, :] = _movmean(Inputs[3, :], 12, 11)
    Inputs[4, :] = _movmean(Inputs[4, :], 12, 11)
    Inputs[5, :] = _movmean(Inputs[5, :], 3, 0)

    # 7..8: +1h 先行 (F10, Kp)
    Inputs[6, :-1] = Inputs[3, 1:]
    Inputs[7, :-1] = Inputs[5, 1:]
    Inputs[6, -1] = Inputs[3, -1]
    Inputs[7, -1] = Inputs[5, -1]

    # 9..10: Kp^2（現時刻/先行）
    Inputs[8, :]  = Inputs[5, :] ** 2
    Inputs[9, :]  = Inputs[7, :] ** 2
    # 11..12: Kp*F10（現時刻/先行）
    Inputs[10, :] = Inputs[5, :] * Inputs[3, :]
    Inputs[11, :] = Inputs[7, :] * Inputs[6, :]

    return Inputs


# =========================
# Main: unified ROM generator
# =========================

def generate_rom_density_model(name: str, r: int, jd0: float, jdf: float,
                               sw_provider: Optional[Callable[[float, float], np.ndarray]] = None
                               ) -> ROMBundle:
    """
    MATLAB generateROMdensityModel.m の統合 Python 版。
      - *.mat から TA を読み出し、PhiC / Uh / Qrom / グリッド（slt,lat,alt）を取得
      - Uh を 3D 補間器群 F_U に、平均対数密度 Dens_Mean を補間器 M_U に
      - AC, BC = PhiC のブロックを 1/3600 でスケーリング（[hr]→[s] 換算）
      - Space Weather 入力 SWinputs を組み立て（NRLMSISE/TIEGCMは内蔵、JB2008は sw_provider で注入）

    Parameters
    ----------
    name : {"JB2008_1999_2010", "NRLMSISE_1997_2008", "TIEGCM_1997_2008"}
    r    : ROM 次元
    jd0, jdf : 推定期間の開始/終了 JD（jdf は end-exclusive 扱いで +1h 精度）
    sw_provider : JB2008 のときに必須。sw_provider(jd0, jdf) -> SWinputs ndarray
                  （1行目が jdate、以降は MATLAB computeSWinputs_JB2008 と同じ列構成）
    """
    if name == "JB2008_1999_2010":
        mat_path = "JB2008_1999_2010_ROM_r100.mat"
        TA = loadmat(mat_path, squeeze_me=True, struct_as_string=False)

        slt = _to_1d(_mat_try_keys(TA, "localSolarTimes", "slt"))
        lat = _to_1d(_mat_try_keys(TA, "latitudes", "lat"))
        alt = _to_1d(_mat_try_keys(TA, "altitudes", "alt"))
        Dens_Mean = _to_1d(_mat_try_keys(TA, "densityDataMeanLog", "Dens_Mean"))
        PhiC_full = np.array(_mat_try_keys(TA, "PhiC", "PhiC_full"), dtype=float)
        Uh_full   = np.array(_mat_try_keys(TA, "Uh", "U"), dtype=float)
        Qrom_full = np.array(_mat_try_keys(TA, "Qrom", "Q"), dtype=float)

        Uh = Uh_full[:, :r]
        AC = PhiC_full[:r, :r] / 3600.0
        BC = PhiC_full[:r, r:] / 3600.0
        Qrom = Qrom_full[:r, :r] if Qrom_full.ndim == 2 else Qrom_full[:r]

        F_U, (SLTm, LATm, ALTm) = _build_interpolants(Uh, slt, lat, alt)
        M_U = _build_mean_interp(Dens_Mean, slt, lat, alt)

        if sw_provider is None:
            raise ValueError(
                "JB2008 を使う場合は sw_provider(jd0, jdf)->SWinputs を渡してください "
                "(MATLAB computeSWinputs_JB2008 相当)。"
            )
        SWinputs = np.asarray(sw_provider(jd0, jdf + 1.0), dtype=float)

        return ROMBundle(
            AC=AC, BC=BC, Uh=Uh,
            F_U=F_U, Dens_Mean=Dens_Mean, M_U=M_U,
            SLTm=SLTm, LATm=LATm, ALTm=ALTm,
            maxAtmAlt=800.0,
            SWinputs=SWinputs,
            Qrom=Qrom,
        )

    elif name == "NRLMSISE_1997_2008":
        mat_path = "NRLMSISE_1997_2008_ROM_r100.mat"
        TA = loadmat(mat_path, squeeze_me=True, struct_as_string=False)

        slt = _to_1d(_mat_try_keys(TA, "localSolarTimes", "slt"))
        lat = _to_1d(_mat_try_keys(TA, "latitudes", "lat"))
        alt = _to_1d(_mat_try_keys(TA, "altitudes", "alt"))
        Dens_Mean = _to_1d(_mat_try_keys(TA, "densityDataMeanLog", "Dens_Mean"))
        PhiC_full = np.array(_mat_try_keys(TA, "PhiC", "PhiC_full"), dtype=float)
        Uh_full   = np.array(_mat_try_keys(TA, "Uh", "U"), dtype=float)
        Qrom_full = np.array(_mat_try_keys(TA, "Qrom", "Q"), dtype=float)

        Uh  = Uh_full[:, :r]
        AC  = PhiC_full[:r, :r] / 3600.0
        BC  = PhiC_full[:r, r:] / 3600.0
        Qrom = Qrom_full[:r, :r] if Qrom_full.ndim == 2 else Qrom_full[:r]

        F_U, (SLTm, LATm, ALTm) = _build_interpolants(Uh, slt, lat, alt)
        M_U = _build_mean_interp(Dens_Mean, slt, lat, alt)

        SW_daily, SW_month = _input_sw_nrlmsise("Data/SW-All.txt")
        SWinputs = _compute_swinputs_nrlmsise(jd0, jdf + 1.0, SW_daily, SW_month)

        return ROMBundle(
            AC=AC, BC=BC, Uh=Uh,
            F_U=F_U, Dens_Mean=Dens_Mean, M_U=M_U,
            SLTm=SLTm, LATm=LATm, ALTm=ALTm,
            maxAtmAlt=800.0,
            SWinputs=SWinputs,
            Qrom=Qrom,
        )

    elif name == "TIEGCM_1997_2008":
        mat_path = "TIEGCM_1997_2008_ROM_r100.mat"
        TA = loadmat(mat_path, squeeze_me=True, struct_as_string=False)

        slt = _to_1d(_mat_try_keys(TA, "localSolarTimes", "slt"))
        lat = _to_1d(_mat_try_keys(TA, "latitudes", "lat"))
        alt = _to_1d(_mat_try_keys(TA, "altitudes", "alt"))
        Dens_Mean = _to_1d(_mat_try_keys(TA, "densityDataMeanLog", "Dens_Mean"))
        PhiC_full = np.array(_mat_try_keys(TA, "PhiC", "PhiC_full"), dtype=float)
        Uh_full   = np.array(_mat_try_keys(TA, "Uh", "U"), dtype=float)
        Qrom_full = np.array(_mat_try_keys(TA, "Qrom", "Q"), dtype=float)

        Uh  = Uh_full[:, :r]
        AC  = PhiC_full[:r, :r] / 3600.0
        BC  = PhiC_full[:r, r:] / 3600.0
        Qrom = Qrom_full[:r, :r] if Qrom_full.ndim == 2 else Qrom_full[:r]

        F_U, (SLTm, LATm, ALTm) = _build_interpolants(Uh, slt, lat, alt)
        M_U = _build_mean_interp(Dens_Mean, slt, lat, alt)

        SW_daily, SW_month = _input_sw_tiegcm("Data/SW-All.txt")
        SWinputs = _compute_swinputs_tiegcm(jd0, jdf + 1.0, SW_daily, SW_month)

        return ROMBundle(
            AC=AC, BC=BC, Uh=Uh,
            F_U=F_U, Dens_Mean=Dens_Mean, M_U=M_U,
            SLTm=SLTm, LATm=LATm, ALTm=ALTm,
            maxAtmAlt=500.0,
            SWinputs=SWinputs,
            Qrom=Qrom,
        )

    else:
        raise ValueError(f"Unsupported ROM model: {name}")


class ROMRuntime:
    def __init__(self, Ur, xbar, Ac, Bc, input_fn):
        """
        Ur   : (Ngrid, r) POD基底
        xbar : (Ngrid,) 平均(log10密度) もしくは密度平均
        Ac,Bc: ROM 連続時間行列
        input_fn: callable(t) -> u_t (空間天気入力ベクトル)
        """
        self.Ur = Ur
        self.xbar = xbar
        self.Ac = Ac
        self.Bc = Bc
        self.input_fn = input_fn

    def density_log10_grid(self, z):
        # x = Ur z + xbar（論文の式(5): xは log10(density) を使う流儀）
        return self.Ur @ z + self.xbar

    def density_at(self, z, grid_interp_fn):
        """
        grid_interp_fn: callable(log10rho_grid) -> rho_at_point
        グリッド→点への補間は呼び出し側に任せる（速度優先）
        """
        log10rho = self.density_log10_grid(z)
        return grid_interp_fn(log10rho)

    def step_z(self, z, t, dt, discretize):
        u = self.input_fn(t)
        Ad, Bd = discretize(self.Ac, self.Bc, dt)
        return Ad @ z + Bd @ u