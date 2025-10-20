# License: GNU GPL v3
from __future__ import annotations

import numpy as np


def _to_float(s: str) -> float:
    s = s.strip()
    return float(s) if s else np.nan

def input_sw_tiegcm(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse CelesTrakの SW-All.txt を TIE-GCM 用の行列へ。
    Returns
    -------
    SWmatDaily: (N_daily + N_pred, 11) ndarray
        cols: [F10.7Daily, F10.7Avg, -, Kp8(3h bins 8 cols)]  ※3列目は未使用(MATLAB同様)
    SWmatMonthlyPred: (N_monthly_pred, 2) ndarray
        cols: [F10.7Daily, F10.7Avg]
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # ---- スキップ: 先頭17行
    idx = 17

    # 観測データ数
    n_daily_obs = int(lines[idx][20:25].strip()); idx += 1  # "BEGIN OBSERVED"の直前行
    idx += 1  # "BEGIN OBSERVED"

    SWaux = np.zeros((n_daily_obs, 11), dtype=float)

    for i in range(n_daily_obs):
        s = lines[idx]; idx += 1
        # F10.7 Daily / Avg （MATLABの 94:98 / 102:106）
        SWaux[i, 0] = _to_float(s[93:98])
        SWaux[i, 1] = _to_float(s[101:106])
        # 8つの3h Kp（各3桁。MATLABは 19:21,22:24,...,40:42）
        kp_slices = [(18,21),(21,24),(24,27),(27,30),(30,33),(33,36),(36,39),(39,42)]
        kp_vals = [ _to_float(s[a:b]) / 10.0 for (a,b) in kp_slices ]
        SWaux[i, 3:11] = kp_vals
        if SWaux[i, 0] == 0.0:
            SWaux[i, 0] = SWaux[i, 1]

    # 空行等を3行スキップ
    idx += 3
    pdt_pnt = int(lines[idx][27:29].strip()); idx += 1
    SWmatDaily = np.zeros((n_daily_obs + pdt_pnt, 11), dtype=float)
    SWmatDaily[:n_daily_obs, :] = SWaux
    idx += 1  # "BEGIN DAILY_PREDICTED"

    # 予測 日次
    for i in range(n_daily_obs, n_daily_obs + pdt_pnt):
        s = lines[idx]; idx += 1
        SWmatDaily[i, 0] = _to_float(s[93:98])   # F10.7 Daily
        SWmatDaily[i, 1] = _to_float(s[101:106]) # F10.7 Avg
        kp_slices = [(18,21),(21,24),(24,27),(27,30),(30,33),(33,36),(36,39),(39,42)]
        kp_vals = [ _to_float(s[a:b]) / 10.0 for (a,b) in kp_slices ]
        SWmatDaily[i, 3:11] = kp_vals

    # 空行等を3行スキップ
    idx += 3
    mpd_pnt = int(lines[idx][29:31].strip()); idx += 1
    SWmatMonthlyPred = np.zeros((mpd_pnt, 2), dtype=float)
    idx += 1  # "BEGIN MONTHLY_PREDICTED"

    for i in range(mpd_pnt):
        s = lines[idx]; idx += 1
        SWmatMonthlyPred[i, 0] = _to_float(s[93:98])   # F10.7 Daily
        SWmatMonthlyPred[i, 1] = _to_float(s[101:106]) # F10.7 Avg

    return SWmatDaily, SWmatMonthlyPred
