# License: GNU GPL v3
from __future__ import annotations

import numpy as np


def _to_float(s: str) -> float:
    s = s.strip()
    return float(s) if s else np.nan

def input_sw_nrlmsise(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    CelesTrakの SW-All.txt を NRLMSISE-00 用にパース。
    Returns
    -------
    SWmatDaily: (N_daily + N_pred, 11) ndarray
        cols:
          0: F10.7 Daily
          1: F10.7 Average
          2: Daily magnetic index (Ap-daily)
          3..10: 3h Ap(×8)  ※各行の 8本の3時間Ap
    SWmatMonthlyPred: (N_monthly_pred, 2) ndarray
        cols: [F10.7 Daily, F10.7 Average]
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    idx = 17  # ヘッダ17行スキップ

    # 観測データ数
    n_daily_obs = int(lines[idx][20:25].strip()); idx += 1
    idx += 1  # "BEGIN OBSERVED"

    SWaux = np.zeros((n_daily_obs, 11), dtype=float)
    ap_slices = [(46,50),(50,54),(54,58),(58,62),(62,66),(66,70),(70,74),(74,78)]

    for i in range(n_daily_obs):
        s = lines[idx]; idx += 1
        SWaux[i, 0] = _to_float(s[93:98])   # F10.7 Daily
        SWaux[i, 1] = _to_float(s[101:106]) # F10.7 Avg
        SWaux[i, 2] = _to_float(s[79:82])   # Daily magnetic index
        SWaux[i, 3:11] = [ _to_float(s[a:b]) for (a,b) in ap_slices ]
        if SWaux[i, 0] == 0.0:
            SWaux[i, 0] = SWaux[i, 1]

    idx += 3
    pdt_pnt = int(lines[idx][27:29].strip()); idx += 1

    SWmatDaily = np.zeros((n_daily_obs + pdt_pnt, 11), dtype=float)
    SWmatDaily[:n_daily_obs, :] = SWaux
    idx += 1  # "BEGIN DAILY_PREDICTED"

    for i in range(n_daily_obs, n_daily_obs + pdt_pnt):
        s = lines[idx]; idx += 1
        SWmatDaily[i, 0] = _to_float(s[93:98])
        SWmatDaily[i, 1] = _to_float(s[101:106])
        SWmatDaily[i, 2] = _to_float(s[79:82])
        SWmatDaily[i, 3:11] = [ _to_float(s[a:b]) for (a,b) in ap_slices ]

    idx += 3
    mpd_pnt = int(lines[idx][29:31].strip()); idx += 1
    SWmatMonthlyPred = np.zeros((mpd_pnt, 2), dtype=float)
    idx += 1  # "BEGIN MONTHLY_PREDICTED"

    for i in range(mpd_pnt):
        s = lines[idx]; idx += 1
        SWmatMonthlyPred[i, 0] = _to_float(s[93:98])   # F10.7 Daily
        SWmatMonthlyPred[i, 1] = _to_float(s[101:106]) # F10.7 Avg

    return SWmatDaily, SWmatMonthlyPred
