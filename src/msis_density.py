#MISIによる密度計算
from __future__ import annotations

import numpy as np
import pandas as pd
from pymsis import msis


def add_msis_density(df: pd.DataFrame, f107: float, f107a: float, ap7: list | tuple) -> pd.DataFrame:
    """DataFrame( t_utc, lat_deg, lon_deg, alt_km )に MSIS密度[kg/m^3] を付与。
    ap7: [ap_daily, ap[0:5]] 計7要素の想定（pymsis v3 仕様）。
    """
    # pymsisは入力を配列で与える（形状合わせに注意）
    times = np.array(df["t_utc"].to_numpy())
    lats = df["lat_deg"].to_numpy()
    lons = df["lon_deg"].to_numpy()
    alts_km = df["alt_km"].to_numpy()

    # 形状：(n_times,)
    f107_arr = np.full_like(lats, fill_value=f107, dtype=float)
    f107a_arr = np.full_like(lats, fill_value=f107a, dtype=float)

    # apは (n_times, 7)
    ap_arr = np.tile(np.array(ap7, dtype=float), (lats.size, 1))

    # 期待形状: (ntime, nlat, nlon, nalt) だが、1次元列をそのまま投げるショートカットAPIあり
    # msis.run は柔軟なブロードキャストをする
    out = msis.run(times, lats, lons, alts_km, f107=f107_arr, f107a=f107a_arr, ap=ap_arr)
    # 出力辞書 out["RHO"] が密度 [kg/m^3]
    rho = out["RHO"].reshape(-1)

    df2 = df.copy()
    df2["rho_kg_m3"] = rho
    return df2