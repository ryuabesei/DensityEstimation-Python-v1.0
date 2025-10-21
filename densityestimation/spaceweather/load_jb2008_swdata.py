# License: GNU GPL v3
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_jb2008_swdata(
    *,
    eop_path: str | Path = "Data/EOP-All.txt",
    solfsmy_path: str | Path = "Data/SOLFSMY.txt",
    dtcfile_path: str | Path = "Data/DTCFILE.txt",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MATLAB loadJB2008SWdata.m のポート。
    Returns
    -------
    eopdata : ndarray
        Earth orientation parameters（CelesTrak拡張版）。読み出しは別のローダに任せるならその戻りを渡す。
        ここでは行列そのものを返す（下流の computeJB2000SWinputs に渡す想定）。
    SOLdata : ndarray (転置済み)
        SOLFSMY.txt（ソーラーインデックス）の内容（MATLAB は readSOLFSMY(... )' として転置している）
    DTCdata : ndarray (転置済み)
        DTCFILE.txt（ジオマグストーム DTC 値）の内容（MATLAB は readDTCFILE(... )' として転置）
    """
    eop_path = Path(eop_path)
    solfsmy_path = Path(solfsmy_path)
    dtcfile_path = Path(dtcfile_path)

    # EOP: 既に別モジュールで読み込み済みならその関数を使ってもOK
    # ここではシンプルに数値表だけ読み込む（区切りは空白/タブ前提）
    # 必要なら densityestimation.data.eop_loader.load_eop_celestrak を呼び出しても良い。
    try:
        # 数値行以外をスキップするため、無効値を 'nan' で埋める読み込み
        eopdata = np.genfromtxt(eop_path, dtype=float)
    except Exception as e:
        raise RuntimeError(f"Failed to read EOP data: {eop_path}") from e

    try:
        # SOLFSMY は MATLAB 実装で転置しているので合わせる
        sol_raw = np.genfromtxt(solfsmy_path, dtype=float)
        # 行方向が指標、列方向が時系列になるよう転置
        SOLdata = sol_raw.T
    except Exception as e:
        raise RuntimeError(f"Failed to read SOLFSMY: {solfsmy_path}") from e

    try:
        dtc_raw = np.genfromtxt(dtcfile_path, dtype=float)
        DTCdata = dtc_raw.T
    except Exception as e:
        raise RuntimeError(f"Failed to read DTCFILE: {dtcfile_path}") from e

    return eopdata, SOLdata, DTCdata
