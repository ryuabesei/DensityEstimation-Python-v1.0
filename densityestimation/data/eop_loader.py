# densityestimation/data/eop_loader.py
# GNU GPL v3

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


@dataclass
class EOPValues:
    xp: float      # [rad]
    yp: float      # [rad]
    dut1: float    # [s]
    lod: float     # [s]
    ddpsi: float   # [rad]
    ddeps: float   # [rad]
    dat: int       # [s] TAI-UTC


def _parse_numeric_rows(path: str) -> np.ndarray:
    """
    CelesTrak EOP-All.txt をヘッダーに依存せず読み込む。
    ・コメント/罫線/空行はスキップ
    ・数値トークンをできるだけ拾い、各行の先頭13要素を採用（不足は0で埋める）
    返り値: shape (N, 13)
      [Year, Month, Day, MJD, x, y, UT1-UTC, LOD, dPsi, dEps, dX, dY, TAI-UTC]
      x,y,dPsi,dEps,dX,dY は arcsec（元ファイル準拠）、UT1/LOD/TAI-UTC は秒
    """
    rows: list[list[float]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # 明らかなヘッダ/罫線/コメントはスキップ
            if s[0].isalpha() or s[0] in "#=-":
                continue

            # 連続する区切りをスペース化して分割
            parts = re.split(r"[,\s]+", s)
            nums: list[float] = []
            for p in parts:
                if p == "":
                    continue
                try:
                    # int/floatをまとめてfloatで受ける
                    nums.append(float(p))
                except Exception:
                    # 非数値トークンが混在ならその行は破棄
                    nums = []
                    break

            if not nums:
                continue

            # 13列に正規化（不足は0で埋め、超過は切り捨て）
            if len(nums) < 13:
                nums = nums + [0.0] * (13 - len(nums))
            else:
                nums = nums[:13]

            # Year, Month の非現実値行はゴミとして弾く（例: 壊れたヘッダー行）
            yr = int(nums[0]) if nums[0].is_integer() else int(nums[0])
            mo = int(nums[1]) if nums[1].is_integer() else int(nums[1])
            if not (1900 <= yr <= 2100 and 1 <= mo <= 12):
                continue

            rows.append(nums)

    if not rows:
        raise ValueError(f"EOP file has no numeric data rows: {path}")

    return np.asarray(rows, dtype=float)


def load_eop_celestrak(path: str, *, full: bool = False) -> np.ndarray:
    """
    CelesTrak の EOP-All.txt を読み込み。
    - full=False: MATLAB inputEOP_Celestrak と同等 (Nx6 行列, 単位: xp/yp/ddpsi/ddeps は [rad])
    - full=True : MATLAB inputEOP_Celestrak_Full と同等 (Nx13 行列, 単位は元ファイル準拠)
                  columns = [Year, Month, Day, MJD, x, y, UT1-UTC, LOD, dPsi, dEps, dX, dY, TAI-UTC]
    """
    data13 = _parse_numeric_rows(path)  # shape (N,13), 単位は元ファイル準拠

    if full:
        return data13

    # Nx6（xp, yp, dut1, lod, dPsi, dEps）に変換、角秒→rad
    rad_per_arcsec = np.pi / (180.0 * 3600.0)
    out = np.zeros((data13.shape[0], 6), dtype=float)
    # 列マッピング（EOP-All 標準）
    x_arcsec = data13[:, 4]
    y_arcsec = data13[:, 5]
    dut1 = data13[:, 6]
    lod = data13[:, 7]
    dpsi_arcsec = data13[:, 8]
    deps_arcsec = data13[:, 9]

    out[:, 0] = x_arcsec * rad_per_arcsec
    out[:, 1] = y_arcsec * rad_per_arcsec
    out[:, 2] = dut1
    out[:, 3] = lod
    out[:, 4] = dpsi_arcsec * rad_per_arcsec
    out[:, 5] = deps_arcsec * rad_per_arcsec
    return out


def compute_eop_celestrak(EOPMat: np.ndarray, jdate: float) -> EOPValues:
    """
    MATLAB computeEOP_Celestrak.m と同等の動作。
    EOPMat は load_eop_celestrak(full=False) の戻り値（shape: (N,6)）を想定。

    行の選択は MATLAB と同様：
      jdate0 = 2437665.5;  # 1962-01-01 00:00:00
      row = floor(jdate - jdate0) + 1
    範囲内なら該当日の行を返す（範囲外はゼロのまま）。

    Leap seconds (TAI-UTC) は MATLAB の DAT テーブルを移植。
    """
    # 初期化（範囲外は 0 のまま、dat は 26 で初期化 = 1992-07-01 時点の値）
    xp = yp = dut1 = lod = ddpsi = ddeps = 0.0
    dat = 26

    # 1962-01-01 00:00:00 の JD
    jdate0 = 2437665.5
    row = int(np.floor(jdate - jdate0) + 1)

    if 1 <= row <= EOPMat.shape[0]:
        xp, yp, dut1, lod, ddpsi, ddeps = EOPMat[row - 1, :]

    # MATLAB の DAT テーブル（3xN）
    DAT = np.array([
        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37],
        [2441317.5, 2441499.5, 2441683.5, 2442048.5, 2442413.5, 2442778.5, 2443144.5, 2443509.5, 2443874.5, 2444239.5, 2444786.5, 2445151.5, 2445516.5, 2446247.5, 2447161.5, 2447892.5, 2448257.5, 2448804.5, 2449169.5, 2449534.5, 2450083.5, 2450630.5, 2451179.5, 2453736.5, 2454832.5, 2456109.5, 2457204.5, 2457754.5],
        [2441499.5, 2441683.5, 2442048.5, 2442413.5, 2442778.5, 2443144.5, 2443509.5, 2443874.5, 2444239.5, 2444786.5, 2445151.5, 2445516.5, 2446247.5, 2447161.5, 2447892.5, 2448257.5, 2448804.5, 2449169.5, 2449534.5, 2450083.5, 2450630.5, 2451179.5, 2453736.5, 2454832.5, 2456109.5, 2457204.5, 2457754.5, np.inf],
    ], dtype=float)

    # jdate ∈ [start, end) を満たす i を探す
    idx = None
    for i in range(DAT.shape[1]):
        if (jdate < DAT[2, i]) and (jdate >= DAT[1, i]):
            idx = i
            break
    if idx is not None:
        dat = int(DAT[0, idx])

    return EOPValues(xp=xp, yp=yp, dut1=dut1, lod=lod, ddpsi=ddpsi, ddeps=ddeps, dat=dat)
