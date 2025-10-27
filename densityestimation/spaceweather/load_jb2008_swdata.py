# densityestimation/spaceweather/load_jb2008_swdata.py
# License: GNU GPL v3
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from densityestimation.data.eop_loader import load_eop_celestrak


def _pkg_root() -> Path:
    # .../densityestimation/spaceweather/load_jb2008_swdata.py から densityestimation/ を指す
    return Path(__file__).resolve().parents[1]


def _resolve(path: str | Path) -> Path:
    """
    与えられたパスが存在しなければ、パッケージルート（densityestimation/）を基準に解決を試みる。
    """
    p = Path(path)
    if p.exists():
        return p
    alt = _pkg_root() / p
    return alt if alt.exists() else p  # 最終的な存在チェックは読み込み時に例外へ

def _load_dtcfile(path: Path) -> np.ndarray:
    """
    DTCFILE.txt 専用ローダ（堅牢版）
    対応フォーマット:
      A) [DTC] YYYY DOY v1 ... v24        -> 26 数値トークン
      B) [DTC] YYYY MM  DD v1 ... v24     -> 27 数値トークン (MM/DD は DOY に変換)
    返り: shape (N, 26) = [year, DOY, h0..h23]
    """
    import re
    from datetime import date

    rows: list[list[float]] = []
    num_pat = re.compile(r"[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+\-]?\d+)?")

    def plausible_year(y: int) -> bool:
        return 1900 <= y <= 2100

    def to_row_year_doy(values: list[float]) -> list[float] | None:
        """
        values が
          - 26 要素: [Y, DOY, 24h] とみなす
          - 27 要素: [Y, M, D, 24h] を DOY に変換
        のどちらかなら [Y, DOY, 24h] を返す。違えば None。
        """
        if len(values) == 26:
            y = int(round(values[0])); doy = int(round(values[1]))
            if not plausible_year(y) or not (1 <= doy <= 366):
                return None
            return [float(y), float(doy)] + [float(v) for v in values[2:26]]

        if len(values) >= 27:
            y = int(round(values[0])); m = int(round(values[1])); d = int(round(values[2]))
            if not (plausible_year(y) and 1 <= m <= 12 and 1 <= d <= 31):
                return None
            try:
                doy = (date(y, m, d) - date(y, 1, 1)).days + 1
            except Exception:
                return None
            return [float(y), float(doy)] + [float(v) for v in values[3:27]]

        return None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.replace("\ufeff", "").strip()
            if not s:
                continue
            s = s.replace(",", " ")
            # Fortran指数 D/d → E
            s = re.sub(r"([0-9])([dD])([+\-]?\d+)", r"\1E\3", s)

            nums = [float(t) for t in num_pat.findall(s)]
            if not nums:
                continue

            # まずは 26/27 ぴったりを優先で判定
            candidates: list[list[float]] = []
            if len(nums) in (26, 27):
                row = to_row_year_doy(nums[:len(nums)])
                if row is not None:
                    candidates.append(row)

            # 行に余計な数が混じる場合に備えて、スライド探索
            if not candidates and len(nums) > 27:
                for i in range(0, len(nums) - 26 + 1):
                    seg = nums[i:i+26]
                    row = to_row_year_doy(seg)
                    if row is not None:
                        candidates.append(row)
                        break
                if not candidates:
                    for i in range(0, len(nums) - 27 + 1):
                        seg = nums[i:i+27]
                        row = to_row_year_doy(seg)
                        if row is not None:
                            candidates.append(row)
                            break

            if candidates:
                rows.append(candidates[0])

    if not rows:
        raise ValueError(f"No DTC rows parsed from {path}")

    return np.asarray(rows, dtype=float)


def _load_solfsm(path: Path) -> np.ndarray:
    """
    SOLFSMY.txt 専用ローダ
    典型: "SOL YYYY MM DD val1 val2 ..." あるいは任意テキスト + 数値列
    返り値: shape (N, M)  （列数 M はファイルに依存）
    """
    import re

    rows: list[list[float]] = []
    max_cols = 0

    # ±, 小数, 指数 (E/e, Fortran D/d) を許容
    num_pat = re.compile(r"""
        (?:
            [+\-]? (?:\d+(?:\.\d*)?|\.\d+) (?:[eEdD][+\-]?\d+)?   # 12, 12.3, .5, 1.2E+03, 1.2D+03
            |
            (?i:NaN|Inf|-Inf)
        )
    """, re.VERBOSE)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            # Fortran指数 D/d → E に正規化（抽出前にやってもOK）
            s = re.sub(r"([0-9])([dD])([+\-]?\d+)", r"\1E\3", s)

            # 数値トークンだけ抽出（行頭の "SOL" などは無視される）
            toks = num_pat.findall(s)
            if not toks:
                continue

            row: list[float] = []
            for t in toks:
                tl = t.lower()
                if tl in ("nan", "+nan", "-nan"):
                    row.append(float("nan"))
                elif tl in ("inf", "+inf"):
                    row.append(float("inf"))
                elif tl in ("-inf",):
                    row.append(float("-inf"))
                else:
                    try:
                        row.append(float(t))
                    except Exception:
                        # 念のため
                        row.append(float("nan"))

            if row:
                rows.append(row)
                max_cols = max(max_cols, len(row))

    if not rows:
        raise ValueError(f"No SOLFSMY rows parsed from {path}")

    # 列数を右側 NaN で揃える
    for i, r in enumerate(rows):
        if len(r) < max_cols:
            rows[i] = r + [float("nan")] * (max_cols - len(r))

    return np.asarray(rows, dtype=float)




def load_jb2008_swdata(
    *,
    eop_path: str | Path = "densityestimation/data/EOP-All.txt",
    solfsmy_path: str | Path = "densityestimation/data/SOLFSMY.txt",
    dtcfile_path: str | Path = "densityestimation/data/DTCFILE.txt",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MATLAB loadJB2008SWdata.m のポート。

    Returns
    -------
    eopdata : ndarray, shape (N, 13)
        CelesTrak 拡張 EOP（Year, Month, Day, MJD, x, y, UT1-UTC, LOD, dPsi, dEps, dX, dY, TAI-UTC）
        ※ 単位は元ファイル準拠（角成分は arcsec、UT1/LOD/TAI-UTC は秒）。
        （Nx6 が欲しければ呼び出し側で load_eop_celestrak(full=False) を利用可）
    SOLdata : ndarray (転置済み)
        SOLFSMY.txt の内容を転置（MATLAB の readSOLFSMY(...)' に合わせる）
    DTCdata : ndarray (転置済み)
        DTCFILE.txt の内容を転置（MATLAB の readDTCFILE(...)' に合わせる）
    """
    eop_path = _resolve(eop_path)
    solfsmy_path = _resolve(solfsmy_path)
    dtcfile_path = _resolve(dtcfile_path)

    # --- EOP ---
    try:
        # CelesTrak 形式に特化した堅牢ローダ（ヘッダや罫線混入に強い）
        eopdata = load_eop_celestrak(str(eop_path), full=True)  # (N, 13)
    except Exception as e:
        raise RuntimeError(f"Failed to read EOP data: {eop_path}") from e

    # --- SOLFSMY ---
    try:
        sol_raw = _load_solfsm(solfsmy_path)
        SOLdata = sol_raw.T  # MATLAB 実装に合わせて転置
    except Exception as e:
        raise RuntimeError(f"Failed to read SOLFSMY: {solfsmy_path}") from e

    # --- DTCFILE ---
    try:
        dtc_raw = _load_dtcfile(dtcfile_path)
        DTCdata = dtc_raw.T  # MATLAB 実装に合わせて転置
    except Exception as e:
        raise RuntimeError(f"Failed to read DTCFILE: {dtcfile_path}") from e

    return eopdata, SOLdata, DTCdata
