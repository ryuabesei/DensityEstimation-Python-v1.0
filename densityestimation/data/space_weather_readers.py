# densityestimation/data/space_weather_readers.py
# 宇宙天気入力ファイル（DTC、SOLFSMY など）を、ヘッダやフォーマットの差異に強い方法で NumPy 配列に読み込む

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional, Union

import numpy as np

PathLike = Union[str, Path]


_FLOAT_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?")

def _iter_lines(path: PathLike) -> Iterable[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            yield line.rstrip("\n\r")


def _parse_floats_from_line(line: str) -> List[float]:
    """行から数値トークン（浮動小数）だけを抽出して float へ。見つからなければ []。"""
    return [float(tok) for tok in _FLOAT_RE.findall(line)]


# -----------------------------------------------------------------------------
# MATLAB: readDTCFILE.m
#   先頭3文字をスキップし、YEAR, DOY, DTC1..DTC24 の計 26 列を返す
#   デフォルトは startRow=1, endRow=inf
# -----------------------------------------------------------------------------
def read_dtcfile(
    filename: PathLike,
    start_row: int = 1,
    end_row: Optional[int] = None,
) -> np.ndarray:
    """
    Read DTCFILE text as matrix.

    Parameters
    ----------
    filename : str | Path
    start_row : 1-based header行数。ここから読み始める（デフォルト 1）
    end_row : 最終行（1-based, inclusive）。None ならファイル末尾まで

    Returns
    -------
    np.ndarray  shape = (N, 26)
      columns = [YEAR, DOY, DTC1, ..., DTC24]
    """
    if start_row < 1:
        raise ValueError("start_row must be >= 1")

    rows: List[List[float]] = []
    ncols_expected = 26  # YEAR, DOY, DTC1..DTC24

    for idx0, line in enumerate(_iter_lines(filename), start=1):
        if idx0 < start_row:
            continue
        if end_row is not None and idx0 > end_row:
            break
        if not line.strip():
            continue

        # MATLAB 版は '%*3s' で先頭3文字を捨てているので一応合わせる
        line_to_parse = line[3:] if len(line) >= 3 else ""

        vals = _parse_floats_from_line(line_to_parse)

        # 期待列数に合わせる（不足は NaN、過剰は切り捨て）
        if len(vals) < ncols_expected:
            vals = vals + [np.nan] * (ncols_expected - len(vals))
        else:
            vals = vals[:ncols_expected]

        rows.append(vals)

    if not rows:
        return np.empty((0, ncols_expected), dtype=float)

    return np.asarray(rows, dtype=float)


# -----------------------------------------------------------------------------
# MATLAB: readSOLFSMY.m
#   デフォルト startRow=5, endRow=inf
#   columns = [YEAR, DOY, JulianDay, F10, F81c, S10, S81c, M10, M81c, Y10, Y81c]
# -----------------------------------------------------------------------------
def read_solf_smy(
    filename: PathLike,
    start_row: int = 5,
    end_row: Optional[int] = None,
) -> np.ndarray:
    """
    Read SOLFSMY text as matrix.

    Parameters
    ----------
    filename : str | Path
    start_row : 1-based 読み始め行（デフォルト 5）
    end_row : 1-based 最終行（inclusive）。None なら末尾まで

    Returns
    -------
    np.ndarray  shape = (N, 11)
      columns = [YEAR, DOY, JulianDay, F10, F81c, S10, S81c, M10, M81c, Y10, Y81c]
    """
    if start_row < 1:
        raise ValueError("start_row must be >= 1")

    rows: List[List[float]] = []
    ncols_expected = 11

    for idx0, line in enumerate(_iter_lines(filename), start=1):
        if idx0 < start_row:
            continue
        if end_row is not None and idx0 > end_row:
            break
        if not line.strip():
            continue

        vals = _parse_floats_from_line(line)

        if len(vals) < ncols_expected:
            vals = vals + [np.nan] * (ncols_expected - len(vals))
        else:
            vals = vals[:ncols_expected]

        rows.append(vals)

    if not rows:
        return np.empty((0, ncols_expected), dtype=float)

    return np.asarray(rows, dtype=float)
