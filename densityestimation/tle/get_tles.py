# Origin: getTLEs.m (Gondelach 2020; based on Aleksander Lidtke 2015)
# License: GNU GPL v3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from sgp4.api import Satrec


@dataclass
class TLEObject:
    noradID: int
    satrecs: List[Satrec]


def _read_tle_lines(path: Path) -> Iterator[Tuple[str, str]]:
    """
    2行1組のTLEを順に返す。空行やヘッダ行はスキップ。
    """
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f.readlines()]

    i = 0
    n = len(lines)
    while i < n:
        # 1行目を探す
        while i < n and not (len(lines[i]) > 0 and lines[i][0] == "1"):
            i += 1
        if i >= n:
            break
        line1 = lines[i]
        i += 1
        # 2行目
        if i < n and len(lines[i]) > 0 and lines[i][0] == "2":
            line2 = lines[i]
            i += 1
            yield (line1, line2)
        else:
            # ペアが壊れている場合はスキップして継続
            continue


def _satrec_from_tle(line1: str, line2: str) -> Optional[Satrec]:
    """
    TLE 2行から Satrec を作成し、MATLAB互換の jdsatepoch（小数を含む）に整形。
    """
    try:
        sat = Satrec.twoline2rv(line1, line2)
    except Exception:
        return None

    # Python版sgp4は整数部/小数部が分かれていることがあるので合算して上書き
    try:
        jd_full = float(getattr(sat, "jdsatepoch", 0.0)) + float(getattr(sat, "jdsatepochF", 0.0))
        setattr(sat, "jdsatepoch", jd_full)  # MATLAB実装と同じフィールド名を使う
    except Exception:
        pass

    return sat


def get_tles(filename: str) -> List[TLEObject]:
    """
    Read TLE data from file and group/sort by NORAD ID.

    Parameters
    ----------
    filename : str
        パス（例: 'TLEdata/estimationObjects.tle'）

    Returns
    -------
    List[TLEObject]
        各要素は .noradID と .satrecs (エポック昇順) を持つ。
    """
    p = Path(filename)
    if not p.exists():
        raise FileNotFoundError(f"TLE file not found: {p}")

    # まず全TLEを読み込んで satrecs / エポック / NORAD をベタ配列で保持
    norad_ids: List[int] = []
    epochs_jd: List[float] = []
    satrecs: List[Satrec] = []

    for l1, l2 in _read_tle_lines(p):
        sat = _satrec_from_tle(l1, l2)
        if sat is None:
            continue
        norad_ids.append(int(getattr(sat, "satnum")))
        epochs_jd.append(float(getattr(sat, "jdsatepoch")))
        satrecs.append(sat)

    if not satrecs:
        return []

    # NORADごとにグルーピング
    unique_ids = sorted(set(norad_ids))
    objects: List[TLEObject] = []
    for nid in unique_ids:
        idxs = [k for k, x in enumerate(norad_ids) if x == nid]

        # その物体のTLEをエポック昇順に並べ替え（MATLAB版の issorted + sort 相当）
        idxs_sorted = sorted(idxs, key=lambda k: epochs_jd[k])
        sat_sorted = [satrecs[k] for k in idxs_sorted]

        objects.append(TLEObject(noradID=nid, satrecs=sat_sorted))

    return objects
