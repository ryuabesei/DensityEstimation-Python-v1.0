# Origin: getTLEsForEstimation.m (Gondelach 2020)
# License: GNU GPL v3
from __future__ import annotations

from pathlib import Path
from typing import List

from sgp4.api import jday  # Julian Date helper from sgp4

from densityestimation.tle.get_tles import TLEObject, get_tles


def _jd_utc_0h(year: int, month: int, day: int) -> float:
    """Julian Date at 00:00:00 UTC for given date (float JD)."""
    jd, fr = jday(year, month, day, 0, 0, 0.0)
    return jd + fr


def get_tles_for_estimation(
    start_year: int,
    start_month: int,
    start_day: int,
    end_year: int,
    end_month: int,
    end_day: int,
    selected_objects: List[int],
    get_tles_from_single_file: bool,
    relative_dir: str = "TLEdata",
) -> List[TLEObject]:
    """
    Read TLEs from files and filter by date window per NORAD ID.

    Parameters
    ----------
    start_*, end_* : int
        フィルタ範囲の日付（UTC, 0時起点）。
    selected_objects : list[int]
        NORAD カタログIDの配列。
    get_tles_from_single_file : bool
        True: 'TLEdata/estimationObjects.tle' から一括読込
        False: 'TLEdata/NNNNN.tle' を各IDごとに読込
    relative_dir : str
        読み込みディレクトリ（既定 'TLEdata'）

    Returns
    -------
    List[TLEObject]
        各要素は .noradID と .satrecs（エポック昇順＆期間内にフィルタ済）、
        可能なら .name と .tle_lines（元の2行文字列タプル）を持つ。
    """
    selected_objects = sorted(selected_objects)
    base = Path(relative_dir)

    # 1) 読み込み
    if get_tles_from_single_file:
        filename = base / "estimationObjects.tle"
        if not filename.exists():
            raise FileNotFoundError(
                f"TLE file estimationObjects.tle was not found in {relative_dir}."
            )
        objects = get_tles(str(filename))
    else:
        objects: List[TLEObject] = []
        for nid in selected_objects:
            p = base / f"{nid:05d}.tle"
            if not p.exists():
                raise FileNotFoundError(
                    f"TLE file {nid:05d}.tle was not found in {relative_dir}."
                )
            objs = get_tles(str(p))  # 通常1要素（そのNORADだけ）が返る
            objects.extend(objs)

    # 2) 日付でフィルタ（両端含む）
    jd_start = _jd_utc_0h(start_year, start_month, start_day)
    jd_end   = _jd_utc_0h(end_year, end_month, end_day)

    # objects は複数NORAD混在のリスト。選択IDごとに抽出してフィルタして上書き。
    out: List[TLEObject] = objects[:]  # 参照を壊さないようにコピー
    for nid in selected_objects:
        # 該当オブジェクトを探す
        idx = next((i for i, o in enumerate(out) if o.noradID == nid), None)
        if idx is None:
            raise ValueError(f"No TLEs found for object {nid:d}.")

        obj = out[idx]
        epochs = [float(getattr(s, "jdsatepoch")) for s in obj.satrecs]

        # 最初に jd_start 以上となるTLEのインデックス
        first = next((k for k, e in enumerate(epochs) if e >= jd_start), None)
        # 最後に jd_end 以下となるTLEのインデックス
        last = None
        for k in range(len(epochs) - 1, -1, -1):
            if epochs[k] <= jd_end:
                last = k
                break

        if first is None or last is None or first > last:
            raise ValueError(
                f"No TLEs found for object {nid:d} between "
                f"{start_day:02d}-{start_month:02d}-{start_year:04d} and "
                f"{end_day:02d}-{end_month:02d}-{end_year:04d}."
            )

        # name / tle_lines を可能なら引き継ぐ（未定義なら渡さない）
        kwargs = dict(noradID=nid, satrecs=obj.satrecs[first : last + 1])
        if hasattr(obj, "name"):
            kwargs["name"] = getattr(obj, "name")
        if hasattr(obj, "tle_lines"):
            kwargs["tle_lines"] = getattr(obj, "tle_lines")

        out[idx] = TLEObject(**kwargs)

    return out

