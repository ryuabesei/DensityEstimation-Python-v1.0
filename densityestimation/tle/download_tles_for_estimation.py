# densityestimation/tle/download_tles_for_estimation.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import requests


class SpaceTrackError(RuntimeError):
    pass


def _build_spacetrack_tle_query(
    start_year: int, start_month: int, start_day: int,
    end_year: int, end_month: int, end_day: int,
    max_apogee_km: float,
    selected_objects: Optional[List[int]] = None
) -> str:
    """
    Space-Track TLE query URL を構築。
    例:
      https://www.space-track.org/basicspacedata/query/class/tle/
        EPOCH/2002-08-01--2002-08-10/
        APOGEE/%3C500/
        NORAD_CAT_ID/27391,25544/
        orderby/NORAD_CAT_ID%20asc/format/tle
    """
    base = "https://www.space-track.org/basicspacedata/query/class/tle/"
    epoch = (
        f"EPOCH/{start_year:04d}-{start_month:02d}-{start_day:02d}"
        f"--{end_year:04d}-{end_month:02d}-{end_day:02d}/"
    )
    # '<' は %3C にエンコードしておく（Space-Track側で許容される）
    apogee = f"APOGEE/%3C{int(max_apogee_km)}/"

    if selected_objects:
        # Space-Trackは整数のカンマ区切りでOK（ゼロ埋め不要）
        obj_list = ",".join(str(int(s)) for s in selected_objects)
        objects = f"NORAD_CAT_ID/{obj_list}/"
    else:
        objects = ""

    ordering = "orderby/NORAD_CAT_ID%20asc/format/tle"
    return base + epoch + apogee + objects + ordering


def _login_spacetrack(session: requests.Session, username: str, password: str) -> None:
    login_url = "https://www.space-track.org/ajaxauth/login"
    r = session.post(login_url, data={"identity": username, "password": password}, timeout=30)
    if r.status_code != 200:
        raise SpaceTrackError(f"Login failed (HTTP {r.status_code})")
    txt = (r.text or "").lower()
    if "unauthorized" in txt or "invalid" in txt:
        raise SpaceTrackError("Login failed: unauthorized/invalid credentials.")


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _parse_tle_lines(lines: List[str]) -> List[Dict[str, str]]:
    """
    Space-Track format=tle の素朴パーサ。
    「1 」「2 」で始まる行をペアにする（0 NAME はスキップ）。
    """
    out: List[Dict[str, str]] = []
    i, n = 0, len(lines)
    while i < n:
        l = lines[i].rstrip("\r\n")
        if l.startswith("1 "):
            line1 = l
            j = i + 1
            # 直後に来るはずの Line2 を探す（防御的にスキャン）
            while j < n and not lines[j].startswith("2 "):
                j += 1
            if j >= n:
                break
            line2 = lines[j].rstrip("\r\n")
            out.append({
                "line1": line1,
                "line2": line2,
                "norad_cat_id": line1[2:7].strip()
            })
            i = j + 1
        else:
            i += 1
    return out


def get_tles_from_file(filename: str | Path) -> List[Dict[str, str]]:
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        return _parse_tle_lines(f.readlines())


def download_tles_for_estimation(
    username: str,
    password: str,
    start_year: int, start_month: int, start_day: int,
    end_year: int, end_month: int, end_day: int,
    max_apogee_km: float,
    selected_objects: Optional[List[int]] = None,
    out_file: str | Path = "TLEdata/estimationObjects.tle",
) -> List[Dict[str, str]]:
    """
    Space-TrackからTLEをダウンロードして out_file に保存。
    返り値は保存ファイルを簡易パースした (line1,line2, norad_cat_id) の配列。
    ※ 後続の処理は get_tles_for_estimation() がファイルを読む前提なので、
       この戻り値そのものは使わなくてもOK。
    """
    url = _build_spacetrack_tle_query(
        start_year, start_month, start_day,
        end_year, end_month, end_day,
        max_apogee_km, selected_objects
    )

    out_path = Path(out_file)
    _ensure_parent_dir(out_path)

    with requests.Session() as s:
        s.headers.update({"User-Agent": "DensityEstimation-Python/1.0"})
        _login_spacetrack(s, username, password)

        r = s.get(url, timeout=120)
        if r.status_code != 200:
            raise SpaceTrackError(f"Query failed (HTTP {r.status_code})")

        text = r.text or ""
        # Space-Trackはエラーでもtextを返す場合あり → TLE先頭の「1 」が無ければ失敗とみなす
        if "error" in text.lower() and "tle" not in text.lower():
            raise SpaceTrackError(f"Query error from Space-Track: {text[:200]}...")

        if "No results returned" in text:
            raise SpaceTrackError("No results returned (check date range / filters / NORAD list).")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text if text.endswith("\n") else text + "\n")

    return get_tles_from_file(out_path)
