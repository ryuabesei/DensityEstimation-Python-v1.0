# Origin: downloadTLEsForEstimation.m (Gondelach 2020)
# License: GNU GPL v3
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import requests

from densityestimation.tle.get_tles import get_tles  # 後でMATLABのgetTLEs.mを移植予定
from densityestimation.tle.get_tles_for_estimation import get_tles_for_estimation


def download_tles_for_estimation(
    username: str,
    password: str,
    start_year: int,
    start_month: int,
    start_day: int,
    end_year: int,
    end_month: int,
    end_day: int,
    max_alt: float,
    selected_objects: Optional[List[int]] = None,
    out_dir: str = "TLEdata",
) -> list:
    """
    Download TLE data from Space-Track.org and parse them into objects.

    Parameters
    ----------
    username, password : str
        Space-Track.org account credentials.
    start_year, start_month, start_day, end_year, end_month, end_day : int
        Epoch range for the TLE query.
    max_alt : float
        Maximum apogee altitude [km] for filtering.
    selected_objects : list[int], optional
        NORAD catalog IDs to include (空なら全衛星)。
    out_dir : str
        保存ディレクトリ（デフォルト: 'TLEdata'）。

    Returns
    -------
    objects : list
        get_tles() により構築された衛星オブジェクトリスト。
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / "estimationObjects.tle"

    # Space-Track.org 認証セッションを確立
    login_url = "https://www.space-track.org/ajaxauth/login"
    query_base = "https://www.space-track.org/basicspacedata/query/class/tle"

    # 期間
    epoch_str = (
        f"EPOCH/{start_year}-{start_month:02d}-{start_day:02d}--"
        f"{end_year}-{end_month:02d}-{end_day:02d}/"
    )

    # 高度制限
    apogee_str = f"APOGEE/%3C{max_alt}/"

    # オブジェクトリスト
    if selected_objects:
        obj_str = "NORAD_CAT_ID/" + ",".join(f"{x:05d}" for x in selected_objects) + "/"
    else:
        obj_str = ""

    ordering = "orderby/NORAD_CAT_ID%20asc/format/tle"

    query_url = "/".join([query_base, epoch_str, apogee_str, obj_str, ordering])
    query_url = query_url.replace("//", "/").replace("https:/", "https://")

    # 認証POST
    session = requests.Session()
    login_payload = {"identity": username, "password": password}
    resp = session.post(login_url, data=login_payload)
    if resp.status_code != 200:
        raise ConnectionError(
            f"Space-Track login failed ({resp.status_code}): {resp.text[:200]}"
        )

    # データ取得
    tle_resp = session.get(query_url)
    if tle_resp.status_code != 200:
        raise ConnectionError(
            f"TLE query failed ({tle_resp.status_code}): {tle_resp.text[:200]}"
        )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(tle_resp.text)

    print(f"[INFO] TLE file saved to: {out_path.resolve()}")

    # MATLABの getTLEs() 相当の関数でパース
    objects = get_tles(out_path)
    return objects


objs = get_tles_for_estimation(
    start_year=2024, start_month=10, start_day=1,
    end_year=2024, end_month=10, end_day=31,
    selected_objects=[25544, 43013],
    get_tles_from_single_file=True,   # or False（個別ファイル）
    relative_dir="TLEdata",
)
