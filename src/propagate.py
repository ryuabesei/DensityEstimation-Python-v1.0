#SGP4伝搬、座標変換
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable

import numpy as np
import pandas as pd
import pymap3d as pm
from sgp4.api import Satrec, jday


@dataclass
class GeoState:
    t: np.ndarray      # datetime(UTC)配列
    lat: np.ndarray    # deg
    lon: np.ndarray    # deg
    alt_km: np.ndarray # km


def _teme_to_geodetic(r_teme_km: np.ndarray, ts: Iterable[datetime]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """TEME位置ベクトル(km)→(lat, lon, alt[km])
    pm.teme2geodeticは(m, rad)系なので合わせて変換
    """
    lats, lons, alts = [], [], []
    for r_km, t in zip(r_teme_km, ts):
        x, y, z = (r_km * 1000.0)  # mへ
        lat, lon, alt_m = pm.teme2geodetic(x, y, z, t)
        lats.append(np.degrees(lat))
        lons.append(np.degrees(lon))
        alts.append(alt_m / 1000.0)
    return np.array(lats), np.array(lons), np.array(alts)


def propagate_tle(l1: str, l2: str, start: datetime, minutes: int, step_sec: int = 60) -> pd.DataFrame:
    sat = Satrec.twoline2rv(l1, l2)
    n_steps = int((minutes * 60) / step_sec) + 1
    ts = np.array([start + timedelta(seconds=i * step_sec) for i in range(n_steps)])

    r_list = []
    for t in ts:
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond * 1e-6)
        e, r, v = sat.sgp4(jd, fr)
        if e != 0:
            raise RuntimeError(f"SGP4 error code {e} at {t.isoformat()}")
        r_list.append(np.array(r))  # km, TEME

    r_arr = np.vstack(r_list)
    lat, lon, alt_km = _teme_to_geodetic(r_arr, ts)

    df = pd.DataFrame({
        "t_utc": ts,
        "lat_deg": lat,
        "lon_deg": lon,
        "alt_km": alt_km,
    })
    return df