from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from sgp4.api import jday

from densityestimation.data.eop_loader import load_eop_celestrak
from densityestimation.estimation.observations import (
    generate_observations_mee,  # ← 既に実装済みのもの
)
from densityestimation.tle.get_tles import TLEObject
from densityestimation.tle.get_tles_for_estimation import get_tles_for_estimation

GM_EARTH_KM3_S2 = 398600.4418

@dataclass
class EstimationConfig:
    year: int
    month: int
    day: int
    nof_days: int
    rom_model: str           # 例: 'JB2008_1999_2010'
    r: int                   # 例: 10
    selected_objects: List[int]
    plot_figures: bool = False
    tle_single_file: bool = True       # estimationObjects.tle から読むか
    tle_dir: str = "TLEdata"
    eop_path: str = "data/EOP-All.txt" # Celestrak形式を想定


def _obs_epochs_jd(year: int, month: int, day: int, nof_days: int, hours_step: float = 1.0) -> np.ndarray:
    jd0, fr0 = jday(year, month, day, 0, 0, 0.0)
    jd0 = jd0 + fr0
    nsteps = int(nof_days * 24 / hours_step) + 1
    return jd0 + np.arange(nsteps) * (hours_step / 24.0)


def run_density_estimation_tle(cfg: EstimationConfig):
    """
    MATLAB: runDensityEstimationTLE(yr,mth,dy,nofDays,ROMmodel,r,selectedObjects,plotFigures)
    の Python 版スケルトン。現状は MEE 観測生成までを自動実行。
    """
    # 1) EOP読込（TEME→J2000 で必要）
    if not Path(cfg.eop_path).exists():
        raise FileNotFoundError(f"EOP file not found: {cfg.eop_path}")
    load_eop_celestrak(cfg.eop_path)

    # 2) TLE読み込み（単一ファイル / 各NORADファイル）
    end_jd_year, end_jd_month, end_jd_day = cfg.year, cfg.month, cfg.day + cfg.nof_days - 1
    # 端数日計算を簡略化（1ヶ月内の例で十分。必要なら datetime で厳密化）
    objects: List[TLEObject] = get_tles_for_estimation(
        start_year=cfg.year, start_month=cfg.month, start_day=cfg.day,
        end_year=end_jd_year, end_month=end_jd_month, end_day=end_jd_day,
        selected_objects=cfg.selected_objects,
        get_tles_from_single_file=cfg.tle_single_file,
        relative_dir=cfg.tle_dir,
    )

    # 3) 観測エポック（1時間刻み）を生成
    obs_epochs = _obs_epochs_jd(cfg.year, cfg.month, cfg.day, cfg.nof_days, hours_step=1.0)

    # 4) MEE観測生成（TLE → TEME → J2000 → MEE）
    mee_obs = generate_observations_mee(objects, obs_epochs, GM_EARTH_KM3_S2)

    # 以降：UKFで ROM 状態/BC を推定（まだ実装中のためフックのみ）
    #  - stateFnc(xx, t_i, t_{i+1})
    #  - measurementFcn(Xp)
    #  - ukf(X_est0, Meas, time, stateFnc, measurementFcn, P0, R, Q)
    # TODO: 実装が揃い次第ここに組み込み

    return {
        "objects": objects,
        "obs_epochs_jd": obs_epochs,
        "mee_obs": mee_obs,
    }
