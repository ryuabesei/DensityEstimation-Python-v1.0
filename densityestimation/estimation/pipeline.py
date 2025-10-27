from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sgp4.api import jday

from densityestimation.data.eop_loader import load_eop_celestrak
from densityestimation.dynamics.derivatives import (
    drag_accel,  # 既存想定（なければ簡易実装）
)
from densityestimation.estimation.observations import (
    EstimationStateSpec,
    generate_observations_mee,  # ← 既実装
    pack_state,
    unpack_state,
)
from densityestimation.orbit.mee import cartesian_to_mee, mee_to_cartesian  # 既存想定
from densityestimation.orbit.propagation import propagate_cartesian  # 既存想定
from densityestimation.rom.dmdc import discretize_lin
from densityestimation.tle.get_tles import TLEObject
from densityestimation.tle.get_tles_for_estimation import get_tles_for_estimation

GM_EARTH_KM3_S2 = 398600.4418


@dataclass
class EstimationConfig:
    year: int
    month: int
    day: int
    nof_days: int
    rom_model: str            # 例: 'JB2008_1999_2010'
    r: int                    # 例: 10
    selected_objects: List[int]
    plot_figures: bool = False
    tle_single_file: bool = True        # estimationObjects.tle から読むか
    tle_dir: str = "TLEdata"
    eop_path: str = "data/EOP-All.txt"  # Python/densityestimation/ 直下で走らせる前提ならこの相対でOK


def _obs_epochs_jd(year: int, month: int, day: int, nof_days: int, hours_step: float = 1.0) -> np.ndarray:
    """開始0時から nof_days 日・hour_step刻みのJD列を生成（両端含む）。"""
    jd0, fr0 = jday(year, month, day, 0, 0, 0.0)
    jd0 = jd0 + fr0
    nsteps = int(round(nof_days * 24.0 / hours_step)) + 1
    return jd0 + np.arange(nsteps, dtype=float) * (hours_step / 24.0)


def _end_ymd(year: int, month: int, day: int, nof_days: int) -> Tuple[int, int, int]:
    """開始日 + (nof_days-1) 日の年月日を厳密に返す。"""
    start = datetime(year, month, day)
    end = start + timedelta(days=max(nof_days - 1, 0))
    return end.year, end.month, end.day


def run_density_estimation_tle(cfg: EstimationConfig) -> Dict[str, Any]:
    """
    MATLAB: runDensityEstimationTLE(yr,mth,dy,nofDays,ROMmodel,r,selectedObjects,plotFigures)
    の Python 版スケルトン。現状は MEE 観測生成までを自動実行。
    """
    # 1) EOP読込（TEME→J2000 の変換で必要）
    if not Path(cfg.eop_path).exists():
        raise FileNotFoundError(f"EOP file not found: {cfg.eop_path}")
    load_eop_celestrak(cfg.eop_path)

    # 2) TLE読み込み（単一ファイル / 個別取得）
    end_y, end_m, end_d = _end_ymd(cfg.year, cfg.month, cfg.day, cfg.nof_days)
    objects: List[TLEObject] = get_tles_for_estimation(
        start_year=cfg.year, start_month=cfg.month, start_day=cfg.day,
        end_year=end_y, end_month=end_m, end_day=end_d,
        selected_objects=cfg.selected_objects,
        get_tles_from_single_file=cfg.tle_single_file,
        relative_dir=cfg.tle_dir,
    )

    # 3) 観測エポック（1時間刻み）を生成（両端含む）
    obs_epochs = _obs_epochs_jd(cfg.year, cfg.month, cfg.day, cfg.nof_days, hours_step=1.0)

    # 4) MEE観測生成（TLE → TEME → J2000 → MEE）
    mee_obs = generate_observations_mee(objects, obs_epochs, GM_EARTH_KM3_S2)

    # 以降：UKFで ROM 状態/BC を推定（この関数では観測生成まで）
    return {
        "objects": objects,
        "obs_epochs_jd": obs_epochs,
        "mee_obs": mee_obs,
    }


def make_process_model(rom_runtime, spec: EstimationStateSpec, grid_interp_fn, dt_sec: float):
    """
    rom_runtime : ROMRuntime (Ac,Bc,Ur,xbar,input_fn などを内包し、step_z と density_at を提供)
    spec        : EstimationStateSpec(n_obj, nz)
    grid_interp_fn : callable(log10rho_grid) -> rho_at_point
                     （ROM側の格子→点値の補間を行う関数。rom_runtime.density_at 内で使うなら不要）
    dt_sec      : タイムステップ（推奨3600）
    """
    dt = float(dt_sec)

    def f(x: np.ndarray, t_epoch: float) -> np.ndarray:
        # 1) 分解
        mee_list, bc_list, z = unpack_state(x, spec)

        # 2) ROM状態を前進（離散化は DMDc 由来の線形近似でOK。必要なら内部で非線形入力も可）
        z_next = rom_runtime.step_z(z, t_epoch, dt, discretize_lin)

        # 3) 各オブジェクトの軌道を前進（ドラッグ含む）
        mee_next = []
        for (p, f, g, h, k, L), BC in zip(mee_list, bc_list):
            # MEE -> r, v （μ 明示）
            r0, v0 = mee_to_cartesian((p, f, g, h, k, L), GM_EARTH_KM3_S2)

            # その時刻における大気密度を返す小さな関数（z と位置から計算）
            def rho_here(r_vec: np.ndarray, v_vec: np.ndarray, t_cur: float) -> float:
                # ROMの再構成→格子→点値（内部で grid_interp_fn を用いる設計でも良い）
                return float(rom_runtime.density_at(z, r_vec, t_cur, grid_interp_fn))

            # 非保存力：ドラッグ加速度クロージャ
            def nonconservative_accel(r_vec: np.ndarray, v_vec: np.ndarray, t_cur: float) -> np.ndarray:
                # BC は m^2/kg、drag_accel は [km/s^2] を返す前提
                return drag_accel(r_vec, v_vec, BC, rho_here)

            # 1ステップ数値積分（オービット伝播器が外力関数を受け取れる想定）
            r1, v1 = propagate_cartesian(r0, v0, dt, nonconservative_accel=nonconservative_accel)

            # r,v -> MEE
            mee1 = cartesian_to_mee(r1, v1, GM_EARTH_KM3_S2)
            mee_next.append(mee1)

        # 4) 再パック（BC はこのモデルでは一定）
        x_next = pack_state(mee_next, bc_list, z_next)
        return x_next

    return f
