from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from sgp4.api import jday

from densityestimation.data.eop_loader import load_eop_celestrak
from densityestimation.dynamics.derivatives import (
    drag_accel,  # 既存想定（なければ簡易実装）
)
from densityestimation.estimation.observations import (
    EstimationStateSpec,
    generate_observations_mee,  # ← 既に実装済みのもの
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


def make_process_model(rom_runtime, spec: EstimationStateSpec, grid_interp_fn, dt_sec):
    """
    rom_runtime : ROMRuntime (Ac,Bc,Ur,xbar,input_fn を内包)
    spec        : EstimationStateSpec(n_obj, nz)
    grid_interp_fn: callable(log10rho_grid) -> rho(point)
    dt_sec      : タイムステップ（推奨3600）
    """
    dt = float(dt_sec)

    def f(x, t_epoch):
        # 1) 分解
        mee_list, bc_list, z = unpack_state(x, spec)

        # 2) ROM状態を前進
        z_next = rom_runtime.step_z(z, t_epoch, dt, discretize_lin)

        # 3) 各オブジェクトの軌道を前進（ドラッグ含む）
        mee_next = []
        for (p,f,g,h,k,L), BC in zip(mee_list, bc_list):
            # MEE -> R,V
            r0,v0, mu = mee_to_cartesian((p,f,g,h,k,L))   # 既存関数に合わせて引数調整
            # 密度推定（ここでは“現在のz”を使用）
            def rho_here():
                # grid補間（lat,LST,altの計算が既存ならそれを使用）
                # 既存の関数に置き換えてOK。ひとまず log10rho_grid -> rho とするダミー:
                rho = rom_runtime.density_at(z, grid_interp_fn)
                return rho

            # ドラッグ加速度（既存のdrag_accelを想定。なければ簡易CdA/mモデル）
            a_drag = drag_accel(r0, v0, BC, rho_here)

            # 1ステップ数値積分（既存のpropagate_cartesianに外力引数があるなら渡す）
            r1, v1 = propagate_cartesian(r0, v0, dt, nonconservative_accel=a_drag)

            # R,V -> MEE
            mee1 = cartesian_to_mee(r1, v1, mu)
            mee_next.append(mee1)

        # 4) 再パック
        x_next = pack_state(mee_next, bc_list, z_next)
        return x_next

    return f