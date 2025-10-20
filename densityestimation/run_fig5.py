# run_fig5.py
# License: GNU GPL v3
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from densityestimation.estimation.conversions import ep2pv  # 6×MEE -> (r,v)
from densityestimation.models.rom.rom_jb2008 import jb2008_initializer
from densityestimation.models.rom.rom_jb2008 import (
    rom_generator as jb2008_rom_generator,
)
from densityestimation.propagation.mee_bc_rom import (
    propagate_state_mee_bc_rom,  # 状態遷移関数
)
from densityestimation.run_density_estimation_tle import (
    GM_EARTH_KM3_S2_ACCURATE,
    EstimationInputs,
    run_density_estimation_tle,
)
from densityestimation.ukf.measurements import fullmee2mee  # 測定関数（MEE抽出）

# ---- 設定用コンテナ ---------------------------------------------------------

@dataclass
class Fig5Config:
    yr: int = 2002
    mth: int = 8
    dy: int = 1
    nof_days: int = 10
    ROMmodel: str = "JB2008_1999_2010"
    r: int = 10
    selected_objects: List[int] = (
        [63,165,614,2153,2622,4221,6073,7337,8744,12138,12388,14483,20774,23278,27391,27392,26405]
    )
    plot_figures: bool = True

    # データファイル
    eop_path: str = "Data/EOP-All.txt"
    tle_dir: str = "TLEdata"
    tle_single_file: bool = True
    bc_path: str = "Data/BCdata.txt"

    # 既存の Fig.5 既知結果を読み込む場合（MATLAB .mat）
    use_precomputed: bool = True
    precomputed_mat: Optional[str] = "mid_data/David_Fig5.mat"  # 例: 既存の結果

# ---- メイン -----------------------------------------------------------------

def run_fig5(cfg: Fig5Config) -> Dict[str, object]:
    # 1) run_density_estimation_tle を使って前処理（TLE・観測・初期化など）
    par = EstimationInputs(
        yr=cfg.yr, mth=cfg.mth, dy=cfg.dy, nof_days=cfg.nof_days,
        ROMmodel=cfg.ROMmodel, r=cfg.r, selected_objects=cfg.selected_objects,
        plot_figures=False,
        eop_path=cgf.eop_path if (cgf := cfg) else cfg.eop_path,  # Py3.8 対応で一度束縛
        tle_dir=cfg.tle_dir, tle_single_file=cfg.tle_single_file,
        bc_path=cfg.bc_path, dt_seconds=3600,
    )

    results = run_density_estimation_tle(
        par,
        rom_generator=jb2008_rom_generator,         # JB2008 ROM
        rom_initializer=jb2008_initializer,         # z0_M を作る初期化
        stateFnc=propagate_state_mee_bc_rom,        # 状態遷移（MEE+BC+ROM）
        measurementFcn=fullmee2mee,                 # 観測空間（MEEのみ）
    )

    jd0 = float(results["jd0"])
    jdf = float(results["jdf"])
    tsec = np.asarray(results["tsec"])
    time_hours = tsec / 3600.0
    mee_meas = np.asarray(results["mee_meas"])
    objects = results["objects"]
    svs = 7
    nop = len(objects)
    r = cfg.r

    # 2) Fig.5 仕様：UKF は回さず、既存の推定結果（X_est, Pv）を .mat から読む or 既にあるものを使用
    if cfg.use_precomputed and cfg.precomputed_mat:
        mat = sio.loadmat(cfg.precomputed_mat, squeeze_me=True)
        # 期待する変数名は MATLAB スクリプトに合わせる（必要に応じてキー名を調整）
        X_est = np.array(mat["X_est"], dtype=float)
        Pv = np.array(mat["Pv"], dtype=float)
        # 時間ベクトルが mat に無い場合、run_* の tsec を使う
    else:
        # その場で UKF を回した結果（既に results["X_est"], results["Pv"] あり）
        X_est = np.array(results["X_est"], dtype=float)
        Pv = np.array(results["Pv"], dtype=float)

    # 初期分散を Pv[:,0] に明示反映（MATLAB と同様の書き方）
    if Pv.shape[1] > 0:
        Pv[:, 0] = np.diag(results["P"])

    # 3) --- プロット（Fig.5 相当） ---
    if cfg.plot_figures:
        # ROM モード推定と 3σ
        plt.figure()
        for i in range(r):
            idx = -r + i
            plt.subplot(int(np.ceil(r / 4.0)), 4, i + 1)
            plt.plot(time_hours, X_est[idx, :], lw=1)
            plt.plot(time_hours, X_est[idx, :] + 3.0 * np.sqrt(Pv[idx, :]), "--k", lw=1)
            plt.plot(time_hours, X_est[idx, :] - 3.0 * np.sqrt(Pv[idx, :]), "--k", lw=1)
            plt.xlabel("Time [hrs]"); plt.ylabel(f"z_{i+1}")
            plt.title(f"Mode {i+1}")
            plt.tight_layout()

        # ROM モードの 3σ
        plt.figure()
        for i in range(r):
            idx = -r + i
            plt.subplot(int(np.ceil(r / 4.0)), 4, i + 1)
            plt.plot(time_hours, 3.0 * np.sqrt(Pv[idx, :]), "k")
            plt.xlabel("Time [hrs]"); plt.ylabel(f"z_{i+1} 3σ")
            plt.title(f"Mode {i+1}")
            plt.tight_layout()

        # BC 推定と 3σ（m^2/kg 単位に戻す）
        plt.figure()
        for i in range(nop):
            idx = svs * (i + 1) - 1  # 0-based: 6(MEE) + 1(BC)
            plt.subplot(int(np.ceil(nop / 2.0)), 2, i + 1)
            plt.plot(time_hours, X_est[idx, :] / 1000.0, lw=1)
            plt.plot(time_hours, X_est[idx, :] / 1000.0 + 3.0 * np.sqrt(Pv[idx, :]) / 1000.0, "--k", lw=1)
            plt.plot(time_hours, X_est[idx, :] / 1000.0 - 3.0 * np.sqrt(Pv[idx, :]) / 1000.0, "--k", lw=1)
            plt.xlabel("Time [hrs]"); plt.ylabel("BC [m$^2$/kg]")
            plt.title(f"Orbit {objects[i].noradID}")
            plt.tight_layout()

        # MEE の各要素の不確かさ（σ）
        plt.figure()
        for j in range(6):
            plt.subplot(2, 3, j + 1)
            for i in range(nop):
                idx = svs * i + j
                plt.plot(time_hours, np.sqrt(Pv[idx, :]))
            plt.xlabel("Time [hrs]")
            ylabs = ["σ_p [km]", "σ_f [-]", "σ_g [-]", "σ_h [-]", "σ_k [-]", "σ_L [rad]"]
            plt.ylabel(ylabs[j])
        plt.tight_layout()

        # 位置誤差（推定 vs TLE 観測）
        plt.figure()
        for k in range(nop):
            xx_est_pv = np.zeros((6, X_est.shape[1]))
            xx_mea_pv = np.zeros((6, mee_meas.shape[1]))
            for j in range(X_est.shape[1]):
                pos, vel = ep2pv(X_est[svs * k: svs * k + 6, j], GM_EARTH_KM3_S2_ACCURATE)
                xx_est_pv[0:3, j] = pos
                xx_est_pv[3:6, j] = vel
                pos_m, vel_m = ep2pv(mee_meas[6 * k: 6 * k + 6, j], GM_EARTH_KM3_S2_ACCURATE)
                xx_mea_pv[0:3, j] = pos_m
                xx_mea_pv[3:6, j] = vel_m
            perr = np.sqrt(np.sum((xx_est_pv[0:3, :] - xx_mea_pv[0:3, :]) ** 2, axis=0))
            plt.subplot(int(np.ceil(nop / 2.0)), 2, k + 1)
            plt.plot(time_hours, perr)
            plt.xlabel("Time [hrs]"); plt.ylabel("Position error [km]")
            plt.title(f"Orbit {objects[k].noradID}, mean={np.mean(perr):.2f}")
        plt.tight_layout()

        # Fig.5 の CHAMP / MSIS 比較は、利用可能なデータが揃った時に追加
        # （プレースホルダ）
        # plot_champ_msis_comparison(...)

        plt.show()

    return dict(
        jd0=jd0, jdf=jdf, tsec=tsec, X_est=X_est, Pv=Pv, objects=objects, mee_meas=mee_meas
    )


if __name__ == "__main__":
    cfg = Fig5Config()
    out = run_fig5(cfg)
