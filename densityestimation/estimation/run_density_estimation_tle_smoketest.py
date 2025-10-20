# -*- coding: utf-8 -*-
"""
スモークテスト：外部データ無しで配線だけ通るか確認
- TLE: ISS(25544) を1体だけ、ローカル TLEdata/estimationObjects.tle に書き出し
- BC: densityestimation/data/BCdata.txt を最小構成で用意
- EOP: densityestimation/data/EOP-All.txt を使用（既にリポジトリに同梱）
- ROM: ダミーROM（定数／ゼロ応答）、UKFは実行しない（stateFnc=None, measurementFcn=None）
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np

# 既存コードのインポート
from densityestimation.estimation.run_density_estimation_tle import (
    EstimationInputs,
    run_density_estimation_tle,
)

# --- ISS TLE（安定の例。手元テスト用） ---
ISS_L1 = "1 25544U 98067A   20344.91667824  .00001264  00000-0  29621-4 0  9991"
ISS_L2 = "2 25544  51.6442  12.2145 0002202  70.9817  48.7153 15.49260293258322"
NORAD_ID = 25544

def _ensure_tle_singlefile(tle_dir: str) -> None:
    os.makedirs(tle_dir, exist_ok=True)
    path = os.path.join(tle_dir, "estimationObjects.tle")
    with open(path, "w", encoding="utf-8") as f:
        f.write(ISS_L1.strip() + "\n")
        f.write(ISS_L2.strip() + "\n")
    print(f"[ok] wrote TLE to {path}")

def _ensure_bcfile(bc_path: str) -> None:
    os.makedirs(os.path.dirname(bc_path), exist_ok=True)
    if not os.path.exists(bc_path):
        with open(bc_path, "w", encoding="utf-8") as f:
            f.write("# NORAD_ID  BC[m^2/kg]\n")
            f.write("# minimal file for smoketest\n")
            f.write(f"{NORAD_ID}  0.010\n")  # 適当な値
        print(f"[ok] wrote BCdata to {bc_path}")
    else:
        # 上書きはしない（既存ファイルを尊重）
        pass

# ---- ROM ダミー実装 ----

@dataclass
class _DummyROM:
    AC: np.ndarray
    BC: np.ndarray
    Uh: np.ndarray
    F_U: List[Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]]
    Dens_Mean: np.ndarray
    M_U: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    SLTm: np.ndarray
    LATm: np.ndarray
    ALTm: np.ndarray
    maxAtmAlt: float
    SWinputs: Dict[str, np.ndarray]
    Qrom: np.ndarray

def _dummy_rom_generator(rom_model: str, r: int, jd0: float, jdf: float) -> _DummyROM:
    """
    generateROMdensityModel の代替。外部ファイル不要なゼロROM。
    - 空間モードはゼロ
    - M_U は定数 -14（~1e-14 kg/m^3）を返す
    - Qrom は対角 1e-8 程度
    """
    # グリッド（形だけ）
    SLTm = np.zeros((4, 4, 2))   # LST × LAT × ALT の3Dグリッド風
    LATm = np.zeros_like(SLTm)
    ALTm = np.zeros_like(SLTm)
    # Uh: r×(grid) 風に見せるだけ（実際は使わない）
    Uh = np.zeros((SLTm.size, r))
    # F_U: r 個の補間関数（全部ゼロを返す）
    def _zero_interp(x, y, z):
        # ブロードキャスト対応：np.zeros の形は (x) と合わせる
        x = np.array(x)
        return np.zeros_like(x, dtype=float)
    F_U = [lambda slt, lat, alt, _=_ : _zero_interp(slt, lat, alt) for _ in range(r)]
    # M_U: 定数（log10密度の平均）-14
    M_U = lambda slt, lat, alt: -14.0 * np.ones_like(np.array(slt), dtype=float)
    # Qrom: 1時間予測誤差（対角）
    Qrom = 1e-8 * np.ones((r, r))
    return _DummyROM(
        AC=np.zeros((r, r)),
        BC=np.zeros((r, r)),
        Uh=Uh,
        F_U=F_U,
        Dens_Mean=-14.0 * np.ones(SLTm.size),
        M_U=M_U,
        SLTm=SLTm,
        LATm=LATm,
        ALTm=ALTm,
        maxAtmAlt=1500.0,
        SWinputs={},
        Qrom=Qrom,
    )

def main():
    # 作業ルート（このスクリプトの親=プロジェクト 直下 を想定）
    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    os.chdir(proj_root)

    # 必須: TLE と BC を（最小構成で）用意
    tle_dir = "TLEdata"  # EstimationInputs のデフォルトと合わせる
    _ensure_tle_singlefile(tle_dir)

    bc_path = os.path.join("densityestimation", "data", "BCdata.txt")
    _ensure_bcfile(bc_path)

    # EOP は同梱ファイルを明示
    eop_path = os.path.join("densityestimation", "data", "EOP-All.txt")

    # 入力（ISSで 0.5 日だけ／r=4／ROM ダミー／UKF無し）
    par = EstimationInputs(
        yr=2020, mth=12, dy=9, nof_days=1,   # TLE epoch 付近の日付なら概ねOK
        ROMmodel="DUMMY", r=4,
        selected_objects=[NORAD_ID],
        plot_figures=False,
        eop_path=eop_path,
        tle_dir=tle_dir,
        tle_single_file=True,
        bc_path=bc_path,
        dt_seconds=3600,  # 1時間ステップ
    )

    # 実行（ROMはダミー、UKF関数は渡さない＝UKFスキップ）
    out = run_density_estimation_tle(
        par,
        rom_generator=_dummy_rom_generator,
        rom_initializer=None,     # 初期 z0 未使用（ゼロのまま）
        stateFnc=None,            # UKF スキップ
        measurementFcn=None,      # UKF スキップ
    )

    # 目視用サマリ
    print("\n=== Smoke Test Summary ===")
    print(f"jd0 .. jdf: {out['jd0']:.6f} .. {out['jdf']:.6f}")
    print(f"#epochs    : {out['tsec'].size}")
    print(f"#objects   : {len(out['objects'])}")
    print(f"X_est shape: {out['X_est'].shape}  (UKF未実行なので初期状態のみ)")
    print(f"ROM Q diag : {np.diag(out['Q'])[-par.r:]} (ROM部Qの対角)")

    # 先頭の観測MEEを少しだけ表示
    mee_sample = out['mee_meas'][:, :3]
    print("\nFirst 3 observation epochs (MEE for the one object):")
    print(mee_sample)

if __name__ == "__main__":
    main()
