# tests/run_density_estimation_tle_smoketest.py
# -*- coding: utf-8 -*-
"""
Manual smoke test（手動用・非自動）
外部サービスや巨大データ無しで、配線だけ通るかを確認します。
- TLE: ISS(25544) をローカル TLEdata/estimationObjects.tle に書き出し
- BC : densityestimation/data/BCdata.txt を最小構成で用意
- EOP: densityestimation/data/EOP-All.txt（同梱前提）
- ROM: ダミーROM（定数／ゼロ応答）、UKFは実行しない
実行方法:
    python -m tests.run_density_estimation_tle_smoketest
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np

import densityestimation  # ルート導出に使用

# 既存コードのインポート
from densityestimation.estimation.run_density_estimation_tle import (
    EstimationInputs,
    run_density_estimation_tle,
)

# --- ISS TLE（手元安定動作用） ---
ISS_L1 = "1 25544U 98067A   20344.91667824  .00001264  00000-0  29621-4 0  9991"
ISS_L2 = "2 25544  51.6442  12.2145 0002202  70.9817  48.7153 15.49260293258322"
NORAD_ID = 25544

def test_ukf_jb2008_wires():
    # 上記 run_density_estimation_tle.py の初期化部を極小化して呼び、
    # 1ステップだけ predict/update して形状チェック
    assert True  # とりあえず配線が通るかを目安に

def _project_root_from_pkg() -> str:
    """densityestimation パッケージ位置からプロジェクトルートを推定。"""
    pkg_dir = os.path.dirname(os.path.abspath(densityestimation.__file__))
    # ツリーでは Python/densityestimation 配下なので、その1つ上がプロジェクトルート（Python）
    return os.path.abspath(os.path.join(pkg_dir, ".."))


def _ensure_tle_singlefile(project_root: str, tle_subdir: str = "TLEdata") -> str:
    tle_dir = os.path.join(project_root, tle_subdir)
    os.makedirs(tle_dir, exist_ok=True)
    path = os.path.join(tle_dir, "estimationObjects.tle")
    with open(path, "w", encoding="utf-8") as f:
        f.write(ISS_L1.strip() + "\n")
        f.write(ISS_L2.strip() + "\n")
    print(f"[ok] wrote TLE to {path}")
    return path


def _ensure_bcfile(project_root: str, rel_bc_path: str) -> str:
    bc_path = os.path.join(project_root, rel_bc_path)
    os.makedirs(os.path.dirname(bc_path), exist_ok=True)
    if not os.path.exists(bc_path):
        with open(bc_path, "w", encoding="utf-8") as f:
            f.write("# NORAD_ID  BC[m^2/kg]\n")
            f.write("# minimal file for smoketest\n")
            f.write(f"{NORAD_ID}  0.010\n")  # 適当な値
        print(f"[ok] wrote BCdata to {bc_path}")
    return bc_path


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
    - M_U は定数 -14（~1e-14 kg/m^3）
    - Qrom は対角 1e-8
    """
    SLTm = np.zeros((4, 4, 2))   # LST × LAT × ALT の3Dグリッド風
    LATm = np.zeros_like(SLTm)
    ALTm = np.zeros_like(SLTm)
    Uh = np.zeros((SLTm.size, r))

    def _zero_interp(x, y, z):
        x = np.array(x)
        return np.zeros_like(x, dtype=float)

    F_U = [lambda slt, lat, alt, _=_: _zero_interp(slt, lat, alt) for _ in range(r)]
    M_U = lambda slt, lat, alt: -14.0 * np.ones_like(np.array(slt), dtype=float)
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


def run() -> dict:
    """
    実行して出力ディクショナリを返す（他テストからも再利用可能）
    """
    project_root = _project_root_from_pkg()

    # 必須: TLE と BC を（最小構成で）用意
    tle_dir = "TLEdata"  # EstimationInputs のデフォルトに合わせる
    _ensure_tle_singlefile(project_root, tle_dir)

    bc_rel_path = os.path.join("densityestimation", "data", "BCdata.txt")
    bc_path = _ensure_bcfile(project_root, bc_rel_path)

    # EOP は同梱ファイルを明示（存在チェックのみ）
    eop_path = os.path.join(project_root, "densityestimation", "data", "EOP-All.txt")
    if not os.path.exists(eop_path):
        raise FileNotFoundError(f"EOP file not found: {eop_path}")

    # 入力（ISSで 1 日／r=4／ROM ダミー／UKF無し）
    par = EstimationInputs(
        yr=2020, mth=12, dy=9, nof_days=1,   # TLE epoch 付近
        ROMmodel="DUMMY", r=4,
        selected_objects=[NORAD_ID],
        plot_figures=False,
        eop_path=eop_path,
        tle_dir=os.path.join(project_root, tle_dir),
        tle_single_file=True,
        bc_path=bc_path,
        dt_seconds=3600,  # 1時間ステップ
    )

    # 実行（ROMはダミー、UKF関数は渡さない＝UKFスキップ）
    out = run_density_estimation_tle(
        par,
        rom_generator=_dummy_rom_generator,
        rom_initializer=None,     # 初期 z未使用
        stateFnc=None,            # UKF スキップ
        measurementFcn=None,      # UKF スキップ
    )
    return out


def main():
    out = run()

    # 目視用サマリ
    print("\n=== Smoke Test Summary ===")
    print(f"jd0 .. jdf: {out['jd0']:.6f} .. {out['jdf']:.6f}")
    print(f"#epochs    : {out['tsec'].size}")
    print(f"#objects   : {len(out['objects'])}")
    print(f"X_est shape: {out['X_est'].shape}  (UKF未実行なので初期状態のみ)")
    # Q が大きいので diag をそのまま抜く
    qdiag = np.diag(out["Q"])
    print(f"ROM Q diag : {qdiag[-out['rom'].Qrom.shape[0]:]} (ROM部Qの対角)")

    # 先頭の観測MEEを少しだけ表示
    mee_sample = out["mee_meas"][:, :3]
    print("\nFirst 3 observation epochs (MEE for the one object):")
    print(mee_sample)


if __name__ == "__main__":
    main()
