# Python/densityestimation/estimation/run_density_estimation_tle.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sgp4.api import jday

from densityestimation.data.bc_loader import load_bc_data
from densityestimation.data.eop_loader import load_eop_celestrak
from densityestimation.estimation.observations import (
    EstimationStateSpec,
    generate_observations_mee,
)
from densityestimation.tle.get_tles_for_estimation import (
    TLEObject,
    get_tles_for_estimation,
)
from densityestimation.tle.sgp4_wrapper import set_eop_matrix

# ---- 地球重力定数（観測MEE計算で使用） ----
GM_EARTH_KM3_S2_ACCURATE = 398600.4418


# ===== 入出力データ構造 =====

@dataclass
class EstimationInputs:
    yr: int
    mth: int
    dy: int
    nof_days: int
    ROMmodel: str
    r: int
    selected_objects: List[int]
    plot_figures: bool = False

    # ファイル系（実プロジェクトの構成に合わせる）
    eop_path: str = "densityestimation/data/EOP-All.txt"
    tle_dir: str = "TLEdata"
    tle_single_file: bool = True
    bc_path: str = "densityestimation/data/BCdata.txt"

    # 伝播（動力学）ステップ [秒]
    dt_seconds: int = 3600

    # ★ 追加: 観測エポックのサンプリング間隔 [時間]
    #   例) 1.0=1時間, 1/6=10分, 0.25=15分
    obs_hours_step: float = 1.0


@dataclass
class ROMBundle:
    """
    必要に応じて rom_generator が返す想定のバンドル（任意）。
    ROMを使わない場合は None で問題なし。
    """
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


# ===== ユーティリティ =====

def _jd_range(yr: int, mth: int, dy: int, nof_days: int) -> Tuple[float, float]:
    jd0, fr0 = jday(yr, mth, dy, 0, 0, 0.0)
    jdf = jd0 + fr0 + float(nof_days)
    return jd0 + fr0, jdf


def _time_vector_seconds(jd0: float, jdf: float, dt: int) -> np.ndarray:
    """両端を含む [0, tf] を dt 刻みで返す秒ベクトル。"""
    tf = (jdf - jd0) * 86400.0
    n = int(np.floor(tf / dt + 1e-9)) + 1
    t = np.arange(n, dtype=float) * float(dt)
    return t


def _obs_epochs_from_time(jd0: float, tsec: np.ndarray) -> np.ndarray:
    """（従来）秒ベクトルからJD配列へ変換。"""
    return jd0 + np.asarray(tsec, float) / 86400.0


def _obs_epochs_by_hours_step(jd0: float, jdf: float, hours_step: float) -> np.ndarray:
    """開始JDから終了JDまでを hours_step[hr] 間隔でサンプリングしてJD配列を返す。"""
    h = max(float(hours_step), 1e-6)  # 0割や負値を防止
    total_hours = (jdf - jd0) * 24.0
    nsteps = int(np.floor(total_hours / h)) + 1
    return jd0 + (np.arange(nsteps, dtype=float) * h) / 24.0


def _end_ymd(yr: int, mth: int, dy: int, nof_days: int) -> Tuple[int, int, int]:
    start = datetime(yr, mth, dy)
    end = start + timedelta(days=max(nof_days - 1, 0))
    return end.year, end.month, end.day


def _select_objects(objects: List[TLEObject], selected_ids: List[int]) -> List[TLEObject]:
    """selected_objects の順序で抽出。無いIDはエラー。"""
    out: List[TLEObject] = []
    for nid in selected_ids:
        found = [o for o in objects if int(o.noradID) == int(nid)]
        if not found:
            raise ValueError(f"No TLEs found for NORAD {nid}.")
        out.append(found[0])
    return out


def _build_RM(objects: List[TLEObject]) -> np.ndarray:
    """
    測定ノイズ共分散の初期対角（物体間独立、MEE 6要素×n_obj）。
    離心率 e に基づくスケーリング（論文の考え方）に近い形。
    """
    blocks = []
    for obj in objects:
        # 初期TLEの離心率を代表値に
        ecco = float(getattr(obj.satrecs[0], "ecco", 0.0))
        c1 = 1.5 * max(4.0 * ecco, 0.0023)
        c2 = 3.0 * max(ecco / 0.004, 1.0)
        blk = np.array(
            [c1 * 1e-8, c2 * 1e-10, c2 * 1e-10, 1e-9, 1e-9, 1e-8],
            dtype=float,
        )
        blocks.append(blk)
    diagv = np.concatenate(blocks, axis=0)
    return np.diag(diagv)


def _assemble_x0(
    objects: List[TLEObject],
    mee_meas: np.ndarray,
    BCdata: np.ndarray,
    r: int,
    svs: int,
    *,
    default_bc: float = 1.0e-2,  # 見つからないときのフォールバック [m^2/kg]
    warn: bool = True,           # 足りないときに警告出力するか
) -> Tuple[np.ndarray, List[float]]:
    """
    初期状態ベクトル x0g を構築（軌道6 + BC + ROM r）。
    BC は Data/BCdata.txt の推定値（m^2/kg）を優先して使用し、
    見つからない場合は default_bc を状態に格納（ファイル自体は変更しない）。
    """
    nop = len(objects)
    x0g = np.zeros((svs * nop + r, 1), dtype=float)

    # 各物体の MEE 初期値（観測の最初の列）
    for i in range(nop):
        x0g[svs * i + 0 : svs * i + 6, 0] = mee_meas[6 * i : 6 * i + 6, 0]

    # BC 推定（m^2/kg）
    bc_used: List[float] = []
    for i, obj in enumerate(objects):
        nid = int(obj.noradID)
        rows = BCdata[BCdata[:, 0] == nid] if BCdata.size > 0 else np.empty((0, 2))
        if rows.shape[0] == 0:
            if warn:
                print(f"[warn] BC not found for NORAD {nid} in BCdata. "
                      f"Using default {default_bc:.6f} m^2/kg.")
            bc = float(default_bc)
        else:
            bc = float(rows[0, 1])  # m^2/kg
        x0g[svs * i + 6, 0] = bc
        bc_used.append(bc)

    return x0g, bc_used


# ===== メイン関数 =====

def run_density_estimation_tle(
    par: EstimationInputs,
    *,
    # ROM生成が必要な場合のフック（未指定なら ROM なしで観測生成のみ実行）
    rom_generator: Optional[Callable[[str, int, float, float], ROMBundle]] = None,
    # ROM 初期化ベクトル z0 を作る関数（Uh'*(log10 rho - mean) 等）
    rom_initializer: Optional[Callable[[ROMBundle, float], np.ndarray]] = None,
    # UKF を走らせる場合のフック。未指定なら観測生成/初期化まで。
    stateFnc: Optional[Callable[[np.ndarray, float, float], np.ndarray]] = None,
    measurementFcn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    MATLAB runDensityEstimationTLE(...) のポート（観測生成～初期状態構築）。
    UKFは stateFnc / measurementFcn が与えられた場合のみ本関数内で実行します。
    """
    # --- 期間設定 ---
    jd0, jdf = _jd_range(par.yr, par.mth, par.dy, par.nof_days)

    # 伝播（動力学）ステップは dt_seconds に従う
    tsec = _time_vector_seconds(jd0, jdf, par.dt_seconds)  # [0:dt:tf]

    # 観測サンプリングは obs_hours_step に従う（従来の1時間固定から分離）
    obs_epochs = _obs_epochs_by_hours_step(jd0, jdf, par.obs_hours_step)

    # --- EOP 読み込み → TEME→J2000 変換へ反映（★重要） ---
    EOPMat = load_eop_celestrak(par.eop_path, full=False)
    set_eop_matrix(EOPMat)

    # --- TLE ---
    end_y, end_m, end_d = _end_ymd(par.yr, par.mth, par.dy, par.nof_days)
    objects_all = get_tles_for_estimation(
        start_year=par.yr, start_month=par.mth, start_day=par.dy,
        end_year=end_y, end_month=end_m, end_day=end_d,
        selected_objects=par.selected_objects,
        get_tles_from_single_file=par.tle_single_file,
        relative_dir=par.tle_dir,
    )
    # 選択ID順に並べ替え（以降の配列整形と一致させる）
    objects = _select_objects(objects_all, par.selected_objects)

    # --- 観測生成（TLE → TEME → J2000 → MEE） ---
    mee_meas = generate_observations_mee(objects, obs_epochs, GM_EARTH_KM3_S2_ACCURATE)

    # --- BC 読み込み・初期状態組み立て ---
    BCdata = load_bc_data(par.bc_path)
    svs = 7  # 6(MEE) + 1(BC)
    x0g, bc_used = _assemble_x0(
        objects, mee_meas, BCdata, par.r, svs,
        default_bc=1.0e-2, warn=True
    )

    # --- ROM 生成（必要な場合のみ） ---
    rom: Optional[ROMBundle] = None
    if rom_generator is not None:
        rom = rom_generator(par.ROMmodel, par.r, jd0, jdf)
        # ROM 初期化（任意）
        if rom_initializer is not None:
            z0 = np.asarray(rom_initializer(rom, jd0), dtype=float).reshape(-1)
            if z0.size != par.r:
                raise ValueError(f"rom_initializer returned size {z0.size}, expected r={par.r}")
            x0g[-par.r :, 0] = z0

    # --- 測定ノイズ RM（ブロック対角） ---
    RM = _build_RM(objects)

    # --- 初期状態共分散 P とプロセス雑音 Q（論文のオーダ） ---
    nop = len(objects)
    Pv = np.zeros((svs * nop + par.r,), dtype=float)
    Qv = np.zeros_like(Pv)

    # 軌道6要素：RM の該当6要素を初期分散に採用
    diag_RM = np.diag(RM)
    for i in range(nop):
        Pv[svs * i + 0 : svs * i + 6] = diag_RM[6 * i : 6 * i + 6]
        # BC 初期分散（1%）
        Pv[svs * i + 6] = (x0g[svs * i + 6, 0] * 0.01) ** 2

        # プロセス雑音（MEE+BC）
        Qv[svs * i + 0] = 1.5e-8
        Qv[svs * i + 1] = 2.0e-14
        Qv[svs * i + 2] = 2.0e-14
        Qv[svs * i + 3] = 1.0e-14
        Qv[svs * i + 4] = 1.0e-14
        Qv[svs * i + 5] = 1.0e-12
        Qv[svs * i + 6] = 1.0e-16  # BC

    # ROM 部分（ROMがある場合）
    if rom is not None and getattr(rom, "Qrom", None) is not None:
        Pv[-par.r :] = 5.0
        Pv[-par.r] = 20.0  # first mode やや大きめ
        Qrom = np.array(rom.Qrom)
        Qv[-par.r :] = (np.diag(Qrom) if Qrom.ndim == 2 else Qrom)[: par.r]
    else:
        # ROM未接続の暫定（小さめに）
        if par.r > 0:
            Pv[-par.r :] = 5.0
            Pv[-par.r] = 20.0
            Qv[-par.r :] = 1e-6

    P = np.diag(Pv)
    Q = np.diag(Qv)

    # --- 本関数では「観測生成～初期化」までを返す。UKFは上位で実行 ---
    result: Dict[str, Any] = dict(
        jd0=jd0,
        jdf=jdf,
        tsec=tsec,
        obs_epochs=obs_epochs,
        objects=objects,
        mee_meas=mee_meas,
        x0=x0g,
        P=P,
        Q=Q,
        RM=RM,
        BC_used=bc_used,
        rom=rom,
        spec=EstimationStateSpec(n_obj=len(objects), nz=par.r),
        svs=svs,
    )

    # --- 互換: stateFnc / measurementFcn が与えられたらここで同化まで走らせたい場合 ---
    if stateFnc is not None and measurementFcn is not None:
        # ここでは“互換フック”のみ提供。SR-UKF/UKF 実装に合わせて上位で配線を推奨。
        result["note"] = "stateFnc/measurementFcn supplied; run your UKF driver with returned x0,P,Q,RM."
    return result
