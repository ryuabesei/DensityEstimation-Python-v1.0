# Port of runDensityEstimationTLE.m
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from sgp4.api import jday

from densityestimation.data.bc_loader import load_bc_data
from densityestimation.data.eop_loader import load_eop_celestrak
from densityestimation.estimation.observations import generate_observations_mee
from densityestimation.models.rom_model import (
    generate_rom_density_model,  # ← 追加：統合ROMジェネレータ
)
from densityestimation.tle.get_tles_for_estimation import (
    TLEObject,
    get_tles_for_estimation,
)
from densityestimation.ukf.srukf import ukf

# ---- 地球重力定数（MATLABの使い分けに合わせた参照値） ----
GM_EARTH_KM3_S2_SGP4 = 398600.8         # 参考：SGP4系の既定（MATLABの mu）
GM_EARTH_KM3_S2_ACCURATE = 398600.4418  # MATLAB で GM*1e-9 として使っているもの


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

    # ファイル系
    eop_path: str = "Data/EOP-All.txt"
    tle_dir: str = "TLEdata"
    tle_single_file: bool = True
    bc_path: str = "Data/BCdata.txt"

    # 時間刻み（秒）
    dt_seconds: int = 3600


@dataclass
class ROMBundle:
    # MATLAB generateROMdensityModel(...) の戻り値たち
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
    # 追加で、初期 ROM 状態を作る補助（任意）
    # e.g., jb2008_initializer: Callable[[...], np.ndarray]


# ===== ユーティリティ =====

def _jd_range(yr: int, mth: int, dy: int, nof_days: int) -> Tuple[float, float]:
    jd0, fr0 = jday(yr, mth, dy, 0, 0, 0.0)
    jdf = jd0 + fr0 + nof_days
    return jd0 + fr0, jdf


def _time_vector_seconds(jd0: float, jdf: float, dt: int) -> np.ndarray:
    tf = (jdf - jd0) * 86400.0
    t = np.arange(0.0, tf + 1e-9, dt, dtype=float)
    return t


def _obs_epochs_from_time(jd0: float, tsec: np.ndarray) -> np.ndarray:
    return jd0 + tsec / 86400.0


def _select_objects(objects: List[TLEObject], selected_ids: List[int]) -> List[TLEObject]:
    out = []
    for nid in selected_ids:
        found = [o for o in objects if o.noradID == nid]
        if not found:
            raise ValueError(f"No TLEs found for object {nid}.")
        out.append(found[0])
    return out


def _build_RM(objects: List[TLEObject]) -> np.ndarray:
    """
    MATLAB の RM 構築を忠実再現。オブジェクトごとに 6×6 の対角を積み上げる。
    """
    blocks = []
    for obj in objects:
        ecco = float(obj.satrecs[0].ecco)
        RMfactor = max(ecco / 0.004, 1.0)
        blk = np.array(
            [
                max(4.0 * ecco, 0.0023),
                RMfactor * 3.0e-10,
                RMfactor * 3.0e-10,
                1.0e-9,
                1.0e-9,
                1.0e-8,
            ],
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
) -> Tuple[np.ndarray, List[float]]:
    """
    初期状態ベクトル x0g を構築（軌道6 + BC + ROM r）
    BC は Data/BCdata.txt の推定値を使用（m^2/kg）、MATLABと同じく x のスロットは [BC*1000]
    """
    nop = len(objects)
    x0g = np.zeros((svs * nop + r, 1), dtype=float)

    # 各物体の MEE 初期値（観測の最初の列）
    for i in range(nop):
        x0g[svs * i + 0 : svs * i + 6, 0] = mee_meas[6 * i : 6 * i + 6, 0]

    # BC 推定
    bc_used = []
    for i, obj in enumerate(objects):
        nid = int(obj.noradID)
        rows = BCdata[BCdata[:, 0] == nid]
        if rows.shape[0] == 0:
            raise ValueError(f"BC not found for NORAD {nid} in BCdata.")
        bc = float(rows[0, 1])  # m^2/kg
        x0g[svs * i + 6, 0] = bc * 1000.0  # MATLAB 同様、状態は [BC*1000]
        bc_used.append(bc)
    return x0g, bc_used


# ===== メイン関数 =====

def run_density_estimation_tle(
    par: EstimationInputs,
    *,
    # 互換性のため残します。省略時は generate_rom_density_model() を使います。
    rom_generator: Optional[Callable[[str, int, float, float], ROMBundle]] = None,
    # 追加: ROM 初期化ベクトル z0 を作る関数（Uh'*(log10 rho - mean) など）
    rom_initializer: Optional[Callable[[ROMBundle, float], np.ndarray]] = None,
    stateFnc: Optional[Callable[[np.ndarray, float, float], np.ndarray]] = None,
    measurementFcn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Dict[str, object]:
    """
    MATLAB runDensityEstimationTLE(...) の忠実ポート。
    既に実装済みの部分は実行、ROM生成・状態遷移はフックを通して差し込めます。

    Parameters
    ----------
    rom_generator : (ROMmodel, r, jd0, jdf) -> ROMBundle
        指定がない場合は densityestimation.models.rom_model.generate_rom_density_model を使用
    rom_initializer : (rom, jd0) -> z0 (shape=(r,))
        例: JB2008 でグリッド密度を作って Uh'*(log10(rho)-Dens_Mean) を返す実装を渡す
    """
    # --- 日時・EOP ---
    jd0, jdf = _jd_range(par.yr, par.mth, par.dy, par.nof_days)
    load_eop_celestrak(par.eop_path)

    # --- TLE ---
    # ※ MATLAB と同様に少し広い収集窓にしたい場合はここを拡張
    objects = get_tles_for_estimation(
        start_year=par.yr,
        start_month=par.mth,
        start_day=1,
        end_year=par.yr,
        end_month=par.mth,
        end_day=par.dy + par.nof_days + 30,  # ゆとり
        selected_objects=par.selected_objects,
        get_tles_from_single_file=par.tle_single_file,
        relative_dir=par.tle_dir,
    )
    # 選択ID順に並べ替え（MATLABと同じ並び前提で後続を組む）
    objects = _select_objects(objects, par.selected_objects)

    # --- 観測エポック・観測生成 ---
    tsec = _time_vector_seconds(jd0, jdf, par.dt_seconds)  # [0:dt:tf]
    obs_epochs = _obs_epochs_from_time(jd0, tsec)  # JD 配列
    mee_meas = generate_observations_mee(
        objects, obs_epochs, GM_EARTH_KM3_S2_ACCURATE
    )

    # --- BC 読み込み・初期状態組み立て ---
    BCdata = load_bc_data(par.bc_path)
    svs = 7  # 6(MEE) + 1(BC)
    x0g, bc_used = _assemble_x0(objects, mee_meas, BCdata, par.r, svs)

    # --- ROM 生成（学習済みモデル読み込み等） ---
    rom: Optional[ROMBundle] = None
    if rom_generator is not None:
        rom = rom_generator(par.ROMmodel, par.r, jd0, jdf)
    else:
        # 統合版ジェネレータを直接使用（JB2008/NRLMSISE/TIEGCM に対応）
        rom = generate_rom_density_model(par.ROMmodel, par.r, jd0, jdf)

    # ROM 初期化（任意）：MATLAB の z0_M 相当を反映
    if rom is not None and rom_initializer is not None:
        z0 = np.asarray(rom_initializer(rom, jd0), dtype=float).reshape(-1)
        if z0.size != par.r:
            raise ValueError(
                f"rom_initializer returned size {z0.size}, expected r={par.r}"
            )
        x0g[-par.r :, 0] = z0

    # --- 測定ノイズ RM ---
    RM = _build_RM(objects)

    # --- 初期状態共分散 P とプロセス雑音 Q ---
    nop = len(objects)
    Pv = np.zeros((svs * nop + par.r,), dtype=float)
    Qv = np.zeros_like(Pv)

    # 軌道6要素：測定ノイズを初期分散に流用
    for i in range(nop):
        RMblk = np.diag(RM)[6 * i : 6 * i + 6]
        Pv[svs * i + 0 : svs * i + 6] = RMblk

        # BC 初期分散（1%）
        Pv[svs * i + 6] = (x0g[svs * i + 6, 0] * 0.01) ** 2

        # プロセス雑音：MATLAB 値を踏襲
        Qv[svs * i + 0] = 1.5e-8
        Qv[svs * i + 1] = 2.0e-14
        Qv[svs * i + 2] = 2.0e-14
        Qv[svs * i + 3] = 1.0e-14
        Qv[svs * i + 4] = 1.0e-14
        Qv[svs * i + 5] = 1.0e-12
        Qv[svs * i + 6] = 1.0e-16  # BC

    # ROM 部分
    if rom is not None and getattr(rom, "Qrom", None) is not None:
        # Pv: 初期 ROM 分散（MATLAB値）
        Pv[-par.r :] = 5.0
        Pv[-par.r] = 20.0  # first mode
        # Qv: ROM 1時間予測誤差の共分散
        Qrom = np.array(rom.Qrom)
        if Qrom.ndim == 1:
            Qv[-par.r :] = Qrom[: par.r]
        else:
            Qv[-par.r :] = np.diag(Qrom)[: par.r]
    else:
        # ROM 未接続なら暫定（小さく置く or 0）
        Pv[-par.r :] = 5.0
        Pv[-par.r] = 20.0
        Qv[-par.r :] = 1e-6

    P = np.diag(Pv)
    Q = np.diag(Qv)

    # --- UKF（stateFnc / measurementFcn が指定されたときのみ実行） ---
    X_est_hist = np.zeros((svs * nop + par.r, tsec.size), dtype=float)
    X_est_hist[:, [0]] = x0g
    Pv_hist = np.zeros_like(X_est_hist)
    Pv_hist[:, 0] = np.diag(P)

    if stateFnc is not None and measurementFcn is not None:
        X_est_hist, Pv_hist = ukf(
            X_est=X_est_hist,
            Meas=mee_meas,
            time=tsec,
            stateFnc=stateFnc,
            measurementFcn=measurementFcn,
            P=P,
            RM=RM,
            Q=Q,
            angle_block=6,  # MEEのL（6,12,18,...行）を wrap
        )

    return dict(
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
        X_est=X_est_hist,
        Pv=Pv_hist,
        BC_used=bc_used,
        rom=rom,
    )
