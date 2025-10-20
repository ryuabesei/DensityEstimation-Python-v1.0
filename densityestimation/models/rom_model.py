# Origin: generateROMdensityModel.m
# License: GNU GPL v3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.interpolate import RegularGridInterpolator
    from scipy.io import loadmat
except Exception as e:
    raise RuntimeError("scipy が必要です（scipy.io, scipy.interpolate）。requirements.txt に追加してください。") from e


@dataclass
class ROMBundle:
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


def _mat_path_for(rom_model: str) -> str:
    if rom_model == "JB2008_1999_2010":
        return "JB2008_1999_2010_ROM_r100.mat"
    if rom_model == "TIEGCM_1997_2008":
        return "TIEGCM_1997_2008_ROM_r100.mat"
    if rom_model == "NRLMSISE_1997_2008":
        return "NRLMSISE_1997_2008_ROM_r100.mat"
    raise ValueError(f"Unknown ROMmodel: {rom_model}")


def _as_1d(a) -> np.ndarray:
    arr = np.array(a).squeeze()
    return arr.astype(float)


def _interp3_linear(xg: np.ndarray, yg: np.ndarray, zg: np.ndarray, V: np.ndarray):
    """MATLAB griddedInterpolant(...,'linear','linear') 相当（線形補間・線形外挿）。"""
    rgi = RegularGridInterpolator(
        (xg, yg, zg), V, bounds_error=False, fill_value=None, method="linear"
    )

    def f(slt: np.ndarray, lat: np.ndarray, alt: np.ndarray) -> np.ndarray:
        pts = np.stack([np.asarray(slt).ravel(), np.asarray(lat).ravel(), np.asarray(alt).ravel()], axis=1)
        out = rgi(pts)
        return out.reshape(np.broadcast_shapes(np.shape(slt), np.shape(lat), np.shape(alt)))
    return f


def _build_from_TA(TA: dict, r: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # 必須フィールド（MATLAB の TA）
    sltm = _as_1d(TA.get("localSolarTimes"))
    latm = _as_1d(TA.get("latitudes"))
    altm = _as_1d(TA.get("altitudes"))
    Dens_Mean = _as_1d(TA.get("densityDataMeanLog"))

    # Uh, PhiC, Qrom は .mat に直接入っているケースと、別関数生成のケースがある
    Uh = np.array(TA.get("Uh") if "Uh" in TA else TA.get("U", None))
    PhiC = np.array(TA.get("PhiC")) if "PhiC" in TA else None
    Qrom = np.array(TA.get("Qrom")) if "Qrom" in TA else None

    if Uh is None or Uh.size == 0:
        raise ValueError("TA から Uh を取得できませんでした（'Uh' が見つかりません）。")
    if PhiC is None or Qrom is None:
        # ここで generateROM_* に相当する処理が必要ですが、未提供なので停止します
        raise NotImplementedError("PhiC / Qrom が見つかりません。generateROM_* 等の出力を含む .mat を使うか、生成関数を実装してください。")

    # r 次に切り詰め
    if Uh.shape[1] < r:
        raise ValueError(f"Uh の列数 {Uh.shape[1]} < r({r})")
    return sltm, latm, altm, Dens_Mean, Uh[:, :r], PhiC


def generate_rom_density_model(
    rom_model: str,
    r: int,
    jd0: float,
    jdf: float,
    *,
    rom_mat_dir: str = ".",
    compute_swinputs: Optional[Callable[[float, float], Dict[str, np.ndarray]]] = None,
) -> ROMBundle:
    """
    MATLAB: [AC,BC,Uh,F_U,Dens_Mean,M_U,SLTm,LATm,ALTm,maxAtmAlt,SWinputs,Qrom]
    """
    mat_path = Path(rom_mat_dir) / _mat_path_for(rom_model)
    if not mat_path.exists():
        raise FileNotFoundError(f"ROM .mat が見つかりません: {mat_path}")

    TA = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)

    sltm, latm, altm, Dens_Mean, Uh, PhiC = _build_from_TA(TA, r)

    # Qrom（ROM 1時間予測誤差の共分散）
    Qrom = np.array(TA.get("Qrom"))
    if Qrom is None or Qrom.size == 0:
        raise NotImplementedError("Qrom が .mat にありません。MATLAB 側で保存して再出力するか、生成してください。")
    Qrom = np.array(Qrom).squeeze()
    if Qrom.ndim == 2 and Qrom.shape[0] != r:
        # 大きいサイズなら先頭 r×r を使用
        Qrom = Qrom[:r, :r]

    # 動的行列（連続時間）→ 1時間刻み扱いなので /3600 は MATLAB と同じ
    AC = (PhiC[:r, :r] / 3600.0).astype(float)
    BC = (PhiC[:r, r:] / 3600.0).astype(float)

    # 3D グリッド
    SLTm, LATm, ALTm = np.meshgrid(sltm, latm, altm, indexing="ij")

    # 各基底の補間関数 F_U{i}
    n_slt, n_lat, n_alt = SLTm.shape
    F_U: List[Callable] = []
    for i in range(r):
        Uhr = Uh[:, i].reshape(n_slt, n_lat, n_alt, order="F") if Uh.size == n_slt * n_lat * n_alt else Uh[:, i].reshape(n_slt, n_lat, n_alt)
        F_U.append(_interp3_linear(sltm, latm, altm, Uhr))

    # 平均 log10 密度の補間
    Mr = Dens_Mean.reshape(n_slt, n_lat, n_alt, order="F") if Dens_Mean.size == n_slt * n_lat * n_alt else Dens_Mean.reshape(n_slt, n_lat, n_alt)
    M_U = _interp3_linear(sltm, latm, altm, Mr)

    # モデル高さ上限
    maxAtmAlt = 800.0 if rom_model in ("JB2008_1999_2010", "NRLMSISE_1997_2008") else 500.0

    # 宇宙天気入力（未実装なら空辞書）
    SWinputs: Dict[str, np.ndarray] = compute_swinputs(jd0, jdf + 1.0) if compute_swinputs else {}

    return ROMBundle(
        AC=AC, BC=BC, Uh=Uh, F_U=F_U, Dens_Mean=Dens_Mean, M_U=M_U,
        SLTm=SLTm, LATm=LATm, ALTm=ALTm, maxAtmAlt=maxAtmAlt, SWinputs=SWinputs, Qrom=Qrom
    )
