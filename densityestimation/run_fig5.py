# densityestimation/run_fig5.py
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# ---- JB2008 密度ラッパ ----
from densityestimation.Astrofunction.jb2008_density import get_density_jb2008_llajd
from densityestimation.Astrofunction.time_utils import gstime

# ---- EOP & 観測準備 ----
from densityestimation.data.eop_loader import load_eop_celestrak
from densityestimation.estimation.run_density_estimation_tle import (
    GM_EARTH_KM3_S2_ACCURATE,
    EstimationInputs,
    run_density_estimation_tle,
)
from densityestimation.orbit.mee import mee_to_cartesian

# ---- NRLMSISE-00 SW ローダ（将来接続用）----
from densityestimation.spaceweather.nrlmsise_loader import input_sw_nrlmsise

# ========= 可搬なリゾルバ =========

def _resolve_jb2008_callable() -> Callable:
    module_candidates = [
        "densityestimation.models.JB2008.jb2008model",  # 大文字
        "densityestimation.models.jb2008.jb2008model",  # 小文字
    ]
    name_candidates = ["JB2008", "jb2008", "jb2008model", "model"]
    last_err = None
    for modname in module_candidates:
        try:
            mod = __import__(modname, fromlist=["*"])
        except Exception as e:
            last_err = e
            continue
        for name in name_candidates:
            fn = getattr(mod, name, None)
            if callable(fn):
                return fn
    raise ImportError(
        "JB2008 callable が見つかりませんでした。"
        f" 探索モジュール: {module_candidates} / 関数候補: {name_candidates}"
        + (f" / 最後のインポート例外: {last_err}" if last_err else "")
    )


def _resolve_compute_swinputs():
    from importlib import import_module
    mod = import_module("densityestimation.spaceweather.jb2008_inputs")
    candidates = [
        "compute_jb2000_swinputs",
        "compute_jb2008_swinputs",
        "compute_jb2008_inputs",
        "compute_jb2000_inputs",
        "compute_jb2000_sw_inputs",  # 実装名
    ]
    for name in candidates:
        fn = getattr(mod, name, None)
        if callable(fn):
            print(f"[INFO] Using JB2008 SW input function: {mod.__name__}.{name}")
            return fn
    raise ImportError(
        "JB2008 宇宙天気入力関数が見つかりませんでした。"
        f" 探索名: {candidates} / モジュール: {mod.__file__}"
    )


def _normalize_sol_dtc(ret) -> Tuple[np.ndarray, np.ndarray]:
    """
    返り値が (sol, dtc) / (eop, sol, dtc) / dict のどれでも正規化。
    """
    if isinstance(ret, tuple):
        if len(ret) == 2:
            sol, dtc = ret
        elif len(ret) >= 3:
            # 末尾2要素を SOL, DTC とみなす（(eop, SOL, DTC)想定）
            sol, dtc = ret[-2], ret[-1]
        else:
            raise ValueError("tuple ですが要素数が不足しています。")
    elif isinstance(ret, dict):
        keys_sol = ["SOL", "sol", "SOLFSMY", "solfsmy"]
        keys_dtc = ["DTC", "dtc", "DTCFILE", "dtcfile"]
        sol = next((ret[k] for k in keys_sol if k in ret), None)
        dtc = next((ret[k] for k in keys_dtc if k in ret), None)
        if sol is None or dtc is None:
            raise ValueError("SWローダが dict を返しましたが、SOL/DTC のキーが見つかりません。")
    else:
        raise ValueError("SWローダの戻り値を解釈できません。(tuple/dict を返してください)")

    return np.asarray(sol, float), np.asarray(dtc, float)


def _resolve_sw_loaders():
    """
    SOLFSMY.txt / DTCFILE.txt のローダを load_jb2008_swdata.py から解決。
    - 一括ローダ: load_jb2008_swdata など（引数0 または 2）
    - 個別ローダ: load_solf_smy / load_dtc_file（引数0 または 1）
    どの形でも使えるように、ここで“bulk”関数にラップして返す。
    """
    from importlib import import_module

    mod = import_module("densityestimation.spaceweather.load_jb2008_swdata")

    # 1) 一括ローダ候補
    bulk_names = ["load_jb2008_swdata", "load_swdata_jb2008", "load_swdata"]
    for nm in bulk_names:
        fn = getattr(mod, nm, None)
        if callable(fn):
            sig = inspect.signature(fn)
            params = list(sig.parameters.values())

            def bulk(sol_path: str, dtc_path: str, fn=fn, params=params):
                # 引数数で分岐
                if len(params) == 0:
                    ret = fn()
                    return _normalize_sol_dtc(ret)
                elif len(params) >= 2:
                    ret = fn()
                    return _normalize_sol_dtc(ret)
                else:
                    # 1引数だけ等の特殊系は、sol_path を渡してみて駄目なら引数なし
                    try:
                        ret = fn(sol_path)
                    except TypeError:
                        ret = fn()
                    return _normalize_sol_dtc(ret)

            return bulk, None, None

    # 2) 個別ローダ候補
    sol_names = ["load_solf_smy", "load_solfsmy", "read_solf_smy", "read_solfsmy"]
    dtc_names = ["load_dtc_file", "read_dtc_file", "load_dtcfile", "read_dtcfile"]
    sol_fn = next((getattr(mod, nm) for nm in sol_names if callable(getattr(mod, nm, None))), None)
    dtc_fn = next((getattr(mod, nm) for nm in dtc_names if callable(getattr(mod, nm, None))), None)

    if sol_fn and dtc_fn:
        sig_sol = inspect.signature(sol_fn)
        sig_dtc = inspect.signature(dtc_fn)
        p_sol = list(sig_sol.parameters.values())
        p_dtc = list(sig_dtc.parameters.values())

        def bulk(sol_path: str, dtc_path: str,
                 sol_fn=sol_fn, dtc_fn=dtc_fn, p_sol=p_sol, p_dtc=p_dtc):
            # sol
            if len(p_sol) == 0:
                sol = sol_fn()
            else:
                sol = sol_fn(sol_path)
            # dtc
            if len(p_dtc) == 0:
                dtc = dtc_fn()
            else:
                dtc = dtc_fn(dtc_path)
            return np.asarray(sol, float), np.asarray(dtc, float)

        return bulk, sol_fn, dtc_fn

    raise ImportError(
        "SWローダ関数が見つかりませんでした。"
        " load_jb2008_swdata.py に load_jb2008_swdata()（引数0/2いずれか）または "
        "load_solf_smy()/load_dtc_file()（引数0/1いずれか）を用意してください。"
    )


# ========= 変換ユーティリティ =========

def _jd_to_day_of_year_2002(jd: np.ndarray) -> np.ndarray:
    JD_2002_0101 = 2452275.5
    return (np.asarray(jd, float) - JD_2002_0101) + 1.0


def _eci_to_ecef_r_gmst(r_eci_km: np.ndarray, jd_ut1: float) -> np.ndarray:
    theta = gstime(jd_ut1)  # [rad]
    c, s = np.cos(theta), np.sin(theta)
    R3 = np.array([[ c,  s, 0.0],
                   [-s,  c, 0.0],
                   [0.0, 0.0, 1.0]], dtype=float)
    return R3 @ np.asarray(r_eci_km, float)


def _ecef_to_geodetic_rough(r_ecef_km: np.ndarray) -> Tuple[float, float, float]:
    x, y, z = map(float, r_ecef_km)
    lon = np.degrees(np.arctan2(y, x))
    rxy = np.hypot(x, y)
    lat = np.degrees(np.arctan2(z, rxy))
    alt = float(np.linalg.norm(r_ecef_km) - 6378.137)
    return lat, lon, alt


def _exp_rho_from_alt_km(alt_km: float) -> float:
    rho0 = 3.0e-12  # @400 km
    H = 50.0
    return rho0 * np.exp(-(alt_km - 400.0) / H)


# ========= 物理モデル呼び出し =========

def _rho_jb2008_true(jd_utc: float, r_eci_km: np.ndarray,
                     eopdata: np.ndarray, solfsmy: np.ndarray, dtcfile: np.ndarray,
                     jb2008_model: Callable, compute_swinputs: Callable) -> float:
    r_ecef = _eci_to_ecef_r_gmst(r_eci_km, jd_utc)
    lat, lon, alt = _ecef_to_geodetic_rough(r_ecef)

    rho_kg_km3 = get_density_jb2008_llajd(
        lon_deg=lon, lat_deg=lat, alt_km=alt, jdate=jd_utc,
        compute_jb2000_swinputs=compute_swinputs,
        jb2008_model=jb2008_model,
        eopdata=eopdata, SOLdata=solfsmy, DTCdata=dtcfile
    )
    return float(rho_kg_km3) / 1e9


def _rho_msis00(jd_utc: float, r_eci_km: np.ndarray, sw_daily: Optional[np.ndarray]) -> float:
    r_ecef = _eci_to_ecef_r_gmst(r_eci_km, jd_utc)
    lat, lon, alt = _ecef_to_geodetic_rough(r_ecef)
    return _exp_rho_from_alt_km(alt)


# ========= 設定と実行 =========

@dataclass
class Fig5Config:
    yr: int = 2002
    mth: int = 8
    dy: int = 1
    nof_days: int = 10
    ROMmodel: str = "JB2008_1999_2010"
    r: int = 10
    selected_objects: List[int] = field(default_factory=lambda: [27391])  # CHAMP
    plot_figures: bool = True
    eop_path: str = "densityestimation/data/EOP-All.txt"
    tle_dir: str = "TLEdata"
    tle_single_file: bool = True
    bc_path: str = "densityestimation/data/BCdata.txt"
    model: str = "JB2008"  # "JB2008" / "NRLMSISE00" / "EXP"


def run_fig5(cfg: Fig5Config) -> Dict[str, object]:
    # 1) 観測準備（TLE→MEE）
    par = EstimationInputs(
        yr=cfg.yr, mth=cfg.mth, dy=cfg.dy, nof_days=cfg.nof_days,
        ROMmodel=cfg.ROMmodel, r=cfg.r, selected_objects=cfg.selected_objects,
        plot_figures=False, eop_path=cfg.eop_path, tle_dir=cfg.tle_dir,
        tle_single_file=cfg.tle_single_file, bc_path=cfg.bc_path, dt_seconds=3600,
    )
    pre = run_density_estimation_tle(par)
    jd = np.asarray(pre["obs_epochs"], float)  # UTC JD
    mee_meas = np.asarray(pre["mee_meas"], float)
    days = _jd_to_day_of_year_2002(jd)

    # 2) JB2008 用の補助データ（ローダは引数0/2どちらでもOK）
    eop_mat = load_eop_celestrak(cfg.eop_path, full=True)
    bulk_loader, _, _ = _resolve_sw_loaders()
    solfsmy, dtcfile = bulk_loader(
        "densityestimation/data/SOLFSMY.txt",
        "densityestimation/data/DTCFILE.txt",
    )

    if solfsmy.shape[0] < 11:
        raise ValueError(
            f"SOLFSMY matrix has too few rows: {solfsmy.shape}. "
            "Expected >= 11 rows (rows are variables, cols are dates)."
        )
    if dtcfile.shape[0] < 26:
        raise ValueError(
            f"DTCFILE matrix has too few rows: {dtcfile.shape}. "
            "Expected >= 26 rows (rows: [year,DOY,h0..h23])."
        )

    # NRLMSISE-00 の SW（未使用なら None）
    try:
        sw_daily, _ = input_sw_nrlmsise("densityestimation/data/SW-All.txt")
    except Exception:
        sw_daily = None

    # 3) JB2008 の関数と SW 入力関数を解決
    jb2008_model = _resolve_jb2008_callable()
    compute_swinputs = _resolve_compute_swinputs()

    # 4) 各時刻で密度を評価
    n = jd.size
    rho_main = np.zeros(n)
    rho_alt  = np.zeros(n)
    rho_obs  = np.zeros(n)

    use = cfg.model.upper()
    for i in range(n):
        mee_i = mee_meas[0:6, i]  # 代表1衛星
        r_i, _v_i, _mu = mee_to_cartesian(mee_i, GM_EARTH_KM3_S2_ACCURATE)

        if use == "JB2008":
            rho_main[i] = _rho_jb2008_true(
                jd[i], r_i, eop_mat, solfsmy, dtcfile, jb2008_model, compute_swinputs
            )
            rho_alt[i]  = _rho_msis00(jd[i], r_i, sw_daily)  # 比較
            rho_obs[i]  = rho_main[i]                        # 将来: 観測密度に置換

        elif use == "NRLMSISE00":
            rho_main[i] = _rho_msis00(jd[i], r_i, sw_daily)
            rho_alt[i]  = rho_main[i]
            rho_obs[i]  = rho_main[i]

        else:  # EXP
            alt_km = float(np.linalg.norm(r_i) - 6378.137)
            rho_main[i] = _exp_rho_from_alt_km(alt_km)
            rho_alt[i]  = rho_main[i]
            rho_obs[i]  = rho_main[i]

    # 5) プロット
    if cfg.plot_figures:
        fig = plt.figure(figsize=(9, 5.5), facecolor="white")
        ax = fig.add_subplot(111)
        label_main = {"JB2008": "JB2008", "NRLMSISE00": "NRLMSISE-00"}.get(use, "EXP")
        ax.plot(days, rho_main, label=label_main, linewidth=1.4)
        ax.plot(days, rho_obs,  label="CHAMP (placeholder)", linewidth=1.0)
        ax.plot(days, rho_alt,  label="MSIS00 (comp)", linewidth=1.0)
        ax.set_xlabel("Day of 2002")
        ax.set_ylabel("Density (kg/m$^3$)")
        ax.set_ylim(5e-13, 1e-11)
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig("fig5_density.png", dpi=150, facecolor="white")
        plt.show()

    return dict(days=days, rho_main=rho_main, rho_alt=rho_alt, rho_obs=rho_obs)


if __name__ == "__main__":
    cfg = Fig5Config()
    out = run_fig5(cfg)
