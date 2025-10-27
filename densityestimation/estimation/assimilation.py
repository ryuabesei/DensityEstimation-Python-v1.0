# Python/densityestimation/estimation/assimilation.py
from __future__ import annotations

import numpy as np

from densityestimation.estimation.measurements import (
    make_measurement_model,
    make_measurement_noise_builder,
)
from densityestimation.estimation.observations import EstimationStateSpec, pack_state
from densityestimation.estimation.pipeline import make_process_model
from densityestimation.estimation.run_density_estimation_tle import (
    run_density_estimation_tle,
)
from densityestimation.tle.sgp4_wrapper import convert_teme_to_j2000
from densityestimation.ukf.srukf import SRUKF

GM_EARTH_KM3_S2 = 398600.4418

def _make_tle_provider(objects, epochs_jd):
    """nearest-newerを優先し、観測時刻tへ後方/前方伝播して (r,v,mu) を返す。"""
    sets = []
    for obj in objects:
        satrecs = list(getattr(obj, "satrecs"))
        tle_epochs = np.array([float(s.jdsatepoch) for s in satrecs])
        order = np.argsort(tle_epochs)
        sets.append(([satrecs[k] for k in order], tle_epochs[order]))

    def _pick_idx(tle_epochs, t_jd):
        newer = np.where(tle_epochs >= t_jd)[0]
        if newer.size > 0:
            return int(newer[0])
        older = np.where(tle_epochs < t_jd)[0]
        if older.size > 0:
            return int(older[-1])
        return int(np.argmin(np.abs(tle_epochs - t_jd)))

    def provider(i_obj: int, t_epoch_jd: float):
        satrecs_i, tle_epochs_i = sets[i_obj]
        k = _pick_idx(tle_epochs_i, float(t_epoch_jd))
        tsince_min = (float(t_epoch_jd) - float(satrecs_i[k].jdsatepoch)) * 24.0 * 60.0
        err, r_teme, v_teme = satrecs_i[k].sgp4_tsince(tsince_min)
        if err != 0:
            raise RuntimeError(f"SGP4 error={err} at obj {i_obj}")
        r, v = convert_teme_to_j2000(np.asarray(r_teme, float), np.asarray(v_teme, float), float(t_epoch_jd))
        return r, v, GM_EARTH_KM3_S2
    return provider

def _block_diag(*arrs):
    arrs = [np.atleast_2d(np.asarray(a, float)) for a in arrs]
    m = sum(a.shape[0] for a in arrs); n = sum(a.shape[1] for a in arrs)
    out = np.zeros((m, n), float)
    r = c = 0
    for a in arrs:
        rr, cc = a.shape
        out[r:r+rr, c:c+cc] = a
        r += rr; c += cc
    return out

def run_assimilation(cfg, rom_runtime, grid_interp_fn):
    # 1) 前段（EOP, TLE, obs, mee）
    pre = run_density_estimation_tle(cfg)
    objects = pre["objects"]
    epochs  = pre["obs_epochs"]   # ← 修正
    mee_obs = pre["mee_meas"]     # ← 修正
    bc0     = pre["BC_used"]      # ← 修正

    # spec は戻り値を使いつつ、ROM次元だけ同期（安全のため）
    spec: EstimationStateSpec = pre["spec"]
    spec = EstimationStateSpec(n_obj=spec.n_obj, nz=rom_runtime.r)

    n_obj = spec.n_obj

    # 2) 初期状態 x0
    mee0 = [mee_obs[6*i:6*(i+1), 0] for i in range(n_obj)]
    z0   = np.zeros(rom_runtime.r)
    x0   = pack_state(mee0, bc0, z0)

    # 3) 初期共分散 P0（論文準拠の例）
    R0_blocks = []
    for i in range(n_obj):
        f, g = float(mee0[i][1]), float(mee0[i][2])
        e = float(np.hypot(f, g))
        c1 = 1.5 * max(4.0*e, 0.0023); c2 = 3.0 * max(e/0.004, 1.0)
        R0_blocks.append(np.diag([c1*1e-8, c2*1e-10, c2*1e-10, 1e-9, 1e-9, 1e-8]))
    P_mee = _block_diag(*R0_blocks)
    P_bc  = np.diag([(0.01*b)**2 for b in bc0])
    P_z   = np.diag([20.0] + [5.0]*(rom_runtime.r-1)) if rom_runtime.r > 0 else np.zeros((0,0))
    P0    = _block_diag(P_mee, P_bc, P_z)

    # 4) プロセス雑音 Q
    Q_mee = np.diag(([1.5e-8, 2e-14, 2e-14, 1e-14, 1e-14, 1e-12] * n_obj))
    Q_bc  = np.diag([1e-16]*n_obj)
    Q_z   = np.diag(rom_runtime.Qz) if getattr(rom_runtime, "Qz", None) is not None and rom_runtime.r > 0 else np.zeros((0,0))
    Q     = _block_diag(Q_mee, Q_bc, Q_z)

    # 5) f, h, R_builder
    f = make_process_model(rom_runtime, spec, grid_interp_fn, dt_sec=cfg.dt_seconds if hasattr(cfg, "dt_seconds") else 3600.0)
    tle_provider = _make_tle_provider(objects, epochs)
    h = make_measurement_model(spec, tle_provider)
    R_builder = make_measurement_noise_builder(spec)

    # 6) UKF 構築（srukf.py のI/Fに合わせて微調整してください）
    ukf = SRUKF(f=f, h=h, Q=Q, R_builder=R_builder, x0=x0, P0=P0)

    # 7) 逐次同化
    Xs = []; Ps = []
    for t in epochs:
        # あなたの SRUKF 実装が step(...) を想定している場合は下の2行を置き換えてください
        x_pred, P_pred = ukf.predict(t)
        z_meas = h(x_pred, t)      # TLEからの観測
        x_upd, P_upd = ukf.update(z_meas, t)  # R は内部で R_builder(x_pred)
        Xs.append(x_upd); Ps.append(P_upd)

    return np.column_stack(Xs), Ps
