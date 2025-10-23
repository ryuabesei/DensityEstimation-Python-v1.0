# Origin: generateObservationsMEE.m (Gondelach 2020; modified Li 2022)
# License: GNU GPL v3 (same as original)
# TLE→SGP4→TEME→J2000→MEE 観測生成
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from densityestimation.orbit.mee import pv2ep  # MATLABの pv2ep.m を1:1移植予定

# 内部依存（同リポジトリ）
from densityestimation.tle.sgp4_wrapper import convert_teme_to_j2000


def generate_observations_mee(objects, obs_epochs, GM_kms) -> np.ndarray:
    nof_objects = len(objects)
    nof_obs = len(obs_epochs)
    meeObs = np.zeros((6 * nof_objects, nof_obs), dtype=float)

    for i in range(nof_objects):
        satrecs = getattr(objects[i], "satrecs", None) or (objects[i].get("satrecs") if isinstance(objects[i], dict) else None)
        if satrecs is None:
            raise AttributeError(f"objects[{i}] に 'satrecs' がありません。")

        # TLE epoch を昇順の numpy 配列に（必要なら並び替え）
        tle_epochs = np.array([s.jdsatepoch for s in satrecs], dtype=float)
        order = np.argsort(tle_epochs)
        satrecs = [satrecs[k] for k in order]
        tle_epochs = tle_epochs[order]

        for j, obs_epoch in enumerate(obs_epochs):
            # ---- まず「観測時刻以前の最新 (<=)」を探す ----
            idx_prev = np.where(tle_epochs <= obs_epoch)[0]
            if idx_prev.size > 0:
                satrec_index = int(idx_prev[-1])
            else:
                # 無ければ「観測時刻以後の最初 (>=)」
                idx_next = np.where(tle_epochs >= obs_epoch)[0]
                if idx_next.size > 0:
                    satrec_index = int(idx_next[0])
                else:
                    # それも無ければ「最も近い方」
                    satrec_index = int(np.argmin(np.abs(tle_epochs - obs_epoch)))

            # 伝播時間 [min]（過去TLEなら正、未来TLEなら負もOK）
            diff_minutes = (obs_epoch - satrecs[satrec_index].jdsatepoch) * 24.0 * 60.0

            e, r_teme, v_teme = satrecs[satrec_index].sgp4_tsince(float(diff_minutes))
            if e != 0:
                raise RuntimeError(f"SGP4 error code: {e} for object {i}, epoch index {j}.")

            r_j2000, v_j2000 = convert_teme_to_j2000(np.array(r_teme), np.array(v_teme), obs_epoch)
            mee = pv2ep(r_j2000, v_j2000, GM_kms)
            meeObs[6 * i:6 * i + 6, j] = np.asarray(mee).reshape(6,)

    return meeObs




@dataclass
class EstimationStateSpec:
    n_obj: int     # 同化に使う物体数
    nz: int        # ROM状態次元 (例: r=10)

def pack_state(mee_list, bc_list, z):
    """
    mee_list: list of (p,f,g,h,k,L) for each object
    bc_list : list of BC (m^2/kg) for each object
    z       : (nz,) ROM reduced state
    return  : (state_vec,)
    """
    parts = []
    for (p,f,g,h,k,L), BC in zip(mee_list, bc_list):
        parts.extend([p,f,g,h,k,L,BC])
    parts.extend(list(z))
    return np.array(parts, dtype=float)

def unpack_state(x, spec: EstimationStateSpec):
    n = spec.n_obj
    nz = spec.nz
    mee_list, bc_list = [], []
    idx = 0
    for _ in range(n):
        p,f,g,h,k,L,BC = x[idx:idx+7]
        mee_list.append((p,f,g,h,k,L))
        bc_list.append(BC)
        idx += 7
    z = x[idx:idx+nz]
    return mee_list, bc_list, z