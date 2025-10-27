from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple, Union

import numpy as np

from densityestimation.orbit.mee import pv2ep  # MATLABの pv2ep.m を1:1移植予定
from densityestimation.tle.sgp4_wrapper import convert_teme_to_j2000


def _as_satrecs(obj: Union[Dict[str, Any], Any]) -> List[Any]:
    """objects[i] が dict でもクラスでも satrecs を取り出せるように統一。"""
    if isinstance(obj, dict):
        sr = obj.get("satrecs")
    else:
        sr = getattr(obj, "satrecs", None)
    if sr is None:
        raise AttributeError("object に 'satrecs' がありません。")
    return list(sr)


def _sorted_by_epoch(satrecs: List[Any]) -> Tuple[List[Any], np.ndarray]:
    """satrec を jdsatepoch 昇順に整列して返す。"""
    tle_epochs = np.array([float(s.jdsatepoch) for s in satrecs], dtype=float)
    order = np.argsort(tle_epochs)
    satrecs_sorted = [satrecs[k] for k in order]
    return satrecs_sorted, tle_epochs[order]


def _pick_tle_index_nearest_newer_first(tle_epochs: np.ndarray, obs_epoch_jd: float) -> int:
    """
    ルール:
      1) 観測時刻 t より新しい/等しい TLE（tle_epoch >= t）の中で最も近いもの（nearest newer）を優先。
      2) もし 1) が無ければ、最も近い過去（tle_epoch < t）の中で最新を採用。
      3) それすら無ければ、絶対差が最小のもの（保険）。
    """
    newer = np.where(tle_epochs >= obs_epoch_jd)[0]
    if newer.size > 0:
        return int(newer[0])  # tle_epochs は昇順なので最初が最も近い新しいTLE
    older = np.where(tle_epochs < obs_epoch_jd)[0]
    if older.size > 0:
        return int(older[-1])  # 最も近い過去
    return int(np.argmin(np.abs(tle_epochs - obs_epoch_jd)))


def generate_observations_mee(objects: Iterable[Union[Dict[str, Any], Any]],
                              obs_epochs: Iterable[float],
                              GM_kms: float) -> np.ndarray:
    """
    各観測時刻 t に対して:
      - 「t 以上」の TLE を優先して選択（nearest newer）
      - その TLE epoch から t まで SGP4 で後方/前方伝播（tsince [min] は負/正どちらも可）
      - TEME → J2000 (ECI) に変換し、pv2ep で (p,f,g,h,k,L) を得る

    Parameters
    ----------
    objects : iterable
        各要素は .satrecs を持つオブジェクト or {"satrecs": [...]}辞書。
        satrec は sgp4.api.Satrec と互換（.jdsatepoch, .sgp4_tsince）。
    obs_epochs : iterable of float
        観測ジュリアン日 (JD)。convert_teme_to_j2000 も JD 想定。
    GM_kms : float
        万有引力定数 μ [km^3/s^2]。

    Returns
    -------
    meeObs : np.ndarray, shape (6 * n_obj, n_obs)
        縦に (p,f,g,h,k,L) を物体ごとに積み上げ、列が観測時刻。
    """
    objects = list(objects)
    obs_epochs = list(obs_epochs)

    n_obj = len(objects)
    n_obs = len(obs_epochs)
    meeObs = np.zeros((6 * n_obj, n_obs), dtype=float)

    # 事前に各オブジェクトの TLE を整列しておく
    satrec_sets: List[Tuple[List[Any], np.ndarray]] = []
    for i in range(n_obj):
        satrecs = _as_satrecs(objects[i])
        if len(satrecs) == 0:
            raise ValueError(f"objects[{i}].satrecs が空です。")
        satrec_sets.append(_sorted_by_epoch(satrecs))

    for j, obs_epoch in enumerate(obs_epochs):
        t_jd = float(obs_epoch)
        for i in range(n_obj):
            satrecs_i, tle_epochs_i = satrec_sets[i]
            idx = _pick_tle_index_nearest_newer_first(tle_epochs_i, t_jd)

            # tsince [min]：観測時刻 − TLE epoch
            # newer（>=）を選ぶと多くは負の値（後方伝播）になる
            diff_minutes = (t_jd - float(satrecs_i[idx].jdsatepoch)) * 24.0 * 60.0

            err_code, r_teme, v_teme = satrecs_i[idx].sgp4_tsince(float(diff_minutes))
            if err_code != 0:
                raise RuntimeError(
                    f"SGP4 error code={err_code} (obj {i}, obs {j}, tsince[min]={diff_minutes:.3f})"
                )

            r_teme = np.asarray(r_teme, dtype=float)
            v_teme = np.asarray(v_teme, dtype=float)

            r_j2000, v_j2000 = convert_teme_to_j2000(r_teme, v_teme, t_jd)
            mee = pv2ep(r_j2000, v_j2000, GM_kms)
            meeObs[6 * i:6 * i + 6, j] = np.asarray(mee, dtype=float).reshape(6,)

    return meeObs


@dataclass
class EstimationStateSpec:
    n_obj: int  # 同化に使う物体数
    nz: int     # ROM状態次元 (例: r=10)


def pack_state(mee_list: Iterable[Tuple[float, float, float, float, float, float]],
               bc_list: Iterable[float],
               z: np.ndarray) -> np.ndarray:
    """
    mee_list: list of (p,f,g,h,k,L) for each object
    bc_list : list of BC (m^2/kg) for each object
    z       : (nz,) ROM reduced state
    return  : (state_vec,)
    """
    mee_list = list(mee_list)
    bc_list = list(bc_list)
    if len(mee_list) != len(bc_list):
        raise ValueError("mee_list と bc_list の長さが一致しません。")

    parts: List[float] = []
    for (p, f, g, h, k, L), BC in zip(mee_list, bc_list):
        parts.extend([float(p), float(f), float(g), float(h), float(k), float(L), float(BC)])
    parts.extend(np.asarray(z, dtype=float).ravel().tolist())
    return np.array(parts, dtype=float)


def unpack_state(x: np.ndarray, spec: EstimationStateSpec):
    n = int(spec.n_obj)
    nz = int(spec.nz)
    x = np.asarray(x, dtype=float).ravel()

    expect_len = 7 * n + nz
    if x.size < expect_len:
        raise ValueError(f"状態ベクトルの長さが不足しています (got={x.size}, expect={expect_len})")

    mee_list, bc_list = [], []
    idx = 0
    for _ in range(n):
        p, f, g, h, k, L, BC = x[idx:idx + 7]
        mee_list.append((p, f, g, h, k, L))
        bc_list.append(BC)
        idx += 7
    z = x[idx:idx + nz]
    return mee_list, bc_list, z
