# Origin: generateObservationsMEE.m (Gondelach 2020; modified Li 2022)
# License: GNU GPL v3 (same as original)
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from densityestimation.orbit.mee import pv2ep  # MATLABの pv2ep.m を1:1移植予定

# 内部依存（同リポジトリ）
from densityestimation.tle.sgp4_wrapper import convert_teme_to_j2000


def generate_observations_mee(objects: Sequence[Any],
                              obs_epochs: Sequence[float],
                              GM_kms: float) -> np.ndarray:
    """
    Generate observations in Modified Equinoctial Elements (MEE) at specified epochs.

    Parameters
    ----------
    objects : sequence
        各要素は `.satrecs` を持つオブジェクト/辞書を想定。
        `.satrecs` は SGP4 の Satrec 相当（.jdsatepoch 属性あり）の配列。
        例: objects[i].satrecs[k].jdsatepoch (float, JD)
    obs_epochs : sequence of float
        観測エポックのユリウス日（JD）。MATLAB版と同じ単位。
    GM_kms : float
        万有引力定数 μ [km^3/s^2]。MATLAB版と同じ引数名/単位。

    Returns
    -------
    meeObs : (6*n_objects, n_epochs) ndarray
        各物体ごとに6成分のMEEを縦に積んだ行列（MATLABと同じ並び）。

    Notes
    -----
    - MATLABの `find([objects(i).satrecs.jdsatepoch]>=obsEpoch,1,'first')`
      と同じロジックで「観測時刻以降で最初のTLE」を選択します。
    - 伝播は SGP4 の `tsince` 分（分単位）で行い、フレームは TEME→J2000 へ変換。
    - `pv2ep` は MATLAB の実装をそのまま式移植する想定（現状はプレースホルダ）。
    """
    nof_objects = len(objects)
    nof_obs = len(obs_epochs)
    meeObs = np.zeros((6 * nof_objects, nof_obs), dtype=float)

    for i in range(nof_objects):
        satrecs = getattr(objects[i], "satrecs", None)
        if satrecs is None and isinstance(objects[i], dict):
            satrecs = objects[i].get("satrecs", None)
        if satrecs is None:
            raise AttributeError(f"objects[{i}] に 'satrecs' がありません。")

        # 各観測エポックでMEEを生成
        for j, obs_epoch in enumerate(obs_epochs):
            # 1) 観測時刻以上で最初のTLEインデックスを取得
            try:
                satrec_index = next(k for k, s in enumerate(satrecs) if s.jdsatepoch >= obs_epoch)
            except StopIteration:
                raise ValueError(
                    f"objects[{i}]: obs_epoch={obs_epoch} 以降の TLE が見つかりません。"
                )

            # 2) 観測エポックとTLEエポックの差（分）を計算
            diff_minutes = (obs_epoch - satrecs[satrec_index].jdsatepoch) * 24.0 * 60.0

            # 3) SGP4でTEME座標のr,vを取得（km, km/s）
            #    Python版は tsince[minutes] 指定のメソッドを使う
            e, r_teme, v_teme = satrecs[satrec_index].sgp4_tsince(float(diff_minutes))
            if e != 0:
                raise RuntimeError(f"SGP4 error code: {e} for object {i}, epoch index {j}.")

            # 4) TEME → J2000(=GCRF) へ変換
            r_j2000, v_j2000 = convert_teme_to_j2000(np.array(r_teme), np.array(v_teme), obs_epoch)

            # 5) r,v → MEE
            mee = pv2ep(r_j2000, v_j2000, GM_kms)

            # 6) 出力スロットに格納
            start = 6 * i
            meeObs[start:start + 6, j] = np.asarray(mee).reshape(6,)

    return meeObs
