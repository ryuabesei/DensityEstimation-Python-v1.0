# densityestimation/Astrofunction/isdecayed.py
# GNU GPL v3
from __future__ import annotations

import numpy as np

from .altitude import altitude  # 既存: ECI位置[km]→高度[km]


def make_decay_event(noo: int, svs: int, threshold_km: float = 120.0):
    """
    SciPy solve_ivp のイベント用コールバックを返す。
    - noo: 物体数
    - svs: 1物体あたりの状態長 (ここでは 7: pos3+vel3+BC1)
    - threshold_km: 崩壊（大気圏再突入）判定の高度[km]
    """
    def event(t: float, x: np.ndarray) -> float:
        # x はフラットな状態ベクトル (長さ >= svs*noo)
        # 各物体の位置(3要素)から高度[km]を計算し、その最小値-120kmを返す
        min_alt = +np.inf
        for i in range(noo):
            pos = x[svs * i : svs * i + 3]
            # altitude(...) は (N,3) 入力なので 2D に
            alt_i = float(altitude(pos.reshape(1, 3))[0])
            if alt_i < min_alt:
                min_alt = alt_i
        return min_alt - threshold_km

    event.terminal = True   # 到達したら積分停止
    event.direction = 0.0   # どちら向きのゼロクロスでもOK
    return event
