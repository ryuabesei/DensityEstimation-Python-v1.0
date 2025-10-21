# tests/test_eop_loader.py
from __future__ import annotations

import numpy as np

from densityestimation.data.eop_loader import compute_eop_celestrak, load_eop_celestrak


def test_eop_load_and_compute(eop_path):
    eop = load_eop_celestrak(str(eop_path), full=False)
    assert eop.ndim == 2 and eop.shape[1] == 6
    # 2020-12-09 00:00:00 UTC の JD (≈ 2459192.5)
    jd = 2459192.5
    vals = compute_eop_celestrak(eop, jd)
    # xp, yp, dut1, lod, ddpsi, ddeps, dat
    assert len(vals.__dict__) == 7
    # 数値として有限であることだけ確認
    a = np.array(list(vals.__dict__.values()), dtype=float)
    assert np.isfinite(a).all()
