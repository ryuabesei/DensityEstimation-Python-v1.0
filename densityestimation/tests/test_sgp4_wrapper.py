# tests/test_sgp4_wrapper.py
from __future__ import annotations

import numpy as np
from sgp4.api import WGS72, Satrec

from densityestimation.tle.sgp4_wrapper import convert_teme_to_j2000

ISS_L1 = "1 25544U 98067A   20344.91667824  .00001264  00000-0  29621-4 0  9991"
ISS_L2 = "2 25544  51.6442  12.2145 0002202  70.9817  48.7153 15.49260293258322"

def test_temetoeci_conversion_ok():
    sat = Satrec.twoline2rv(ISS_L1, ISS_L2, WGS72)
    # tsince=0 (TLE epoch)
    e, r_teme, v_teme = sat.sgp4_tsince(0.0)
    assert e == 0
    jd = sat.jdsatepoch
    r_eci, v_eci = convert_teme_to_j2000(np.array(r_teme), np.array(v_teme), jd)
    assert np.all(np.isfinite(r_eci))
    assert np.all(np.isfinite(v_eci))
    rnorm = np.linalg.norm(r_eci)
    assert 6500.0 < rnorm < 7500.0
