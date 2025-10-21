# tests/test_observations_and_pipeline.py
from __future__ import annotations

import numpy as np
from sgp4.api import WGS72, Satrec

from densityestimation.estimation.observations import generate_observations_mee
from densityestimation.estimation.run_density_estimation_tle import (
    GM_EARTH_KM3_S2_ACCURATE,
    EstimationInputs,
    run_density_estimation_tle,
)
from densityestimation.tle.get_tles_for_estimation import TLEObject

ISS_L1 = "1 25544U 98067A   20344.91667824  .00001264  00000-0  29621-4 0  9991"
ISS_L2 = "2 25544  51.6442  12.2145 0002202  70.9817  48.7153 15.49260293258322"
NORAD_ID = 25544

def _one_object() -> TLEObject:
    # get_tles_for_estimation と互換の簡易オブジェクト
    sat = Satrec.twoline2rv(ISS_L1, ISS_L2, WGS72)
    return TLEObject(noradID=NORAD_ID, satrecs=[sat], tle_lines=(ISS_L1, ISS_L2))

def test_generate_observations_mee_simple():
    obj = _one_object()
    jd0 = obj.satrecs[0].jdsatepoch
    obs = [jd0, jd0 + 1.0/24.0]  # 0h と +1h
    mee = generate_observations_mee([obj], obs, GM_EARTH_KM3_S2_ACCURATE)
    assert mee.shape == (6, 2)
    assert np.isfinite(mee).all()

def test_pipeline_e2e_smoke(tmp_path, tle_dir, bc_path, eop_path, dummy_rom_generator, monkeypatch):
    # 作業ディレクトリをテスト用に
    monkeypatch.chdir(tmp_path)

    # run_density_estimation_tle は TLE をファイルから読むので、
    # tests/conftest.py が作った TLEdata/estimationObjects.tle をそのまま使う
    par = EstimationInputs(
        yr=2020, mth=12, dy=9, nof_days=1,
        ROMmodel="DUMMY", r=4,
        selected_objects=[NORAD_ID],
        plot_figures=False,
        eop_path=str(eop_path),
        tle_dir=str(tle_dir),
        tle_single_file=True,
        bc_path=str(bc_path),
        dt_seconds=3600,
    )

    out = run_density_estimation_tle(
        par,
        rom_generator=dummy_rom_generator,
        rom_initializer=None,
        stateFnc=None,
        measurementFcn=None,
    )

    # 最低限の整合チェック
    assert "mee_meas" in out and out["mee_meas"].shape[0] == 6
    assert "X_est" in out and out["X_est"].shape[0] == (6 + 1) * 1 + par.r  # 6(MEE)+1(BC)×#obj + r
    assert out["tsec"].size >= 2
    assert np.isfinite(out["mee_meas"]).all()
