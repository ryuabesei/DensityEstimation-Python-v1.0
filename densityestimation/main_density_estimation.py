from __future__ import annotations

from densityestimation.estimation.pipeline import (
    EstimationConfig,
    run_density_estimation_tle,
)

if __name__ == "__main__":
    cfg = EstimationConfig(
        year=2002, month=8, day=1,
        nof_days=10,
        rom_model="JB2008_1999_2010",
        r=10,
        selected_objects=[63,165,614,2153,2622,4221,6073,7337,8744,12138,12388,14483,20774,23278,27391,27392,26405],
        plot_figures=False,
        tle_single_file=True,
        tle_dir="TLEdata",
        eop_path="Data/EOP-All.txt",
    )
    out = run_density_estimation_tle(cfg)
    print("[OK] MEE observations shape:", out["mee_obs"].shape)
