#最小実行スクリプト
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt

from src.config import EnvParams, PropagationCfg
from src.msis_density import add_msis_density
from src.propagate import propagate_tle

# サンプルTLE（ISS, 適当な例。必要なら data/tle/iss.tle を読み込むように変更してOK）
L1 = "1 25544U 98067A   21226.49389238  .00001429  00000-0  34174-4 0  9998"
L2 = "2 25544  51.6437  54.3833 0001250 307.1355 142.9078 15.48901431297630"

if __name__ == "__main__":
    env = EnvParams()
    cfg = PropagationCfg(step_sec=60, duration_min=180)

    start = datetime(2021, 8, 14, 12, 0, 0, tzinfo=timezone.utc)
    df = propagate_tle(L1, L2, start=start, minutes=cfg.duration_min, step_sec=cfg.step_sec)
    df = add_msis_density(df, f107=env.f107, f107a=env.f107a, ap7=env.ap)

    # 保存
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "demo_iss_density.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # 簡易プロット（高度・密度の時間変化）
    t = df["t_utc"]
    plt.figure()
    plt.plot(t, df["alt_km"])  # 色指定はデフォルトのまま
    plt.xlabel("UTC time")
    plt.ylabel("Altitude [km]")
    plt.title("ISS Altitude vs Time")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.semilogy(t, df["rho_kg_m3"])  # 密度は対数で見ると把握しやすい
    plt.xlabel("UTC time")
    plt.ylabel("Density [kg/m^3]")
    plt.title("MSIS Density vs Time")
    plt.tight_layout()
    plt.show()