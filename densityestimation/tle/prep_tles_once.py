# /code/tools/prep_tles_once.py などに保存して一度だけ実行
from densityestimation.tle.download_tles_celestrak import download_tles_for_estimation

objs = download_tles_for_estimation(
    username="", password="",              # 互換のため残っているだけ。未使用
    start_year=2000, start_month=1, start_day=1,   # 無視される
    end_year=2030,  end_month=1, end_day=1,        # 無視される
    max_alt=5000.0,                       # フィルタ回避のため十分大きく
    selected_objects=[27391],             # まずはエラーの当該ID（GRACE-A）
    out_dir="TLEdata",                    # run_fig5 の par.tle_dir と一致させる
)
print(f"prepared {len(objs)} object(s)")
