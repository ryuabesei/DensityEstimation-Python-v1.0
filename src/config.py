#環境変数の設定
from dataclasses import dataclass


@dataclass
class EnvParams:
    # とりあえず固定値（後でSWPC等から取得/時系列化に拡張）
    f107: float = 150.0   # 日変化指数（例）
    f107a: float = 150.0  # 81日移動平均（例）
    ap: list = (4, 4, 4, 4, 4, 4, 4)  # ap(7) = [ap_daily, ap[0:6]]の簡易例

@dataclass
class PropagationCfg:
    step_sec: int = 60  # 1分刻み
    duration_min: int = 180  # 3時間ぶん