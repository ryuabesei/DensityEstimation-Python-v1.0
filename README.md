
DensityEstimation-Python-v1.0

熱圏密度推定（TLE同化 × 動的ROM）を Python で再現・拡張するプロジェクトです。
MIT Gondelach (2020) 系の流れを参考に、SGP4 伝播 / JB2008・NRLMSISE・TIE-GCM 入力生成 / POD+DMDc によるROM / UKF同化 を一体化。Dockerで開発環境を統一し、論文図（Fig.5）やTLE同化のスモークを再現できます。

🔗 リンク

（任意で）GitHubリポジトリ: <your-repo-url>

参考論文PDF（プロジェクト内）: docs/Space Weather - 2020 - Gondelach - Real-Time Thermospheric Density Estimation ....pdf など

✨ 機能一覧 (MVP)

TLE伝播/観測生成: SGP4で衛星軌道を伝播し、MEEや抗力に基づく観測を生成

宇宙天気入力生成: JB2008/NRLMSISE/TIE-GCM向けの F10.7/Kp/ap/DSTDTC などを整形

ROM学習(POD+DMDc): 高次元密度場を次元削減し、連続系 (Ac,Bc) の動的ROMを構築

UKF同化: ROM状態＋補助パラメータをUKFで更新、TLE観測に同化

再現図出力: 論文 Fig.5 の再現プロットを出力（検証用）

🧩 主要画面/スクリプト（まとめ）

python -m densityestimation.run_fig5 : Fig.5再現（入力ローダの動作検証）

python -m densityestimation.estimation.run_density_estimation_tle : TLE同化パイプライン

python scripts/demo_iss.py : ISSを題材に動作デモ（例）

🛠️ 技術スタック

Language: Python (3.11+)

Core: NumPy / SciPy / pandas / sgp4 / filterpy / (pymsis・spiceypy等)

Test: pytest

Dev: Docker / docker-compose

🚀 開発環境セットアップ（Docker推奨）

このプロジェクトは Docker で開発環境を統一します。
すでに ./Python をコンテナの /code にマウントする docker-compose.yml を同梱しています。

1) 必要なもの

Docker Desktop

VS Code（推奨） + Dev Containers拡張（任意）

2) リポジトリのクローン
git clone <your-repo-url>
cd <your-repo>

3) 環境変数の設定（任意）

Python/densityestimation/constants.py でデータディレクトリ既定を持っていますが、環境変数で上書きできます。

# 宇宙天気/EOP/TLE/ROMデータの基底ディレクトリ
export DENSITYEST_DATA_DIR=/code/densityestimation/data

4) Dockerコンテナの起動

VS Code: 左下の緑アイコン → “Reopen in Container”

ターミナル:

docker-compose up -d --build
docker-compose exec app bash   # 例: サービス名が app の場合

5) 依存の同期（必要なら）

requirements.txt に基づきビルド時に導入されます。追加がある場合のみ：

pip install -r requirements.txt

📦 データの配置

Python/densityestimation/data/ 配下に、以下の生データまたは同等のファイルを配置してください（名称は既存コードと整合）。

宇宙天気:

SOLFSMY.txt（太陽活動インデックス）

DTCFILE.txt（JB2008 DSTDTC 等）

SW-All.txt（統合インデックス。必要に応じて）

EOP/時刻系:

EOP-All.txt

nut80.dat

kernel.txt（SPICEカーネル一覧）+ SPICEカーネル本体（Data/等に配置する場合はパスをkernel.txtで解決）

物体データ:

BCdata.txt（弾道係数）

TLE:

TLEdata/estimationObjects.tle（同梱サンプル or tle/download_tles_for_estimation.pyで取得）

別の場所に置く場合は DENSITYEST_DATA_DIR を設定し、ローダが解決できるようにしてください。

🧪 実行方法
1) Fig.5 再現（ローダ＆前処理の確認）
cd Python
python -m densityestimation.run_fig5
# 出力: fig5_density.png など（プロットやログで確認）

2) 最小ROM学習 → 予測（JB2008を教師に）

小さなグリッド・短い時間列でテスト（例: 1h刻み 1~2日）

入口は models/generators/jb2008_top.py / rom/dmdc.py を想定

# 例: Python/in-container
python - <<'PY'
from densityestimation.models.generators.jb2008_top import train_rom_from_jb2008
from densityestimation.rom.rom_runtime import propagate_one_step
# … grid/times/sw_inputs を準備して train_rom_from_jb2008(...)
# rom.step(...) / rom.predict(...) で動作を確認
print("ROM OK")
PY

3) TLE同化パイプライン（スモーク）
cd Python
python -m densityestimation.estimation.run_density_estimation_tle
# or
pytest densityestimation/tests/run_density_estimation_tle_smoketest.py -q

🧭 ディレクトリ構成（抜粋・コメント付き）
├── docker-compose.yml                          #~ ルート用compose。Pythonサービスをビルド・起動（./Python を /code にマウント）
└── Python
    ├── densityestimation
    │   ├── __init__.py                         #~ パッケージ初期化（エクスポート/バージョン管理）
    │   ├── Astrofunction
    │   │   ├── altitude.py                     #~ 地心半径↔高度などの高度計算ユーティリティ
    │   │   ├── density_rom.py                  #~ ROMから密度を復元/補間する関数群
    │   │   ├── fundarg.py                      #~ 太陽・月の基本引数/GMSTなど天文関数
    │   │   ├── gc2gd.py                        #~ 地心座標→測地座標（緯度経度高度）変換
    │   │   ├── gravity_model.py                #~ 地球重力場（EGM等）の加速度計算
    │   │   ├── isdecayed.py                    #~ TLEオブジェクトの減衰（再突入）判定
    │   │   ├── jb2008_density.py               #~ JB2008ベースの密度計算ラッパ
    │   │   ├── sgp4_constants.py               #~ SGP4で用いる各種定数
    │   │   ├── spice_loader.py                 #~ SPICEカーネルのロード/フレーム変換ヘルパ
    │   │   └── time_utils.py                   #~ 時刻系変換（UTC/TAI/TT/GMST）・日付ユーティリティ
    │   ├── constants.py                        #~ 物理定数・地球定数・モデル共通設定
    │   ├── data
    │   │   ├── bc_loader.py                    #~ 物体のBC(弾道係数)テーブル読み込み
    │   │   ├── BCdata.txt                      #~ 弾道係数データ（参照値）
    │   │   ├── DTCFILE.txt                     #~ JB2008向けDSTDTC等の時間列データ
    │   │   ├── eop_loader.py                   #~ EOP(地球自転極運動)の読み込み/補間
    │   │   ├── EOP-All.txt                     #~ EOPデータ本体
    │   │   ├── kernel.txt                      #~ 使用するSPICEカーネルのリスト/パス
    │   │   ├── nut80.dat                       #~ 章動テーブル（IAU1980）
    │   │   ├── SOLFSMY.txt                     #~ 太陽活動(SOLFSMY)入力ファイル
    │   │   ├── space_weather_readers.py        #~ F10.7/Kp/ap等の宇宙天気インデックス読み込み
    │   │   └── SW-All.txt                      #~ 宇宙天気インデックスの統合データ
    │   ├── dynamics
    │   │   └── derivatives.py                  #~ 軌道力学の微分方程式（重力・SRP・抗力の合力）
    │   ├── estimation
    │   │   ├── __init__.py                     #~ 推定モジュール初期化
    │   │   ├── assimilation.py                 #~ UKF同化のメインループ/状態更新ロジック
    │   │   ├── measurements.py                 #~ 観測ノイズ/測定モデル（MEE表現等）
    │   │   ├── observations.py                 #~ TLE→観測生成（SGP4逆伝播・MEE変換）
    │   │   ├── pipeline.py                     #~ 同化パイプラインの構成/ジョブ管理
    │   │   └── run_density_estimation_tle.py   #~ TLE同化実行スクリプト（CLIエントリ）
    │   ├── grid.py                             #~ ROMの空間グリッド（緯度/LST/高度）の定義・補間
    │   ├── main_density_estimation.py          #~ プロジェクト全体の実行エントリ（統合ランナー）
    │   ├── models
    │   │   ├── __init__.py                     #~ モデル層初期化
    │   │   ├── density_rom.py                  #~ ROMモデルのデータ構造/入出力API
    │   │   ├── generators
    │   │   │   ├── jb2008_top.py               #~ JB2008ベースのROM生成（POD/DMDc）
    │   │   │   ├── nrlmsise.py                 #~ NRLMSISE-00ベースのROM生成
    │   │   │   └── tiegcm.py                   #~ TIE-GCMベースのROM生成
    │   │   ├── JB2008
    │   │   │   ├── __init__.py                 #~ JB2008サブパッケージ初期化
    │   │   │   ├── constants.py                #~ JB2008固有定数
    │   │   │   ├── finddays.py                 #~ 日数換算/暦関数
    │   │   │   ├── iers.py                     #~ IERS関連の補助計算
    │   │   │   ├── invjday.py                  #~ ユリウス日→暦日変換
    │   │   │   ├── jb2008model.py              #~ JB2008モデル本体の計算ルーチン
    │   │   │   ├── mjday.py                    #~ Modified Julian Day計算
    │   │   │   ├── sat_const.py                #~ 衛星力学で使う定数群
    │   │   │   ├── sign_.py                    #~ 数値演算ユーティリティ（符号等）
    │   │   │   └── timediff.py                 #~ 時刻差・時間系補助計算
    │   │   └── rom_model.py                    #~ ROMの状態方程式(Ac,Bc)・時間発展API
    │   ├── orbit
    │   │   ├── __init__.py                     #~ 軌道モジュール初期化
    │   │   ├── mee.py                          #~ MEE(修正春分点要素)←→他表現の相互変換
    │   │   └── propagation.py                  #~ SGP4/数値積分による軌道伝播
    │   ├── rom
    │   │   ├── __init__.py                     #~ ROMユーティリティ初期化
    │   │   ├── den_cal.py                      #~ ROM密度の計算・平均/軌道平均などの生成
    │   │   ├── dmdc.py                         #~ DMDc実装（離散→連続(Ac,Bc)変換含む）
    │   │   └── rom_runtime.py                  #~ 推定時のROM時間更新・入力u組み立て
    │   ├── run_fig5.py                         #~ 論文Figure 5の再現プロット/検証スクリプト
    │   ├── spaceweather
    │   │   ├── jb2008_inputs.py                #~ JB2008用の宇宙天気入力(指数/非線形項)生成
    │   │   ├── load_jb2008_swdata.py           #~ JB2008インデックス(DTC等)一括ローダ
    │   │   ├── nrlmsise_inputs.py              #~ NRLMSISE-00用の宇宙天気入力生成
    │   │   ├── nrlmsise_loader.py              #~ NRLMSISE関連データの読み込み
    │   │   ├── tiegcm_inputs.py                #~ TIE-GCM用の強制力入力(非線形項含む)生成
    │   │   └── tiegcm_loader.py                #~ TIE-GCM学習データ/設定のローダ
    │   ├── tests
    │   │   ├── conftest.py                     #~ pytest共通フィクスチャ/設定
    │   │   ├── run_density_estimation_tle_smoketest.py #~ TLE同化のスモークテスト
    │   │   ├── test_eop_loader.py              #~ EOPローダの単体テスト
    │   │   ├── test_jb2008_basic.py            #~ JB2008モデルの基本検証
    │   │   ├── test_observations_and_pipeline.py#~ 観測生成とパイプライン結合テスト
    │   │   └── test_sgp4_wrapper.py            #~ SGP4ラッパ/座標変換のテスト
    │   ├── tle
    │   │   ├── __init__.py                     #~ TLEモジュール初期化
    │   │   ├── download_tles_for_estimation.py #~ Space-Trackから同化対象TLEをダウンロード
    │   │   ├── get_tles_for_estimation.py      #~ 期間/高度条件でTLE集合を抽出
    │   │   ├── get_tles.py                     #~ TLE取得の汎用関数
    │   │   ├── prep_tles_once.py               #~ 取得TLEの前処理（重複/外れ値処理など）
    │   │   └── sgp4_wrapper.py                 #~ SGP4呼び出し＋MEE/ECI変換の統合ラッパ
    │   └── ukf
    │       ├── __init__.py                     #~ UKFパッケージ初期化
    │       ├── srukf.py                        #~ Square-Root UKFの実装
    │       └── unscented.py                    #~ Unscented変換/シグマ点生成ユーティリティ
    ├── docker-compose.yml                      #~ （同上）開発用compose（/Python配下をコンテナにマウント）
    ├── Dockerfile                              #~ ルート用Dockerfile（ベース環境のビルド定義）
    ├── fig5_density.png                        #~ run_fig5の出力図（検証用プロット）
    ├── README.md                               #~ プロジェクト説明・実行手順・背景
    ├── requirements.txt                        #~ 依存パッケージ一覧（sgp4/spiceypy/filterpy等）
    ├── sample.py                               #~ 簡単な実行サンプル/動作確認スクリプト
    ├── scripts
    │   └── demo_iss.py                         #~ ISSを例にした伝播/密度計算デモ
    ├── src
    │   ├── config.py                           #~ スクリプト系の設定（パス/定数）
    │   ├── msis_density.py                     #~ MSISによる密度計算の単体スクリプト
    │   ├── propagate.py                        #~ 単発の軌道伝播ユーティリティ（CLI想定）
    │   └── tle.py                              #~ TLE読取/整形の簡易ツール
    └── TLEdata
        └── estimationObjects.tle               #~ 同化対象のTLE集合（ダウンロード済みサンプル）



ーーーーーーーーーーーーーーーーーーーーー
##　以下のディレクトリでも試す
.
├── Dockerfile                             # コンテナのビルド定義（OS/依存/作業ディレクトリ/エントリ）
├── docker-compose.yml                     # サービス起動/ボリューム/環境変数の統合管理
├── pyproject.toml                         # パッケージ設定・依存・エントリポイント（console_scripts）
├── README.md                              # プロジェクト概要・実行方法・依存・データ取得手順
├── requirements.txt                       # 依存パッケージ固定（pyproject未使用時/ロック代替）
├── data/                                  # データ全般（コードと分離）
│   ├── raw/                               # “生”データ（外部配布のまま）
│   │   ├── eop/                           # EOP/うるう秒などの時刻系補正データ
│   │   ├── spice/                         # SPICEカーネル（*.bsp, *.bpc, *.tpc, *.tf, *.tls）
│   │   ├── spaceweather/                  # 宇宙天気入力（SOLFSMY/DTCFILE/SW-All）
│   │   ├── tle/                           # TLEファイル（estimationObjects.tle 等）
│   │   └── objects/                       # 物体特性（BCdata.txt 等）
│   ├── interim/                           # 中間生成物（前処理・解析途中）
│   └── processed/                         # 最終生成物（再利用可能な成果）
│       └── rom/                           # 事前学習ROM（*_ROM_r100.(mat|npz|h5)）
├── assets/                                # 結果物や補助資料
│   ├── figures/                           # 生成図（run_fig5 出力など）
│   └── notes/                             # 参考メモ・Tips・引用抜粋
├── docs/                                  # 仕様・設計・数式メモ・APIドキュメント
├── scripts/                               # ワンショット実行（CLIでない軽量ユーティリティ）
├── tests/                                 # pytestテスト一式（単体・結合・スモーク）
└── densityestimation/                     # パッケージ本体
    ├── __init__.py                        # パッケージ初期化（バージョン/公開API）
    ├── config/
    │   ├── __init__.py                    # configサブパッケージ初期化
    │   └── constants.py                   # 物理定数・地球定数・共通設定値
    ├── astro/                             # 天文/座標/時間・重力などの物理基盤
    │   ├── __init__.py                    # astro初期化
    │   ├── altitude.py                    # 緯度経度高度・地心半径・高度換算
    │   ├── coords.py                      # 座標変換（TEME/ECI/J2000/測地系など）
    │   ├── gravity.py                     # 地球重力場モデル/加速度（EGM等）
    │   ├── time.py                        # 時刻系変換（UTC/TT/TAI/GMST/基本引数）
    │   └── spice_loader.py                # SPICEカーネル読み込み/フレーム解決
    ├── spaceweather/                      # 宇宙天気入力の生成・読込
    │   ├── __init__.py                    # 初期化
    │   ├── readers.py                     # F10.7/Kp/ap等のファイル読取ヘルパ
    │   ├── loaders.py                     # 各モデル用の宇宙天気データ一括ローダ
    │   ├── jb2008_inputs.py               # JB2008用の入力ベクトル生成（DSTDTC等）
    │   ├── nrlmsise_inputs.py             # NRLMSISE-00用の入力生成
    │   └── tiegcm_inputs.py               # TIE-GCM用の入力生成（強制力成分など）
    ├── io/                                # ファイルI/O（外部データの取り回し）
    │   ├── __init__.py                    # 初期化
    │   ├── eop_loader.py                  # EOPデータ読込/補間インターフェース
    │   ├── bc_loader.py                   # 弾道係数(BC)テーブル読込
    │   └── tle/                           # TLE関連I/O
    │       ├── __init__.py                # 初期化
    │       ├── download.py                # Space-Track等からのTLE取得
    │       ├── prepare.py                 # TLE前処理（期間抽出/重複除去/検証）
    │       └── sgp4_wrapper.py            # SGP4呼び出し＋状態/フレーム変換ラッパ
    ├── orbit/                             # 軌道力学（状態表現・伝播）
    │   ├── __init__.py                    # 初期化
    │   ├── mee.py                         # MEE（修正要素）⇔ 他表現の変換
    │   ├── propagation.py                 # 数値/SGP4伝播の統合API
    │   └── dynamics.py                    # 連立微分方程式（重力/抗力/SRP等の合力）
    ├── models/                            # 密度モデル群（経験/物理/ROM）
    │   ├── __init__.py                    # 初期化
    │   ├── jb2008/                        # JB2008実装
    │   │   ├── __init__.py                # 初期化
    │   │   ├── core.py                    # JB2008密度計算コア
    │   │   ├── constants.py               # JB2008固有定数
    │   │   ├── iers.py                    # IERS関連補助（UT1/極運動等）
    │   │   ├── timeutils.py               # MJD/ユリウス日/時刻差ユーティリティ
    │   │   └── utils.py                   # finddays/signなどの補助関数
    │   └── rom/                           # ROM（縮約モデル）
    │       ├── __init__.py                # 初期化
    │       ├── core.py                    # ROMの状態方程式/入出力（Ac,Bc,観測）
    │       ├── dmdc.py                    # DMDc学習/連続化・安定化
    │       ├── runtime.py                 # 推定時のROM時間発展・入力組立
    │       └── generators/                # 学習データ→ROM生成器
    │           ├── jb2008.py              # JB2008由来ROM生成（POD/DMDc）
    │           ├── nrlmsise.py            # NRLMSISE由来ROM生成
    │           └── tiegcm.py              # TIE-GCM由来ROM生成
    ├── estimation/                        # 同化・推定（UKF等）
    │   ├── __init__.py                    # 初期化
    │   ├── assimilation.py                # 同化メインループ（予報→観測更新）
    │   ├── observations.py                # 観測生成（TLE→MEE等）/H行列相当
    │   ├── measurements.py                # 測定モデル/ノイズモデル(R/Q)
    │   └── filters/                       # フィルタ実装群
    │       ├── __init__.py                # 初期化
    │       └── ukf.py                     # SR-UKF/Unscented変換/シグマ点
    ├── cli/                               # 実行用エントリ（ユーザ向け）
    │   ├── run_tle_assim.py               # TLE同化ジョブのCLI
    │   ├── run_fig5.py                    # 論文Fig.5再現（検証用プロット）
    │   └── demo_iss.py                    # ISSサンプル（伝播/密度の例）
    └── utils/
        ├── __init__.py                    # 初期化
        └── sgp4_constants.py              # SGP4関連の定数/パラメータ
