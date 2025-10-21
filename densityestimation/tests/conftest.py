# tests/conftest.py
from __future__ import annotations

import pathlib
import sys

# --- まず import 前に sys.path を適切に通す（どの作業ディレクトリでも動くように） ---
THIS_FILE = pathlib.Path(__file__).resolve()

def _ensure_sys_path() -> tuple[pathlib.Path, pathlib.Path]:
    """
    returns: (project_root, package_parent)
      - project_root: リポジトリのルート候補
      - package_parent: sys.path に追加したディレクトリ
    """
    # 1) 上に辿って densityestimation/ が見つかれば、その親を追加
    for p in [THIS_FILE] + list(THIS_FILE.parents):
        if (p / "densityestimation").exists():
            pkg_parent = p  # ここを sys.path に入れれば import densityestimation が通る
            if str(pkg_parent) not in sys.path:
                sys.path.insert(0, str(pkg_parent))
            return p, pkg_parent

    # 2) それが無ければ、上に辿って Python/densityestimation/ を探し、Python を追加
    for p in [THIS_FILE] + list(THIS_FILE.parents):
        cand = p / "Python" / "densityestimation"
        if cand.exists():
            pkg_parent = p / "Python"
            if str(pkg_parent) not in sys.path:
                sys.path.insert(0, str(pkg_parent))
            return p, pkg_parent

    raise RuntimeError(
        "tests/conftest.py: densityestimation パッケージの場所が見つかりませんでした。"
        "プロジェクト直下に 'densityestimation/' または 'Python/densityestimation/' があるか確認してください。"
    )

PROJECT_ROOT, PACKAGE_PARENT = _ensure_sys_path()

import numpy as np
import pytest

import densityestimation as _de  # パッケージ実体から data パスを確定
from densityestimation.data.eop_loader import load_eop_celestrak
from densityestimation.tle.sgp4_wrapper import set_eop_matrix

# === 固定TLE（ISS, 2020-12-09 付近） ===
ISS_L1 = "1 25544U 98067A   20344.91667824  .00001264  00000-0  29621-4 0  9991"
ISS_L2 = "2 25544  51.6442  12.2145 0002202  70.9817  48.7153 15.49260293258322"
NORAD_ID = 25544


@pytest.fixture(scope="session")
def repo_root() -> pathlib.Path:
    # プロジェクトのルート候補（単に返すだけ）
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def data_dir() -> pathlib.Path:
    # 実際に import できた densityestimation パッケージの隣にある data を使う
    pkg_dir = pathlib.Path(_de.__file__).resolve().parent
    d = pkg_dir / "data"
    assert (d / "EOP-All.txt").exists(), f"EOP-All.txt が見つかりません: {d / 'EOP-All.txt'}"
    assert (d / "nut80.dat").exists(), f"nut80.dat が見つかりません: {d / 'nut80.dat'}"
    return d


@pytest.fixture()
def tle_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    # テスト用に 1ファイルにまとめた TLE を作る
    d = tmp_path / "TLEdata"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "estimationObjects.tle", "w", encoding="utf-8") as f:
        f.write(ISS_L1 + "\n")
        f.write(ISS_L2 + "\n")
    return d


@pytest.fixture()
def bc_path(tmp_path: pathlib.Path) -> pathlib.Path:
    # テスト用の最小 BC ファイル（ tmp/.../densityestimation/data/BCdata.txt ）
    d = tmp_path / "densityestimation" / "data"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "BCdata.txt"
    with open(p, "w", encoding="utf-8") as f:
        f.write("# NORAD_ID  BC[m^2/kg]\n")
        f.write(f"{NORAD_ID}  0.010\n")
    return p


@pytest.fixture(scope="session")
def eop_path(data_dir: pathlib.Path) -> pathlib.Path:
    return data_dir / "EOP-All.txt"


@pytest.fixture(autouse=True, scope="session")
def _eop_matrix_set(eop_path: pathlib.Path):
    # 1回だけ EOP を読み込んで TEME→J2000 変換系が使えるようにする
    eop = load_eop_celestrak(str(eop_path), full=False)
    set_eop_matrix(eop)
    return eop


# --- ダミーROM（UKF無しテスト用） ---
class _DummyROM:
    def __init__(self, r: int):
        self.AC = np.zeros((r, r))
        self.BC = np.zeros((r, r))
        self.Uh = np.zeros((8, r))
        self.F_U = [lambda slt, lat, alt: np.zeros_like(np.asarray(slt)) for _ in range(r)]
        self.Dens_Mean = -14.0 * np.ones(8)
        self.M_U = lambda slt, lat, alt: -14.0 * np.ones_like(np.asarray(slt))
        self.SLTm = np.zeros((2, 2, 2))
        self.LATm = np.zeros((2, 2, 2))
        self.ALTm = np.zeros((2, 2, 2))
        self.maxAtmAlt = 1500.0
        self.SWinputs = {}
        self.Qrom = 1e-8 * np.eye(r)


@pytest.fixture()
def dummy_rom_generator():
    def _gen(rom_model: str, r: int, jd0: float, jdf: float):
        return _DummyROM(r)
    return _gen
