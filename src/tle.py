#TLE読み込み
from pathlib import Path
from typing import List, Tuple


def load_tles_from_file(path: str | Path) -> List[Tuple[str, str, str]]:
    """2行要素をファイルから読み込んで [(name, l1, l2), ...] を返す。
    name行が無い形式なら 'NONAME' を割り振る。
    """
    p = Path(path)
    lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    out: List[Tuple[str, str, str]] = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("1 ") and i + 1 < len(lines) and lines[i+1].startswith("2 "):
            out.append(("NONAME", lines[i], lines[i+1]))
            i += 2
        elif i + 2 < len(lines) and lines[i+1].startswith("1 ") and lines[i+2].startswith("2 "):
            out.append((lines[i], lines[i+1], lines[i+2]))
            i += 3
        else:
            i += 1
    return out