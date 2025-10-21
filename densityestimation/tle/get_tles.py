# densityestimation/tle/get_tles.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sgp4.api import WGS72, Satrec


@dataclass
class TLEObject:
    noradID: int
    satrecs: List[Satrec]
    # 任意情報（テストやデバッグで便利）
    name: Optional[str] = None
    tle_lines: Optional[Tuple[str, str]] = None  # ← 追加（元の2行を保持）


def _clean(line: str) -> str:
    return line.rstrip("\r\n")


def _parse_tle_block(lines: List[str]) -> Tuple[Optional[str], str, str]:
    """
    与えられた行リストから 1 つの (name?, L1, L2) ブロックを取り出す。
    - 3行ブロック (name, L1, L2) または 2行ブロック (L1, L2) を許容。
    - コメント行/空行はスキップ。
    """
    # 空やコメント(#, 0-9, '1', '2' 以外のものなど)を除去しつつ先頭へ
    buf = []
    for s in lines:
        s = _clean(s)
        if not s:
            continue
        # そのまま一旦バッファへ
        buf.append(s)

    if not buf:
        raise ValueError("No TLE content found.")

    # 先頭が '1' 始まりなら 2 行ブロック、そうでなければ 3 行ブロックとみなす
    if buf[0].startswith("1 ") and len(buf) >= 2 and buf[1].startswith("2 "):
        return None, buf[0], buf[1]
    elif len(buf) >= 3 and buf[1].startswith("1 ") and buf[2].startswith("2 "):
        return buf[0], buf[1], buf[2]
    else:
        raise ValueError("Malformed TLE block (expected 2 or 3 consecutive lines).")


def get_tles(path: str, *, whichconst=WGS72) -> List[TLEObject]:
    """
    単一ファイルから複数オブジェクトの TLE を読み込み、NORAD ID ごとに Satrec リストへ集約。
    - name 行がある形式/ない形式どちらも許容
    - 同一 NORAD ID が複数回出てくる場合はエポック順に並べる
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"TLE file not found: {path}")

    with p.open("r", encoding="utf-8", errors="ignore") as f:
        raw = f.read().splitlines()

    objects: Dict[int, TLEObject] = {}

    i = 0
    n = len(raw)
    while i < n:
        # 次の有効ブロックを探す
        # スライスで渡してパース（例外時は次行から再トライ）
        try:
            # 最大3行見るので i..i+3 を渡す
            name, l1, l2 = _parse_tle_block(raw[i:i+3])
        except Exception:
            i += 1
            continue

        # パース位置を進める（3行 or 2行）
        if name is None:
            i += 2
        else:
            i += 3

        # Satrec 生成
        try:
            sat = Satrec.twoline2rv(l1, l2, whichconst)
        except Exception:
            # 壊れた TLE はスキップ（安全側）
            continue

        # NORAD ID は行2先頭のカラムを使っても良いが、python-sgp4 で sat.satnum に入っている
        nid = int(getattr(sat, "satnum", 0))
        if nid == 0:
            # フォールバック: 2行目の3-7桁
            try:
                nid = int(l2[2:7])
            except Exception:
                continue

        # 既存があれば追加し、最後に epoch でソート
        if nid not in objects:
            objects[nid] = TLEObject(noradID=nid, satrecs=[], name=name, tle_lines=(l1, l2))
        # name は最初のものを優先（必要なら更新してもよい）
        if objects[nid].tle_lines is None:
            objects[nid].tle_lines = (l1, l2)
        if objects[nid].name is None and name is not None:
            objects[nid].name = name

        objects[nid].satrecs.append(sat)

    # エポック昇順に並べる
    out: List[TLEObject] = []
    for nid, obj in objects.items():
        obj.satrecs.sort(key=lambda s: float(getattr(s, "jdsatepoch", 0.0)))
        out.append(obj)

    # NORAD 昇順で返す
    out.sort(key=lambda o: o.noradID)
    return out
