# 선택 사항인 tagdata/의 자동완성·색·도움말 자료를 읽는다.
# UI 서버(moss_ui.py)가 처음 쓸 때 한 번 읽는다(약 1~2초, 메모리 약 150MB).
import bisect
import colorsys
import hashlib
import json
import os
import re
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "tagdata")
TYPE_NAMES = {0: "일반", 1: "작가", 3: "작품", 4: "캐릭터", 5: "메타"}
# 분류 색이 없는 작가·작품·캐릭터·메타 태그는 danbooru 어두운 테마의 종류 색(HWtagger 단부루 탭과 같음)
TYPE_COLORS = {1: (255, 138, 139), 3: (199, 151, 255), 4: (53, 198, 74), 5: (234, 208, 132)}
CAT_KO = {}  # 분류 이름은 HWtagger 의 영어 대문자 그대로 쓴다

_lock = threading.Lock()
_db = None


def norm(t):
    """비교용: 소문자, 밑줄 → 띄어쓰기, 공백 정리."""
    return re.sub(r"\s+", " ", str(t).replace("_", " ")).strip().lower()


def _load():
    global _db
    with _lock:
        if _db is not None:
            return _db
        with open(os.path.join(DATA, "tags.json"), encoding="utf-8") as f:
            rows = json.load(f)["tags"]
        with open(os.path.join(DATA, "colors.json"), encoding="utf-8") as f:
            colors = json.load(f)
        with open(os.path.join(DATA, "help.json"), encoding="utf-8") as f:
            helpd = json.load(f)["help"]
        info = {}       # 태그 → (횟수, 종류, 별칭들)
        alias = {}      # 별칭 → 태그
        for name, cnt, typ, al in rows:
            if name not in info:
                info[name] = (cnt, typ, al)
            for a in al:
                alias.setdefault(a, name)
        names = sorted(info)          # 앞글자 찾기(bisect)
        anames = sorted(alias)
        by_count = [r[0] for r in rows]  # 단어 중간 찾기용(횟수순)
        _db = {"info": info, "alias": alias, "names": names, "anames": anames, "by_count": by_count,
               "cats": {k: tuple(v) for k, v in colors["cats"].items()}, "tag2cat": colors["tag2cat"], "help": helpd}
        return _db


def _hash_color(t):
    """분류 색이 없는 일반 태그·처음 보는 태그: 이름으로 정한 고유 색(어두운 바탕에서 읽히는 밝기)."""
    h = int(hashlib.md5(t.encode("utf-8")).hexdigest()[:8], 16)
    r, g, b = colorsys.hls_to_rgb((h % 360) / 360, 0.68, 0.55)
    return int(r * 255), int(g * 255), int(b * 255)


def meta(tag):
    db = _load()
    t = norm(tag)
    canon = t if t in db["info"] else db["alias"].get(t)
    cnt, typ, al = db["info"].get(canon, (0, None, []))
    cat = db["tag2cat"].get(canon or t)
    if cat and cat in db["cats"]:
        color = db["cats"][cat]
    elif typ in TYPE_COLORS:
        color = TYPE_COLORS[typ]
    else:
        color = _hash_color(canon or t)
    return {"tag": t, "canon": canon, "known": canon is not None, "count": cnt, "type": TYPE_NAMES.get(typ, ""),
            "cat": cat or "", "color": list(color), "aliases": al[:6]}


def metas(tags):
    return {t: meta(t) for t in tags[:400]}


def help_of(tag):
    db = _load()
    m = meta(tag)
    h = db["help"].get(m["canon"] or m["tag"]) or {}
    return {**m, "summary": h.get("s", ""), "related": h.get("r", [])}


def _prefix(sorted_list, q, limit):
    i = bisect.bisect_left(sorted_list, q)
    out = []
    while i < len(sorted_list) and sorted_list[i].startswith(q) and len(out) < limit:
        out.append(sorted_list[i])
        i += 1
    return out


def complete(q, n=15):
    """앞글자가 같은 태그(횟수순) → 별칭 → 단어 중간(' '+q). 같은 태그는 한 번만."""
    db = _load()
    q = norm(q)
    if not q:
        return []
    seen, out = set(), []

    def add(tag, via=""):
        if tag in seen or len(out) >= n:
            return
        seen.add(tag)
        m = meta(tag)
        out.append({"tag": tag, "count": m["count"], "type": m["type"], "cat": m["cat"], "color": m["color"], "via": via})

    pre = _prefix(db["names"], q, 4000)  # 한 글자면 수천 개라 횟수로 고른다
    for t in sorted(pre, key=lambda x: -db["info"][x][0])[:n]:
        add(t)
    if len(out) < n:
        for a in sorted(_prefix(db["anames"], q, 400), key=lambda x: -db["info"].get(db["alias"][x], (0,))[0]):
            add(db["alias"][a], via=a)
    if len(out) < n and len(q) >= 2:
        key = " " + q
        for t in db["by_count"][:120000]:
            if key in t:
                add(t)
                if len(out) >= n:
                    break
    return out
