# 슬라이더 LoRA 그림 쌍 처리 (PIL 만 쓴다. UI 서버와 학습 파일이 함께 쓴다)
# 쌍 폴더 하나의 그림을 이름순(숫자는 크기순, 대소문자 무시: 탐색기와 같음)으로 늘어놓고 앞에서부터 2장씩 묶는다.
# 예) 1,2,3,4 → (1,2) (3,4). first="neg"(기본)면 앞 그림이 − 방향(원본 → 바뀐 그림 순), "pos" 면 앞 그림이 + 방향. 홀수면 마지막 1장은 쓰지 않는다.
# 학습 전에 짝마다 [+ | −] 를 가로로 붙인 묶음 그림을 만든다. 두 그림은 같은 크기(간격의 배수, 면적 ≤ 해상도²)로
# 맞추고, 묶음 크기가 정확히 버킷 크기가 되게 해서 sd-scripts 가 자르거나 늘리지 않게 한다(업스케일 안 함 + 간격 배수).
# 학습은 묶음을 반으로 나눠 따로 VAE 에 넣고, 같은 노이즈·타임스텝으로 + 는 LoRA +1, − 는 LoRA −1 로 학습한다.
# 마스크 합성(mask = "file" / "auto"): 두 그림이 바꾸려는 곳 말고도 다르면(같은 시드로 LoRA 를 걸고 뽑은 쌍처럼 세부가 다시
# 그려진 경우) 그 차이도 배운다. 그래서 마스크 밖은 기준 그림(keep 쪽) 픽셀로 채워 두 그림을 똑같이 만든다.
#   마스크 파일: 쌍의 어느 그림이든 이름 뒤에 _mask 를 붙인 그림(1_mask.png, 흰색 = 바꿀 곳). 쌍 짝짓기에서는 빠진다.
#   auto: 마스크 파일이 없는 쌍은 두 그림이 크게 다른 곳 중 세로 범위(top~bottom, 그림 높이 비율) 안만 쓴다.
#     범위 끝은 일직선이 아니라, 그 높이 근처(±SEAM_WINDOW)에서 두 그림 차이가 가장 작은 칸을 따라 굽이치는 곡선이다
#     (최소 비용 경로, 사진 이어 붙이기의 경계 찾기와 같다). 그래야 경계에서 배꼽·허리선이 겹치는 일이 줄어든다.
#   경계 어긋남(seam): 합성 경계 띠에서 두 원본이 크게 다른 픽셀 비율(%). 크면 경계를 어디에 두어도 맞지 않는 쌍이다.
#   쌍별 설정(pair_masks.json, 쌍 폴더): 쌍마다 마스크를 끄거나(off) 자동 범위·기준을 따로 준다(auto). 키는 쌍의 앞 그림 이름.
#     쌍별 auto 는 전체 설정이 꺼져 있어도 그 쌍에 마스크를 쓴다. 마스크 파일이 있으면 범위 대신 파일을 쓴다.
import hashlib
import json
import math
import os
import re

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
ASPECT_TOLERANCE = 0.02  # 두 그림의 가로세로 비가 이만큼 넘게 다르면 경고(− 그림을 + 크기로 늘려 맞춘다)
MANIFEST = "pairs.json"
MASK_SUFFIX = "_mask"
OVERRIDES = "pair_masks.json"  # 쌍 폴더 안. 쌍별 마스크 설정 {앞 그림 이름(소문자): {mode, top, bottom, keep}}
OVERRIDE_MODES = ("off", "auto")
EXCLUDE_DIR = "excluded"  # 쌍 폴더 안. 쌍 미리보기에서 뺀 쌍(그림·캡션·마스크)을 옮겨 둔다. 학습은 쌍 폴더 바로 아래 그림만 읽는다
MASK_MODES = ("off", "file", "auto")
AUTO_DIFF = 20       # auto: 픽셀의 |+ − −| 평균(0~255)이 이보다 크면 다른 픽셀
AUTO_CELL = 8        # auto: 이 크기 칸으로 나눠 판단
AUTO_CELL_FRAC = 0.3  # auto: 칸 안에서 다른 픽셀이 이 비율 이상이면 바뀐 칸. 다시 그려진 가는 선은 넘지 못하고 형태 변화(살·배경)만 넘는다
AUTO_GROW = 4        # auto: 바뀐 칸을 이 칸 수만큼 넓힌다(윤곽 포함)
SEAM_CELL = 4        # auto: 곡선 경계를 찾는 칸 크기(px). 한 칸 옆으로 갈 때 위아래로 최대 한 칸 움직인다(45°)
SEAM_WINDOW = 0.10   # auto: 범위 끝 높이에서 위아래로 이 비율(그림 높이) 안에서 경계를 찾는다
SEAM_DRIFT = 0.5     # auto: 지정 높이에서 한 칸 멀어질 때마다 더하는 비용(차이가 비슷하면 지정 높이 쪽)
SEAM_STEP = 2.0      # auto: 위아래로 한 칸 움직일 때 더하는 비용(쓸데없이 출렁이지 않게)
SEAM_OUTSIDE = 0.3   # auto: 경계 바깥에 남는 바뀐 칸(비율 × 255)에 매기는 비용 배수
SEAM_BAD = 38        # 경계 어긋남: 두 원본 차이(0~255)가 이보다 큰 픽셀을 어긋난 것으로 센다
SEAM_WARN = 3       # 경계 어긋남이 이 % 이상이면 학습 로그·화면에서 알린다(6% 이상은 화면에서 빨강).
CAPTION_RULE = 2      # 묶음 서명에 넣는다. 2(2026-11-13): 빈 캡션이면 캡션 파일을 쓰지 않는다
SEAM_VERSION = 3     # 묶음 서명에 넣는다(경계 방식·서명 방식이 바뀌면 다시 만든다)


def is_mask_name(name):
    return os.path.splitext(os.path.basename(name))[0].lower().endswith(MASK_SUFFIX)


def find_mask(*image_paths):
    """쌍 그림 중 하나의 이름 + _mask 그림 경로."""
    for p in image_paths:
        stem = os.path.splitext(p)[0]
        for ext in sorted(IMAGE_EXTENSIONS):
            for cand in (stem + MASK_SUFFIX + ext, stem + MASK_SUFFIX.upper() + ext):
                if os.path.isfile(cand):
                    return cand.replace("\\", "/")
    return None


def mask_options(mode="off", keep="neg", top=0.45, bottom=1.0):
    mode = mode if mode in MASK_MODES else "off"
    top = min(max(float(top), 0.0), 1.0)
    bottom = min(max(float(bottom), 0.0), 1.0)
    return {"mode": mode, "keep": "pos" if keep == "pos" else "neg", "top": top, "bottom": bottom}


def clean_override(o):
    """쌍별 마스크 설정을 고쳐 돌려준다. 틀리면 None."""
    if not isinstance(o, dict) or o.get("mode") not in OVERRIDE_MODES:
        return None
    if o["mode"] == "off":
        return {"mode": "off"}
    try:
        top, bottom = float(o.get("top", 0.45)), float(o.get("bottom", 1.0))
    except (TypeError, ValueError):
        return None
    top, bottom = min(max(top, 0.0), 1.0), min(max(bottom, 0.0), 1.0)
    if not top < bottom:
        return None
    return {"mode": "auto", "top": round(top, 4), "bottom": round(bottom, 4), "keep": "pos" if o.get("keep") == "pos" else "neg"}


def load_overrides(folder):
    """쌍 폴더의 쌍별 마스크 설정. 없거나 읽을 수 없으면 빈 dict."""
    try:
        with open(os.path.join(folder, OVERRIDES), encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    out = {}
    for k, v in data.items():
        c = clean_override(v)
        if c:
            out[str(k).lower()] = c
    return out


def _replace_retry(tmp, path):
    """Windows 에서 다른 프로그램이 읽는 중이면 os.replace 가 PermissionError 를 낸다(FS-001). 잠깐 기다려 다시 한다."""
    import time
    for i in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if i == 9:
                raise
            time.sleep(0.05 * (i + 1))


def save_overrides(folder, data):
    path = os.path.join(folder, OVERRIDES)
    if not data:
        if os.path.exists(path):
            os.remove(path)
        return
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(dict(sorted(data.items())), f, ensure_ascii=False, indent=1)
        _replace_retry(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def pair_key(front_image):
    return os.path.basename(front_image).lower()


def natural_key(name):
    """이름순 정렬 키: 숫자 부분은 크기로 비교한다(2 < 10). 짝수 칸은 글자, 홀수 칸은 숫자라 서로 비교되지 않는다."""
    return [int(t) if i % 2 else t.lower() for i, t in enumerate(re.split(r"(\d+)", name))]


def list_images(folder):
    """폴더 바로 아래 그림 경로를 이름순으로."""
    if not folder or not os.path.isdir(folder):
        return []
    names = [n for n in os.listdir(folder)
             if os.path.splitext(n)[1].lower() in IMAGE_EXTENSIONS and not is_mask_name(n) and os.path.isfile(os.path.join(folder, n))]
    return [os.path.join(folder, n) for n in sorted(names, key=natural_key)]


def read_caption(image_path, caption_ext):
    path = os.path.splitext(image_path)[0] + caption_ext
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    return None


def image_size(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im) if im.getexif().get(0x0112) else im
        return im.size


def half_size(w, h, resolution, steps):
    """한쪽 그림의 학습 크기: 면적 ≤ resolution², 간격 배수로 내림, 늘리지 않음."""
    scale = min(1.0, math.sqrt(resolution * resolution / (w * h)))
    hw = max(steps, int(w * scale) // steps * steps)
    hh = max(steps, int(h * scale) // steps * steps)
    return hw, hh


def match_pairs(folder, first="neg", caption_ext=".txt"):
    """쌍 폴더의 그림을 이름순으로 2장씩 묶고 점검한다. 크기는 그림 머리만 읽는다."""
    images = list_images(folder)
    count = len(images) // 2
    overrides = load_overrides(folder)
    pairs, mismatched = [], []
    for idx in range(count):
        a, b = images[2 * idx], images[2 * idx + 1]
        p, n = (a, b) if first == "pos" else (b, a)
        try:
            ps, ns = image_size(p), image_size(n)
        except Exception as e:
            mismatched.append({"name": f"{os.path.basename(a)} · {os.path.basename(b)}", "note": f"읽을 수 없음: {e}"})
            continue
        item = _make_item(idx, a, p, n, ps, ns, caption_ext, overrides)
        if item.get("aspect_mismatch"):
            mismatched.append({"name": f"{item['pos_name']} ↔ {item['neg_name']}", "note": f"가로세로 비가 다름 {ps[0]}×{ps[1]} / {ns[0]}×{ns[1]}"})
        pairs.append(item)
    return {
        "pairs": pairs,
        "leftover": [os.path.basename(x) for x in images[2 * count:]],  # 홀수일 때 짝 없이 남는 마지막 그림
        "mismatched": mismatched,
        "image_count": len(images),
    }


def pair_item(folder, first, index, caption_ext=".txt"):
    """index 번(1부터) 쌍 하나만 (그 두 그림의 크기만 읽는다). 없으면 None."""
    images = list_images(folder)
    if not 1 <= index <= len(images) // 2:
        return None
    a, b = images[2 * index - 2], images[2 * index - 1]
    p, n = (a, b) if first == "pos" else (b, a)
    return _make_item(index - 1, a, p, n, image_size(p), image_size(n), caption_ext, load_overrides(folder))


def _pair_at(folder, index, names):
    """index 번(1부터) 쌍의 두 그림(이름순 앞, 뒤). names 는 화면이 본 두 그림 이름: 그사이 목록이 바뀌었으면 LookupError."""
    images = list_images(folder)
    if not 1 <= index <= len(images) // 2:
        raise LookupError(f"{index}번 쌍이 없습니다")
    a, b = images[2 * index - 2], images[2 * index - 1]
    if sorted(os.path.basename(x).lower() for x in (a, b)) != sorted(str(x).lower() for x in names):
        raise LookupError("쌍 목록이 바뀌었습니다. 다시 읽은 뒤 해 주세요")
    return a, b


def set_override(folder, index, names, override):
    """쌍 하나의 쌍별 마스크 설정을 저장한다. override 가 None 이면 지운다(전체 설정을 따름). 반환: 저장된 값."""
    a, _b = _pair_at(folder, index, names)
    data = load_overrides(folder)
    c = clean_override(override) if override is not None else None
    if override is not None and c is None:
        raise ValueError("쌍별 마스크 설정이 올바르지 않습니다(범위는 0~100%, 위 < 아래)")
    if c:
        data[pair_key(a)] = c
    else:
        data.pop(pair_key(a), None)
    save_overrides(folder, data)
    return c


def save_mask_file(folder, index, names, png_bytes):
    """화면에서 그린 마스크를 쌍의 마스크 파일로 저장한다(흰색 = 바꿀 곳). png_bytes 가 None 이면 마스크 파일을 지운다.
    이미 있는 마스크 파일이 있으면 그 파일을 덮어쓰고, 없으면 '앞 그림 이름_mask.png' 로 만든다. 반환: 경로 또는 None."""
    import io
    a, b = _pair_at(folder, index, names)
    existing = find_mask(a, b)
    if png_bytes is None:
        if existing:
            os.remove(existing)
        return None
    with Image.open(io.BytesIO(png_bytes)) as im:
        m = im.convert("L")
    m = m.resize(image_size(a), Image.BILINEAR)
    path = existing if existing and existing.lower().endswith(".png") else os.path.splitext(a)[0] + MASK_SUFFIX + ".png"
    tmp = path + ".tmp.png"
    try:
        m.save(tmp, "PNG")
        _replace_retry(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    if existing and os.path.normcase(existing) != os.path.normcase(path) and os.path.exists(existing):
        os.remove(existing)  # 다른 형식(jpg 등)의 예전 마스크는 새 png 로 바꾼다
    return path.replace("\\", "/")


def _pair_files(a, b):
    """한 쌍의 두 그림과 딸린 파일: 이름(확장자 앞)이 같은 파일(캡션 등)과 _mask 그림."""
    folder = os.path.dirname(a)
    stems = set()
    for img in (a, b):
        stem = os.path.splitext(os.path.basename(img))[0].lower()
        stems |= {stem, stem + MASK_SUFFIX}
    return [os.path.join(folder, n) for n in sorted(os.listdir(folder))
            if os.path.splitext(n)[0].lower() in stems and os.path.isfile(os.path.join(folder, n))]


def move_pair(folder, index, names, restore=False):
    """index 번(1부터) 쌍을 쌍 폴더 안의 EXCLUDE_DIR 로 옮긴다(restore=True 면 거기서 쌍 폴더로 되돌린다).
    names 는 화면이 본 두 그림 이름(앞, 뒤): 목록이 그사이 바뀌었으면 옮기지 않는다. 반환: 옮긴 파일 이름 목록."""
    ex = os.path.join(folder, EXCLUDE_DIR)
    src, dst = (ex, folder) if restore else (folder, ex)
    a, b = _pair_at(src, index, names)
    files = _pair_files(a, b)
    clash = [os.path.basename(f) for f in files if os.path.exists(os.path.join(dst, os.path.basename(f)))]
    if clash:
        raise FileExistsError("옮길 곳에 같은 이름의 파일이 있습니다: " + ", ".join(clash))
    os.makedirs(dst, exist_ok=True)
    moved = []
    try:
        for f in files:
            os.replace(f, os.path.join(dst, os.path.basename(f)))
            moved.append(f)
    except OSError:
        for f in moved:  # 반쯤 옮겨진 쌍이 남지 않게 되돌린다
            try:
                os.replace(os.path.join(dst, os.path.basename(f)), f)
            except OSError:
                pass
        raise
    src_ovr = load_overrides(src)  # 쌍별 마스크 설정도 함께 옮긴다
    o = src_ovr.pop(pair_key(a), None)
    if o:
        save_overrides(src, src_ovr)
        dst_ovr = load_overrides(dst)
        dst_ovr[pair_key(a)] = o
        save_overrides(dst, dst_ovr)
    if restore and not os.listdir(ex):
        os.rmdir(ex)
    return [os.path.basename(f) for f in files]


def _make_item(idx, a, p, n, ps, ns, caption_ext, overrides=None):
    """쌍 하나의 정보. a 는 묶음 앞 그림(이름에 쓴다), p / n 은 + / − 그림."""
    stem = os.path.splitext(os.path.basename(a))[0]
    item = {"index": idx + 1, "name": f"{idx + 1:04d}_{stem}", "pos": p.replace("\\", "/"), "neg": n.replace("\\", "/"),
            "pos_name": os.path.basename(p), "neg_name": os.path.basename(n), "pos_size": ps, "neg_size": ns}
    cap = read_caption(p, caption_ext)
    if cap is None:
        cap = read_caption(n, caption_ext)
    item["caption"] = cap
    item["mask"] = find_mask(p, n)
    item["key"] = pair_key(a)
    item["override"] = (overrides or {}).get(item["key"])
    if abs(ps[0] / ps[1] - ns[0] / ns[1]) > ASPECT_TOLERANCE * (ps[0] / ps[1]):
        item["aspect_mismatch"] = True
    elif ps != ns:
        item["size_differs"] = True
    return item


def _signature(pairs, resolution, steps, fallback, mask=None):
    """입력 서명. 마스크는 쌍마다 실제로 쓸 옵션(쌍별 설정 반영)과 그때 쓰는 마스크 파일만 넣는다
    (마스크를 쓰지 않는 쌍의 마스크 파일이 바뀌어도 다시 만들지 않는다)."""
    h = hashlib.sha1(json.dumps([resolution, steps, fallback, SEAM_VERSION, CAPTION_RULE], ensure_ascii=False).encode("utf-8"))
    for it in pairs:
        eff = effective_mask(it, mask)
        h.update(json.dumps(eff, sort_keys=True).encode("utf-8"))
        for k in ("pos", "neg", "mask"):
            if not it.get(k) or (k == "mask" and not eff):
                continue
            st = os.stat(it[k])
            h.update(f"{it[k]}|{st.st_size}|{st.st_mtime_ns}".encode("utf-8"))
        h.update((it["caption"] or "").encode("utf-8"))
    return h.hexdigest()


def _load_rgb(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im.convert("RGBA"), mask=im.convert("RGBA").split()[-1])
            return bg
        return im.convert("RGB")


def _load_mask(path):
    """마스크 파일: 투명도가 있으면 알파, 없으면 밝기. 흰색(불투명) = 바꿀 곳."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            alpha = im.convert("RGBA").split()[-1]
            if alpha.getextrema()[0] < 255:
                return alpha
        return im.convert("L")


def seam_path(diff, frac, outside="above"):
    """frac(그림 높이 비율) 근처에서 왼쪽 끝부터 오른쪽 끝까지 이어지는 합성 경계. 반환: [(x, y)] 점 목록(px).
    비용 = 경계가 지나는 칸의 차이 + 경계 바깥(outside 쪽, 기준 그림을 쓰는 곳)에 남는 바뀐 칸.
    경계 위에서 두 그림이 같아도, 바로 바깥에 바뀐 부분(옮겨진 배꼽·허리선)이 남으면 두 그림의 것이 함께 보인다.
    그래서 바깥에 남는 바뀐 칸에 비용을 매겨, 경계가 바뀐 부분을 모두 안에 담는 쪽으로 간다."""
    w, h = diff.size
    cw, ch = max(1, math.ceil(w / SEAM_CELL)), max(1, math.ceil(h / SEAM_CELL))
    small = diff.resize((cw, ch), Image.BOX)
    cost = list(small.getdata())
    # 바뀐 칸은 자동 마스크와 같은 기준(AUTO_CELL 칸의 AUTO_CELL_FRAC 이상)으로 판단해 경로 칸 크기로 늘린다.
    # 경로 칸(4px)으로 바로 재면 다시 그려진 가는 선도 칸을 넘어 경계가 늘 창 끝으로 밀린다
    big = diff.point(lambda v: 255 if v > AUTO_DIFF else 0).resize((max(1, w // AUTO_CELL), max(1, h // AUTO_CELL)), Image.BOX)
    big = big.point(lambda v: 255 if v >= 255 * AUTO_CELL_FRAC else 0)
    changed = list(big.resize((cw, ch), Image.NEAREST).getdata())
    r0 = min(ch - 1, max(0, int(round(frac * ch))))
    win = max(2, int(round(SEAM_WINDOW * ch)))
    lo, hi = max(0, r0 - win), min(ch - 1, r0 + win)
    rows = range(lo, hi + 1)
    # out[c][r]: 경계를 r 에 둘 때 창 안에서 바깥에 남는 바뀐 칸의 합(칸 안 바뀐 픽셀 비율 × 255)
    out = []
    for c in range(cw):
        col, acc = {}, 0.0
        order = rows if outside == "above" else reversed(rows)
        for r in order:
            col[r] = acc
            v = changed[r * cw + c]
            acc += v
        out.append(col)
    unit = lambda r, c: cost[r * cw + c] + SEAM_DRIFT * abs(r - r0) + SEAM_OUTSIDE * out[c][r]
    inf = float("inf")
    prev = {r: unit(r, 0) for r in rows}
    back = []
    for c in range(1, cw):
        cur, bk = {}, {}
        for r in rows:
            best, br = prev[r], r
            for rr in (r - 1, r + 1):
                v = prev.get(rr, inf) + SEAM_STEP
                if v < best:
                    best, br = v, rr
            cur[r] = best + unit(r, c)
            bk[r] = br
        back.append(bk)
        prev = cur
    r = min(prev, key=prev.get)
    path = [r]
    for bk in reversed(back):
        r = bk[r]
        path.append(r)
    path.reverse()
    pts = [(0, path[0] * SEAM_CELL + SEAM_CELL / 2)]
    pts += [(c * SEAM_CELL + SEAM_CELL / 2, r * SEAM_CELL + SEAM_CELL / 2) for c, r in enumerate(path)]
    pts.append((w, path[-1] * SEAM_CELL + SEAM_CELL / 2))
    return pts


def _band(size, top_pts, bottom_pts):
    """두 경로 사이(위 경로 아래, 아래 경로 위)를 255 로 칠한 L 그림. 경로가 None 이면 그림 끝."""
    w, h = size
    top_pts = top_pts or [(0, 0), (w, 0)]
    bottom_pts = bottom_pts or [(0, h), (w, h)]
    img = Image.new("L", size, 0)
    poly = list(top_pts) + list(reversed(bottom_pts))
    ImageDraw.Draw(img).polygon(poly, fill=255)
    return img


def auto_mask(pos, neg, top, bottom):
    """두 그림이 형태로 다른 곳(가는 선 차이는 뺌) 중 두 곡선 경계 사이. 반환: L 그림(255 = 바꿀 곳)."""
    w, h = pos.size
    dl = ImageChops.difference(pos, neg).convert("L")
    top_pts = seam_path(dl, top) if top > 0 else None
    bottom_pts = seam_path(dl, bottom, "below") if bottom < 1 else None
    band = _band(pos.size, top_pts, bottom_pts)
    diff = ImageChops.multiply(dl.point(lambda v: 255 if v > AUTO_DIFF else 0), band)
    cw, ch = max(1, w // AUTO_CELL), max(1, h // AUTO_CELL)
    cells = diff.resize((cw, ch), Image.BOX).point(lambda v: 255 if v >= 255 * AUTO_CELL_FRAC else 0)
    cells = cells.filter(ImageFilter.MaxFilter(AUTO_GROW * 2 + 1))
    cells = cells.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))  # 바뀐 칸 사이 작은 틈 메움
    return ImageChops.multiply(cells.resize((w, h), Image.BILINEAR), band)  # 넓힌 칸도 곡선 경계를 넘지 않는다


def seam_score(pos, neg, m):
    """합성 경계 띠(부드럽게 섞이는 곳)에서 두 원본이 크게 다른 픽셀 비율(%). 띠가 없으면 0."""
    band = m.point(lambda v: 255 if 13 <= v <= 242 else 0)
    n = band.histogram()[255]
    if not n:
        return 0.0
    bad = ImageChops.difference(pos, neg).convert("L").point(lambda v: 255 if v > SEAM_BAD else 0)
    return round(ImageChops.multiply(band, bad).histogram()[255] * 100.0 / n, 1)


def effective_mask(item, mask):
    """이 쌍에 실제로 쓸 마스크 옵션(쌍별 설정이 있으면 그것). 마스크를 쓰지 않으면 None."""
    mask = mask_options(**(mask or {}))
    o = item.get("override")
    if o:
        if o["mode"] == "off":
            return None
        mask = {**mask, "mode": "auto", "top": o["top"], "bottom": o["bottom"], "keep": o["keep"]}
    if mask["mode"] == "off" or (mask["mode"] == "file" and not item.get("mask")):
        return None
    return mask


def pair_mask(item, pos, neg, mask, blur=True):
    """쌍의 마스크(L, 경계를 부드럽게) 또는 None. mask 는 effective_mask 결과. 마스크 파일이 있으면 그것을 쓴다."""
    if not mask:
        return None
    w = pos.size[0]
    if item.get("mask"):
        m = _load_mask(item["mask"]).resize(pos.size, Image.BILINEAR)
    elif mask["mode"] == "auto":
        m = auto_mask(pos, neg, mask["top"], mask["bottom"])
    else:
        return None
    return m.filter(ImageFilter.GaussianBlur(max(2.0, w * 0.01))) if blur else m


def raw_pair(item, resolution, steps):
    """합성 전 + / − 그림(학습 크기)."""
    pw, ph = item["pos_size"]
    hw, hh = half_size(pw, ph, resolution, steps)
    return _load_rgb(item["pos"]).resize((hw, hh), Image.LANCZOS), _load_rgb(item["neg"]).resize((hw, hh), Image.LANCZOS)


def render_pair(item, resolution, steps, mask=None, info=None):
    """학습에 들어갈 + / − 그림(같은 크기)과 마스크(없으면 None). 마스크 밖은 기준 그림(keep) 픽셀이다.
    info(dict)를 주면 seam(경계 어긋남 %), area(마스크 넓이 %)를 채운다."""
    pos, neg = raw_pair(item, resolution, steps)
    mask = effective_mask(item, mask)
    m = pair_mask(item, pos, neg, mask)
    if info is not None:
        info["seam"] = seam_score(pos, neg, m) if m is not None else None
        info["area"] = round(sum(i * c for i, c in enumerate(m.histogram())) * 100.0 / (255 * m.size[0] * m.size[1]), 1) if m is not None else None
    if m is not None:
        if mask["keep"] == "neg":
            pos = Image.composite(pos, neg, m)
        else:
            neg = Image.composite(neg, pos, m)
    return pos, neg, m


def build_pairs(folder, first, out_dir, resolution, steps, caption_ext=".txt", fallback_caption="", log=print, mask=None):
    """묶음 그림·캡션을 out_dir 에 만든다. 입력과 설정이 같으면 다시 만들지 않는다. 반환: (짝 수, 최대 묶음 면적, 최대 변)."""
    report = match_pairs(folder, first, caption_ext)
    pairs = report["pairs"]
    if not pairs:
        raise ValueError(f"쌍을 만들 그림이 없습니다 (그림 {report['image_count']}장, 2장 이상 필요)")
    mask = mask_options(**(mask or {}))
    os.makedirs(out_dir, exist_ok=True)
    sig = _signature(pairs, resolution, steps, fallback_caption, mask)
    manifest_path = os.path.join(out_dir, MANIFEST)
    try:
        with open(manifest_path, encoding="utf-8") as f:
            old = json.load(f)
    except Exception:
        old = {}
    if old.get("signature") == sig:
        log(f"slider pairs: {len(pairs)} pairs up to date in {out_dir}")
        return old["count"], old["max_area"], old["max_side"]

    for name in os.listdir(out_dir):  # 이전 묶음을 지운다 (이 폴더는 학습 파일이 만든 것만 둔다)
        if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS | {".txt", ".npz"}:
            os.remove(os.path.join(out_dir, name))
    max_area = max_side = 0
    used = set()
    masked = 0
    seams = []
    for it in pairs:
        info = {}
        left, right, m = render_pair(it, resolution, steps, mask, info)
        masked += m is not None
        if info.get("seam") is not None and info["seam"] >= SEAM_WARN:
            seams.append(f"#{it['index']} {os.path.basename(it['pos'])} {info['seam']:.0f}%")
        hw, hh = left.size
        combo = Image.new("RGB", (hw * 2, hh))
        combo.paste(left, (0, 0))
        combo.paste(right, (hw, 0))
        name = it["name"]
        while name.lower() in used:
            name += "_"
        used.add(name.lower())
        combo.save(os.path.join(out_dir, name + ".png"))
        # 빈 캡션 파일은 sd-scripts 가 'caption file is empty' assert 로 학습을 멈춘다(2026-11-13 실제 오류).
        # 캡션 파일이 없으면 경고만 하고 빈 캡션으로 학습하므로, 빈 캡션이면 파일을 쓰지 않는다
        caption = it["caption"] if (it["caption"] or "").strip() else fallback_caption  # 빈 캡션 파일도 '없음'으로
        if caption and caption.strip():
            with open(os.path.join(out_dir, name + ".txt"), "w", encoding="utf-8") as f:
                f.write(caption)
        max_area = max(max_area, hw * 2 * hh)
        max_side = max(max_side, hw * 2, hh)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"signature": sig, "count": len(pairs), "max_area": max_area, "max_side": max_side}, f)
    n_ovr = sum(1 for it in pairs if it.get("override"))
    if mask["mode"] != "off" or masked:
        log(f"slider pairs: mask {mask['mode']} (keep {mask['keep']}, auto rows {mask['top']:.2f}~{mask['bottom']:.2f}, "
            f"per-pair settings {n_ovr}): {masked}/{len(pairs)} pairs masked")
        if seams:
            log(f"slider pairs: seam mismatch >= {SEAM_WARN}% (check in pair preview): " + ", ".join(seams))
    log(f"slider pairs: built {len(pairs)} pairs in {out_dir}")
    return len(pairs), max_area, max_side


def dataset_toml(user_dataset, pairs_dir, max_area, max_side, steps, repeats):
    """묶음 폴더 하나를 쓰는 데이터셋 설정. 사용자 dataset.toml 의 [general]·첫 [[datasets]] 값을 이어받는다."""
    general = dict(user_dataset.get("general", {}))
    ds = dict((user_dataset.get("datasets") or [{}])[0])
    ds.pop("subsets", None)
    general["caption_extension"] = ".txt"  # 묶음 캡션은 .txt 로 만든다
    # 묶음 크기가 정확히 버킷이 되도록: 업스케일 안 함, 면적 한도 ≥ 가장 큰 묶음, 뒤집기·자르기 끔(뒤집으면 +·− 가 바뀐다)
    reso = int(math.ceil(math.sqrt(max_area) / steps)) * steps
    for k in ("flip_aug", "random_crop", "color_aug"):
        general.pop(k, None)
        ds.pop(k, None)
    ds.update(resolution=reso, enable_bucket=True, bucket_no_upscale=True, bucket_reso_steps=steps,
              min_bucket_reso=steps, max_bucket_reso=max(reso, int(math.ceil(max_side / steps)) * steps))
    lines = ["[general]"]
    lines += [f"{k} = {json.dumps(v, ensure_ascii=False)}" for k, v in general.items()]
    lines += ["", "[[datasets]]"]
    lines += [f"{k} = {json.dumps(v, ensure_ascii=False)}" for k, v in ds.items()]
    lines += ["", "[[datasets.subsets]]", f"image_dir = {json.dumps(pairs_dir.replace(chr(92), '/'))}", f"num_repeats = {int(repeats)}",
              "flip_aug = false", "random_crop = false"]
    return "\n".join(lines) + "\n"
