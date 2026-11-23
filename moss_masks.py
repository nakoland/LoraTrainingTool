# 마스크 학습용 마스크 (2026-11-12). UI 서버(moss_ui.py)와 학습(moss_train.py → moss_extras.prepare_args)이 함께 쓴다.
# 마스크는 그림과 같은 폴더에 같은 이름 + .msk 로 둔다(a.png → a.msk, 2026-11-12 사용자 결정). 내용은 흑백 PNG 이고
# 흰색 = 배울 곳, 검은색 = loss 에서 빠지는 곳. .msk 는 그림 확장자가 아니라 sd-scripts·UI 가 학습 그림으로 세지 않는다.
# (예전 방식) 데이터셋 설정의 conditioning_data_dir 마스크 폴더도 읽는다(같은 이름의 그림 파일). 둘 다 있으면 .msk 가 우선.
# sd-scripts 는 빨강 채널을 쓴다(load_image → RGB → [:, :, 0]).
#
# sd-scripts 의 마스크 학습은 conditioning_data_dir(ControlNet 데이터셋)로만 마스크를 받고 제약이 많다:
#  - 그 폴더의 모든 그림에 마스크가 있어야 한다(없으면 assert 로 학습이 시작되지 않음)
#  - 한 [[datasets]] 안의 폴더가 모두 conditioning_data_dir 을 가져야 한다
#  - 마스크 크기가 그림과 같아야 한다(다르면 assert)
#  - masked_loss 가 꺼져 있는데 conditioning_data_dir 이 있으면 데이터셋 설정 오류
# prepare_dataset 이 학습 직전에 .mask_fill/<출력 이름>/ 에 폴더마다 마스크를 모아(없는 그림은 흰색 = 그림 전체를 배움,
# 크기가 다르면 그림 크기로) 그쪽을 conditioning_data_dir 로 가리키는 데이터셋 설정을 만든다.
# 마스크 학습이 꺼져 있으면 conditioning_data_dir 만 뺀 설정을 쓴다.
#
# 투명 배경(2026-11-20 사용자 결정): sd-scripts 는 그림을 RGB 로 읽어(load_image → convert("RGB")) 투명도를 버리고, 투명한 곳
# 밑에 남은 픽셀(지우개로 지운 원래 배경·흰 얼룩)을 그대로 학습한다. transparent_mask(기본 켬)면 그림의 투명도를 마스크에 곱해
# 투명한 곳은 학습하지 않는다 — 투명 배경 그림이 있으면 마스크 학습이 꺼져 있어도 이 마스크만으로 마스크 학습을 켠다
# (그때 .msk 는 쓰지 않음: 투명도만). 반투명 가장자리는 그 값만큼만 배운다.
import json
import os
import threading

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FILL_ROOT = os.path.join(HERE, ".mask_fill")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
MASK_EXT = ".msk"
ASPECT_TOLERANCE = 0.02  # 가로세로 비가 이보다 다르면 다른 그림의 마스크로 보고 쓰지 않는다(그림 전체를 배움)
FILL_VERSION = 3
ALPHA_OPAQUE = 250        # 알파가 이보다 작으면 투명으로 센다
TRANSPARENT_MIN = 0.001   # 투명한 픽셀이 이 비율 이상이어야 투명 배경 그림으로 본다(가장자리 몇 점은 무시)
_alpha_cache = {}         # {절대 경로: (크기, 수정 시각, 투명 비율)} — UI 서버는 점검마다 부르므로 기억한다
_alpha_lock = threading.Lock()


def has_alpha_mode(im):
    return im.mode in ("RGBA", "LA", "PA", "RGBa", "La") or (im.mode == "P" and "transparency" in im.info)


def alpha_fraction(path):
    """그림의 투명한 픽셀 비율(알파 < ALPHA_OPAQUE). 투명도가 없는 형식이면 헤더만 읽고 0."""
    try:
        st = os.stat(path)
    except OSError:
        return 0.0
    key = os.path.abspath(path)
    with _alpha_lock:
        hit = _alpha_cache.get(key)
    if hit and hit[0] == st.st_size and hit[1] == st.st_mtime_ns:
        return hit[2]
    frac = 0.0
    try:
        with Image.open(path) as im:
            if has_alpha_mode(im):
                a = im.convert("RGBA").getchannel("A")
                hist = a.histogram()
                frac = sum(hist[:ALPHA_OPAQUE]) / max(1, a.size[0] * a.size[1])
    except OSError:
        frac = 0.0
    with _alpha_lock:
        _alpha_cache[key] = (st.st_size, st.st_mtime_ns, frac)
    return frac


def is_transparent(path):
    return alpha_fraction(path) >= TRANSPARENT_MIN


def transparent_images(image_dir):
    """학습 폴더의 {투명 배경 그림 경로: 투명 비율}."""
    out = {}
    for img in list_images(image_dir):
        f = alpha_fraction(img)
        if f >= TRANSPARENT_MIN:
            out[img] = f
    return out


def alpha_channel(path):
    with Image.open(path) as im:
        return im.convert("RGBA").getchannel("A")


def is_image(name):
    return os.path.splitext(name)[1].lower() in IMAGE_EXTS


def list_images(folder):
    """폴더 바로 아래의 그림(sd-scripts glob_images 와 같이 하위 폴더는 보지 않음), 이름순."""
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    return sorted((os.path.join(folder, n) for n in names if is_image(n) and os.path.isfile(os.path.join(folder, n))), key=str.lower)


def msk_path(image):
    """그림 옆의 마스크 파일 경로: a.png → a.msk."""
    return os.path.splitext(image)[0] + MASK_EXT


def mask_index(mask_dir):
    """(예전 방식) 마스크 폴더의 {이름(확장자 앞, 소문자): 경로}. sd-scripts 처럼 같은 이름이 여럿이면 정렬해 첫 번째."""
    out = {}
    if not mask_dir or not os.path.isdir(mask_dir):
        return out
    for p in list_images(mask_dir):
        out.setdefault(os.path.splitext(os.path.basename(p))[0].lower(), p)
    return out


def folder_masks(image_dir, legacy_dir=""):
    """학습 폴더의 {그림 경로: 마스크 경로}. .msk 가 우선이고, 없으면 예전 마스크 폴더의 같은 이름 그림."""
    legacy = mask_index(legacy_dir)
    try:
        names = {n.lower() for n in os.listdir(image_dir)}
    except OSError:
        return {}
    out = {}
    for img in list_images(image_dir):
        stem = os.path.splitext(os.path.basename(img))[0]
        if (stem + MASK_EXT).lower() in names:
            out[img] = msk_path(img)
        elif stem.lower() in legacy:
            out[img] = legacy[stem.lower()]
    return out


def aspect_ok(mask_size, image_size):
    (mw, mh), (iw, ih) = mask_size, image_size
    return mw > 0 and mh > 0 and iw > 0 and ih > 0 and abs((mw / mh) / (iw / ih) - 1.0) <= ASPECT_TOLERANCE


def mask_channel(path, size=None):
    """학습이 쓰는 값(빨강 채널)을 L 그림으로. size 를 주면 그 크기로 늘리거나 줄인다. .msk 도 내용이 PNG 라 그대로 열린다."""
    with Image.open(path) as im:
        m = im.convert("RGB").getchannel("R")
    if size and m.size != tuple(size):
        m = m.resize(tuple(size), Image.BILINEAR)
    return m


def save_mask(image_l, path):
    """L 그림을 .msk(내용은 PNG)로 원자 저장."""
    tmp = path + ".tmp"
    image_l.save(tmp, format="PNG")
    os.replace(tmp, path)


def _fill_folder(image_dir, legacy_dir, out_dir, use_msk=True, use_alpha=False):
    """학습용 마스크 폴더 하나를 만든다. 바뀐 그림만 다시 쓴다. → {masks, filled, resized, bad_aspect, alpha}
    use_msk: .msk(·예전 마스크 폴더)를 쓴다(마스크 학습 켬). use_alpha: 투명 배경 그림은 투명도를 마스크에 곱한다."""
    os.makedirs(out_dir, exist_ok=True)
    sig_path = os.path.join(out_dir, "_fill.json")
    try:
        with open(sig_path, encoding="utf-8") as f:
            old = json.load(f)
        if old.get("version") != FILL_VERSION:
            old = {}
    except (OSError, ValueError):
        old = {}
    old_files = old.get("files", {})
    masks = folder_masks(image_dir, legacy_dir) if use_msk else {}
    files, stats = {}, {"masks": 0, "filled": 0, "resized": 0, "bad_aspect": [], "alpha": 0}
    for img in list_images(image_dir):
        stem = os.path.splitext(os.path.basename(img))[0]
        st = os.stat(img)
        m = masks.get(img)
        alpha = use_alpha and is_transparent(img)
        key = [st.st_size, st.st_mtime_ns, bool(alpha)]
        if m:
            ms = os.stat(m)
            key += [os.path.abspath(m), ms.st_size, ms.st_mtime_ns]
        dest = os.path.join(out_dir, stem + ".png")
        prev = old_files.get(stem.lower())
        if not (prev and prev["key"] == key and os.path.isfile(dest)):
            with Image.open(img) as im:
                size = im.size
            stat, out = "filled", None
            if m:
                with Image.open(m) as mi:
                    msize = mi.size
                if not aspect_ok(msize, size):
                    stat = "bad_aspect"
                else:
                    out = mask_channel(m, size)
                    stat = "masks" if msize == size else "resized"
            if out is None:
                out = Image.new("L", size, 255)  # 마스크가 없는 그림: 그림 전체를 배운다
            if alpha:  # 투명한 곳은 배우지 않는다(마스크 × 투명도)
                from PIL import ImageChops
                out = ImageChops.multiply(out, alpha_channel(img))
            tmp = dest + ".tmp.png"
            out.save(tmp, compress_level=1)
            os.replace(tmp, dest)
            prev = {"key": key, "stat": stat, "alpha": bool(alpha)}
        files[stem.lower()] = prev
        stats["alpha"] += bool(prev.get("alpha"))
        if prev["stat"] == "bad_aspect":
            stats["filled"] += 1
            stats["bad_aspect"].append(os.path.basename(img))
        else:
            stats[prev["stat"]] += 1
    keep = {os.path.normcase(os.path.join(out_dir, os.path.splitext(os.path.basename(p))[0] + ".png")) for p in list_images(image_dir)}
    for p in list_images(out_dir):  # 지워진 그림의 마스크(남으면 sd-scripts 가 '남는 마스크' 경고)
        if os.path.normcase(p) not in keep:
            os.remove(p)
    with open(sig_path, "w", encoding="utf-8") as f:
        json.dump({"version": FILL_VERSION, "image_dir": image_dir, "legacy_dir": legacy_dir, "files": files}, f)
    return stats


def _toml_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise TypeError(f"데이터셋 설정 값 형식을 옮길 수 없습니다: {v!r}")


def dataset_toml(data):
    """tomllib 로 읽은 데이터셋 설정을 다시 TOML 로 쓴다([general] · [[datasets]] · [[datasets.subsets]] 만)."""
    lines = []
    if data.get("general"):
        lines += ["[general]"] + [f"{k} = {_toml_value(v)}" for k, v in data["general"].items()] + [""]
    for ds in data.get("datasets", []):
        lines.append("[[datasets]]")
        lines += [f"{k} = {_toml_value(v)}" for k, v in ds.items() if k != "subsets"]
        lines.append("")
        for s in ds.get("subsets", []):
            lines.append("[[datasets.subsets]]")
            lines += [f"{k} = {_toml_value(v)}" for k, v in s.items()]
            lines.append("")
    return "\n".join(lines)


def resolve(path, base):
    if not path:
        return ""
    path = os.path.expandvars(os.path.expanduser(str(path)))
    return os.path.normpath(path if os.path.isabs(path) else os.path.join(base, path))


def prepare_dataset(dataset_config, masked, output_name, log=print, alpha=True):
    """학습 직전에 부른다. → (새 데이터셋 설정 경로 또는 None(원래 설정 그대로), 투명도를 마스크로 쓴 그림 수).
    투명도를 쓴 그림이 있으면 부르는 쪽이 마스크 학습(masked_loss)을 켜야 한다."""
    import tomllib
    with open(dataset_config, "rb") as f:
        data = tomllib.load(f)
    base = os.getcwd()  # sd-scripts 는 데이터셋 설정의 상대 경로를 작업 폴더 기준으로 읽는다
    subs = [s for ds in data.get("datasets", []) for s in ds.get("subsets", [])]
    has_legacy = any(s.get("conditioning_data_dir") for s in subs)
    masks = {i: folder_masks(resolve(s.get("image_dir"), base), resolve(s.get("conditioning_data_dir"), base)) for i, s in enumerate(subs)}
    total = sum(len(m) for m in masks.values())
    n_alpha = sum(len(transparent_images(resolve(s.get("image_dir"), base))) for s in subs) if alpha else 0
    use_alpha = n_alpha > 0
    if not total and not has_legacy and not use_alpha:
        if masked:
            log("masks: 마스크 학습이 켜져 있지만 마스크(.msk)가 하나도 없어 그림 전체를 배웁니다")
        return None, 0
    out_root = os.path.join(FILL_ROOT, output_name or "default")
    if not masked and not has_legacy and not use_alpha:  # .msk 만 있으면 sd-scripts 는 모르므로 설정을 바꿀 것이 없다
        log(f"masks: 마스크 학습이 꺼져 있어 마스크 {total}장을 쓰지 않습니다(그림 전체를 배움)")
        return None, 0
    if not masked and not use_alpha:
        for s in subs:
            s.pop("conditioning_data_dir", None)
        log(f"masks: 마스크 학습이 꺼져 있어 마스크 {total}장을 쓰지 않습니다(그림 전체를 배움)")
    else:
        if not masked:
            log(f"masks: 투명 배경 그림 {n_alpha}장 — 투명한 곳은 학습하지 않도록 투명도를 마스크로 씁니다"
                + (f"(마스크 학습이 꺼져 있어 .msk {total}장은 쓰지 않음)" if total else ""))
        for i, s in enumerate(subs):
            image_dir = resolve(s.get("image_dir"), base)
            legacy = resolve(s.get("conditioning_data_dir"), base) if masked else ""
            out = os.path.join(out_root, f"{i:02d}_{os.path.basename(image_dir) or 'folder'}")
            st = _fill_folder(image_dir, legacy if legacy and os.path.isdir(legacy) else "", out, use_msk=masked, use_alpha=use_alpha)
            s["image_dir"] = image_dir.replace("\\", "/")
            s["conditioning_data_dir"] = out.replace("\\", "/")
            note = f"마스크 {st['masks']}장" + (f", 크기 맞춤 {st['resized']}장" if st["resized"] else "") + \
                   (f", 마스크 없음(그림 전체) {st['filled']}장" if st["filled"] else "") + \
                   (f", 투명한 곳 빼기 {st['alpha']}장" if st["alpha"] else "")
            if legacy and not os.path.isdir(legacy):
                note += f" — 예전 마스크 폴더가 없습니다: {legacy}"
            if st["bad_aspect"]:
                note += f" — 가로세로 비가 달라 쓰지 않은 마스크 {len(st['bad_aspect'])}장: {', '.join(st['bad_aspect'][:5])}"
            log(f"masks: {os.path.basename(image_dir)}: {note}")
    for s in subs:  # 상대 경로가 새 설정 파일 위치 기준으로 바뀌지 않게
        if s.get("image_dir"):
            s["image_dir"] = resolve(s["image_dir"], base).replace("\\", "/")
    os.makedirs(out_root, exist_ok=True)
    path = os.path.join(out_root, "dataset.toml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(dataset_toml(data))
    return path, n_alpha
