# MOSS-Image LoRA 학습 UI (로컬 웹 서버)
# 사용법: run.bat  (포트를 바꾸려면 run.bat --port 7866)
# - 설정 toml / dataset.toml / prompts.txt 를 주석을 보존한 채 편집
# - 데이터셋 점검(캡션 누락·캐시·해상도), 설정 검사
# - 학습 실행/중지, 로그·진행률·loss 그래프·샘플 이미지 표시
# 학습 프로세스의 출력은 파일로 직접 쓰므로 UI 를 닫았다 다시 열어도 학습은 계속되고 다시 붙는다.

import argparse
import codecs
import collections
import ctypes
import datetime
import importlib.metadata
import importlib.util
import json
import mimetypes
import os
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import tomllib
import urllib.parse
import webbrowser
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    import toml as toml_lib  # sd-scripts 가 설정을 읽는 라이브러리. 저장 전 검증에도 쓴다
except ImportError:
    toml_lib = None

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "sd-scripts")  # install.bat 이 받아 둔 sd-scripts
PYTHON = os.path.join(ROOT, "venv", "Scripts", "python.exe")
STATE_FILE = os.path.join(HERE, ".moss_ui_state.json")
RUN_LOG_DIR = os.path.join(HERE, "ui_logs")
DEFAULT_CONFIG = os.path.join(HERE, "moss_lora_32gb.toml")
PROFILE_DIR = os.path.join(HERE, "profiles")  # <이름>/<이름>.toml + 데이터셋·프롬프트 사본
PROFILE_NAME_RE = re.compile(r"^\w[\w .\-]{0,79}$")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
CAPTION_SEPARATOR = "---===---"
MAX_SCAN_IMAGES = 20000

# 설정 파일에 없는 키를 새로 쓸 때 넣을 섹션 (sd-scripts 는 섹션을 평탄화하므로 위치는 가독성용)
SECTION_FOR_KEY = {
    "model": ["pretrained_model_name_or_path", "text_encoder", "vae", "tokenizer_cache_dir", "moss_max_token_length"],
    "dataset": ["dataset_config", "max_data_loader_n_workers", "persistent_data_loader_workers"],
    "output": [
        "output_dir", "output_name", "logging_dir", "save_model_as", "save_every_n_epochs", "save_last_n_epochs",
        "sample_prompts", "sample_every_n_epochs", "sample_at_first", "save_state",
    ],
    "network": ["network_module", "network_dim", "network_alpha", "network_train_unet_only", "network_args", "network_weights"],
    "optimizer": [
        "optimizer_type", "optimizer_args", "learning_rate", "lr_scheduler", "lr_warmup_steps", "max_train_epochs",
        "seed", "max_grad_norm", "gradient_accumulation_steps",
    ],
    "timestep": [
        "timestep_sampling", "logit_mean", "logit_std", "discrete_flow_shift", "sigmoid_scale", "weighting_scheme",
        "timestep_cutoff", "timestep_uniform_ratio",
    ],
    "memory": [
        "mixed_precision", "gradient_checkpointing", "cache_latents", "cache_latents_to_disk", "cache_text_encoder_outputs",
        "cache_text_encoder_outputs_to_disk", "convrot_int8", "convrot_int8_text_encoder", "blocks_to_swap",
        "unsloth_offload_checkpointing", "attn_mode", "split_attn", "compile",
    ],
}
KEY_SECTION = {k: s for s, keys in SECTION_FOR_KEY.items() for k in keys}

if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# 주석 보존 TOML 편집
# ---------------------------------------------------------------------------

HEADER_RE = re.compile(r"^\s*(\[\[?)\s*([A-Za-z0-9_.\-]+)\s*\]\]?\s*(#.*)?$")
KEY_RE = re.compile(r"^(\s*)([A-Za-z0-9_\-]+)\s*=")
COMMENTED_KEY_RE = re.compile(r"^(\s*)#\s*([A-Za-z0-9_\-]+)\s*=")
TOML_NUMBER_RE = re.compile(r"^[+-]?(\d[\d_]*)(\.\d[\d_]*)?([eE][+-]?\d+)?$")


def split_tables(lines):
    """파일을 테이블 단위로 나눈다. 반환: [{name, index, start, end}] (start=헤더 줄, 루트는 start=-1).
    배열 테이블([[x]])은 같은 이름이 나온 순서로 index 를 붙인다."""
    tables = [{"name": "", "index": 0, "start": -1, "end": len(lines), "array": False}]
    counts = collections.Counter()
    for i, line in enumerate(lines):
        m = HEADER_RE.match(line)
        if not m:
            continue
        is_array = m.group(1) == "[["
        name = m.group(2)
        idx = counts[name]
        counts[name] += 1
        tables[-1]["end"] = i
        tables.append({"name": name, "index": idx, "start": i, "end": len(lines), "array": is_array})
    return tables


def toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(v) for v in value) + "]"
    if isinstance(value, dict) and "raw" in value:  # 숫자를 사용자가 쓴 표기(1e-4 등) 그대로
        raw = str(value["raw"]).strip()
        if not TOML_NUMBER_RE.match(raw):
            raise ValueError(f"숫자가 아닙니다: {raw}")
        return raw
    return json.dumps(str(value), ensure_ascii=False)


def toml_set(lines, table, index, key, value, new_section=None):
    """table[index] 안의 key 값을 바꾼다. value=None 이면 줄을 주석 처리한다(기본값 사용).
    키가 없으면 주석 처리된 같은 키 줄을 되살리고, 그것도 없으면 테이블 끝에 추가한다."""
    tables = split_tables(lines)
    target = None
    if table is None:
        # 설정 파일: 섹션은 의미가 없으므로 어느 섹션에 있든 그 줄을 고친다
        for t in tables:
            for i in range(t["start"] + 1, t["end"]):
                m = KEY_RE.match(lines[i])
                if m and m.group(2) == key:
                    target = t
                    break
            if target:
                break
        if target is None:
            sec = new_section or KEY_SECTION.get(key, "extra")
            target = next((t for t in tables if t["name"] == sec), None)
            if target is None:
                for t in tables:
                    for i in range(t["start"] + 1, t["end"]):
                        m = COMMENTED_KEY_RE.match(lines[i])
                        if m and m.group(2) == key:
                            target = t
                            break
                    if target:
                        break
            if target is None:
                if value is None:
                    return lines
                while lines and lines[-1].strip() == "":
                    lines.pop()
                lines += ["", f"[{sec}]", f"{key} = {toml_value(value)}"]
                return lines
    else:
        target = next((t for t in tables if t["name"] == table and t["index"] == index), None)
        if target is None:
            raise ValueError(f"테이블이 없습니다: {table}[{index}]")

    rng = range(target["start"] + 1, target["end"])
    for i in rng:
        m = KEY_RE.match(lines[i])
        if m and m.group(2) == key:
            if value is None:
                lines[i] = m.group(1) + "# " + lines[i][len(m.group(1)):]
            else:
                lines[i] = f"{m.group(1)}{key} = {toml_value(value)}"
            return lines
    if value is None:
        return lines
    for i in rng:
        m = COMMENTED_KEY_RE.match(lines[i])
        if m and m.group(2) == key:
            lines[i] = f"{m.group(1)}{key} = {toml_value(value)}"
            return lines
    # 테이블 끝(뒤쪽 빈 줄 앞)에 추가. 들여쓰기는 테이블 헤더를 따른다
    indent = ""
    if target["start"] >= 0:
        indent = re.match(r"^(\s*)", lines[target["start"]]).group(1)
    pos = target["end"]
    while pos - 1 > target["start"] and lines[pos - 1].strip() == "":
        pos -= 1
    lines.insert(pos, f"{indent}{key} = {toml_value(value)}")
    return lines


def toml_remove_table(lines, table, index):
    t = next((t for t in split_tables(lines) if t["name"] == table and t["index"] == index), None)
    if t is None or t["start"] < 0:
        return lines
    return lines[: t["start"]] + lines[t["end"]:]


def toml_append_subset(lines, image_dir):
    while lines and lines[-1].strip() == "":
        lines.pop()
    lines += ["", "  [[datasets.subsets]]", f"  image_dir = {toml_value(image_dir)}", "  num_repeats = 1"]
    return lines


def validate_toml(text):
    tomllib.loads(text)
    if toml_lib is not None:
        toml_lib.loads(text)


def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text_atomic(path, text):
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        for attempt in range(10):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:  # Windows: 다른 프로세스가 읽는 중이면 잠깐 실패한다
                if attempt == 9:
                    raise
                time.sleep(0.05)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def flatten_config(data):
    flat = {}
    for k, v in data.items():
        if isinstance(v, dict):
            flat.update(v)
        else:
            flat[k] = v
    return flat


# ---------------------------------------------------------------------------
# 경로·데이터셋 분석
# ---------------------------------------------------------------------------

def norm(p):
    return os.path.normcase(os.path.abspath(p)) if p else ""


def resolve_path(p):
    if not p:
        return p
    p = os.path.expandvars(os.path.expanduser(str(p)))
    return p if os.path.isabs(p) else os.path.join(ROOT, p)  # 학습은 저장소 루트에서 실행된다


def path_info(p, kind="file"):
    if not p:
        return {"state": "empty"}
    rp = resolve_path(p)
    if kind == "dir":
        if os.path.isdir(rp):
            return {"state": "ok"}
        return {"state": "missing" if not os.path.exists(rp) else "notdir"}
    if os.path.isfile(rp):
        size = os.path.getsize(rp)
        name = os.path.basename(rp).lower()
        info = {"state": "ok", "size": size}
        if "int8_convrot" in name or "convrot" in name:
            info["note"] = "ConvRot INT8 (자동 감지)"
        elif "gptq" in name or "w4a8" in name:
            info["note"] = "GPTQ 파일은 지원하지 않음"
            info["state"] = "bad"
        elif re.search(r"-0000[2-9]-of-", name):
            info["note"] = "첫 번째 샤드(00001)를 지정해야 함"
            info["state"] = "bad"
        return info
    return {"state": "missing" if not os.path.exists(rp) else "notfile"}


def list_images(d):
    # sd-scripts 는 image_dir 바로 아래만 읽는다(하위 폴더를 훑는 옵션이 없다)
    out = []
    for f in os.listdir(d):
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS and os.path.isfile(os.path.join(d, f)):
            out.append(os.path.join(d, f))
    return sorted(out)


DEFAULT_SOURCE_ROOT = os.path.join(HERE, "datasets").replace("\\", "/")
TRAINING_DIR_RE = re.compile(r"^trainn?ing$", re.I)
REPEATS_RE = re.compile(r"^(\d+)_")  # kohya 방식 폴더 이름: <반복 수>_<이름>
EXCLUDED_DIRS = {"discarded", "original_format"}  # 전처리 도구가 남긴 폴더. 학습 대상이 아니다
CAPTION_EXTENSIONS = {".txt", ".caption"}
MAX_SUBFOLDER_DEPTH = 3
MAX_SUBFOLDER_WALK = 300
_sets_cache = {}


def dir_stats(path, cap_ext):
    """폴더 한 단계만 본다: 이미지 수, 내용 있는 캡션과 짝이 맞는 이미지 수, 하위 폴더 이름."""
    stems, caps, subdirs = [], set(), []
    other_captions = 0
    try:
        with os.scandir(path) as it:
            for e in it:
                if e.is_dir(follow_symlinks=False):
                    subdirs.append(e.name)
                    continue
                stem, ext = os.path.splitext(e.name)
                ext = ext.lower()
                if ext in IMAGE_EXTENSIONS:
                    stems.append(stem)
                elif ext == cap_ext:
                    if e.stat().st_size:
                        caps.add(stem)
                elif ext in CAPTION_EXTENSIONS:
                    other_captions += 1
    except OSError:
        pass
    name = os.path.basename(path)
    m = REPEATS_RE.match(name)
    return {"images": len(stems), "captions": sum(s in caps for s in stems), "other_captions": other_captions,
            "subdirs": sorted(subdirs, key=str.lower), "repeats": int(m.group(1)) if m else None,
            "excluded": name.lower() in EXCLUDED_DIRS}


def folder_verdict(st, cap_ext, scan=None):
    """폴더 하나가 학습에 바로 쓸 만한지 판정한다. scan 은 선택한 폴더만 주는 scan_subset 결과.
    level: ok / warn / bad / skip(학습 제외) / none(이미지 없음)"""
    images = st["images"]
    if not images:
        return {"level": "none", "notes": ["이 폴더에는 바로 쓸 이미지가 없습니다"]}
    if st["excluded"]:
        return {"level": "skip", "notes": ["전처리 도구가 남긴 폴더라 학습 대상이 아닙니다"]}
    missing = scan["no_caption_count"] + scan["empty_caption_count"] if scan else images - st["captions"]
    level, notes = "ok", []
    if missing >= images:
        level = "bad"
        notes.append(f"캡션({cap_ext})이 하나도 없습니다 — 전부 빈 캡션으로 학습됩니다")
    elif missing:
        level = "warn"
        notes.append(f"캡션 없는 이미지 {missing}장 — 이 이미지들은 빈 캡션으로 학습됩니다")
    else:
        detail = f" · 평균 {scan['avg_caption_chars']}자" if scan else ""
        notes.append(f"이미지 {images}장 모두 캡션 있음{detail}")
    if missing and st["other_captions"]:
        notes.append(f"다른 확장자 캡션 파일 {st['other_captions']}개가 있습니다 — 캡션 확장자 설정을 확인하세요")
    if scan and scan["bad_separator"]:
        level = "bad"
        notes.append(f"캡션 구분자 줄 오류 {len(scan['bad_separator'])}장 — 구분자 뒤에는 # 주석만 올 수 있습니다")
    if st["repeats"] is None:
        level = "warn" if level == "ok" else level
        notes.append("폴더 이름에 반복 수가 없어 반복 1 로 들어갑니다")
    if scan and scan["small_count"]:
        notes.append(f"저해상도 이미지 {scan['small_count']}장")
    return {"level": level, "notes": notes}


def subfolder_report(path, cap_ext):
    """선택한 폴더 아래에서 이미지가 든 폴더를 모은다. usable 은 일괄 등록 대상(N_이름 형식)."""
    rows, walked, truncated = [], 0, False
    for base, dirs, _files in os.walk(path):
        dirs[:] = sorted((d for d in dirs if not d.startswith(".")), key=str.lower)
        if base == path:
            continue
        walked += 1
        if walked > MAX_SUBFOLDER_WALK:
            truncated = True
            break
        rel = os.path.relpath(base, path)
        st = dir_stats(base, cap_ext)
        if st["excluded"] or rel.count(os.sep) + 1 >= MAX_SUBFOLDER_DEPTH:
            dirs[:] = []
        if not st["images"]:
            continue
        usable = st["repeats"] is not None and not st["excluded"]
        verdict = folder_verdict(st, cap_ext)
        if not usable and not st["excluded"]:
            verdict = {"level": "skip", "notes": ["폴더 이름이 <반복>_<이름> 형식이 아니라 일괄 등록에서 뺍니다 — 이 폴더를 직접 고르면 쓸 수 있습니다"]}
        rows.append({"name": rel.replace("\\", "/"), "path": base.replace("\\", "/"), "images": st["images"],
                     "captions": st["captions"], "repeats": st["repeats"], "usable": usable, "verdict": verdict})
    return rows, truncated


def browse_dir(path, cap_ext, detail=None):
    path = os.path.abspath(path)
    st = dir_stats(path, cap_ext)
    entries = []
    for name in st["subdirs"]:
        p = os.path.join(path, name)
        s = dir_stats(p, cap_ext)
        entries.append({"name": name, "path": p.replace("\\", "/"), "images": s["images"], "captions": s["captions"],
                        "subdirs": len(s["subdirs"]), "repeats": s["repeats"],
                        "is_training": bool(TRAINING_DIR_RE.match(name))})
    parent = os.path.dirname(path)
    info = {"path": path.replace("\\", "/"), "parent": parent.replace("\\", "/") if parent != path else None,
            "images": st["images"], "captions": st["captions"], "repeats": st["repeats"],
            "preview": [p.replace("\\", "/") for p in list_images(path)[:8]] if st["images"] else [],
            "entries": entries}
    if detail is not None:
        general = {"caption_extension": cap_ext, "enable_multiline_caption": detail["multiline"]}
        scan = scan_subset({"image_dir": path}, general, detail["min_reso"]) if st["images"] else None
        info["verdict"] = folder_verdict(st, cap_ext, scan)
        info["subfolders"], info["subfolders_truncated"] = subfolder_report(path, cap_ext)
    return info


def find_training_sets(root, cap_ext, force=False):
    """root 아래의 training 폴더를 찾아, 그 안에서 이미지가 든 폴더를 모두 모은다.
    반환 항목: topic(=root 바로 아래 폴더), group(이미지 폴더의 부모 = 세트), 폴더 통계."""
    key = (norm(root), cap_ext)
    hit = _sets_cache.get(key)
    if hit and not force and time.time() - hit[0] < 60:
        return hit[1]
    root = os.path.abspath(root)
    found = []
    training_dirs = []
    for base, dirs, _files in os.walk(root):
        depth = os.path.relpath(base, root).count(os.sep) if base != root else -1
        dirs[:] = sorted((d for d in dirs if not d.startswith(".")), key=str.lower)
        hits = [d for d in dirs if TRAINING_DIR_RE.match(d)]
        for d in hits:
            training_dirs.append(os.path.join(base, d))
        # training 안쪽은 아래에서 따로 훑는다 / 너무 깊이는 내려가지 않는다
        dirs[:] = [] if depth >= 4 else [d for d in dirs if d not in hits]
    for tdir in training_dirs:
        for base, dirs, _files in os.walk(tdir):
            depth = os.path.relpath(base, tdir).count(os.sep) if base != tdir else -1
            dirs[:] = sorted((d for d in dirs if not d.startswith(".")), key=str.lower)
            if depth >= 3:
                dirs[:] = []
            st = dir_stats(base, cap_ext)
            if not st["images"]:
                continue
            rel = os.path.relpath(base, root).replace("\\", "/")
            found.append({
                "path": base.replace("\\", "/"), "rel": rel, "topic": rel.split("/")[0],
                "group": os.path.relpath(os.path.dirname(base), root).replace("\\", "/"),
                "name": os.path.basename(base), "images": st["images"], "captions": st["captions"],
                "repeats": st["repeats"],
            })
    _sets_cache[key] = (time.time(), found)
    return found


def image_size(path):
    try:
        from PIL import Image

        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


def image_thumb(path, size=220):
    import io

    from PIL import Image

    with Image.open(path) as im:
        im.thumbnail((size, size))
        buf = io.BytesIO()
        im.convert("RGB").save(buf, "JPEG", quality=82)
    return buf.getvalue()


def split_caption(cap, multiline):
    """캡션 파일 내용을 학습에 쓰이는 후보로 나눈다. 두 번째 값은 구분자 줄 오류 여부.
    여러 줄 캡션이 꺼져 있으면 sd-scripts 는 첫 줄만 읽는다."""
    if not multiline:
        return [c for c in [cap.split("\n")[0].strip()] if c], False
    cands, cur, bad = [], [], False
    for line in cap.split("\n"):
        s = line.strip()
        if s.startswith(CAPTION_SEPARATOR):
            if s[len(CAPTION_SEPARATOR):].split("#", 1)[0].strip():
                bad = True
            cands.append("\n".join(cur).strip())
            cur = []
        else:
            cur.append(line)
    cands.append("\n".join(cur).strip())
    return [c for c in cands if c], bad


def subset_caption_options(subset, general):
    cap_ext = subset.get("caption_extension", general.get("caption_extension", ".caption"))
    if not cap_ext.startswith("."):
        cap_ext = "." + cap_ext
    return cap_ext, subset.get("enable_multiline_caption", general.get("enable_multiline_caption", False))


def dataset_images(config_path):
    """샘플 프롬프트 화면용: 저장된 데이터셋 폴더의 이미지와 캡션 후보를 모은다."""
    cfg = flatten_config(tomllib.loads(read_text(config_path)))
    ds = cfg.get("dataset_config")
    if not ds or not os.path.isfile(resolve_path(ds)):
        return {"folders": []}
    _data, general, d0, subsets = load_dataset(resolve_path(ds))
    merged = {**general, **{k: v for k, v in d0.items() if k != "subsets"}}
    folders = []
    for sub in subsets:
        rp = resolve_path(sub.get("image_dir", ""))
        if not rp or not os.path.isdir(rp):
            continue
        cap_ext, multiline = subset_caption_options(sub, merged)
        _image_roots.add(os.path.abspath(rp))
        images = []
        for img in list_images(rp):
            try:
                with open(os.path.splitext(img)[0] + cap_ext, "r", encoding="utf-8") as f:
                    captions = split_caption(f.read(), multiline)[0]
            except (OSError, UnicodeDecodeError):
                captions = []
            images.append({"path": img.replace("\\", "/"), "name": os.path.basename(img), "captions": captions})
        folders.append({"path": rp.replace("\\", "/"), "name": os.path.basename(rp), "images": images})
    return {"folders": folders}


def cached_dropout_rate(npz_path):
    """TE 캐시에 저장된 caption_dropout_rate. npz 는 zip 이고 값은 .npy 끝의 float32 하나다. 읽지 못하면 None."""
    try:
        with zipfile.ZipFile(npz_path) as z:
            return struct.unpack("<f", z.read("caption_dropout_rate.npy")[-4:])[0]
    except (OSError, KeyError, zipfile.BadZipFile, struct.error):
        return None


def scan_subset(subset, general, min_reso):
    image_dir = subset.get("image_dir", "")
    res = {"image_dir": image_dir, "ok": False}
    rp = resolve_path(image_dir)
    if not image_dir or not os.path.isdir(rp):
        res["error"] = "폴더가 없습니다"
        return res
    cap_ext, multiline = subset_caption_options(subset, general)
    images = list_images(rp)
    names_in_dir = None
    no_caption, empty_caption, small, bad_sep = [], [], [], []
    cand_counts, cap_lens = [], []
    latent_cached = te_cached = 0
    te_sample = None
    sizes = collections.Counter()
    for img in images:
        stem = os.path.splitext(img)[0]
        cap_path = stem + cap_ext
        base_dir = os.path.dirname(img)
        if names_in_dir is None or names_in_dir[0] != base_dir:
            names_in_dir = (base_dir, set(os.listdir(base_dir)))
        files = names_in_dir[1]
        bname = os.path.basename(stem)
        if os.path.basename(cap_path) in files:
            try:
                with open(cap_path, "r", encoding="utf-8") as f:
                    cap = f.read()
            except UnicodeDecodeError:
                cap = ""
            text = cap.strip()
            if not text:
                empty_caption.append(os.path.basename(img))
            else:
                cands, bad = split_caption(cap, multiline)
                if bad:
                    bad_sep.append(os.path.basename(img))
                cand_counts.append(max(1, len(cands)))
                cap_lens.extend(len(c) for c in cands)
        else:
            no_caption.append(os.path.basename(img))
        if any(f.startswith(bname + "_") and f.endswith("_moss.npz") for f in files):
            latent_cached += 1
        if bname + "_moss_te.npz" in files:
            te_cached += 1
            te_sample = te_sample or stem + "_moss_te.npz"
        if len(images) <= 3000:
            sz = image_size(img)
            if sz:
                sizes[f"{sz[0]}x{sz[1]}"] += 1
                if max(sz) < min_reso:
                    small.append(f"{os.path.basename(img)} ({sz[0]}x{sz[1]})")
    res.update(
        ok=True,
        caption_extension=cap_ext,
        num_images=len(images),
        truncated=len(images) >= MAX_SCAN_IMAGES,
        num_repeats=int(subset.get("num_repeats", 1)),
        no_caption=no_caption[:200],
        no_caption_count=len(no_caption),
        empty_caption=empty_caption[:200],
        empty_caption_count=len(empty_caption),
        bad_separator=bad_sep[:50],
        small=small[:100],
        small_count=len(small),
        latent_cached=latent_cached,
        te_cached=te_cached,
        te_dropout=cached_dropout_rate(te_sample) if te_sample else None,
        multi_candidate=sum(1 for c in cand_counts if c > 1),
        avg_caption_chars=round(sum(cap_lens) / len(cap_lens)) if cap_lens else 0,
        max_caption_chars=max(cap_lens) if cap_lens else 0,
        top_sizes=sizes.most_common(6),
        preview=[img for img in images[:12]],
    )
    return res


def load_dataset(path):
    data = tomllib.loads(read_text(path))
    general = data.get("general", {})
    datasets = data.get("datasets", [])
    ds = datasets[0] if datasets else {}
    return data, general, ds, ds.get("subsets", [])


def analyze(config_path):
    """설정 전체를 점검해 오류/경고/정보 목록과 데이터셋 통계를 돌려준다."""
    issues = []

    def add(level, msg, tab="run"):
        issues.append({"level": level, "msg": msg, "tab": tab})

    try:
        cfg = flatten_config(tomllib.loads(read_text(config_path)))
    except Exception as e:
        return {"issues": [{"level": "error", "msg": f"설정 파일을 읽을 수 없습니다: {e}"}], "paths": {}, "dataset": None}

    paths = {}
    for key, kind in [
        ("pretrained_model_name_or_path", "file"), ("text_encoder", "file"), ("vae", "file"),
        ("tokenizer_cache_dir", "dir"), ("dataset_config", "file"), ("sample_prompts", "file"),
        ("output_dir", "dir"), ("logging_dir", "dir"),
    ]:
        paths[key] = path_info(cfg.get(key), kind)
    for key in ["pretrained_model_name_or_path", "text_encoder", "vae", "dataset_config"]:
        st = paths[key]["state"]
        if st == "empty":
            add("error", f"{key} 가 비어 있습니다 (필수)", "dataset" if key == "dataset_config" else "train")
        elif st != "ok":
            add("error", f"{key}: {paths[key].get('note') or '파일이 없습니다'} — {cfg.get(key)}", "dataset" if key == "dataset_config" else "train")
    if paths["tokenizer_cache_dir"]["state"] == "empty":
        add("info", "tokenizer_cache_dir 가 없어 토크나이저를 Hugging Face 에서 내려받습니다(오프라인이면 실패)", "train")
    elif paths["tokenizer_cache_dir"]["state"] != "ok":
        add("warn", f"tokenizer_cache_dir 폴더가 없습니다: {cfg.get('tokenizer_cache_dir')}", "train")
    if cfg.get("sample_prompts") and paths["sample_prompts"]["state"] != "ok":
        add("error", f"sample_prompts 파일이 없습니다: {cfg.get('sample_prompts')}", "output")
    if paths["output_dir"]["state"] == "missing":
        add("info", "output_dir 가 아직 없습니다 (학습 시작 시 생성됨)", "output")

    # 옵션 조합 검사 (docs/moss_image_train_network.md 기준)
    workers = cfg.get("max_data_loader_n_workers", 8)
    if cfg.get("persistent_data_loader_workers") and workers == 0:
        add("error", "persistent_data_loader_workers 는 max_data_loader_n_workers 0 과 함께 쓸 수 없습니다", "dataset")
    if not cfg.get("persistent_data_loader_workers") and workers > 0:
        add("warn", "Windows 에서는 persistent_data_loader_workers 를 켜야 에포크마다 작업 프로세스를 다시 만들지 않습니다(수 배 느려짐)", "dataset")
    if workers > 2:
        add("info", f"max_data_loader_n_workers={workers}: 2 보다 늘려도 빨라지지 않습니다(문서 권장 2)", "dataset")
    if cfg.get("cache_text_encoder_outputs") and not cfg.get("network_train_unet_only"):
        add("error", "cache_text_encoder_outputs 는 network_train_unet_only 가 필요합니다", "memory")
    if not cfg.get("network_train_unet_only"):
        add("warn", "텍스트 인코더도 학습합니다 — 공식 권장은 텍스트 인코더 고정, bf16 이면 VRAM 약 32GB", "train")
    if cfg.get("cache_text_encoder_outputs_to_disk") and not cfg.get("cache_text_encoder_outputs"):
        add("warn", "cache_text_encoder_outputs_to_disk 는 cache_text_encoder_outputs 와 함께 켜야 합니다", "memory")
    if cfg.get("cache_latents_to_disk") and not cfg.get("cache_latents"):
        add("warn", "cache_latents_to_disk 는 cache_latents 와 함께 켜야 합니다", "memory")
    bts = int(cfg.get("blocks_to_swap") or 0)
    if bts > 40:
        add("error", f"blocks_to_swap 최대값은 40 입니다 (현재 {bts})", "memory")
    if bts and (cfg.get("unsloth_offload_checkpointing") or cfg.get("cpu_offload_checkpointing")):
        add("error", "blocks_to_swap 은 unsloth/cpu_offload_checkpointing 과 함께 쓸 수 없습니다", "memory")
    if not cfg.get("gradient_checkpointing") and not cfg.get("unsloth_offload_checkpointing"):
        add("warn", "gradient_checkpointing 이 꺼져 있습니다 (사실상 필수)", "memory")
    if cfg.get("fp8_base") or cfg.get("fp8_base_unet"):
        add("warn", "fp8_base 는 MOSS-Image 에서 무시됩니다 — convrot_int8 을 쓰세요", "memory")
    if cfg.get("mixed_precision", "no") == "fp16":
        add("warn", "mixed_precision fp16 은 권장하지 않습니다 (bf16 모델)", "memory")
    if cfg.get("attn_mode") == "xformers":
        if not cfg.get("split_attn"):
            add("error", "attn_mode xformers 는 split_attn 이 필요합니다", "memory")
        add("warn", "RTX 50xx 에서 xformers 가 FA3 커널을 골라 트레이스백 없이 종료될 수 있습니다 — torch 권장", "memory")
    if cfg.get("attn_mode") == "sageattn":
        add("error", "attn_mode sageattn 은 학습을 지원하지 않습니다", "memory")
    if cfg.get("attn_mode", "flash") == "flash":  # 키가 없으면 moss_train.py 가 flash 로 실행한다
        if not flash_info()["installed"]:
            add("error", "Flash Attention 2 에 flash_attn 패키지가 없습니다 — 메모리·속도 탭에서 끄거나 설치하세요", "memory")
        if cfg.get("mixed_precision", "no") not in ("bf16", "fp16"):
            add("error", "Flash Attention 2 는 bf16·fp16 만 지원합니다 — 정밀도를 bf16 으로 두거나 끄세요", "memory")
    ts = cfg.get("timestep_sampling", "sigmoid")
    if ts == "sigmoid":
        add("info", "timestep_sampling 기본(sigmoid)은 사전학습 분포와 많이 다릅니다 — truncated_logit_normal 또는 shift 3.0 권장", "train")
    if ts == "shift" and float(cfg.get("discrete_flow_shift", 1.0)) == 1.0:
        add("info", "shift 샘플링에 discrete_flow_shift 1.0 은 이동이 없습니다 — 사전학습 근사값은 3.0", "train")
    dim = float(cfg.get("network_dim") or 4)
    alpha = float(cfg.get("network_alpha", 1)) or dim  # 0 이면 sd-scripts 는 rank 로 쓴다
    lr = float(cfg.get("learning_rate", 0) or 0)
    strength = alpha / dim * lr
    if strength >= 4e-4:  # anima_lora channel_scaling.md: 32/128, lr 1e-4, AdamW 에서 프롬프트 세부를 잃었다(30장 1세트)
        add("info", f"alpha ÷ rank × 학습률이 {strength:g} 입니다 — anima_lora 기록에서 4e-4(32/128, lr 1e-4, AdamW)는 화풍은 강하지만 프롬프트 세부가 사라졌습니다", "train")
    if cfg.get("lr_warmup_steps") and cfg.get("lr_scheduler", "constant") == "constant":
        add("error", "constant 스케줄러에는 워밍업을 쓸 수 없어 학습이 시작되지 않습니다 — constant_with_warmup 을 고르거나 워밍업을 0 으로 두세요", "train")
    if not triton_info()["installed"]:
        if cfg.get("convrot_int8") or cfg.get("convrot_int8_text_encoder"):
            add("warn", "convrot_int8 에 triton 이 없어 느린 bf16 대체 경로로 동작합니다 (pip install triton-windows)", "memory")
        if cfg.get("compile"):
            add("error", "compile 에는 triton 이 필요합니다 (pip install triton-windows)", "memory")
    if not cfg.get("save_every_n_epochs") and not cfg.get("save_every_n_steps"):
        add("info", "중간 저장이 꺼져 있습니다 — 마지막에만 저장됩니다", "output")

    # 데이터셋
    dataset = None
    if paths["dataset_config"]["state"] == "ok":
        try:
            ds_path = resolve_path(cfg["dataset_config"])
            _data, general, ds, subsets = load_dataset(ds_path)
            reso = ds.get("resolution", 1024)
            r = reso if isinstance(reso, list) else [reso, reso]
            if any(int(v) % 16 for v in r):
                add("error", f"resolution 은 16 의 배수여야 합니다 ({reso})", "dataset")
            steps = ds.get("bucket_reso_steps", 64)
            if steps % 16:
                add("error", f"bucket_reso_steps 는 16 의 배수여야 합니다 ({steps})", "dataset")
            if not general.get("enable_multiline_caption", ds.get("enable_multiline_caption")):
                add("info", "enable_multiline_caption 이 꺼져 있습니다 — MOSS 는 긴 자연어(여러 줄) 캡션에 맞춰져 있습니다", "dataset")
            batch = int(ds.get("batch_size", general.get("batch_size", 1)) or 1)
            min_reso = int(ds.get("min_bucket_reso", 256))
            subs = []
            total = 0
            for sub in subsets:
                if "recursive" in sub:
                    add("error", f"sd-scripts 는 recursive 키를 지원하지 않아 학습이 시작되지 않습니다 — dataset.toml 에서 지우세요 ({os.path.basename(sub.get('image_dir', ''))})", "dataset")
                s = scan_subset(sub, {**general, **{k: v for k, v in ds.items() if k != "subsets"}}, min_reso)
                subs.append(s)
                name = os.path.basename(sub.get("image_dir", ""))
                if "caption_dropout_rate" in sub or "caption_dropout_rate" in ds:
                    add("warn", f"caption_dropout_rate 가 폴더나 [[datasets]] 에도 있어 공통 값보다 우선합니다 — dataset.toml 에서 지우세요 ({name})", "dataset")
                rate = float({**general, **ds, **sub}.get("caption_dropout_rate") or 0)
                te = s.get("te_dropout")  # 캐시 검사는 이 값을 비교하지 않아 학습은 캐시를 만들 때의 값을 그대로 쓴다
                if te is not None and abs(te - rate) > 1e-6 and cfg.get("cache_text_encoder_outputs_to_disk"):
                    add("warn", f"캡션 드롭아웃 {rate:g} 가 반영되지 않습니다 — TE 캐시는 {te:g} 로 만들어졌으니 *_moss_te.npz 를 지워 다시 만드세요 ({name})", "dataset")
                if not s["ok"]:
                    add("error", f"이미지 폴더가 없습니다: {sub.get('image_dir')}", "dataset")
                    continue
                if s["num_images"] == 0:
                    add("error", f"이미지가 없습니다: {sub.get('image_dir')}", "dataset")
                if s["no_caption_count"]:
                    add("warn", f"캡션 파일이 없는 이미지 {s['no_caption_count']}장 ({os.path.basename(sub.get('image_dir', ''))})", "dataset")
                if s["empty_caption_count"]:
                    add("warn", f"빈 캡션 {s['empty_caption_count']}장 ({os.path.basename(sub.get('image_dir', ''))})", "dataset")
                if s["bad_separator"]:
                    add("error", f"캡션 구분자 줄 뒤에 # 주석 외 문자가 있습니다: {', '.join(s['bad_separator'][:3])}", "dataset")
                if s["small_count"]:
                    if ds.get("enable_bucket") and ds.get("bucket_no_upscale"):
                        add("info", f"긴 변이 {min_reso} 보다 작은 이미지 {s['small_count']}장 — 업스케일 안 함이라 그 크기로 학습됩니다", "dataset")
                    else:
                        add("info", f"긴 변이 min_bucket_reso({min_reso}) 보다 작은 이미지 {s['small_count']}장", "dataset")
                total += s["num_images"] * s["num_repeats"]
            if not subsets:
                add("error", "dataset.toml 에 [[datasets.subsets]] 가 없습니다", "dataset")
            epochs = int(cfg.get("max_train_epochs") or 0)
            per_epoch = -(-total // batch) if total else 0
            dataset = {
                "path": ds_path,
                "subsets": subs,
                "batch_size": batch,
                "samples_per_epoch": total,
                "steps_per_epoch": per_epoch,
                "epochs": epochs,
                "total_steps": per_epoch * epochs,
            }
            # experiments/int8_ab 실측(RTX 5090): 튜닝 후 int8 이 약 12% 빠르지만 첫 실행엔 버킷당 약 28초 autotune
            # 실측은 배치 1 이라 기준은 스텝이 아니라 처리하는 이미지 수다
            total_images = per_epoch * epochs * batch
            if total_images >= 5000 and not cfg.get("convrot_int8"):
                add("info", f"이미지 {total_images}장을 처리하는 긴 학습 — DiT INT8 을 켜면 약 12% 빨라집니다(배치 1 실측, 품질 차이는 시드 차이 이하)", "memory")
            if 0 < total_images < 1500 and cfg.get("convrot_int8"):
                add("info", f"이미지 {total_images}장을 처리하는 짧은 학습 — 처음 보는 버킷은 INT8 튜닝(버킷당 약 30초) 때문에 bf16 보다 느릴 수 있습니다", "memory")
        except Exception as e:
            add("error", f"dataset.toml 을 읽을 수 없습니다: {e}", "dataset")

    order = {"error": 0, "warn": 1, "info": 2}
    issues.sort(key=lambda x: order[x["level"]])
    return {"issues": issues, "paths": paths, "dataset": dataset}


# ---------------------------------------------------------------------------
# 학습 프로세스 관리
# ---------------------------------------------------------------------------

PROGRESS_RE = re.compile(r"(?<![A-Za-z_])(?<!Denoising )steps:\s*(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([^<\]]*)<([^,\]]*),\s*([^,\]]*)(?:,\s*avr_loss=([-+\d.eE]+))?")
EPOCH_RE = re.compile(r"^epoch (\d+)/(\d+)")
BATCH_RE = re.compile(r"batch size per device\D*(\d+)")
RESUME_RE = re.compile(r"^moss_resume step=(\d+) total=(\d+)")  # moss_train.py 가 남긴다
STOPPED_RE = re.compile(r"^moss_stopped (.+)$")
MODEL_NAME_RE = re.compile(r"^(.+?)(?:-step\d{8}|-\d{6})?$")  # <이름>, <이름>-000001, <이름>-step00000100
SAMPLE_NAME_RE = re.compile(r"^(?:(.*)_)?(?:e\d{6}|\d{6})_\d+_\d{14}(?:_\d+)?$")  # <이름>_e000001_00_<시각>_<시드>
MAX_GROUP_SAMPLES = 300
STOP_FILE = os.path.join(HERE, ".moss_stop_request")
TRAIN_SCRIPT = os.path.join(HERE, "moss_train.py")
STOP_TIMEOUT = 900  # 이 시간 안에 저장하고 끝나지 않으면 강제로 끝낸다(샘플 생성 중일 수 있다)
STEP_TIME_WINDOW = 30  # 최근 몇 스텝으로 스텝당 시간을 계산할지


def clock_seconds(text):
    """tqdm 시간 표기(mm:ss, h:mm:ss)를 초로 바꾼다."""
    try:
        seconds = 0
        for part in text.strip().split(":"):
            seconds = seconds * 60 + int(part)
        return seconds
    except ValueError:
        return None


def pid_alive(pid):
    if not pid:
        return False
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not h:
        return False
    try:
        code = ctypes.c_ulong()
        if not k32.GetExitCodeProcess(h, ctypes.byref(code)):
            return False
        return code.value == STILL_ACTIVE
    finally:
        k32.CloseHandle(h)


def process_create_time(pid):
    """PID 재사용 오인을 막기 위한 프로세스 생성 시각(FILETIME 정수)."""
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x1000, False, int(pid))
    if not h:
        return None
    try:
        ft = [ctypes.c_ulonglong() for _ in range(4)]
        if not k32.GetProcessTimes(h, *[ctypes.byref(x) for x in ft]):
            return None
        return ft[0].value
    finally:
        k32.CloseHandle(h)


class TrainRunner:
    def __init__(self):
        self.lock = threading.Lock()
        self.proc = None
        self.pid = None
        self.log_path = None
        self.settings_dir = None
        self.config_path = None
        self.started = None
        self.exit_code = None
        self.lines = collections.deque(maxlen=4000)
        self.seq = 0  # 마지막 줄 번호
        self.partial = ""  # 아직 줄바꿈이 안 온 마지막 줄(tqdm \r 갱신)
        self.progress = {}
        self.epoch = None
        self.batch = None
        self.step_marks = collections.deque(maxlen=STEP_TIME_WINDOW + 1)  # (스텝, 경과 초)
        self.stop_requested = False
        self.stopping = False
        self.resume_from = None  # (이어 시작한 스텝, 총 스텝)
        self.saved_state = None
        self._offset = 0
        self._decoder = None
        self._restore()
        threading.Thread(target=self._tail_loop, daemon=True).start()

    # 상태 파일: UI 를 다시 켰을 때 실행 중인 학습에 다시 붙기 위함
    def _restore(self):
        try:
            st = json.loads(read_text(STATE_FILE))
        except Exception:
            return
        if pid_alive(st.get("pid")) and process_create_time(st["pid"]) == st.get("ctime"):
            self.pid = st["pid"]
            self.log_path = st.get("log")
            self.config_path = st.get("config")
            self.started = st.get("started")
            self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        elif st.get("log") and os.path.exists(st["log"]):
            # 끝난 학습의 로그를 보여 준다
            self.log_path = st["log"]
            self.config_path = st.get("config")
            self.started = st.get("started")
            self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
            self.exit_code = st.get("exit_code", "?")

    def _save_state(self, **extra):
        st = {"pid": self.pid, "log": self.log_path, "config": self.config_path, "started": self.started}
        if self.pid:
            st["ctime"] = process_create_time(self.pid)
        st.update(extra)
        write_text_atomic(STATE_FILE, json.dumps(st, ensure_ascii=False))

    def running(self):
        if self.proc is not None:
            return self.proc.poll() is None
        return pid_alive(self.pid)

    def start(self, config_path, extra_args, resume=None):
        with self.lock:
            if self.running():
                raise RuntimeError("이미 학습이 실행 중입니다")
            if not os.path.isfile(PYTHON):
                raise RuntimeError(f"venv 파이썬이 없습니다: {PYTHON}")
            os.makedirs(RUN_LOG_DIR, exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_path = os.path.join(RUN_LOG_DIR, f"train_{stamp}.log")
            self.settings_dir = os.path.join(RUN_LOG_DIR, f"train_{stamp}_settings")
            try:
                snapshot_settings(config_path, self.settings_dir)
                snapshot_note = f"설정 사본: {self.settings_dir}"
            except (OSError, ValueError) as e:  # 사본이 없어도 학습은 할 수 있다
                self.settings_dir = None
                snapshot_note = f"설정 사본을 만들지 못했습니다: {e}"
            cmd = build_command(config_path, extra_args, resume)
            if os.path.exists(STOP_FILE):
                os.remove(STOP_FILE)
            # COLUMNS: 터미널이 없으면 rich 로그가 80칸에서 줄바꿈돼 읽기 어렵다
            # TRITON_CACHE_AUTOTUNING: convrot_int8 의 버킷별 autotune 결과를 디스크에 남겨 다음 학습부터 건너뜀
            env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", COLUMNS="160", MOSS_STOP_FILE=STOP_FILE,
                       TRITON_CACHE_AUTOTUNING=os.environ.get("TRITON_CACHE_AUTOTUNING", "1"))
            logf = open(self.log_path, "wb")
            logf.write(("> " + subprocess.list2cmdline(cmd) + "\n" + snapshot_note + "\n\n").encode("utf-8"))
            logf.flush()
            flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
            self.proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, creationflags=flags)
            logf.close()  # 자식이 핸들을 물려받았다
            self.pid = self.proc.pid
            self.config_path = config_path
            self.started = time.time()
            self.exit_code = None
            self.stop_requested = False
            self.stopping = False
            self.resume_from = None
            self.saved_state = None
            self.lines.clear()
            self.partial = ""
            self.progress = {}
            self.epoch = None
            self.batch = None
            self.step_marks.clear()
            self._offset = 0
            self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
            self._save_state()

    def stop(self, force=False):
        """스텝이 도는 중이면 moss_train.py 가 상태를 저장하고 끝나게 요청한다. 스텝 전이거나 force 면 바로 끝낸다."""
        with self.lock:
            if not self.running():
                return False
            self.stop_requested = True
            if force or self.stopping or not self.progress.get("step"):
                self._kill()
                return "killed"
            self.stopping = True
            write_text_atomic(STOP_FILE, "stop")
            timer = threading.Timer(STOP_TIMEOUT, self._stop_timeout, [self.pid])
            timer.daemon = True
            timer.start()
            return "saving"

    def _kill(self):
        subprocess.run(["taskkill", "/PID", str(self.pid), "/T", "/F"], capture_output=True)

    def _stop_timeout(self, pid):
        with self.lock:
            if self.stopping and self.pid == pid and self.running():
                self._push_line("[UI] 저장이 끝나지 않아 강제로 종료합니다")
                self._kill()

    def _push_line(self, text):
        self.seq += 1
        self.lines.append((self.seq, text))
        m = EPOCH_RE.match(text.strip())
        if m:
            self.epoch = [int(m.group(1)), int(m.group(2))]
        m = BATCH_RE.search(text)
        if m:
            self.batch = int(m.group(1))
        m = RESUME_RE.match(text.strip())
        if m:
            self.resume_from = (int(m.group(1)), int(m.group(2)))
        m = STOPPED_RE.match(text.strip())
        if m:
            self.saved_state = m.group(1)

    def _feed(self, text):
        text = self.partial + text
        parts = re.split(r"(\r\n|\n|\r)", text)
        # parts: [seg, sep, seg, sep, ..., last]
        for i in range(0, len(parts) - 1, 2):
            seg, sep = parts[i], parts[i + 1]
            if sep == "\r":
                # tqdm 갱신: 다음 내용이 같은 줄을 덮어쓴다 — 진행률만 갱신
                self._parse_progress(seg)
                continue
            self._parse_progress(seg)
            if seg.strip():
                self._push_line(seg)
        self.partial = parts[-1]
        if self.partial:
            self._parse_progress(self.partial)

    def _parse_progress(self, seg):
        m = PROGRESS_RE.search(seg)
        if m:
            step, elapsed = int(m.group(2)), clock_seconds(m.group(4))
            if elapsed is not None and (not self.step_marks or step > self.step_marks[-1][0]):
                self.step_marks.append((step, elapsed))
            pct, shown, total = int(m.group(1)), step, int(m.group(3))
            if self.resume_from:  # tqdm 은 이어 시작한 지점을 모르고 0 부터 센다
                shown, total = self.resume_from[0] + step, self.resume_from[1]
                pct = shown * 100 // total
            self.progress = {
                "pct": pct, "step": shown, "total": total,
                "elapsed": m.group(4).strip(), "eta": m.group(5).strip(), "rate": m.group(6).strip(),
                "avr_loss": float(m.group(7)) if m.group(7) else None,
                "step_time": self._step_time(),
            }

    def _step_time(self):
        """최근 스텝의 평균 시간(초). 샘플 생성·저장처럼 한 번 크게 튀는 구간은 뺀다."""
        marks = list(self.step_marks)
        gaps = [(b[1] - a[1]) / (b[0] - a[0]) for a, b in zip(marks, marks[1:])]
        if len(gaps) >= 5:
            gaps.remove(max(gaps))
        return sum(gaps) / len(gaps) if gaps else None

    def _tail_loop(self):
        while True:
            try:
                self._tail_once()
            except Exception as e:
                print("tail error:", e)
            time.sleep(0.5)

    def _tail_once(self):
        path = self.log_path
        if not path or not os.path.exists(path) or self._decoder is None:
            return
        with open(path, "rb") as f:
            f.seek(self._offset)
            data = f.read(4 * 1024 * 1024)
        if data:
            self._offset += len(data)
            with self.lock:
                self._feed(self._decoder.decode(data))
        if self.exit_code is None and self.pid and not self.running():
            code = self.proc.poll() if self.proc is not None else "?"
            with self.lock:
                if self.partial.strip():
                    self._push_line(self.partial)
                    self.partial = ""
                self.exit_code = "중지됨" if self.stop_requested else code
                self.stopping = False
                if os.path.exists(STOP_FILE):
                    os.remove(STOP_FILE)
                self._push_line(f"[UI] 학습 프로세스 종료 (exit code: {self.exit_code})")
            self._save_state(exit_code=self.exit_code)

    def status(self, since):
        with self.lock:
            lines = [t for s, t in self.lines if s > since]
            first = self.lines[0][0] if self.lines else 0
            return {
                "running": self.running(),
                "pid": self.pid,
                "exit_code": self.exit_code,
                "started": self.started,
                "config": self.config_path,
                "log_path": self.log_path,
                "seq": self.seq,
                "reset": bool(since > self.seq or (since and since < first - 1)),
                "lines": lines[-2000:],
                "partial": self.partial[-400:],
                "progress": self.progress,
                "epoch": self.epoch,
                "batch": self.batch,
                "stopping": self.stopping,
                "saved_state": self.saved_state,
            }


def resume_states(config_path):
    """이어서 학습할 수 있는 상태 폴더. 출력 폴더의 *-state 중 train_state.json 이 있는 것을 최신순으로."""
    cfg = flatten_config(tomllib.loads(read_text(config_path)))
    out = resolve_path(cfg.get("output_dir"))
    if not out or not os.path.isdir(out):
        return []
    ds_path = resolve_path(cfg.get("dataset_config"))
    batch = None
    if ds_path and os.path.isfile(ds_path):
        general, ds = load_dataset(ds_path)[1:3]
        batch = ds.get("batch_size", general.get("batch_size", 1))
    rows = []
    for e in os.scandir(out):
        state_file = os.path.join(e.path, "train_state.json")
        if not e.name.endswith("-state") or not os.path.isfile(state_file):
            continue
        try:
            st = json.loads(read_text(state_file))
        except (OSError, ValueError):
            continue
        saved = st.get("settings", {})
        warnings = []
        for key, label, now in (("network_dim", "Rank", cfg.get("network_dim")), ("network_alpha", "Alpha", cfg.get("network_alpha")), ("batch_size", "배치", batch)):
            if saved.get(key) is not None and now is not None and float(saved[key]) != float(now):
                warnings.append(f"{label}: 저장 당시 {saved[key]:g}, 지금 {float(now):g} — 그대로 이어가면 위치가 어긋나거나 시작에 실패합니다")
        rows.append({"path": e.path.replace("\\", "/"), "name": e.name, "step": st.get("current_step"), "total": st.get("total_steps"), "epoch": st.get("current_epoch"),
                     "time": os.path.getmtime(state_file), "size": sum(f.stat().st_size for f in os.scandir(e.path) if f.is_file()), "warnings": warnings})
    return sorted(rows, key=lambda r: -r["time"])


def build_command(config_path, extra_args, resume=None):
    cmd = [
        PYTHON, "-m", "accelerate.commands.launch", "--num_cpu_threads_per_process", "1", "--num_processes", "1",
        "--num_machines", "1", "--mixed_precision", "bf16", "--dynamo_backend", "no",
        TRAIN_SCRIPT, "--config_file", config_path,
    ]
    if resume:  # skip_until_initial_step: 에포크 처음부터가 아니라 멈춘 스텝 다음부터 잇는다
        cmd += ["--resume", resume, "--skip_until_initial_step"]
    if extra_args:
        import shlex

        cmd += shlex.split(extra_args, posix=True)
    return cmd


# ---------------------------------------------------------------------------
# TensorBoard 로그 / 샘플 / 결과물
# ---------------------------------------------------------------------------

_accumulators = {}


def list_runs(logging_dir):
    d = resolve_path(logging_dir)
    if not d or not os.path.isdir(d):
        return []
    runs = []
    for name in os.listdir(d):
        p = os.path.join(d, name)
        if not os.path.isdir(p):
            continue
        for base, _dirs, files in os.walk(p):
            if any(f.startswith("events.out.tfevents") for f in files):
                runs.append({"name": os.path.relpath(base, d).replace("\\", "/"), "path": base,
                             "mtime": max(os.path.getmtime(os.path.join(base, f)) for f in files)})
    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return runs


def read_scalars(run_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    key = norm(run_path)
    ea = _accumulators.get(key)
    if ea is None:
        ea = EventAccumulator(run_path, size_guidance={"scalars": 0})
        _accumulators[key] = ea
    ea.Reload()  # 이어서 읽는다
    out = {}
    for tag in ea.Tags().get("scalars", []):
        out[tag] = [[e.step, e.value, e.wall_time] for e in ea.Scalars(tag)]
    return out


def list_files(d, exts, limit=200):
    d = resolve_path(d)
    if not d or not os.path.isdir(d):
        return []
    out = []
    for f in os.listdir(d):
        p = os.path.join(d, f)
        if os.path.splitext(f)[1].lower() in exts and os.path.isfile(p):
            st = os.stat(p)
            out.append({"name": f, "path": p, "size": st.st_size, "mtime": st.st_mtime})
    out.sort(key=lambda x: x["mtime"], reverse=True)
    return out[:limit]


def output_groups(out):
    """출력 폴더의 LoRA·샘플을 output_name 별로 묶어 최신순으로 준다. 출력 폴더는 학습끼리 함께 쓴다."""
    groups = {}

    def group(name):
        return groups.setdefault(name, {"name": name, "models": [], "samples": []})

    for f in list_files(out, {".safetensors", ".ckpt", ".pt"}, None):
        group(MODEL_NAME_RE.match(os.path.splitext(f["name"])[0])[1])["models"].append(f)
    for f in list_files(os.path.join(out, "sample") if out else "", IMAGE_EXTENSIONS, None):
        m = SAMPLE_NAME_RE.match(os.path.splitext(f["name"])[0])
        group(m[1] or "" if m else "")["samples"].append(f)
    for g in groups.values():
        g["mtime"] = max(f["mtime"] for f in g["models"] + g["samples"])
        g["size"] = sum(f["size"] for f in g["models"])
        g["sample_count"] = len(g["samples"])
        del g["samples"][MAX_GROUP_SAMPLES:]
    return sorted(groups.values(), key=lambda g: g["mtime"], reverse=True)


def browse_dialog(kind, initial):
    """tkinter 파일 대화상자를 별도 프로세스로 띄운다 (서버 스레드에서 Tk 를 돌리지 않기 위해)."""
    script = (
        "import sys,tkinter as tk\n"
        "from tkinter import filedialog\n"
        "r=tk.Tk();r.withdraw();r.attributes('-topmost',True)\n"
        "k,init=sys.argv[1],sys.argv[2]\n"
        "import os\n"
        "d=init if os.path.isdir(init) else os.path.dirname(init)\n"
        "if not os.path.isdir(d): d=None\n"
        "p=filedialog.askdirectory(parent=r,initialdir=d) if k=='dir' else filedialog.askopenfilename(parent=r,initialdir=d)\n"
        "sys.stdout.buffer.write((p or '').encode('utf-8'))\n"
    )
    r = subprocess.run([sys.executable, "-c", script, kind, initial or ""], capture_output=True, timeout=600)
    return r.stdout.decode("utf-8").strip()


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

runner = None
sysmon = None
PORT = 7865
SYSTEM_POLL_SECONDS = 2
PREVIEW_SCRIPT = os.path.join(HERE, "moss_preview.py")
PREVIEW_LOG = os.path.join(RUN_LOG_DIR, "preview_worker.log")
PREVIEW_TIMEOUT = 600  # CPU VAE 는 그림이 크거나 학습 중이면 오래 걸린다


class MemoryStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)]
    _fields_ += [(n, ctypes.c_ulonglong) for n in ("total_phys", "avail_phys", "total_page", "avail_page", "total_virtual", "avail_virtual", "avail_extended")]


def read_ram():
    m = MemoryStatus()
    m.dwLength = ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        return None
    return {"used_gb": (m.total_phys - m.avail_phys) / 1024**3, "total_gb": m.total_phys / 1024**3}


def read_gpu():
    fields = "name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=5, creationflags=subprocess.CREATE_NO_WINDOW).stdout
        name, util, used, total, temp, power = [v.strip() for v in out.splitlines()[0].split(",")]

        def num(v):
            return float(v) if re.fullmatch(r"[\d.]+", v) else None

        return {"name": name, "util": num(util), "vram_used_gb": num(used) / 1024, "vram_total_gb": num(total) / 1024, "temp": num(temp), "power": num(power)}
    except (OSError, subprocess.SubprocessError, ValueError, TypeError, IndexError):
        return None


class SystemMonitor:
    """GPU·RAM 사용량. nvidia-smi 는 프로세스를 띄우므로 화면이 보고 있을 때만 일정 간격으로 읽는다."""

    def __init__(self):
        self.data = {"gpu": None, "ram": None}
        self.last_request = 0.0
        threading.Thread(target=self._loop, daemon=True).start()

    def read(self):
        self.last_request = time.time()
        return self.data

    def _loop(self):
        while True:
            if time.time() - self.last_request < 10:
                self.data = {"gpu": read_gpu(), "ram": read_ram()}
            time.sleep(SYSTEM_POLL_SECONDS)


def config_files():
    out = []
    for f in sorted(os.listdir(HERE)):
        if not f.endswith(".toml"):
            continue
        p = os.path.join(HERE, f)
        try:
            data = tomllib.loads(read_text(p))
        except Exception:
            continue
        if "datasets" in data:
            continue  # 데이터셋 설정은 제외
        out.append(p)
    if os.path.isdir(PROFILE_DIR):
        for name in sorted(os.listdir(PROFILE_DIR), key=str.lower):
            p = os.path.join(PROFILE_DIR, name, name + ".toml")
            if os.path.isfile(p):
                out.append(p)
    return out


def is_config_path(path):
    """설정 파일은 도구 폴더 바로 아래 또는 profiles/<이름>/ 의 .toml 만 다룬다."""
    parent = os.path.dirname(os.path.abspath(path))
    return path.endswith(".toml") and (norm(parent) == norm(HERE) or norm(os.path.dirname(parent)) == norm(PROFILE_DIR))


def load_all(config_path):
    text = read_text(config_path)
    cfg = flatten_config(tomllib.loads(text))
    result = {"config_path": config_path, "config": cfg, "configs": config_files(), "dataset": None, "prompts": None,
              "triton": triton_info(), "flash": flash_info()}
    ds = cfg.get("dataset_config")
    if ds and os.path.isfile(resolve_path(ds)):
        data, general, d0, subsets = load_dataset(resolve_path(ds))
        result["dataset"] = {
            "path": resolve_path(ds),
            "general": general,
            "dataset": {k: v for k, v in d0.items() if k != "subsets"},
            "subsets": subsets,
        }
    sp = cfg.get("sample_prompts")
    if sp and os.path.isfile(resolve_path(sp)):
        result["prompts"] = {"path": resolve_path(sp), "text": read_text(resolve_path(sp))}
    return result


def save_all(body):
    config_path = body["config_path"]
    if not is_config_path(config_path):
        raise ValueError("설정 파일은 도구 폴더의 .toml 만 저장할 수 있습니다")
    saved = []
    # 설정 파일
    text = read_text(config_path)
    lines = text.split("\n")
    for key, value in body.get("config", {}).items():
        if not re.match(r"^[A-Za-z0-9_]+$", key):
            raise ValueError(f"잘못된 키: {key}")
        lines = toml_set(lines, None, 0, key, value)
    new = "\n".join(lines)
    validate_toml(new)
    if new != text:
        write_text_atomic(config_path, new)
        saved.append(config_path)
    cfg = flatten_config(tomllib.loads(new))

    # 데이터셋 설정
    ds_body = body.get("dataset")
    if ds_body and cfg.get("dataset_config"):
        ds_path = resolve_path(cfg["dataset_config"])
        text = read_text(ds_path)
        lines = text.split("\n")
        for key, value in ds_body.get("general", {}).items():
            if not any(t["name"] == "general" for t in split_tables(lines)):
                lines = ["[general]", ""] + lines
            lines = toml_set(lines, "general", 0, key, value)
        for key, value in ds_body.get("dataset", {}).items():
            lines = toml_set(lines, "datasets", 0, key, value)
        for idx in sorted(ds_body.get("remove_subsets", []), reverse=True):
            lines = toml_remove_table(lines, "datasets.subsets", idx)
        n_existing = sum(1 for t in split_tables(lines) if t["name"] == "datasets.subsets")
        for i, sub in enumerate(ds_body.get("subsets", [])):
            if i >= n_existing:
                lines = toml_append_subset(lines, sub.get("image_dir") or "")
            for key, value in sub.items():
                lines = toml_set(lines, "datasets.subsets", i, key, value)
        new = "\n".join(lines)
        validate_toml(new)
        if new != text:
            write_text_atomic(ds_path, new)
            saved.append(ds_path)

    # 샘플 프롬프트
    if body.get("prompts") is not None and cfg.get("sample_prompts"):
        sp = resolve_path(cfg["sample_prompts"])
        old = read_text(sp) if os.path.exists(sp) else None
        if old != body["prompts"]:
            write_text_atomic(sp, body["prompts"])
            saved.append(sp)
    return saved


def triton_info():
    if importlib.util.find_spec("triton") is None:
        return {"installed": False, "version": ""}
    for dist in ("triton-windows", "triton"):
        try:
            return {"installed": True, "version": importlib.metadata.version(dist)}
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"installed": True, "version": ""}


def flash_info():
    if importlib.util.find_spec("flash_attn") is None:
        return {"installed": False, "version": ""}
    try:
        return {"installed": True, "version": importlib.metadata.version("flash-attn").split("+")[0]}
    except importlib.metadata.PackageNotFoundError:
        return {"installed": True, "version": ""}


def save_profile(body):
    """현재 설정을 저장 안 된 변경까지 포함해 profiles/<이름>/ 에 새 설정으로 만든다. 원본 파일은 바꾸지 않는다.
    출력 이름도 <이름> 으로 맞춘다. 그대로 두면 다른 LoRA 의 결과 파일을 덮어쓰게 된다."""
    name = str(body.get("name", "")).strip()
    if not PROFILE_NAME_RE.match(name) or name.endswith("."):
        raise ValueError("이름에는 글자·숫자·공백·. - _ 만 쓸 수 있습니다")
    src = body["config_path"]
    if not is_config_path(src):
        raise ValueError("설정 파일은 도구 폴더의 .toml 만 다룹니다")
    if {"dataset_config", "sample_prompts"} & set(body.get("config", {})):
        raise ValueError("데이터셋·프롬프트 파일 경로를 바꾼 뒤에는 먼저 저장하세요")
    dest = os.path.join(PROFILE_DIR, name)
    if os.path.exists(dest):
        raise ValueError(f"이미 있는 이름입니다: {name}")
    os.makedirs(dest)
    try:
        text = read_text(src)
        cfg = flatten_config(tomllib.loads(text))
        lines = text.split("\n")
        for key, stem in (("dataset_config", "dataset"), ("sample_prompts", "prompts")):
            old = resolve_path(cfg.get(key))
            if old and os.path.isfile(old):
                copy = os.path.join(dest, f"{name}_{stem}{os.path.splitext(old)[1]}")
                shutil.copy2(old, copy)
                lines = toml_set(lines, None, 0, key, copy.replace("\\", "/"))
        lines = toml_set(lines, None, 0, "output_name", name)
        new = "\n".join(lines)
        validate_toml(new)
        config_path = os.path.join(dest, name + ".toml")
        write_text_atomic(config_path, new)
        save_all({**body, "config_path": config_path})
    except Exception:
        shutil.rmtree(dest, ignore_errors=True)
        raise
    return config_path


def snapshot_settings(config_path, dest):
    """학습에 쓴 설정 파일을 복사해 둔다. 원본은 다음 편집 때 덮어써진다."""
    cfg = flatten_config(tomllib.loads(read_text(config_path)))
    os.makedirs(dest, exist_ok=True)
    for p in [config_path] + [resolve_path(cfg[k]) for k in ("dataset_config", "sample_prompts") if cfg.get(k)]:
        if os.path.isfile(p):
            shutil.copy2(p, dest)


class PreviewError(Exception):
    """미리보기 설정이 학습에서도 실패할 값이거나 처리에 실패했을 때. 메시지를 화면에 그대로 보여 준다."""


class PreviewWorker:
    """자료 미리보기 작업 프로세스(moss_preview.py). UI 서버에 torch 를 올리지 않으려고 따로 띄우고, 처음 쓸 때 시작한다."""

    def __init__(self):
        self.lock = threading.Lock()
        self.proc = None
        self.pending = {}
        self.seq = 0

    def _start(self):
        os.makedirs(RUN_LOG_DIR, exist_ok=True)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONUTF8": "1"}
        with open(PREVIEW_LOG, "w", encoding="utf-8") as log:
            self.proc = subprocess.Popen([PYTHON, "-u", PREVIEW_SCRIPT], cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                                         text=True, encoding="utf-8", creationflags=subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS)
        threading.Thread(target=self._read, args=(self.proc,), daemon=True).start()

    def _read(self, proc):
        for line in proc.stdout:
            msg = json.loads(line)
            job = self.pending.pop(msg["id"], None)
            if job:
                job["result"] = msg
                job["done"].set()
        with self.lock:  # 프로세스가 끝나면 기다리는 요청을 모두 깨운다
            for job in self.pending.values():
                job["done"].set()
            self.pending.clear()

    def call(self, op, **args):
        job = {"done": threading.Event(), "result": None}
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                self._start()
            self.seq += 1
            self.pending[self.seq] = job
            self.proc.stdin.write(json.dumps({"id": self.seq, "op": op, **args}, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
        if not job["done"].wait(PREVIEW_TIMEOUT):
            raise PreviewError("미리보기 처리 시간이 초과되었습니다")
        if job["result"] is None:
            raise PreviewError(f"미리보기 프로세스가 끝났습니다. 다시 시도하세요 ({os.path.basename(PREVIEW_LOG)})")
        if job["result"].get("error"):
            raise PreviewError(job["result"]["error"])
        return job["result"]


def preview_options(q, path):
    """화면의 버킷 값(저장 전 값 포함)과, 화면에 없는 보간 방식·VAE 경로를 합친다."""
    one = lambda k, default="": q.get(k, [""])[0].strip() or default
    try:
        res = [int(v) for v in re.split(r"[,x ]+", one("resolution")) if v]
        if len(res) not in (1, 2):
            raise ValueError
        opts = {"resolution": res * (3 - len(res)), "enable_bucket": one("enable_bucket") == "1", "no_upscale": one("no_upscale") == "1",
                "min": int(one("min", 256)), "max": int(one("max", 1024)), "steps": int(one("steps", 64))}
    except ValueError:
        raise PreviewError("해상도와 버킷 값을 숫자로 입력하세요")
    cfg = flatten_config(tomllib.loads(read_text(q.get("config", [DEFAULT_CONFIG])[0])))
    ds_path = resolve_path(cfg.get("dataset_config"))
    general, ds, subsets = load_dataset(ds_path)[1:] if ds_path and os.path.isfile(ds_path) else ({}, {}, [])
    sub = next((s for s in subsets if norm(resolve_path(s.get("image_dir"))) == norm(os.path.dirname(path))), {})
    opts["interp"] = next((d["resize_interpolation"] for d in (sub, ds, general, cfg) if d.get("resize_interpolation")), None)
    opts["vae"] = resolve_path(cfg.get("vae"))
    return opts


def allowed_image(path):
    """이미지 제공은 샘플 폴더·데이터셋 폴더 안의 이미지 확장자만 허용한다."""
    if os.path.splitext(path)[1].lower() not in IMAGE_EXTENSIONS:
        return False
    p = norm(path)
    roots = []
    try:
        cfg_path = runner.config_path if runner.config_path and os.path.exists(runner.config_path) else DEFAULT_CONFIG
        for cp in {cfg_path, DEFAULT_CONFIG}:
            cfg = flatten_config(tomllib.loads(read_text(cp)))
            if cfg.get("output_dir"):
                roots.append(resolve_path(cfg["output_dir"]))
            if cfg.get("dataset_config") and os.path.isfile(resolve_path(cfg["dataset_config"])):
                for s in load_dataset(resolve_path(cfg["dataset_config"]))[3]:
                    if s.get("image_dir"):
                        roots.append(resolve_path(s["image_dir"]))
    except Exception:
        pass
    for extra in _image_roots:
        roots.append(extra)
    return any(p.startswith(norm(r) + os.sep) for r in roots)


_image_roots = set()  # 현재 화면에서 연 설정의 폴더들 (/api/load 때 등록)
preview = PreviewWorker()


class Handler(BaseHTTPRequestHandler):
    server_version = "MossUI/1.0"

    def log_message(self, fmt, *args):
        pass

    def _host_ok(self):
        host = self.headers.get("Host", "")
        if host not in (f"127.0.0.1:{PORT}", f"localhost:{PORT}"):
            return False
        origin = self.headers.get("Origin")
        if origin and origin not in (f"http://127.0.0.1:{PORT}", f"http://localhost:{PORT}"):
            return False
        return True

    def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        extra = dict(extra or {})
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", extra.pop("Cache-Control", "no-store"))
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in extra.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self._host_ok():
            return self._send(403, {"error": "forbidden"})
        url = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(url.query)
        if "config" in q:
            # 설정 파일은 도구 폴더의 .toml 만 다룬다
            cp = q["config"][0]
            if not is_config_path(cp) or not os.path.isfile(cp):
                q["config"] = [DEFAULT_CONFIG]
        try:
            if url.path == "/":
                html = read_text(os.path.join(HERE, "moss_ui.html"))
                return self._send(200, html, "text/html; charset=utf-8",
                                  {"Content-Security-Policy": "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'"})
            if url.path == "/api/load":
                data = load_all(q.get("config", [DEFAULT_CONFIG])[0])
                c = data["config"]
                if c.get("output_dir"):
                    _image_roots.add(resolve_path(c["output_dir"]))
                if data["dataset"]:
                    for s in data["dataset"]["subsets"]:
                        if s.get("image_dir"):
                            _image_roots.add(resolve_path(s["image_dir"]))
                return self._send(200, data)
            if url.path == "/api/check":
                cp = q.get("config", [DEFAULT_CONFIG])[0]
                res = analyze(cp)
                if res.get("dataset"):
                    for s in res["dataset"]["subsets"]:
                        if s.get("ok"):
                            _image_roots.add(resolve_path(s["image_dir"]))
                return self._send(200, res)
            if url.path == "/api/status":
                return self._send(200, {**runner.status(int(q.get("since", ["0"])[0])), "system": sysmon.read()})
            if url.path == "/api/runs":
                return self._send(200, list_runs(q.get("logging_dir", [""])[0]))
            if url.path == "/api/scalars":
                run = q.get("run", [""])[0]
                logging_dir = resolve_path(q.get("logging_dir", [""])[0])
                if not run or not norm(run).startswith(norm(logging_dir) + os.sep):
                    return self._send(400, {"error": "bad run"})
                return self._send(200, read_scalars(run))
            if url.path == "/api/outputs":
                out = resolve_path(q.get("output_dir", [""])[0])
                if out:
                    _image_roots.add(out)
                return self._send(200, {"groups": output_groups(out)})
            if url.path == "/api/command":
                cp = q.get("config", [DEFAULT_CONFIG])[0]
                return self._send(200, {"command": subprocess.list2cmdline(build_command(cp, q.get("extra", [""])[0], q.get("resume", [""])[0]))})
            if url.path == "/api/resume_states":
                return self._send(200, resume_states(q.get("config", [DEFAULT_CONFIG])[0]))
            if url.path in ("/api/training_sets", "/api/dir"):
                root = q.get("root", [DEFAULT_SOURCE_ROOT])[0] or DEFAULT_SOURCE_ROOT
                ext = q.get("ext", [".txt"])[0] or ".txt"
                ext = ext if ext.startswith(".") else "." + ext
                if url.path == "/api/training_sets":
                    if not os.path.isdir(root):
                        return self._send(404, {"error": f"폴더가 없습니다: {root}"})
                    _image_roots.add(os.path.abspath(root))
                    return self._send(200, {"root": os.path.abspath(root).replace("\\", "/"),
                                            "sets": find_training_sets(root, ext, q.get("force", ["0"])[0] == "1")})
                path = q.get("path", [root])[0] or root
                if not os.path.isdir(path):
                    return self._send(404, {"error": f"폴더가 없습니다: {path}"})
                _image_roots.add(os.path.abspath(path))
                detail = None
                if q.get("detail", ["0"])[0] == "1":
                    detail = {"multiline": q.get("multiline", ["0"])[0] == "1", "min_reso": int(q.get("min_reso", ["256"])[0])}
                return self._send(200, browse_dir(path, ext, detail))
            if url.path == "/api/dataset_images":
                return self._send(200, dataset_images(q.get("config", [DEFAULT_CONFIG])[0]))
            if url.path == "/api/thumb":
                p = q.get("path", [""])[0]
                if not p or not os.path.isfile(p) or not allowed_image(p):
                    return self._send(404, {"error": "not found"})
                try:
                    data = image_thumb(p)
                except Exception:
                    return self._send(404, {"error": "not an image"})
                return self._send(200, data, "image/jpeg", {"Cache-Control": "private, max-age=3600"})
            if url.path in ("/api/preview/plan", "/api/preview/image"):
                p = q.get("path", [""])[0]
                kind = q.get("kind", [""])[0]
                if not p or not os.path.isfile(p) or not allowed_image(p):
                    return self._send(404, {"error": "not found"})
                if url.path == "/api/preview/plan":
                    return self._send(200, preview.call("plan", path=p, opts=preview_options(q, p)))
                if kind not in ("processed", "vae"):
                    return self._send(400, {"error": "bad kind"})
                with open(preview.call("image", path=p, kind=kind, opts=preview_options(q, p))["file"], "rb") as f:
                    data = f.read()
                return self._send(200, data, "image/png")
            if url.path == "/api/image":
                p = q.get("path", [""])[0]
                if not p or not os.path.isfile(p) or not allowed_image(p):
                    return self._send(404, {"error": "not found"})
                with open(p, "rb") as f:
                    data = f.read()
                return self._send(200, data, mimetypes.guess_type(p)[0] or "application/octet-stream")
            return self._send(404, {"error": "not found"})
        except PreviewError as e:
            return self._send(422, {"error": str(e)})
        except ConnectionError:
            return None  # 미리보기를 닫아 브라우저가 VAE 요청을 취소한 경우
        except Exception as e:
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def do_POST(self):
        if not self._host_ok():
            return self._send(403, {"error": "forbidden"})
        if not self.headers.get("Content-Type", "").startswith("application/json"):
            return self._send(415, {"error": "json only"})
        try:
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
            url = urllib.parse.urlparse(self.path)
            if url.path == "/api/save":
                return self._send(200, {"saved": save_all(body)})
            if url.path == "/api/save_profile":
                return self._send(200, {"config_path": save_profile(body)})
            if url.path == "/api/start":
                cp = body.get("config_path", DEFAULT_CONFIG)
                if not is_config_path(cp):
                    return self._send(400, {"error": "도구 폴더의 설정만 실행할 수 있습니다"})
                res = analyze(cp)
                errors = [i["msg"] for i in res["issues"] if i["level"] == "error"]
                if errors and not body.get("force"):
                    return self._send(409, {"error": "설정 오류가 있어 시작하지 않았습니다", "issues": errors})
                resume = body.get("resume") or None
                if resume and not any(norm(r["path"]) == norm(resume) for r in resume_states(cp)):
                    return self._send(400, {"error": "이어서 학습할 상태 폴더를 찾을 수 없습니다"})
                runner.start(cp, body.get("extra_args", ""), resume)
                return self._send(200, {"ok": True, "settings_dir": runner.settings_dir})
            if url.path == "/api/stop":
                return self._send(200, {"stopped": runner.stop(bool(body.get("force")))})
            if url.path == "/api/browse":
                return self._send(200, {"path": browse_dialog(body.get("kind", "file"), body.get("initial", ""))})
            if url.path == "/api/open":
                p = resolve_path(body.get("path", ""))
                if p and os.path.exists(p):
                    target = p if os.path.isdir(p) else os.path.dirname(p)
                    os.startfile(target)
                    return self._send(200, {"ok": True})
                return self._send(404, {"error": "없는 경로입니다"})
            return self._send(404, {"error": "not found"})
        except Exception as e:
            return self._send(400 if isinstance(e, (ValueError, RuntimeError)) else 500, {"error": f"{e}"})


def main():
    global runner, sysmon, PORT
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7865)
    ap.add_argument("--no_browser", action="store_true")
    a = ap.parse_args()
    PORT = a.port
    # Windows 에서 SO_REUSEADDR 는 같은 포트에 두 번째 서버가 조용히 붙게 한다 → 끄고 이미 실행 중이면 알린다
    ThreadingHTTPServer.allow_reuse_address = False
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    except OSError:
        print(f"포트 {PORT} 를 이미 다른 프로그램(이미 켜 둔 UI 등)이 쓰고 있습니다: http://127.0.0.1:{PORT}/")
        print("기존 창을 닫고 다시 실행하거나 --port 로 다른 포트를 지정하세요.")
        return 1
    runner = TrainRunner()
    sysmon = SystemMonitor()
    url = f"http://127.0.0.1:{PORT}/"
    print(f"MOSS 학습 UI: {url}  (종료: Ctrl+C — 실행 중인 학습은 계속됩니다)")
    if not a.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    sys.exit(main())
