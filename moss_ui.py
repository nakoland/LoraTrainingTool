# MOSS-Image LoRA 학습 UI (로컬 웹 서버)
# 사용법: run.bat (포트 변경은 run.bat --port 7866)
# - 설정 toml / dataset.toml / prompts.txt 를 주석을 보존한 채 편집
# - 데이터셋 점검(캡션 누락·캐시·해상도), 설정 검사
# - 학습 실행/중지, 로그·진행률·loss 그래프·샘플 이미지 표시
# 학습 프로세스의 출력은 파일로 직접 쓰므로 UI 를 닫았다 다시 열어도 학습은 계속되고 다시 붙는다.

import argparse
import codecs
import collections
import ctypes
import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import mimetypes
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import tomllib
import urllib.parse
import uuid
import webbrowser
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import moss_cache
import moss_dop
import moss_shuffle
import moss_notify
import moss_save
import moss_tags
import moss_masks
import slider_pairs

try:
    import toml as toml_lib  # sd-scripts 가 설정을 읽는 라이브러리. 저장 전 검증에도 쓴다
except ImportError:
    toml_lib = None

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "sd-scripts")
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
        "save_from_pct", "save_max_count",
        "sample_prompts", "sample_every_n_epochs", "sample_at_first", "save_state", "sample_preset", "sample_negative",
        "sample_dit", "sample_turbo_lora", "sample_turbo_strength", "sample_look_lora", "sample_look_strength",
    ],
    "network": ["network_module", "network_dim", "network_alpha", "network_train_unet_only", "network_args", "network_weights"],
    "optimizer": [
        "optimizer_type", "optimizer_args", "learning_rate", "lr_scheduler", "lr_warmup_steps", "max_train_epochs",
        "seed", "max_grad_norm", "gradient_accumulation_steps", "lora_ema_decay",
    ],
    "timestep": [
        "timestep_sampling", "logit_mean", "logit_std", "discrete_flow_shift", "sigmoid_scale", "weighting_scheme",
        "timestep_cutoff", "timestep_uniform_ratio", "clean_share",
    ],
    "masked": ["masked_loss", "masked_prior_weight", "transparent_mask"],
    "triggers": ["trigger_style", "trigger_general"],
    "dop": ["dop", "dop_trigger", "dop_class", "dop_every", "dop_weight"],
    "caption": ["caption_shuffle", "caption_shuffle_mode", "caption_shuffle_count", "caption_shuffle_style", "caption_shuffle_general"],
    "memory": [
        "mixed_precision", "gradient_checkpointing", "cache_latents", "cache_latents_to_disk", "cache_text_encoder_outputs",
        "cache_text_encoder_outputs_to_disk", "convrot_int8", "convrot_int8_text_encoder", "blocks_to_swap",
        "unsloth_offload_checkpointing", "attn_mode", "split_attn", "compile",
    ],
    "slider": [
        "slider", "slider_method", "slider_target_class", "slider_positive", "slider_negative", "slider_anchor",
        "slider_guidance", "slider_anchor_strength", "slider_pair_dir", "slider_pair_first",
        "slider_pair_repeats", "slider_pair_loss", "slider_pair_scale", "slider_pair_noise", "slider_pair_center", "slider_pair_diff_weight", "slider_pair_hold", "slider_layers", "slider_skip_last_blocks",
        "slider_pair_mask", "slider_pair_mask_keep", "slider_pair_mask_top", "slider_pair_mask_bottom",
        "slider_sigma_min", "slider_sigma_max", "slider_clean_share", "slider_sample_multipliers",
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


_text_override = threading.local()  # 점검이 저장 안 된 값으로 볼 때 {정규화 경로: 글} (analyze(overrides=))


def _text_key(path):
    return os.path.normcase(os.path.abspath(path))


def read_text(path):
    ov = getattr(_text_override, "map", None)
    if ov:
        text = ov.get(_text_key(path))
        if text is not None:
            return text
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


def is_inside(path, root):
    """path 가 root 폴더 안(하위 폴더 포함)에 있는지."""
    return bool(path and root) and norm(path).startswith(norm(root) + os.sep)


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
SLIDER_DIR_RE = re.compile(r"^sliders?$", re.I)  # 그림 쌍 슬라이더 자료: <주제>/slider/<세트>/a·b
SET_DIR_RE = {"training": TRAINING_DIR_RE, "slider": SLIDER_DIR_RE}
EXCLUDED_DIRS = {"discarded", "original_format", slider_pairs.EXCLUDE_DIR}  # 전처리 도구가 남긴 폴더·쌍 미리보기에서 뺀 쌍. 학습 대상이 아니다
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
    return {"images": len(stems), "captions": sum(s in caps for s in stems), "other_captions": other_captions,
            "subdirs": sorted(subdirs, key=str.lower), "excluded": name.lower() in EXCLUDED_DIRS}


def folder_verdict(st, cap_ext, scan=None):
    """폴더 하나가 학습에 바로 쓸 만한지 판정한다. scan 은 선택한 폴더만 주는 scan_subset 결과.
    level: ok / warn / bad / skip(학습 제외) / none(이미지 없음)"""
    images = st["images"]
    if not images:
        return {"level": "none", "notes": ["이 폴더에는 바로 쓸 이미지가 없습니다"]}
    if st["excluded"]:
        return {"level": "skip", "notes": ["전처리 도구가 남긴 폴더나 쌍 미리보기에서 뺀 쌍의 폴더라 학습 대상이 아닙니다"]}
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
    if scan and scan["small_count"]:
        notes.append(f"저해상도 이미지 {scan['small_count']}장")
    return {"level": level, "notes": notes}


def subfolder_report(path, cap_ext):
    """선택한 폴더 아래에서 이미지가 든 폴더를 모은다. usable 은 일괄 등록 대상(학습 제외 폴더가 아닌 것)."""
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
        usable = not st["excluded"]
        verdict = folder_verdict(st, cap_ext)
        rows.append({"name": rel.replace("\\", "/"), "path": base.replace("\\", "/"), "images": st["images"],
                     "captions": st["captions"], "usable": usable, "verdict": verdict})
    return rows, truncated


def browse_dir(path, cap_ext, detail=None):
    path = os.path.abspath(path)
    st = dir_stats(path, cap_ext)
    entries = []
    for name in st["subdirs"]:
        p = os.path.join(path, name)
        s = dir_stats(p, cap_ext)
        entries.append({"name": name, "path": p.replace("\\", "/"), "images": s["images"], "captions": s["captions"],
                        "subdirs": len(s["subdirs"]), "excluded": s["excluded"],
                        "is_training": bool(TRAINING_DIR_RE.match(name)), "is_slider": bool(SLIDER_DIR_RE.match(name))})
    parent = os.path.dirname(path)
    info = {"path": path.replace("\\", "/"), "parent": parent.replace("\\", "/") if parent != path else None,
            "images": st["images"], "captions": st["captions"], "excluded": st["excluded"],
            "preview": [p.replace("\\", "/") for p in list_images(path)[:8]] if st["images"] else [],
            "entries": entries}
    if detail is not None:
        general = {"caption_extension": cap_ext, "enable_multiline_caption": detail["multiline"]}
        scan = scan_subset({"image_dir": path}, general, detail["low_area"]) if st["images"] else None
        info["verdict"] = folder_verdict(st, cap_ext, scan)
        info["subfolders"], info["subfolders_truncated"] = subfolder_report(path, cap_ext)
    return info


def find_training_sets(root, cap_ext, force=False, kind="training"):
    """root 아래의 training 폴더(kind="slider" 면 slider 폴더)를 찾아, 그 안에서 이미지가 든 폴더를 모두 모은다.
    반환 항목: topic(=root 바로 아래 폴더), group(이미지 폴더의 부모 = 세트), 폴더 통계."""
    dir_re = SET_DIR_RE.get(kind, TRAINING_DIR_RE)
    key = (norm(root), cap_ext, kind)
    hit = _sets_cache.get(key)
    if hit and not force and time.time() - hit[0] < 60:
        return hit[1]
    root = os.path.abspath(root)
    found = []
    training_dirs = []
    for base, dirs, _files in os.walk(root):
        depth = os.path.relpath(base, root).count(os.sep) if base != root else -1
        dirs[:] = sorted((d for d in dirs if not d.startswith(".")), key=str.lower)
        hits = [d for d in dirs if dir_re.match(d)]
        for d in hits:
            training_dirs.append(os.path.join(base, d))
        # training 안쪽은 아래에서 따로 훑는다 / 너무 깊이는 내려가지 않는다
        dirs[:] = [] if depth >= 4 else [d for d in dirs if d not in hits]
    for tdir in training_dirs:
        for base, dirs, _files in os.walk(tdir):
            depth = os.path.relpath(base, tdir).count(os.sep) if base != tdir else -1
            dirs[:] = sorted((d for d in dirs if not d.startswith(".") and d.lower() != slider_pairs.EXCLUDE_DIR), key=str.lower)
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
                "excluded": st["excluded"],
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


def split_caption(cap, multiline, separator=CAPTION_SEPARATOR):
    """캡션 파일 내용을 학습에 쓰이는 후보로 나눈다. 두 번째 값은 구분자 줄 오류 여부.
    여러 줄 캡션이 꺼져 있으면 sd-scripts 는 첫 줄만 읽는다."""
    if not multiline:
        return [c for c in [cap.split("\n")[0].strip()] if c], False
    cands, cur, bad = [], [], False
    for line in cap.split("\n"):
        s = line.strip()
        if s.startswith(separator):
            if s[len(separator):].split("#", 1)[0].strip():
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
    """저장된 데이터셋 폴더의 이미지와 캡션 후보를 모은다."""
    cfg = flatten_config(tomllib.loads(read_text(config_path)))
    if cfg.get("slider") and str(cfg.get("slider_method") or "pair") == "pair":
        # 그림 쌍 슬라이더: 학습 그림은 쌍 폴더다(데이터셋 설정에 학습 폴더가 남아 있어도 쓰지 않음, 2026-11-13)
        _cfg, pdir, cap_ext, _reso, _steps = _pair_context(config_path)
        if not pdir:
            return {"folders": []}
        _image_roots.add(os.path.abspath(pdir))
        images = []
        for img in slider_pairs.list_images(pdir):
            try:
                with open(os.path.splitext(img)[0] + cap_ext, "r", encoding="utf-8") as f:
                    captions = split_caption(f.read(), False)[0]
            except (OSError, UnicodeDecodeError):
                captions = []
            images.append({"path": img.replace("\\", "/"), "name": os.path.basename(img), "captions": captions,
                           "mask": None, "size": image_size(img)})
        return {"folders": [{"path": pdir.replace("\\", "/"), "name": os.path.basename(pdir), "images": images, "legacy_dir": "", "pair": True}],
                "masked_loss": False, "masked_prior_weight": 0}
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
        # 마스크 학습: 그림 옆의 <이름>.msk (예전 방식의 마스크 폴더 conditioning_data_dir 도 읽음)
        legacy = legacy_mask_dir(sub)
        if legacy:
            _image_roots.add(os.path.abspath(legacy))
        masks = moss_masks.folder_masks(rp, legacy)
        images = []
        for img in list_images(rp):
            try:
                with open(os.path.splitext(img)[0] + cap_ext, "r", encoding="utf-8") as f:
                    captions = split_caption(f.read(), multiline)[0]
            except (OSError, UnicodeDecodeError):
                captions = []
            m = masks.get(img)
            af = moss_masks.alpha_fraction(img)
            images.append({"path": img.replace("\\", "/"), "name": os.path.basename(img), "captions": captions,
                           "mask": m.replace("\\", "/") if m else None, "size": image_size(img),  # 크기: 저해상도 보기·2배 확대
                           "alpha": round(af, 4) if af >= moss_masks.TRANSPARENT_MIN else 0})  # 투명 배경 비율(투명한 곳은 학습 안 함)
        folders.append({"path": rp.replace("\\", "/"), "name": os.path.basename(rp), "images": images,
                        "caption_separator": sub.get("caption_separator", merged.get("caption_separator")) or ",",
                        "legacy_dir": legacy.replace("\\", "/") if legacy else ""})
    return {"folders": folders, "masked_loss": bool(cfg.get("masked_loss")), "transparent_mask": transparent_on(cfg),
            "masked_prior_weight": cfg.get("masked_prior_weight") or 0}


def legacy_mask_dir(sub):
    """(예전 방식) 데이터셋 설정의 마스크 폴더(conditioning_data_dir). 없으면 ''."""
    return os.path.normpath(resolve_path(sub["conditioning_data_dir"])) if sub.get("conditioning_data_dir") else ""


_mask_report_cache = {}


def transparent_on(cfg):
    """transparent_mask(투명한 곳은 학습 안 함): 키가 없으면 켬(moss_extras.transparent_mask_on 과 같음)."""
    v = cfg.get("transparent_mask", "on")
    return not (v is False or str(v).strip().lower() in ("off", "false", "0", "no"))


def mask_report(image_dir, legacy_dir=""):
    """학습 폴더의 마스크 점검: {masks, missing, bad_aspect: [이름]}. 폴더들의 수정 시각이 같으면 기억한 값
    (.msk 를 더하거나 지우면 폴더 시각이 바뀐다. 헤더만 읽어도 장수가 많으면 느림)."""
    try:
        key = (norm(image_dir), norm(legacy_dir), os.stat(image_dir).st_mtime_ns,
               os.stat(legacy_dir).st_mtime_ns if legacy_dir and os.path.isdir(legacy_dir) else 0)
    except OSError:
        return None
    if key in _mask_report_cache:
        return _mask_report_cache[key]
    from PIL import Image
    masks = moss_masks.folder_masks(image_dir, legacy_dir)
    rep = {"masks": 0, "missing": 0, "bad_aspect": []}
    for img in list_images(image_dir):
        m = masks.get(img)
        if not m:
            rep["missing"] += 1
            continue
        try:
            with Image.open(img) as a, Image.open(m) as b:
                ok = moss_masks.aspect_ok(b.size, a.size)
        except OSError:
            ok = False
        if ok:
            rep["masks"] += 1
        else:
            rep["bad_aspect"].append(os.path.basename(img))
    _mask_report_cache[key] = rep
    return rep


def cached_dropout_rate(npz_path):
    """TE 캐시에 저장된 caption_dropout_rate. npz 는 zip 이고 값은 .npy 끝의 float32 하나다. 읽지 못하면 None."""
    try:
        with zipfile.ZipFile(npz_path) as z:
            return struct.unpack("<f", z.read("caption_dropout_rate.npy")[-4:])[0]
    except (OSError, KeyError, zipfile.BadZipFile, struct.error):
        return None


def reso_area(resolution):
    """해상도 설정(1024 / [1024, 768] / "1024,768")의 면적. 저해상도 = 그림 면적이 이보다 작은 것(학습에서 원래 크기로 쓰임)."""
    try:
        r = [int(v) for v in re.split(r"[,x ]+", str(resolution).strip("[]")) if v.strip()] if not isinstance(resolution, (list, tuple)) else [int(v) for v in resolution]
    except (TypeError, ValueError):
        return 0
    return r[0] * (r[1] if len(r) > 1 else r[0]) if r else 0


def dop_report(cfg, general, ds, subsets, add):
    """DOP(moss_dop): 트리거·조건 점검, 트리거가 든 그림 수와 바뀐 캡션 예."""
    if not (cfg.get("cache_text_encoder_outputs") or cfg.get("cache_text_encoder_outputs_to_disk")):
        add("error", "DOP 는 TE 출력 캐시가 필요합니다 — 메모리 탭에서 'TE 출력 캐시'를 켜세요", "train")
    if not cfg.get("network_train_unet_only"):
        add("error", "DOP 는 'DiT 만 학습'이 필요합니다", "train")
    if int(cfg.get("blocks_to_swap") or 0) > 0:
        add("error", "DOP 는 블록 스왑과 함께 쓸 수 없습니다 — 메모리 탭에서 블록 스왑을 0 으로", "train")
    try:
        pairs = moss_dop.dop_pairs(cfg.get("dop_trigger"), cfg.get("dop_class"))
    except ValueError as e:
        add("error", str(e), "train")
        return
    if not pairs:
        add("error", "DOP 가 켜져 있지만 '뺄 트리거'가 비어 있습니다", "train")
        return
    hit = total = 0
    example = None
    per = []
    mixed = {**general, **{k: v for k, v in ds.items() if k != "subsets"}}
    for sub in subsets:
        rp = resolve_path(sub.get("image_dir", ""))
        if not os.path.isdir(rp):
            continue
        cap_ext, multiline = subset_caption_options(sub, mixed)
        n = h = 0
        for img in list_images(rp):
            n += 1
            try:
                with open(os.path.splitext(img)[0] + cap_ext, "r", encoding="utf-8") as f:
                    cands, _bad = split_caption(f.read(), multiline)
            except (OSError, UnicodeDecodeError):
                continue
            outs = [(c, moss_dop.dop_caption(c, pairs)) for c in cands]
            got = [o for o in outs if o[1] is not None]
            if got:
                h += 1
                example = example or got[0]
        hit += h
        total += n
        per.append(f"{os.path.basename(sub.get('image_dir', ''))} {h}/{n}")
    if not hit:
        add("error", f"DOP: 캡션에 트리거({cfg.get('dop_trigger')})가 든 그림이 없습니다 — 트리거 철자를 캡션과 맞추세요", "train")
        return
    ex = f" 예: '{example[0][:80]}' → '{example[1][:80]}'" if example else ""
    add("info", f"DOP: 트리거가 든 그림 {hit}/{total}장({', '.join(per)}), 나머지는 평소처럼 학습. 배치 1 로 학습합니다.{ex}", "train")


def shuffle_report(cfg, general, ds, subsets, add):
    """셔플 미리보기와 캡션에 들어 있는 트리거를 점검한다."""
    style, general_triggers = moss_shuffle.resolve_groups(cfg.get("caption_shuffle_style"), cfg.get("caption_shuffle_general"),
                                                         cfg.get("caption_shuffle_keep"), cfg.get("dop_trigger"))
    keep = style + general_triggers
    from_dop = ("caption_shuffle_style" not in cfg and "caption_shuffle_general" not in cfg
                and not cfg.get("caption_shuffle_keep") and bool(keep))
    try:
        mode = moss_shuffle.shuffle_mode(cfg.get("caption_shuffle_mode"))
    except ValueError as e:
        if cfg.get("caption_shuffle"):
            add("error", str(e), "train")
        mode = moss_shuffle.DEFAULT_MODE
    by_epoch = mode == moss_shuffle.DEFAULT_MODE
    if by_epoch:
        count = int(cfg.get("max_train_epochs") or 0) or None
        if cfg.get("max_train_epochs") is not None and (count is None or count < 1):
            if cfg.get("caption_shuffle"):
                add("error", "캡션 셔플: 에포크 수는 1 이상이어야 합니다", "train")
            count = None
    else:
        try:
            count = moss_shuffle.fixed_count(cfg.get("caption_shuffle_count"))
        except (TypeError, ValueError):
            count = moss_shuffle.DEFAULT_COUNT
    mixed = {**general, **{k: v for k, v in ds.items() if k != "subsets"}}
    total = with_keep = n_out = 0
    found, example, multi = set(), None, 0
    for sub in subsets:
        rp = resolve_path(sub.get("image_dir", ""))
        if not os.path.isdir(rp):
            continue
        cap_ext, multiline = subset_caption_options(sub, mixed)
        sep = sub.get("caption_separator", mixed.get("caption_separator")) or ","
        multi_sep = sub.get("multiline_caption_separator", mixed.get("multiline_caption_separator")) or CAPTION_SEPARATOR
        wildcard = sub.get("enable_wildcard", mixed.get("enable_wildcard", False))
        for img in list_images(rp):
            total += 1
            try:
                with open(os.path.splitext(img)[0] + cap_ext, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                cs = [text] if not multiline and wildcard else split_caption(text, multiline, multi_sep)[0]
            except (OSError, UnicodeDecodeError):
                cs = []
            multi += sum("\n" in c for c in cs)
            tags = {moss_shuffle.norm(t) for c in cs for t in c.split(sep)}
            got = [k for k in keep if k in tags]
            found.update(got)
            hit = bool(got)
            with_keep += hit
            show_example = example is None or (hit and not example["hit"])
            if by_epoch and len(cs) <= 1:
                available = moss_shuffle.epoch_variant_count(cs[0] if cs else "", general_triggers, style, sep)
                image_count = min(count or 1, available)
                vs = moss_shuffle.combine_variants(cs, general_triggers, min(8, image_count), sep,
                                                   style=style, by_epoch=True, image=img) if show_example else []
            else:
                vs = moss_shuffle.combine_variants(cs, general_triggers, count or 1, sep,
                                                   style=style, by_epoch=by_epoch, image=img)
                image_count = len(vs)
            n_out += image_count
            if show_example:
                example = {"variants": vs[:8], "count": image_count if count is not None else None,
                           "hit": hit, "image": img, "separator": sep}
    missing = [k for k in keep if k not in found]
    rep_ = {"style": style, "general": general_triggers, "from_dop": from_dop,
            "mode": mode, "count": count, "images": total, "with_keep": with_keep,
            "candidates": n_out if count is not None else None, "missing": missing, "example": example}
    if not cfg.get("caption_shuffle"):
        return rep_
    if not (cfg.get("cache_text_encoder_outputs") or cfg.get("cache_text_encoder_outputs_to_disk")):
        add("info", "캡션 셔플: TE 출력 캐시가 꺼져 있어 학습 중 선택된 캡션을 인코딩합니다", "train")
    if not keep:
        add("warn", "캡션 셔플: 스타일·일반 트리거가 비어 있어 모든 태그를 한 그룹으로 섞습니다", "train")
    elif missing:
        add("warn", f"캡션 셔플: 고정할 트리거 {', '.join(missing)} 가 어느 캡션에도 태그로 없습니다 — 태그 전체가 같아야 합니다(대소문자·밑줄 무시)", "train")
    src = " (DOP 의 뺄 트리거)" if from_dop else ""
    avg = n_out / total if total else 0
    mode_label = "에포크별" if by_epoch else "고정 후보"
    estimate = f"그림 {total}장 → 후보 {n_out}개(평균 {avg:.1f})" if count is not None else "후보 수는 학습 시작 시 에포크 수로 결정"
    add("info", f"캡션 셔플({mode_label}): {estimate}, 스타일 {', '.join(style) or '없음'}, "
                f"일반 {', '.join(general_triggers) or '없음'}{src} — "
                f"트리거가 든 그림 {with_keep}/{total}장. TE 캐시를 만드는 시간이 후보 수만큼 늘어납니다"
                + (f". 여러 줄(문장) 캡션 {multi}개는 섞지 않습니다" if multi else ""), "train")
    return rep_


def scan_subset(subset, general, low_area):
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
    # 그림이 캐시보다 새로운(그림을 고친) latent 캐시: 학습 시작 때 지우고 다시 만든다(moss_cache.remove_stale_latents)
    stale = moss_cache.stale_latents(rp)
    stale_stems = {os.path.splitext(os.path.basename(i))[0].lower() for _n, i in stale}
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
        if bname.lower() not in stale_stems and any(f.startswith(bname + "_") and f.endswith("_moss.npz") for f in files):
            latent_cached += 1
        if bname + "_moss_te.npz" in files:
            te_cached += 1
            te_sample = te_sample or stem + "_moss_te.npz"
        if len(images) <= 3000:
            sz = image_size(img)
            if sz:
                sizes[f"{sz[0]}x{sz[1]}"] += 1
                if sz[0] * sz[1] < low_area:
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
        latent_stale=sorted({os.path.basename(i) for _n, i in stale}, key=str.lower),
        te_cached=te_cached,
        te_dropout=cached_dropout_rate(te_sample) if te_sample else None,
        multi_candidate=sum(1 for c in cand_counts if c > 1),
        avg_caption_chars=round(sum(cap_lens) / len(cap_lens)) if cap_lens else 0,
        max_caption_chars=max(cap_lens) if cap_lens else 0,
        top_sizes=sizes.most_common(6),
        preview=[img for img in images[:12]],
    )
    return res


def parse_dataset(text):
    data = tomllib.loads(text)
    general = data.get("general", {})
    datasets = data.get("datasets", [])
    ds = datasets[0] if datasets else {}
    return data, general, ds, ds.get("subsets", [])


def load_dataset(path):
    return parse_dataset(read_text(path))


def analyze(config_path, overrides=None):
    """설정 전체를 점검해 오류/경고/정보 목록과 데이터셋 통계를 돌려준다.
    overrides = {경로: 글} 이면 그 파일들은 디스크 대신 그 글로 읽는다(화면의 저장 안 된 값, pending_texts)."""
    if overrides:
        _text_override.map = {_text_key(p): t for p, t in overrides.items()}
        try:
            return _analyze(config_path)
        finally:
            _text_override.map = None
    return _analyze(config_path)


def _analyze(config_path):
    issues = []

    def add(level, msg, tab="run", key=None):
        issues.append({"level": level, "msg": msg, "tab": tab, "key": key})

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
    ov = getattr(_text_override, "map", None) or {}
    if cfg.get("sample_prompts") and _text_key(resolve_path(cfg["sample_prompts"])) in ov:
        paths["sample_prompts"] = {"state": "ok", "size": len(ov[_text_key(resolve_path(cfg["sample_prompts"]))].encode("utf-8"))}  # 저장하면 만들어짐
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
    # moss_image_train_network.assert_extra_args: 디스크 저장이 켜져 있으면 TE 출력 캐시도 켠다 → 그 뒤 'DiT 만 학습' 이 아니면 assert
    te_cache = cfg.get("cache_text_encoder_outputs") or cfg.get("cache_text_encoder_outputs_to_disk")
    if te_cache and not cfg.get("network_train_unet_only"):
        add("error", "TE 출력 캐시(디스크 저장만 켜도 자동으로 켜짐)는 'DiT 만 학습'이 필요해 학습이 시작되지 않습니다 — "
                     "학습 설정 탭에서 'DiT 만 학습'을 켜거나(권장), 텍스트 인코더도 학습하려면 메모리 탭의 TE 출력 캐시와 디스크 저장을 모두 끄세요", "train")
    if not cfg.get("network_train_unet_only"):
        add("warn", "텍스트 인코더도 학습합니다 — 공식 권장은 텍스트 인코더 고정, bf16 이면 VRAM 약 32GB", "train")
    if cfg.get("cache_text_encoder_outputs_to_disk") and not cfg.get("cache_text_encoder_outputs"):
        add("info", "TE 출력 캐시 디스크 저장이 켜져 있어 학습할 때 TE 출력 캐시도 자동으로 켜집니다", "memory")
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
    share_key = "slider_clean_share" if cfg.get("slider") else "clean_share"  # 슬라이더는 전용 칸만 쓴다(moss_extras.prepare_args)
    if cfg.get(share_key):
        try:
            text = str(cfg[share_key]).replace(" ", "").lower()
            if text.startswith("auto"):  # moss_extras.resolve_share: 'auto' 또는 'auto:처음:나중', 에포크 수에 맞춤
                parts = [float(x) for x in text[4:].lstrip(":").split(":") if x != ""]
                if len(parts) > 2 or any(not 0 < v < 100 for v in parts):
                    raise ValueError
                if not cfg.get("max_train_epochs"):
                    add("warn", "깨끗한 끝 비율 auto 는 에포크 수(max_train_epochs)로 계산합니다 — 에포크 수가 없어 10에포크 기준으로 계산합니다", "train")
            else:
                parts = [float(x) for x in text.split(":") if x != ""]  # moss_extras.parse_share 와 같은 형식
                if len(parts) not in (1, 3, 4, 5):
                    raise ValueError
                if any(not 0 < v < 100 for i, v in enumerate(parts) if i in (0, 2)):
                    raise ValueError
            if cfg.get("timestep_sampling") != "truncated_logit_normal":
                add("info", f"깨끗한 끝 비율을 쓰면 타임스텝 샘플링이 {cfg.get('timestep_sampling') or '기본'} 대신 truncated_logit_normal 로 바뀝니다",
                    "dataset" if cfg.get("slider") else "train")
        except Exception:
            add("error", f"깨끗한 끝 비율 '{cfg[share_key]}' 형식이 틀렸습니다 — auto, auto:30:60, 40 또는 30:7:60:8:3 처럼 쓰세요",
                "dataset" if cfg.get("slider") else "train")
    if cfg.get("sample_dit") and not os.path.isfile(resolve_path(cfg["sample_dit"])):
        add("error", f"샘플 모델 파일이 없습니다: {cfg['sample_dit']}", "output")
    if cfg.get("sample_preset") == "turbo":
        if not cfg.get("sample_turbo_lora") or not os.path.isfile(resolve_path(cfg["sample_turbo_lora"])):
            add("error", f"샘플 방식이 터보인데 터보 LoRA 파일이 없습니다: {cfg.get('sample_turbo_lora') or '(비어 있음)'}", "output")
        if cfg.get("sample_look_lora") and not os.path.isfile(resolve_path(cfg["sample_look_lora"])):
            add("warn", f"SFT 룩 LoRA 파일이 없어 빼고 그립니다: {cfg['sample_look_lora']}", "output")
        if cfg.get("sample_dit"):
            add("info", "터보 샘플은 학습 모델(Base)로 그립니다 — 샘플 모델(SFT)은 쓰지 않습니다(터보 LoRA 가 Base 에서 증류됨)", "output")
    if cfg.get("lr_warmup_steps") and cfg.get("lr_scheduler", "constant") == "constant":
        add("error", "constant 스케줄러에는 워밍업을 쓸 수 없어 학습이 시작되지 않습니다 — constant_with_warmup 을 고르거나 워밍업을 0 으로 두세요", "train")
    if not triton_info()["installed"]:
        if cfg.get("convrot_int8") or cfg.get("convrot_int8_text_encoder"):
            add("warn", "convrot_int8 에 triton 이 없어 느린 bf16 대체 경로로 동작합니다 (pip install triton-windows)", "memory")
        if cfg.get("compile"):
            add("error", "compile 에는 triton 이 필요합니다 (pip install triton-windows)", "memory")
    if cfg.get("compile"):
        if not msvc_available():
            add("error", "compile 에는 MSVC C++ 컴파일러가 필요합니다 — Visual Studio Build Tools 2022 의 'C++를 사용한 데스크톱 개발'을 설치하세요", "memory")
        if cfg.get("slider"):
            add("info", "슬라이더 학습에서 compile 은 시험하지 않았습니다(배율 ±1 마다 한 번씩 더 컴파일할 수 있음)", "memory")
    keep = moss_save.save_epochs(cfg.get("max_train_epochs"), cfg.get("save_every_n_epochs"), cfg.get("save_from_pct"), cfg.get("save_max_count"))
    if keep is not None and cfg.get("save_every_n_epochs"):
        add("info", f"저장: 에포크 LoRA {f'{keep[0]}~{keep[-1]}' if keep else '없음'} + 최종({cfg.get('max_train_epochs')}) = {len(keep) + 1}개", "output")
    if not cfg.get("save_every_n_epochs") and not cfg.get("save_every_n_steps"):
        add("info", "중간 저장이 꺼져 있습니다 — 마지막에만 저장됩니다", "output")

    # 슬라이더 LoRA (moss_slider.py). text: 캡션 대신 프롬프트 3개로 학습, pair: + / − 그림 쌍
    slider = bool(cfg.get("slider"))
    pair_mode = slider and cfg.get("slider_method", "pair") == "pair"
    if slider and cfg.get("dop"):
        add("info", "DOP 는 슬라이더에서 쓰지 않아 저절로 꺼집니다(설정 파일 값은 일반 LoRA 일 때만 씀)", "train")
    if slider and (cfg.get("masked_loss") or cfg.get("masked_prior_weight")):
        add("info", "마스크 학습은 슬라이더에서 쓰지 않아 저절로 꺼집니다(그림 쌍은 슬라이더 자료 탭의 쌍 마스크를 씀)", "train")
    if slider:
        same_file, same_text = prompt_twins(config_path, cfg)
        if same_file:
            add("warn", f"샘플 프롬프트 파일을 일반 LoRA 설정({', '.join(same_file)})과 함께 씁니다 — 슬라이더 샘플이 그 프롬프트로 그려지고, "
                "여기서 고치면 그 설정의 프롬프트도 바뀝니다. 새 슬라이더 설정을 만들어 쓰세요", "output")
        elif same_text:
            add("warn", f"샘플 프롬프트가 일반 LoRA 설정({', '.join(same_text)})과 똑같습니다 — 일반 LoRA 설정을 복사해 만든 것 같습니다. "
                "바꿀 속성이 잘 보이는 프롬프트로 고치세요", "output")
    pair_report = None
    if slider:
        if pair_mode:
            pdir = cfg.get("slider_pair_dir")
            if not pdir:
                add("error", "그림 쌍: 쌍 폴더가 비어 있습니다 — 왼쪽에서 slider 폴더 안의 그림 폴더를 고르세요", "dataset")
            elif not os.path.isdir(resolve_path(pdir)):
                add("error", f"그림 쌍: 쌍 폴더가 없습니다 — {pdir}", "dataset")
            else:
                pair_report = slider_pairs.match_pairs(resolve_path(pdir), cfg.get("slider_pair_first") or "neg")
                _image_roots.add(resolve_path(pdir))
                if not pair_report["pairs"]:
                    add("error", f"그림 쌍: 쌍을 만들 그림이 없습니다 (그림 {pair_report['image_count']}장, 2장 이상 필요)", "dataset")
                if pair_report["leftover"]:
                    add("warn", f"그림 쌍: 그림이 {pair_report['image_count']}장(홀수)이라 마지막 {pair_report['leftover'][0]} 은 쓰지 않습니다. "
                        "중간에 한 장이 빠졌다면 그 뒤 쌍이 모두 어긋나므로 매칭 미리보기로 확인하세요", "dataset")
                for m in pair_report["mismatched"][:5]:
                    add("warn", f"그림 쌍: {m['name']} — {m['note']}. − 그림을 + 크기로 늘려 맞추므로 위치가 어긋날 수 있습니다", "dataset")
                mopt = pair_mask_cfg(cfg)
                with_mask = sum(1 for it in pair_report["pairs"] if it.get("mask"))
                ovr = [it["override"] for it in pair_report["pairs"] if it.get("override")]
                if ovr:
                    n_off = sum(1 for o in ovr if o["mode"] == "off")
                    add("info", f"그림 쌍: 쌍별 마스크 설정 {len(ovr)}개(끔 {n_off}, 자동 {len(ovr) - n_off}). 전체 설정보다 우선합니다", "dataset")
                if mopt is None:
                    add("error", "그림 쌍 마스크: 범위 위·아래는 0~100% 숫자이고 위가 아래보다 작아야 합니다", "dataset")
                elif mopt["mode"] == "off":
                    unused = sum(1 for it in pair_report["pairs"] if it.get("mask") and not (it.get("override") or {}).get("mode") == "auto")
                    if unused:
                        add("warn", f"그림 쌍: 마스크 파일이 {unused}개 있지만 마스크 합성이 꺼져 있어 쓰지 않습니다(쌍별 설정을 '자동'으로 한 쌍만 씁니다)", "dataset")
                elif mopt["mode"] == "file":
                    if with_mask < len(pair_report["pairs"]):
                        add("warn", f"그림 쌍 마스크(파일만): 마스크 파일이 없는 쌍 {len(pair_report['pairs']) - with_mask}개는 그림 그대로 학습합니다", "dataset")
                else:
                    add("info", f"그림 쌍 마스크(자동): 마스크 파일 {with_mask}개, 나머지 {len(pair_report['pairs']) - with_mask}쌍은 "
                        f"그림 높이 {mopt['top'] * 100:g}~{mopt['bottom'] * 100:g}% 안에서 두 그림이 형태로 다른 곳을 씁니다. 쌍 미리보기에서 확인하세요", "dataset")
                no_cap = sum(1 for it in pair_report["pairs"] if it["caption"] is None)
                if no_cap:
                    fb = cfg.get("slider_target_class") or ""
                    add("info", f"그림 쌍: 캡션이 없는 쌍 {no_cap}개는 '{fb}' 를 캡션으로 씁니다" if fb else f"그림 쌍: 캡션이 없는 쌍 {no_cap}개는 빈 캡션으로 학습합니다", "dataset")
            if cfg.get("cache_latents"):
                add("info", "그림 쌍: 두 그림을 따로 인코딩하므로 latent 캐시는 쓰지 않습니다(스텝마다 VAE 인코딩)", "memory")
            if (cfg.get("slider_pair_loss") or "diff") == "diff":
                add("info", "그림 쌍(원본 대비 차이): 한 스텝에 원본 예측 2번 + 학습 2번이라 일반 학습의 약 2.5배 걸립니다", "train")
                try:
                    if float(cfg.get("slider_pair_scale") or 1.0) <= 0:
                        add("error", "그림 쌍: 차이 배수는 0 보다 커야 합니다", "train")
                except (TypeError, ValueError):
                    add("error", "그림 쌍: 차이 배수는 숫자여야 합니다", "train")
            else:
                kind = "각자 복원 + 차이 가중(Fizgig 방식)" if cfg.get("slider_pair_loss") == "wrecon" else "각자 복원(논문·ai-toolkit 방식)"
                noise = "그림마다 따로" if cfg.get("slider_pair_noise") == "separate" else "같은 노이즈"
                add("info", f"그림 쌍 {kind}, 두 그림 노이즈: {noise}. 두 그림이 다른 곳만 남지만, 쌍이 적으면 그 그림들의 얼굴·화풍까지 "
                            "외울 수 있습니다", "train")
                add("info", "그림 쌍: 한 스텝에 + 와 − 를 각각 학습하므로 일반 학습의 약 2배 걸립니다", "train")
        else:
            pos, neg = str(cfg.get("slider_positive") or ""), str(cfg.get("slider_negative") or "")
            if not (pos or neg):
                add("error", "슬라이더: + 방향이나 − 방향 프롬프트 중 하나는 있어야 합니다", "dataset")
            elif pos == neg:
                add("error", "슬라이더: + 방향과 − 방향 프롬프트가 같습니다", "dataset")
            add("info", "슬라이더(프롬프트 방식): 캡션은 쓰지 않고 그림은 노이즈 입력으로만 씁니다. 한 스텝이 일반 학습의 약 3배 걸립니다", "train")
        try:
            lo, hi = float(cfg.get("slider_sigma_min", 0.0)), float(cfg.get("slider_sigma_max", 1.0))
            if not (0.0 <= lo < hi <= 1.0):
                add("error", f"슬라이더: 노이즈 구간은 0 ≤ 아래 < 위 ≤ 1 이어야 합니다 ({lo}, {hi})", "train")
            elif lo > 0.0 or hi < 1.0:
                add("info", f"슬라이더: 노이즈 σ {lo}~{hi} 구간에서만 학습합니다(타임스텝 탭의 분포를 이 구간으로 옮김)", "train")
        except (TypeError, ValueError):
            add("error", "슬라이더: 노이즈 구간 값은 숫자여야 합니다", "train")
        if (cfg.get("slider_layers") or "attn") == "attn":
            add("info", "슬라이더: 어텐션(q·k·v·o)만 학습합니다(FFN 제외, network_args 의 exclude_patterns 에 더함)", "train")
        try:
            skip = int(cfg.get("slider_skip_last_blocks") or 0)
            if not 0 <= skip <= 41:
                add("error", f"슬라이더: 뒤 블록 제외는 0 ~ 41 이어야 합니다 ({skip})", "train")
            elif skip:
                add("info", f"슬라이더: 블록 0~{41 - skip} 만 학습합니다(뒤 {skip}개 블록 {42 - skip}~41 제외)", "train")
        except (TypeError, ValueError):
            add("error", "슬라이더: 뒤 블록 제외는 정수여야 합니다", "train")
        if bts:
            add("error", "슬라이더 학습은 블록 스왑을 지원하지 않습니다 (한 스텝에 순전파를 여러 번 함)", "memory")
        try:
            mults = [float(x) for x in str(cfg.get("slider_sample_multipliers", "-1,1")).replace(" ", "").split(",") if x]
            if not mults:
                raise ValueError
        except ValueError:
            add("error", f"슬라이더: 샘플 배율은 쉼표로 구분한 숫자여야 합니다 ({cfg.get('slider_sample_multipliers')})", "output")

    # 데이터셋
    dataset = shuffle = None
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
            dop_on = bool(cfg.get("dop")) and not slider and not pair_mode
            if dop_on:
                batch = 1  # moss_dop.prepare_args 가 배치 1 데이터셋 설정을 만든다
            subs = []
            total = 0
            # 마스크 학습(moss_extras → moss_masks.prepare_dataset): 그림 옆 .msk(예전 방식 마스크 폴더도)를 학습 직전에
            # .mask_fill/ 에 모아 쓴다. 마스크 없는 그림은 그림 전체, 크기가 다른 마스크는 맞추고, 마스크 학습이 꺼져 있으면 쓰지 않는다
            prior_on = float(cfg.get("masked_prior_weight") or 0) > 0
            masked_on = bool(cfg.get("masked_loss")) or prior_on
            if not pair_mode:
                mask_total = 0
                for s in subsets:
                    name, legacy = os.path.basename(s.get("image_dir", "")), legacy_mask_dir(s)
                    if legacy and not os.path.isdir(legacy):
                        add("info", f"데이터셋 설정에 예전 방식 마스크 폴더({s['conditioning_data_dir']})가 남아 있지만 그 폴더가 없어 무시합니다 — "
                                    f"마스크는 이제 그림 옆 .msk 라 필요 없습니다. 데이터셋 탭 '마스크 폴더(예전 방식)' 칸을 비우면 사라집니다 ({name})", "dataset")
                        legacy = ""
                    rep = mask_report(resolve_path(s.get("image_dir", "")), legacy)
                    if not rep:
                        continue
                    mask_total += rep["masks"] + len(rep["bad_aspect"])
                    if masked_on and rep["bad_aspect"]:
                        add("warn", f"가로세로 비가 그림과 달라 쓰지 않는 마스크 {len(rep['bad_aspect'])}장(그림 전체를 배움): "
                                    f"{', '.join(rep['bad_aspect'][:4])}{' …' if len(rep['bad_aspect']) > 4 else ''} ({name})", "dataset", "mask")
                    if masked_on and rep["missing"] and (rep["masks"] or rep["bad_aspect"]):
                        add("info", f"마스크가 없는 그림 {rep['missing']}장은 그림 전체를 배웁니다 ({name})", "dataset", "mask")
                # 투명 배경(2026-11-20): sd-scripts 는 투명도를 버리고 투명한 곳 밑에 남은 픽셀(지운 배경·얼룩)을 학습한다
                alpha_on, alpha_total = transparent_on(cfg), 0
                for s in subsets:
                    rp = resolve_path(s.get("image_dir", ""))
                    ti = moss_masks.transparent_images(rp) if os.path.isdir(rp) else {}
                    if not ti:
                        continue
                    alpha_total += len(ti)
                    names = ", ".join(f"{os.path.basename(k)} {v * 100:.0f}%" for k, v in sorted(ti.items())[:5]) + (" …" if len(ti) > 5 else "")
                    name = os.path.basename(s.get("image_dir", ""))
                    if alpha_on:
                        add("warn", f"투명 배경 그림 {len(ti)}장({names}) — 학습은 투명도를 버리고 투명한 곳 밑에 남은 픽셀을 읽으므로, "
                                    f"투명한 곳은 학습하지 않도록 투명도를 마스크로 씁니다" + ("" if masked_on else "(마스크 학습이 저절로 켜짐)")
                                    + f". 이미지 처리 탭 '처리 후'가 밑에 남은 픽셀입니다 ({name})", "dataset", "alpha")
                    else:
                        add("warn", f"투명 배경 그림 {len(ti)}장({names}) — '투명한 곳은 배우지 않음'이 꺼져 있어 투명한 곳 밑에 남은 픽셀"
                                    f"(지운 배경·얼룩)까지 학습합니다. 이미지 처리 탭 '처리 후'가 실제 학습 그림입니다 ({name})", "dataset", "alpha")
                if mask_total and not masked_on:
                    add("info", f"마스크 {mask_total}장이 있지만 '마스크 학습'이 꺼져 있어 쓰지 않습니다(그림 전체를 배움)"
                                + (" — 투명 배경 그림은 투명도만 마스크로 씁니다" if alpha_on and alpha_total else ""), "dataset", "mask")
                if masked_on and not mask_total and not (alpha_on and alpha_total) and not any(s.get("alpha_mask") for s in subsets):
                    add("warn", "마스크 학습이 켜져 있지만 마스크가 하나도 없어 효과가 없습니다 — 이미지 처리 탭에서 마스크를 만드세요", "train")
            for sub in ([] if pair_mode else subsets):  # 그림 쌍은 dataset.toml 의 폴더를 쓰지 않는다(해상도·배치만)
                if "recursive" in sub:
                    add("error", f"sd-scripts 는 recursive 키를 지원하지 않아 학습이 시작되지 않습니다 — dataset.toml 에서 지우세요 ({os.path.basename(sub.get('image_dir', ''))})", "dataset")
                s = scan_subset(sub, {**general, **{k: v for k, v in ds.items() if k != "subsets"}}, reso_area(ds.get("resolution", 1024)))
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
                if s["no_caption_count"] and not slider:
                    add("warn", f"캡션 파일이 없는 이미지 {s['no_caption_count']}장 ({os.path.basename(sub.get('image_dir', ''))})", "dataset")
                if s["empty_caption_count"] and not slider:
                    add("warn", f"빈 캡션 {s['empty_caption_count']}장 ({os.path.basename(sub.get('image_dir', ''))})", "dataset")
                if s["bad_separator"]:
                    add("error", f"캡션 구분자 줄 뒤에 # 주석 외 문자가 있습니다: {', '.join(s['bad_separator'][:3])}", "dataset")
                if s["latent_stale"] and cfg.get("cache_latents_to_disk"):
                    st = s["latent_stale"]
                    add("info", f"고친 그림 {len(st)}장의 latent 캐시가 그림보다 오래되어 학습 시작 때 다시 만듭니다: "
                                f"{', '.join(st[:4])}{' …' if len(st) > 4 else ''} ({name})", "dataset", "stale")
                if s["small_count"]:
                    if ds.get("enable_bucket") and ds.get("bucket_no_upscale"):
                        add("info", f"면적이 해상도²보다 작은 저해상도 이미지 {s['small_count']}장 — 업스케일 안 함이라 그 크기로 학습됩니다(이미지 처리 탭에서 원본 유지 2배 확대 가능)", "dataset", "small")
                    else:
                        add("info", f"면적이 해상도²보다 작은 저해상도 이미지 {s['small_count']}장 — 학습 때 늘려서 씁니다", "dataset", "small")
                total += s["num_images"] * s["num_repeats"]
            if dop_on:
                dop_report(cfg, general, ds, subsets, add)
            if not slider and not pair_mode:
                shuffle = shuffle_report(cfg, general, ds, subsets, add)
            if not subsets and not pair_mode:
                add("error", "dataset.toml 에 [[datasets.subsets]] 가 없습니다", "dataset")
            if pair_mode and pair_report:
                total = len(pair_report["pairs"]) * max(1, int(cfg.get("slider_pair_repeats") or 1))
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
            if pair_mode:
                dataset["pairs"] = {k: (len(v) if isinstance(v, list) else v) for k, v in (pair_report or {}).items()} if pair_report else None
        except Exception as e:
            add("error", f"dataset.toml 을 읽을 수 없습니다: {e}", "dataset")

    order = {"error": 0, "warn": 1, "info": 2}
    issues.sort(key=lambda x: order[x["level"]])
    return {"issues": issues, "paths": paths, "dataset": dataset, "shuffle": shuffle}


# ---------------------------------------------------------------------------
# 학습 프로세스 관리
# ---------------------------------------------------------------------------

PROGRESS_RE = re.compile(r"(?<![A-Za-z_])(?<!Denoising )steps:\s*(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([^<\]]*)<([^,\]]*),\s*([^,\]]*)(?:,\s*avr_loss=([-+\d.eE]+))?")
EPOCH_RE = re.compile(r"^epoch (\d+)/(\d+)")
# 에포크가 시작될 때 sd-scripts 가 남기는 줄(시각은 rich 로그 머리). 남은 시간에 샘플·저장 시간을 더하는 데 쓴다
EPOCH_INC_RE = re.compile(r"epoch is incremented\. current_epoch: \d+, epoch: (\d+)")
LOG_TIME_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) ")
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


def _log_tail(path, size=65536):
    """로그 끝부분 글(예전 상태 파일에서 오류 줄을 찾을 때)."""
    try:
        with open(path, "rb") as f:
            f.seek(max(0, os.path.getsize(path) - size))
            return f.read().decode("utf-8", "replace").replace("\r", "\n")
    except OSError:
        return ""


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
        # (에포크, 그 에포크가 시작된 벽시계 초, 그때의 tqdm 경과 초). tqdm 경과에는 샘플 생성·저장 시간이 빠져 있어
        # (sd-scripts 가 샘플 뒤 진행 막대를 unpause) 두 값의 차이가 에포크마다 드는 샘플·저장 시간이다
        self.epoch_marks = collections.deque(maxlen=12)
        self._log_time = None
        self.stop_requested = False
        self.stopping = False
        self.resume_from = None  # (이어 시작한 스텝, 총 스텝)
        self.resume_path = None  # 이어 시작한 상태 폴더. 학습이 끝까지 가면 지운다
        self.cleaned = []  # 그렇게 지운 폴더
        self.saved_state = None
        # 끝난 결과(2026-11-13): outcome = done / error / stopped / None(모름), error = 마지막 예외 한 줄,
        # error_ack = 사용자가 확인함(헤더의 붉은 배경을 끔). 상태 파일에 남아 UI 를 다시 켜도 유지된다
        self.outcome = None
        self.error = ""
        self.error_ack = False
        self.shutdown_at = None  # 학습 파일이 건 PC 끄기 예정 시각(로그 'moss_shutdown <초>'). 취소하면 None
        self.on_finish = None  # 학습 프로세스가 끝나면 부름(outcome). 연속 학습(TrainQueue)이 다음을 시작한다
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
            self.resume_path = st.get("resume")
            self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        elif st.get("log") and os.path.exists(st["log"]):
            # 끝난 학습의 로그를 보여 준다
            self.log_path = st["log"]
            self.config_path = st.get("config")
            self.started = st.get("started")
            self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
            self.exit_code = st.get("exit_code", "?")
            self.outcome, self.error, self.error_ack = st.get("outcome"), st.get("error", ""), bool(st.get("error_ack"))
            if self.outcome is None:  # 예전 상태 파일: 종료 코드로 정한다(모르면 None)
                c = self.exit_code
                self.outcome = "done" if c == 0 else "stopped" if c == "중지됨" else "error" if isinstance(c, int) else None
                if self.outcome == "error":
                    self.error = moss_notify.error_summary(_log_tail(st["log"]).splitlines())
            if self.outcome == "error" and (not self.error or self.error.startswith("종료 코드")):
                # 예외 줄이 없으면 학습 파이썬의 종료 코드를 풀이한다(예: Windows 종료로 끊김)
                self.error = moss_notify.exit_reason(_log_tail(st["log"]).splitlines()) or self.error

    def _save_state(self, **extra):
        st = {"pid": self.pid, "log": self.log_path, "config": self.config_path, "started": self.started, "resume": self.resume_path,
              "outcome": self.outcome, "error": self.error, "error_ack": self.error_ack}
        if self.pid:
            st["ctime"] = process_create_time(self.pid)
        st.update(extra)
        write_text_atomic(STATE_FILE, json.dumps(st, ensure_ascii=False))

    def running(self):
        if self.proc is not None:
            return self.proc.poll() is None
        return pid_alive(self.pid)

    def start(self, config_path, extra_args, resume=None, note=None):
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
            logf.write(("> " + subprocess.list2cmdline(cmd) + "\n" + snapshot_note + "\n" + (f"{note}\n" if note else "") + "\n").encode("utf-8"))
            logf.flush()
            flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
            self.proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, creationflags=flags)
            logf.close()  # 자식이 핸들을 물려받았다
            self.pid = self.proc.pid
            self.config_path = config_path
            self.started = time.time()
            self.exit_code = None
            self.outcome, self.error, self.error_ack = None, "", False
            self.shutdown_at = None
            self.stop_requested = False
            self.stopping = False
            self.resume_from = None
            self.resume_path = resume
            self.cleaned = []
            self.saved_state = None
            self.lines.clear()
            self.epoch_marks.clear()
            self._log_time = None
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
        subprocess.run(["taskkill", "/PID", str(self.pid), "/T", "/F"], capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)

    def _stop_timeout(self, pid):
        with self.lock:
            if self.stopping and self.pid == pid and self.running():
                self._push_line("[UI] 저장이 끝나지 않아 강제로 종료합니다")
                self._kill()

    def _push_line(self, text):
        self.seq += 1
        self.lines.append((self.seq, text))
        m = LOG_TIME_RE.match(text)
        if m:
            try:
                self._log_time = time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
            except ValueError:
                pass
        m = EPOCH_INC_RE.search(text)
        if m and self._log_time is not None:
            ep = int(m.group(1))
            if not self.epoch_marks or ep > self.epoch_marks[-1][0]:  # 데이터 로더 작업 프로세스마다 같은 줄이 나온다
                self.epoch_marks.append((ep, self._log_time, self.step_marks[-1][1] if self.step_marks else 0))
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
        m = moss_notify.SHUTDOWN_RE.match(text.strip())
        if m and int(m.group(1)) > time.time() - 5:  # 지난 로그를 다시 읽을 때(UI 재시작)는 이미 지난 예약을 띄우지 않는다
            self.shutdown_at = int(m.group(1))

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
            # 남은 시간에 더할 샘플·저장 시간: 에포크마다 드는 시간 × 남은 에포크 끝 수(지금 에포크의 끝 포함)
            over = self._epoch_overhead()
            self.progress["epoch_extra"] = over
            self.progress["extra"] = over * max(0, self.epoch[1] - self.epoch[0] + 1) if over is not None and self.epoch else None

    def _epoch_overhead(self):
        """에포크마다 스텝 밖에서 드는 시간(샘플 생성·저장, 초). 최근 5개 에포크의 중앙값. 아직 모르면 None.
        2026-11-14: 남은 시간이 스텝 시간만 세어, 에포크마다 샘플 4장을 그리는 학습이 실제의 절반으로 보였다
        (59분 표시·실제 약 2시간 → 사용자가 맞춰 둔 예약 종료가 학습 70% 에서 PC 를 껐다)."""
        marks = list(self.epoch_marks)
        over = [max(0.0, (b[1] - a[1]) - (b[2] - a[2])) / (b[0] - a[0]) for a, b in zip(marks, marks[1:]) if b[0] > a[0]]
        over = sorted(over[-5:])
        return over[len(over) // 2] if over else None

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
                # UI 를 다시 켜 붙은 학습은 종료 코드를 모르므로 진행률로 끝까지 갔는지 본다
                done = code == 0 or (code == "?" and self.progress.get("total") and self.progress.get("step") == self.progress["total"])
                cleanup = self.resume_path if done and not self.stop_requested and not self.saved_state else None
                # 결과와 알림: 학습 파일이 이미 알렸으면(moss_notified 줄) 다시 보내지 않는다(강제 종료·충돌만 여기서)
                texts = [t for _s, t in self.lines]
                stopped = self.stop_requested or any(t.startswith("moss_stopped ") for t in texts[-50:])
                self.outcome = "stopped" if stopped else "done" if done else "error"
                self.error = moss_notify.error_summary(texts[-400:]) if self.outcome == "error" else ""
                if self.outcome == "error" and not self.error:
                    self.error = moss_notify.exit_reason(texts[-80:]) or (
                        f"종료 코드 {code}" if code != "?" else "이유를 알 수 없이 끝났습니다(UI 밖에서 종료)")
                notified = any(moss_notify.NOTIFIED_RE.match(t) for t in texts[-50:])
                name = os.path.splitext(os.path.basename(self.config_path or ""))[0]
                if not notified and self.outcome == "done":
                    moss_notify.notify("MOSS 학습 완료", f"{name} — 학습이 끝났습니다")
                elif not notified and self.outcome == "error":
                    moss_notify.notify("MOSS 학습 오류", f"{name} — {self.error}", error=True)
            if cleanup:
                removed, failed = remove_resumed_states(cleanup)
                with self.lock:
                    self.cleaned = removed
                    for p in removed:
                        self._push_line(f"[UI] 이어하기를 마쳐 상태 폴더를 지웠습니다: {p}")
                    for p, err in failed:
                        self._push_line(f"[UI] 상태 폴더를 지우지 못했습니다: {p} ({err})")
            self._save_state(exit_code=self.exit_code)
            cb = self.on_finish
            if cb:
                try:
                    cb(self.outcome)
                except Exception as e:
                    print("on_finish error:", e)

    def cancel_shutdown(self):
        ok = moss_notify.cancel_shutdown()
        with self.lock:
            self.shutdown_at = None
            self._push_line("[UI] PC 끄기를 취소했습니다" if ok else "[UI] 취소할 PC 끄기가 없었습니다(이미 취소됐거나 걸리지 않음)")
        return ok

    def ack_error(self):
        """오류 종료를 확인함 → 헤더의 붉은 배경을 끈다(다음 학습 시작 때 다시 초기화)."""
        with self.lock:
            self.error_ack = True
        self._save_state(exit_code=self.exit_code)
        return True

    def status(self, since):
        with self.lock:
            lines = [t for s, t in self.lines if s > since]
            first = self.lines[0][0] if self.lines else 0
            return {
                "running": self.running(),
                "pid": self.pid,
                "exit_code": self.exit_code,
                "outcome": self.outcome,
                "error": self.error,
                "error_ack": self.error_ack,
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
                "shutdown_after": moss_notify.shutdown_requested(),
                "shutdown_at": self.shutdown_at if self.shutdown_at and self.shutdown_at > time.time() - 2 else None,
                "cleaned": self.cleaned,
            }


QUEUE_FILE = os.path.join(HERE, ".moss_queue.json")
QUEUE_GAP = 5  # 앞 학습이 끝나고 다음을 시작하기까지(초): GPU 메모리가 비게


class TrainQueue:
    """연속 학습(2026-11-15 사용자 요청): 저장된 설정 여러 개를 차례로 학습하고, 모두 끝나면(스위치가 켜져 있으면) PC 를 끈다.
    항목 = {id, config, status: wait/running/done/error/stopped/unknown, error, log, started, ended}. 파일(QUEUE_FILE)에 남는다.
    다음 학습은 UI 서버가 시작하므로 앱 창을 닫으면 지금 학습 뒤로 이어지지 않는다(창 닫기 알림이 알린다).
    중간 학습이 오류로 끝나도 다음으로 간다. 사용자가 중지하면 대기열도 멈춘다.
    PC 끄기: 대기가 남아 있는 동안 moss_notify.QUEUE_MORE_FLAG 를 두어 학습 파일이 끄지 않게 하고, 마지막 학습은 학습 파일이
    (정상 완료 때) 직접 끈다. 마지막이 오류였거나 남은 항목이 시작 전 점검에서 빠져 스위치가 그대로면 서버가 끝에 끈다."""

    def __init__(self, runner):
        self.runner = runner
        self.lock = threading.RLock()
        self.items = []
        self.active = False
        self.note = ""
        self._seq = 0
        self._load()
        runner.on_finish = self._on_finish

    def _load(self):
        try:
            st = json.loads(read_text(QUEUE_FILE))
        except Exception:
            st = {}
        self.items = [i for i in st.get("items", []) if isinstance(i, dict) and i.get("config")]
        self.active, self.note = bool(st.get("active")), st.get("note", "")
        self._seq = max([int(i.get("id", 0)) for i in self.items] + [0])
        cur = self._current()
        if cur and not (self.runner.running() and norm(self.runner.config_path or "") == norm(cur["config"])):
            # UI 서버가 꺼진 사이 그 학습이 끝났다: 결과를 상태 파일에서 읽고, 다음은 저절로 시작하지 않는다
            same = self.runner.log_path and norm(self.runner.log_path) == norm(cur.get("log") or "")
            cur["status"] = (self.runner.outcome if same and self.runner.outcome else "unknown")
            cur["error"] = self.runner.error if same else ""
            if self.active and self.pending():
                self.note = "UI 가 꺼진 사이 학습이 끝나 연속 학습을 멈췄습니다. 남은 학습은 '연속 학습 시작'으로 이어 갑니다"
            self.active = False
        self._save()

    def _save(self):
        more = self.active and self.pending() > 0
        try:
            if more:
                write_text_atomic(moss_notify.QUEUE_MORE_FLAG, "1")
            elif os.path.exists(moss_notify.QUEUE_MORE_FLAG):
                os.remove(moss_notify.QUEUE_MORE_FLAG)
            write_text_atomic(QUEUE_FILE, json.dumps({"items": self.items, "active": self.active, "note": self.note}, ensure_ascii=False))
        except OSError as e:
            print("queue save error:", e)

    def _current(self):
        return next((i for i in self.items if i.get("status") == "running"), None)

    def pending(self):
        """아직 시작하지 않은 항목 수."""
        return sum(1 for i in self.items if i.get("status") == "wait")

    def snapshot(self):
        with self.lock:
            return {"items": [dict(i) for i in self.items], "active": self.active, "note": self.note}

    def set_items(self, rows):
        """화면의 목록(순서·추가·삭제)을 받는다. 시작한 항목(실행 중·끝남)은 id 로 찾아 그대로 두고, 실행 중 항목은 뺄 수 없다."""
        with self.lock:
            old = {str(i["id"]): i for i in self.items}
            new = []
            for r in rows or []:
                rid = str(r.get("id") or "")
                if rid in old:
                    new.append(old.pop(rid))
                    continue
                cp = r.get("config", "")
                if not is_config_path(cp) or not os.path.isfile(cp):
                    raise ValueError(f"도구 폴더의 설정 파일이 아닙니다: {cp}")
                # 중지한 학습 이어하기(2026-11-15 사용자 요청): 그 설정의 이어하기 목록에 있는 상태 폴더만 받는다
                resume = r.get("resume") or None
                if resume:
                    st = next((s for s in resume_states(cp) if norm(s["path"]) == norm(resume)), None)
                    if st is None:
                        raise ValueError(f"이어서 학습할 상태 폴더를 찾을 수 없습니다: {resume}")
                    resume = st["path"]
                self._seq += 1
                new.append({"id": self._seq, "config": cp, "resume": resume, "status": "wait", "error": ""})
            if any(i.get("status") == "running" for i in old.values()):
                raise ValueError("지금 학습 중인 항목은 목록에서 뺄 수 없습니다(먼저 중지하세요)")
            self.items = new
            if not self.pending():
                self.active = False
            self._save()
            return self.snapshot()

    def check(self, items):
        """시작 전 점검: (오류 목록, 같은 이름 결과 목록)."""
        problems, conflicts, names = [], [], {}
        for i in items:
            label = os.path.splitext(os.path.basename(i["config"]))[0]
            if not os.path.isfile(i["config"]):
                problems.append(f"{label}: 설정 파일이 없습니다")
                continue
            errs = [x["msg"] for x in analyze(i["config"])["issues"] if x["level"] == "error"]
            if errs:
                problems.append(f"{label}: {errs[0]}" + (f" (외 {len(errs) - 1}개)" if len(errs) > 1 else ""))
            resume = i.get("resume")
            if resume:
                st = next((s for s in resume_states(i["config"]) if norm(s["path"]) == norm(resume)), None)
                if st is None:
                    problems.append(f"{label}: 이어서 학습할 상태 폴더가 없습니다 ({os.path.basename(resume)})")
                elif st.get("warnings"):
                    problems.append(f"{label}: 이어하기 — {st['warnings'][0]}")
            cfg = flatten_config(tomllib.loads(read_text(i["config"])))
            key = (norm(resolve_path(cfg.get("output_dir")) or ""), cfg.get("output_name") or "last")
            if key in names:
                problems.append(f"{label}: 출력 이름 '{key[1]}' 이 {names[key]} 와 같아 앞 결과를 덮어씁니다 — 출력 이름을 바꾸세요")
            names[key] = label
            c = None if resume else existing_outputs(i["config"])  # 이어하기는 같은 이름에 이어 쓰는 것이 정상(/api/start 와 같음)
            if c:
                conflicts.append({"config": label, **c})
        return problems, conflicts

    def start(self, overwrite=False):
        with self.lock:
            waiting = [i for i in self.items if i.get("status") == "wait"]
            if not waiting:
                raise ValueError("대기 중인 학습이 없습니다 — 목록에 설정을 더하세요")
        problems, conflicts = self.check(waiting)  # analyze 는 수 초 걸릴 수 있어 잠금 밖에서
        if problems:
            return {"problems": problems}
        if conflicts and not overwrite:
            return {"conflicts": conflicts}
        with self.lock:
            self.active, self.note = True, ""
            self._save()
        threading.Thread(target=self._next, daemon=True).start()
        return {"ok": True}

    def pause(self):
        """지금 학습은 그대로 두고 그 뒤로 시작하지 않는다."""
        with self.lock:
            self.active = False
            self.note = "연속 학습을 멈췄습니다(지금 학습은 계속). 남은 학습은 '연속 학습 시작'으로 이어 갑니다" if self.pending() else ""
            self._save()
            return self.snapshot()

    def _on_finish(self, outcome):
        with self.lock:
            cur = self._current()
            if cur:
                cur["status"] = outcome or "unknown"
                cur["error"] = self.runner.error or ""
                cur["ended"] = time.time()
            # 대기열 밖에서 시작한 학습이 끝나도 대기열이 켜져 있으면 다음을 시작한다(그 학습 때문에 기다리던 경우)
            if cur and outcome == "stopped" and self.active:
                self.active = False
                self.note = "학습을 중지해 연속 학습도 멈췄습니다. 남은 학습은 '연속 학습 시작'으로 이어 갑니다" if self.pending() else ""
            go = self.active
            self._save()
        if go:
            threading.Timer(QUEUE_GAP, self._next).start()

    def _next(self):
        while True:
            with self.lock:
                if not self.active or self.runner.running() or self._current():
                    return
                item = next((i for i in self.items if i.get("status") == "wait"), None)
                if item is None:
                    self.active = False
                    done = sum(1 for i in self.items if i.get("status") == "done")
                    bad = sum(1 for i in self.items if i.get("status") in ("error", "unknown"))
                    self.note = f"연속 학습을 모두 마쳤습니다: 완료 {done}개" + (f", 오류 {bad}개" if bad else "")
                    self._save()
                    break
            label = os.path.splitext(os.path.basename(item["config"]))[0]
            problems = self.check([item])[0] if os.path.isfile(item["config"]) else [f"{label}: 설정 파일이 없습니다"]
            with self.lock:
                if not self.active or item.get("status") != "wait":
                    return
                if problems:  # 대기 중에 설정이 바뀌어 오류가 생김 → 건너뛴다
                    item.update(status="error", error="시작 전 점검: " + problems[0], ended=time.time())
                    self._save()
                    continue
                try:
                    studio.stop()  # 스튜디오 모델이 올라가 있으면 VRAM 을 비운다
                    pos = [i["id"] for i in self.items].index(item["id"]) + 1
                    item.update(status="running", error="", started=time.time())
                    self._save()  # 학습 파일이 시작할 때 남은 학습이 있는지(QUEUE_MORE_FLAG)가 맞도록 먼저 쓴다
                    resume = item.get("resume") or None
                    how = f" (이어하기: {os.path.basename(resume)})" if resume else ""
                    self.runner.start(item["config"], "", resume, note=f"연속 학습 {pos}/{len(self.items)}: {label}{how}")
                    item["log"] = self.runner.log_path
                    self._save()
                    return
                except Exception as e:
                    item.update(status="error", error=f"시작하지 못했습니다: {e}", ended=time.time())
                    self._save()
        # 모두 끝남: 알림, 스위치가 남아 있으면(마지막 학습이 오류였거나 마지막 항목이 시작 전에 빠짐) 여기서 PC 를 끈다
        moss_notify.notify("MOSS 연속 학습 완료", self.note)
        if moss_notify.shutdown_requested() and moss_notify.shutdown_if_requested("연속 학습"):
            with self.runner.lock:
                self.runner.shutdown_at = int(time.time()) + moss_notify.SHUTDOWN_DELAY
                self.runner._push_line("[UI] 연속 학습을 모두 마쳐 PC 를 끕니다")


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


STOP_STATE_RE = re.compile(r"^(.+)-step\d{8}-state$")  # moss_train.py 가 중지할 때 만든 상태 폴더


def delete_state_dir(path):
    """상태 폴더 하나를 지운다. train_state.json 이 있는 *-state 폴더만 지운다."""
    if not os.path.basename(path).endswith("-state") or not os.path.isfile(os.path.join(path, "train_state.json")):
        raise ValueError(f"이어하기 상태 폴더가 아닙니다: {path}")
    shutil.rmtree(path)


def remove_resumed_states(path):
    """이어하기가 끝까지 간 뒤 그 상태 폴더를 지운다. 같은 이름으로 그보다 먼저 중지해 남은 상태 폴더도
    이미 이어 쓴 지점이라 함께 지운다(에포크마다 저장한 상태는 남긴다). 반환: (지운 경로, [(경로, 오류)])."""
    targets = [path]
    m = STOP_STATE_RE.match(os.path.basename(path))
    name = m[1] if m else MODEL_NAME_RE.match(os.path.basename(path)[: -len("-state")])[1]
    try:
        limit = os.path.getmtime(os.path.join(path, "train_state.json"))
        for e in os.scandir(os.path.dirname(path)):
            m = STOP_STATE_RE.match(e.name)
            state_file = os.path.join(e.path, "train_state.json")
            if m and m[1] == name and norm(e.path) != norm(path) and os.path.isfile(state_file) and os.path.getmtime(state_file) < limit:
                targets.append(e.path)
    except OSError:
        pass
    removed, failed = [], []
    for p in targets:
        try:
            delete_state_dir(p)
            removed.append(p.replace("\\", "/"))
        except (OSError, ValueError) as e:
            failed.append((p.replace("\\", "/"), str(e)))
    return removed, failed


def existing_outputs(config_path):
    """같은 output_name 으로 이미 출력 폴더에 남은 LoRA·샘플·상태 폴더 수. 없으면 None."""
    cfg = flatten_config(tomllib.loads(read_text(config_path)))
    out = resolve_path(cfg.get("output_dir"))
    name = cfg.get("output_name") or "last"
    if not out or not os.path.isdir(out):
        return None
    g = next((g for g in output_groups(out) if g["name"] == name), None)
    states = sum(1 for e in os.scandir(out) if e.is_dir() and e.name.endswith("-state") and MODEL_NAME_RE.match(e.name[: -len("-state")])[1] == name)
    models, samples = (len(g["models"]), len(g["samples"])) if g else (0, 0)
    if not (models or samples or states):
        return None
    return {"name": name, "output_dir": out.replace("\\", "/"), "models": models, "samples": samples, "states": states}


# ---------------------------------------------------------------------------
# 결과 탭: 실행 목록 (UI 로 시작한 학습 = ui_logs/train_<시각>.log + train_<시각>_settings/)
# ---------------------------------------------------------------------------

RUN_LOG_RE = re.compile(r"^train_(\d{8}_\d{6})\.log$")
RUN_NOTES = os.path.join(RUN_LOG_DIR, "run_notes.json")
RUN_RATINGS = os.path.join(RUN_LOG_DIR, "run_ratings.json")
RUN_SLACK = 180  # 로그가 끝난 뒤 이만큼(초)까지 생긴 파일도 그 실행의 것으로 본다
_run_cache = {}
_run_annotations_lock = threading.Lock()


def _cmd_arg(cmd, flag):
    m = re.search(rf'{flag}\s+(?:"([^"]+)"|(\S+))', cmd)
    return (m[1] or m[2]) if m else None


def _parse_run_log(path):
    """로그 하나에서 명령·진행·끝난 모양을 읽는다. 크기·시각이 같으면 기억한 값을 쓴다."""
    st = os.stat(path)
    key = (st.st_size, st.st_mtime)
    hit = _run_cache.get(path)
    if hit and hit[0] == key:
        return hit[1]
    with open(path, "rb") as f:
        text = f.read(32 * 1024 * 1024).decode("utf-8", "replace")
    first = text.split("\n", 1)[0]
    cmd = first[2:] if first.startswith("> ") else ""
    prog = None
    for m in PROGRESS_RE.finditer(text):
        prog = m
    epochs = re.findall(r"(?m)^epoch (\d+)/(\d+)", text)
    res = re.search(r"(?m)^moss_resume step=(\d+) total=(\d+)", text)
    info = {
        "cmd": cmd, "config": _cmd_arg(cmd, "--config_file"), "resume": _cmd_arg(cmd, "--resume"),
        "step": None, "total": None, "loss": None, "elapsed": None, "epoch": None,
        "stopped": bool(re.search(r"(?m)^moss_stopped ", text)),
        "error": "Traceback (most recent call last)" in text[-200000:],
    }
    if prog:
        step, total = int(prog[2]), int(prog[3])
        if res:  # tqdm 은 이어 시작한 지점을 모르고 0 부터 센다
            step, total = int(res[1]) + step, int(res[2])
        info.update(step=step, total=total, elapsed=prog[4].strip(), loss=float(prog[7]) if prog[7] else None)
    if epochs:
        info["epoch"] = [int(epochs[-1][0]), int(epochs[-1][1])]
    _run_cache[path] = (key, info)
    return info


def _run_annotations(path):
    try:
        values = json.loads(read_text(path))
        return values if isinstance(values, dict) else {}
    except (OSError, ValueError):
        return {}


def _set_run_annotation(path, run_id, value):
    values = _run_annotations(path)
    if value is None:
        if run_id not in values:
            return
        del values[run_id]
    else:
        values[run_id] = value
    write_text_atomic(path, json.dumps(values, ensure_ascii=False, indent=1))


def _out_listing(out, cache):
    """출력 폴더의 LoRA·샘플·상태 폴더(한 번만 읽음)."""
    if out in cache:
        return cache[out]
    models, samples, states = [], [], []
    if out and os.path.isdir(out):
        for e in os.scandir(out):
            if e.is_file() and os.path.splitext(e.name)[1].lower() in {".safetensors", ".ckpt", ".pt"}:
                models.append({"name": e.name, "path": e.path.replace("\\", "/"), "size": e.stat().st_size, "mtime": e.stat().st_mtime,
                               "group": MODEL_NAME_RE.match(os.path.splitext(e.name)[0])[1]})
            elif e.is_dir() and e.name.endswith("-state") and os.path.isfile(os.path.join(e.path, "train_state.json")):
                stf = os.path.join(e.path, "train_state.json")
                try:
                    tst = json.loads(read_text(stf))
                except (OSError, ValueError):
                    continue
                states.append({"name": e.name, "path": e.path.replace("\\", "/"), "mtime": os.path.getmtime(stf),
                               "group": MODEL_NAME_RE.match(e.name[: -len("-state")])[1], "step": tst.get("current_step"),
                               "total": tst.get("total_steps"), "epoch": tst.get("current_epoch"),
                               "size": sum(f.stat().st_size for f in os.scandir(e.path) if f.is_file())})
        sd = os.path.join(out, "sample")
        if os.path.isdir(sd):
            for e in os.scandir(sd):
                if not e.is_file() or os.path.splitext(e.name)[1].lower() not in IMAGE_EXTENSIONS:
                    continue
                m = re.match(r"^(?:(.*)_)?(e\d{6}|\d{6})_(\d+)_\d{14}(?:_\d+)?$", os.path.splitext(e.name)[0])
                if m:
                    samples.append({"path": e.path.replace("\\", "/"), "group": m[1] or "", "col": m[2], "idx": int(m[3]),
                                    "mtime": e.stat().st_mtime, "size": e.stat().st_size})
    cache[out] = (models, samples, states)
    return cache[out]


def _run_files(info, cache):
    """실행 시간 안에 그 이름으로 생긴 LoRA·샘플·상태 폴더."""
    models, samples, states = _out_listing(info["output_dir"], cache)
    lo, hi, name = info["started"] - 5, info["ended"] + RUN_SLACK, info["output_name"]
    pick = lambda xs: [x for x in xs if x["group"] == name and lo <= x["mtime"] <= hi]
    return pick(models), pick(samples), pick(states)


def run_settings_files(run_id):
    """실행의 (로그 경로, 설정 사본 폴더 또는 None)."""
    if not re.fullmatch(r"\d{8}_\d{6}", run_id or ""):
        raise ValueError("실행 id 가 올바르지 않습니다")
    log = os.path.join(RUN_LOG_DIR, f"train_{run_id}.log")
    if not os.path.isfile(log):
        raise FileNotFoundError("실행 기록이 없습니다")
    sdir = os.path.join(RUN_LOG_DIR, f"train_{run_id}_settings")
    return log, (sdir if os.path.isdir(sdir) else None)


def _run_config(log_info, sdir):
    """학습 당시 설정(사본이 있으면 사본, 없으면 지금 원본)과 그 데이터셋 설정."""
    orig = log_info["config"]
    cp = os.path.join(sdir, os.path.basename(orig)) if sdir and orig else None
    snap = bool(cp and os.path.isfile(cp))
    if not snap:
        cp = orig if orig and os.path.isfile(orig) else None
    cfg = flatten_config(tomllib.loads(read_text(cp))) if cp else {}
    ds_path = None
    if cfg.get("dataset_config"):
        cand = os.path.join(sdir, os.path.basename(cfg["dataset_config"])) if snap else None
        ds_path = cand if cand and os.path.isfile(cand) else resolve_path(cfg["dataset_config"])
    return cfg, snap, ds_path


def run_info(run_id, cache, notes, ratings):
    log, sdir = run_settings_files(run_id)
    li = _parse_run_log(log)
    cfg, snap, ds_path = _run_config(li, sdir)
    started = datetime.datetime.strptime(run_id, "%Y%m%d_%H%M%S").timestamp()
    live = bool(runner.log_path and norm(runner.log_path) == norm(log) and runner.running())
    ended = time.time() if live else os.path.getmtime(log)
    # 끝난 모양: 진행 중 / 중지됨(이어하기 저장) / 완료 / 오류 / 중단(이유 모름)
    code = runner.exit_code if runner.log_path and norm(runner.log_path) == norm(log) else None
    if live:
        status = "running"
    elif li["stopped"] or code == "중지됨":
        status = "stopped"
    elif code == 0 or (li["total"] and li["step"] == li["total"]):
        status = "done"
    elif li["error"] or (isinstance(code, int) and code != 0):
        status = "error"
    else:
        status = "ended"
    dirs = []
    if cfg.get("slider") and str(cfg.get("slider_method") or "pair") == "pair" and cfg.get("slider_pair_dir"):
        dirs = [str(cfg["slider_pair_dir"])]
    elif ds_path and os.path.isfile(ds_path):
        try:
            dirs = [str(x.get("image_dir")) for x in load_dataset(ds_path)[3] if x.get("image_dir")]
        except Exception:
            pass
    out = resolve_path(cfg.get("output_dir")) if cfg.get("output_dir") else None
    info = {"id": run_id, "started": started, "ended": ended, "status": status, "config_path": (li["config"] or "").replace("\\", "/"),
            "config_exists": bool(li["config"] and os.path.isfile(li["config"])), "snapshot": snap, "resumed_from": li["resume"],
            "output_dir": out, "output_name": cfg.get("output_name") or "last", "slider": bool(cfg.get("slider")),
            "dirs": dirs, "step": li["step"], "total": li["total"], "loss": li["loss"], "epoch": li["epoch"],
            "max_epochs": cfg.get("max_train_epochs"), "rank": cfg.get("network_dim"), "alpha": cfg.get("network_alpha"),
            "lr": cfg.get("learning_rate"), "optimizer": cfg.get("optimizer_type"), "notes": notes.get(run_id, ""),
            "rating": ratings.get(run_id)}
    if out:
        _image_roots.add(out)
        models, samples, states = _run_files(info, cache)
        # 미리보기: 마지막으로 그린 칸(에포크·스텝)의 프롬프트별 그림, 그리고 첫 칸
        cols = {}
        for x in samples:
            cols.setdefault(x["col"], []).append(x)
        order = sorted(cols.values(), key=lambda xs: max(y["mtime"] for y in xs))
        last = sorted(order[-1], key=lambda y: y["idx"])[:6] if order else []
        info["previews"] = [{"path": x["path"], "idx": x["idx"], "col": x["col"]} for x in last]
        info["sample_count"] = len(samples)
        info["models"] = [{"name": m["name"], "size": m["size"]} for m in sorted(models, key=lambda m: m["mtime"])]
        info["model_bytes"] = sum(m["size"] for m in models)
        st = max(states, key=lambda x: x["mtime"]) if states else None
        info["state"] = {k: st[k] for k in ("path", "name", "step", "total", "epoch", "size")} if st else None
    else:
        info.update(previews=[], sample_count=0, models=[], model_bytes=0, state=None)
    return info


def list_train_runs():
    cache, runs = {}, []
    notes, ratings = _run_annotations(RUN_NOTES), _run_annotations(RUN_RATINGS)
    if os.path.isdir(RUN_LOG_DIR):
        for f in sorted(os.listdir(RUN_LOG_DIR), reverse=True):
            m = RUN_LOG_RE.match(f)
            if not m:
                continue
            try:
                runs.append(run_info(m[1], cache, notes, ratings))
            except Exception as e:  # 깨진 기록 하나 때문에 목록 전체가 안 나오지 않게
                runs.append({"id": m[1], "status": "broken", "error": str(e), "started": 0, "previews": [], "models": [], "state": None})
    return runs


def run_notes_save(run_id, text):
    text = str(text or "")[:4000]
    with _run_annotations_lock:
        run_settings_files(run_id)
        _set_run_annotation(RUN_NOTES, run_id, text if text.strip() else None)


def run_rating_save(run_id, rating):
    if type(rating) is not int or not 1 <= rating <= 5:
        raise ValueError("평가는 1부터 5까지 고를 수 있습니다")
    with _run_annotations_lock:
        run_settings_files(run_id)
        _set_run_annotation(RUN_RATINGS, run_id, rating)
    return {"rating": rating}


def run_settings_compare(run_id, restore=False):
    """학습 당시 설정 사본과 지금 원본 파일을 비교한다. restore 면 사본을 원본 자리에 되돌린다."""
    log, sdir = run_settings_files(run_id)
    li = _parse_run_log(log)
    orig = li["config"]
    if not orig or not is_config_path(orig):
        raise ValueError("이 실행의 설정 파일이 도구 설정이 아닙니다")
    if not sdir or not os.path.isfile(os.path.join(sdir, os.path.basename(orig))):
        return {"config_path": orig.replace("\\", "/"), "exists": os.path.isfile(orig), "snapshot": False, "diff": [], "files": []}
    snap_cfg_path = os.path.join(sdir, os.path.basename(orig))
    snap_cfg = flatten_config(tomllib.loads(read_text(snap_cfg_path)))
    pairs = [("설정", snap_cfg_path, orig)]  # (이름, 사본, 원본)
    for k, label in (("dataset_config", "데이터셋 설정"), ("sample_prompts", "샘플 프롬프트")):
        if snap_cfg.get(k):
            src = os.path.join(sdir, os.path.basename(str(snap_cfg[k])))
            if os.path.isfile(src):
                pairs.append((label, src, resolve_path(snap_cfg[k])))
    diff, files = [], []
    if os.path.isfile(orig):
        now = flatten_config(tomllib.loads(read_text(orig)))
        for k in sorted(set(snap_cfg) | set(now)):
            if snap_cfg.get(k) != now.get(k):
                diff.append({"key": k, "then": snap_cfg.get(k), "now": now.get(k)})
    for label, src, dst in pairs[1:]:
        same = os.path.isfile(dst) and read_text(src) == read_text(dst)
        if not same:
            files.append({"label": label, "path": dst.replace("\\", "/"), "exists": os.path.isfile(dst)})
    if restore:
        if runner.running() and runner.config_path and norm(runner.config_path) == norm(orig):
            raise RuntimeError("이 설정으로 학습하는 중에는 되돌릴 수 없습니다")
        for label, src, dst in pairs:
            if label != "설정" and not is_inside(dst, HERE):  # 도구 밖의 파일은 건드리지 않는다
                continue
            tmp = dst + ".restore.tmp"
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)
    return {"config_path": orig.replace("\\", "/"), "exists": os.path.isfile(orig), "snapshot": True, "diff": diff, "files": files}


def recycle(paths):
    """파일·폴더를 Windows 휴지통으로 옮긴다(SHFileOperationW + FOF_ALLOWUNDO)."""
    import ctypes
    from ctypes import wintypes

    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [("hwnd", wintypes.HWND), ("wFunc", wintypes.UINT), ("pFrom", ctypes.c_void_p), ("pTo", ctypes.c_void_p),
                    ("fFlags", ctypes.c_uint16), ("fAnyOperationsAborted", wintypes.BOOL), ("hNameMappings", ctypes.c_void_p),
                    ("lpszProgressTitle", ctypes.c_void_p)]

    paths = [os.path.abspath(p) for p in paths if os.path.exists(p)]
    if not paths:
        return
    text = "\0".join(paths) + "\0\0"
    buf = (ctypes.c_wchar * len(text))()
    buf[:] = text
    flags = 0x0040 | 0x0010 | 0x0004 | 0x0400  # ALLOWUNDO | NOCONFIRMATION | SILENT | NOERRORUI
    op = SHFILEOPSTRUCTW(None, 3, ctypes.cast(buf, ctypes.c_void_p), None, flags, False, None, None)
    r = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
    if r != 0 or op.fAnyOperationsAborted:
        raise OSError(f"휴지통으로 옮기지 못했습니다 (코드 {r})")


def run_delete(run_id, confirm=False):
    """실행이 만든 LoRA·샘플·이어하기 상태와 실행 기록을 휴지통으로. 학습 폴더(그림·캡션)는 건드리지 않는다."""
    log, sdir = run_settings_files(run_id)
    info = run_info(run_id, {}, {}, {})
    if info["status"] == "running":
        raise RuntimeError("학습 중인 실행은 지울 수 없습니다")
    models, samples, states = _run_files(info, {}) if info.get("output_dir") else ([], [], [])
    targets = [m["path"] for m in models] + [x["path"] for x in samples] + [x["path"] for x in states] + [log] + ([sdir] if sdir else [])
    size = sum(m["size"] for m in models) + sum(x["size"] for x in samples) + sum(x["size"] for x in states)
    res = {"models": len(models), "samples": len(samples), "states": len(states), "bytes": size, "folder": info.get("output_dir")}
    if confirm:
        with _run_annotations_lock:
            recycle(targets)
            _set_run_annotation(RUN_NOTES, run_id, None)
            _set_run_annotation(RUN_RATINGS, run_id, None)
        res["deleted"] = True
    return res


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
        # 슬라이더 학습은 샘플 번호를 (프롬프트, 배율)로 읽을 수 있게 <이름>_slider.json 을 남긴다 (moss_slider.py)
        info = os.path.join(out, "sample", f"{g['name']}_slider.json") if out else ""
        if info and os.path.isfile(info):
            try:
                with open(info, encoding="utf-8") as f:
                    g["slider"] = json.load(f)
            except (OSError, ValueError):
                pass
        g["mtime"] = max(f["mtime"] for f in g["models"] + g["samples"])
        g["size"] = sum(f["size"] for f in g["models"])
        g["sample_count"] = len(g["samples"])
        del g["samples"][MAX_GROUP_SAMPLES:]
    return sorted(groups.values(), key=lambda g: g["mtime"], reverse=True)


def pair_mask_cfg(cfg):
    """설정의 그림 쌍 마스크 옵션. 범위 값이 틀리면 None."""
    try:
        top = float(cfg.get("slider_pair_mask_top", 0.45))
        bottom = float(cfg.get("slider_pair_mask_bottom", 1.0))
    except (TypeError, ValueError):
        return None
    if not 0 <= top < bottom <= 1:
        return None
    return slider_pairs.mask_options(cfg.get("slider_pair_mask") or "off", cfg.get("slider_pair_mask_keep") or "neg", top, bottom)


def _pair_context(config_path):
    """저장된 설정의 (설정, 쌍 폴더, 캡션 확장자, 해상도, 간격). 쌍 폴더가 없으면 폴더 None."""
    if not is_config_path(config_path) or not os.path.isfile(config_path):
        config_path = DEFAULT_CONFIG
    cfg = flatten_config(tomllib.loads(read_text(config_path)))
    pdir = cfg.get("slider_pair_dir")
    pdir = resolve_path(pdir) if pdir and os.path.isdir(resolve_path(pdir)) else None
    caption_ext, reso, steps = ".txt", 1024, 64
    if cfg.get("dataset_config") and os.path.isfile(resolve_path(cfg["dataset_config"])):
        _data, general, ds, _subs = load_dataset(resolve_path(cfg["dataset_config"]))
        caption_ext = general.get("caption_extension", ds.get("caption_extension", ".txt"))
        r = ds.get("resolution", general.get("resolution", 1024))
        reso = int(max(r) if isinstance(r, (list, tuple)) else r)
        steps = int(ds.get("bucket_reso_steps", general.get("bucket_reso_steps", 64)))
    return cfg, pdir, caption_ext, reso, steps


def slider_pair_list(config_path, excluded=False):
    """그림 쌍 미리보기용 목록. 저장된 설정의 쌍 폴더를 짝지어 준다. excluded 면 제외 폴더(쌍 폴더/excluded)의 쌍."""
    cfg, pdir, caption_ext, _reso, _steps = _pair_context(config_path)
    if not pdir:
        return {"pairs": [], "error": "저장된 설정에 쌍 폴더가 없습니다"}
    _image_roots.add(pdir)
    ex = os.path.join(pdir, slider_pairs.EXCLUDE_DIR)
    out = slider_pairs.match_pairs(ex if excluded else pdir, cfg.get("slider_pair_first") or "neg", caption_ext)
    out["mask"] = slider_pairs.mask_options() if excluded else (pair_mask_cfg(cfg) or slider_pairs.mask_options())
    for it in out["pairs"]:
        it["eff"] = None if excluded else slider_pairs.effective_mask(it, out["mask"])
    out["excluded_count"] = len(slider_pairs.list_images(ex)) // 2
    out["excluded_dir"] = ex.replace("\\", "/")
    return out


def slider_pair_move(config_path, index, names, restore):
    """쌍 하나를 제외 폴더로 옮기거나(restore=False) 되돌린다. 그림·같은 이름의 캡션·_mask 그림을 함께 옮긴다."""
    cfg, pdir, _caption_ext, _reso, _steps = _pair_context(config_path)
    if not pdir:
        raise FileNotFoundError("저장된 설정에 쌍 폴더가 없습니다")
    return slider_pairs.move_pair(pdir, int(index), names, bool(restore))


_pair_render_cache = collections.OrderedDict()
_pair_info_cache = {}  # 경계 어긋남·마스크 넓이(작아서 모두 기억)


def slider_pair_image(config_path, index, side):
    """학습에 들어갈 쌍 그림(마스크 합성 후) 한쪽 또는 마스크를 PNG 로. 최근 몇 쌍은 기억한다.
    마스크 그리기용 side: rawpos·rawneg(합성 전 그림), automask(자동 마스크, 부드럽게 하기 전), filemask(마스크 파일).
    side = "info" 면 PNG 대신 {index, masked, seam, area} (경계 어긋남 %, 마스크 넓이 %)."""
    import io
    cfg, pdir, caption_ext, reso, steps = _pair_context(config_path)
    if not pdir:
        raise FileNotFoundError("저장된 설정에 쌍 폴더가 없습니다")
    item = slider_pairs.pair_item(pdir, cfg.get("slider_pair_first") or "neg", index, caption_ext)
    if item is None:
        raise FileNotFoundError(f"{index}번 쌍이 없습니다")
    mask = pair_mask_cfg(cfg) or slider_pairs.mask_options()
    if side in ("rawpos", "rawneg", "automask", "filemask"):  # 마스크 그리기용: 합성 전 그림, 부드럽게 하기 전 마스크
        img = _pair_edit_image(item, reso, steps, mask, side)
        buf = io.BytesIO()
        img.save(buf, "PNG", compress_level=1)
        return buf.getvalue()
    stats = [(os.stat(it).st_mtime_ns, os.stat(it).st_size) for it in (item["pos"], item["neg"], item["mask"]) if it]
    key = json.dumps([item["pos"], item["neg"], item["mask"], stats, reso, steps, mask, item.get("override"), slider_pairs.SEAM_VERSION], sort_keys=True)
    if side == "info" and key in _pair_info_cache:  # 전체 점검은 그림을 다시 만들지 않는다
        return {"index": index, "masked": _pair_info_cache[key].get("seam") is not None, **_pair_info_cache[key]}
    imgs = _pair_render_cache.get(key)
    if imgs is None:
        info = {}
        imgs = (*slider_pairs.render_pair(item, reso, steps, mask, info), info)
        _pair_render_cache[key] = imgs
        _pair_info_cache[key] = info
        while len(_pair_render_cache) > 4:
            _pair_render_cache.popitem(last=False)
    else:
        _pair_render_cache.move_to_end(key)
    if side == "info":
        return {"index": index, "masked": imgs[2] is not None, **imgs[3]}
    img = {"pos": imgs[0], "neg": imgs[1], "mask": imgs[2]}[side]
    if img is None:
        raise FileNotFoundError("이 쌍에는 마스크가 없습니다")
    buf = io.BytesIO()
    img.save(buf, "PNG", compress_level=1)
    return buf.getvalue()


def _pair_edit_image(item, reso, steps, mask, side):
    from PIL import Image
    pos, neg = slider_pairs.raw_pair(item, reso, steps)
    if side == "rawpos":
        return pos
    if side == "rawneg":
        return neg
    if side == "filemask":
        if not item["mask"]:
            raise FileNotFoundError("이 쌍에는 마스크 파일이 없습니다")
        return slider_pairs._load_mask(item["mask"]).resize(pos.size, Image.BILINEAR)
    eff = slider_pairs.effective_mask({**item, "mask": None}, mask)  # 범위: 쌍별 설정 → 전체 설정
    rng = eff if eff and eff["mode"] == "auto" else mask
    return slider_pairs.auto_mask(pos, neg, rng["top"], rng["bottom"])


def slider_pair_override(config_path, index, names, override):
    cfg, pdir, _caption_ext, _reso, _steps = _pair_context(config_path)
    if not pdir:
        raise FileNotFoundError("저장된 설정에 쌍 폴더가 없습니다")
    return slider_pairs.set_override(pdir, int(index), names, override)


def slider_pair_mask_save(config_path, index, names, data_url):
    """화면에서 그린 마스크(PNG data URL)를 마스크 파일로 저장한다. data_url 이 None 이면 마스크 파일을 지운다."""
    import base64
    cfg, pdir, _caption_ext, _reso, _steps = _pair_context(config_path)
    if not pdir:
        raise FileNotFoundError("저장된 설정에 쌍 폴더가 없습니다")
    png = None
    if data_url is not None:
        if not isinstance(data_url, str) or not data_url.startswith("data:image/png;base64,"):
            raise ValueError("PNG 그림만 저장할 수 있습니다")
        png = base64.b64decode(data_url.split(",", 1)[1], validate=True)
    return slider_pairs.save_mask_file(pdir, int(index), names, png)


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
        "ft=[('그림','*.png *.jpg *.jpeg *.webp *.bmp'),('모든 파일','*.*')]\n"
        "p=filedialog.askdirectory(parent=r,initialdir=d) if k=='dir' else filedialog.askopenfilename(parent=r,initialdir=d,filetypes=ft) if k=='image' else filedialog.askopenfilename(parent=r,initialdir=d)\n"
        "sys.stdout.buffer.write((p or '').encode('utf-8'))\n"
    )
    # 앱 창(pythonw)에서 콘솔 창이 깜빡이지 않게 CREATE_NO_WINDOW (Tk 대화상자는 그대로 뜬다)
    r = subprocess.run([sys.executable, "-c", script, kind, initial or ""], capture_output=True, timeout=600, creationflags=subprocess.CREATE_NO_WINDOW)
    return r.stdout.decode("utf-8").strip()


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

runner = None
queue = None  # TrainQueue(연속 학습)
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


def profile_date(name, created):
    for match in re.finditer(r"(?<!\d)(?:20\d{2}-\d{2}-\d{2}|(?:20)?\d{6})(?!\d)", name):
        date = match.group()
        try:
            return datetime.date.fromisoformat("20" + date if len(date) == 6 else date).isoformat()
        except ValueError:
            continue
    return datetime.date.fromtimestamp(created).isoformat()


def config_kinds(files):
    """학습 종류, 수정 시각과 날짜별 목록에 쓰는 프로필 날짜를 반환한다."""
    out = {}
    for p in files:
        try:
            cfg = flatten_config(tomllib.loads(read_text(p)))
            info = {"kind": "slider" if cfg.get("slider") else "normal", "mtime": os.path.getmtime(p)}
            folder = os.path.dirname(p)
            if norm(os.path.dirname(folder)) == norm(PROFILE_DIR):
                stat = os.stat(folder)
                created = getattr(stat, "st_birthtime", stat.st_ctime if os.name == "nt" else stat.st_mtime)
                info.update(created=created, profile_date=profile_date(os.path.basename(folder), created))
        except Exception:
            continue
        out[p] = info
    return out


def prompt_twins(config_path, cfg):
    """슬라이더 설정의 샘플 프롬프트가 일반 LoRA 설정과 같은 파일이거나 내용이 같으면 그 설정 이름들 → (같은 파일, 같은 내용).
    일반 LoRA 설정을 복사해 슬라이더로 바꾸면 캐릭터 프롬프트가 그대로 따라와 슬라이더 샘플이 그것으로 그려졌다."""
    sp = resolve_path(cfg.get("sample_prompts"))
    if not sp or not os.path.isfile(sp):
        return [], []
    mine = read_text(sp).strip()
    same_file, same_text = [], []
    for p in config_files():
        if norm(p) == norm(config_path):
            continue
        try:
            other = flatten_config(tomllib.loads(read_text(p)))
        except Exception:
            continue
        if other.get("slider"):
            continue
        op = resolve_path(other.get("sample_prompts"))
        if not op or not os.path.isfile(op):
            continue
        name = os.path.splitext(os.path.basename(p))[0]
        if norm(op) == norm(sp):
            same_file.append(name)
        elif mine and read_text(op).strip() == mine:
            same_text.append(name)
    return same_file, same_text


def is_config_path(path):
    """설정 파일은 도구 바로 아래 또는 profiles/<이름>/ 의 .toml 만 다룬다."""
    parent = os.path.dirname(os.path.abspath(path))
    return path.endswith(".toml") and (norm(parent) == norm(HERE) or norm(os.path.dirname(parent)) == norm(PROFILE_DIR))


def load_all(config_path):
    text = read_text(config_path)
    cfg = flatten_config(tomllib.loads(text))
    files = config_files()
    config_path = next((p for p in files if norm(p) == norm(config_path)), config_path)
    result = {"config_path": config_path, "config": cfg, "configs": files, "kinds": config_kinds(files), "dataset_root": DEFAULT_SOURCE_ROOT,
              "dataset": None, "prompts": None,
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


DATASET_TRIGGER_KEYS = (
    "trigger_style", "trigger_general", "dop_trigger",
    "caption_shuffle_style", "caption_shuffle_general", "caption_shuffle_keep",
)
_save_lock = threading.Lock()


def random_dataset_caption(dataset):
    _data, general, ds, subsets = dataset
    merged = {**general, **{k: v for k, v in ds.items() if k != "subsets"}}
    images = []
    for sub in subsets:
        folder = resolve_path(sub.get("image_dir"))
        if not folder or not os.path.isdir(folder):
            continue
        ext, multiline = subset_caption_options(sub, merged)
        separator = sub.get("multiline_caption_separator", merged.get("multiline_caption_separator")) or CAPTION_SEPARATOR
        images.extend((os.path.splitext(img)[0] + ext, multiline, separator) for img in list_images(folder))
    random.shuffle(images)
    for path, multiline, separator in images:
        try:
            captions, _bad = split_caption(read_text(path), multiline, separator)
        except (OSError, UnicodeDecodeError):
            continue
        captions = [cap for cap in captions if cap.lstrip("#").strip()]
        if captions:
            return random.choice(captions)
    return ""


def replace_sample_prompt(path, text, caption):
    """긍정 프롬프트를 하나로 바꾸고 첫 샘플의 생성 옵션을 사용한다."""
    def replace_line(line):
        safe = caption.replace("\n", "\\n").replace(" --", " —").lstrip("#").lstrip()
        options = line.partition(" --")[2]
        return safe + (" --" + options if options else "")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        prompts = json.loads(text or "[]")
        first = prompts[0] if prompts else {}
        prompt = replace_line(first.strip()) if isinstance(first, str) else {**first, "prompt": caption}
        return json.dumps([prompt] if caption else [], ensure_ascii=False, indent=2) + "\n"
    if ext == ".toml":
        data = tomllib.loads(text or "")
        settings = data.setdefault("prompt", {})
        subsets = settings.get("subset") or [{}]
        first = dict(subsets[0])
        (settings if "prompt" in settings else first)["prompt"] = caption
        settings["subset"] = [first] if caption else []
        return toml_lib.dumps(data)
    if ext != ".txt":
        return text
    lines = []
    inserted = False
    for line in (text or "").splitlines():
        if not line.strip() or line.startswith("#"):
            lines.append(line)
        elif caption and not inserted:
            lines.append(replace_line(line.strip()))
            inserted = True
    if caption and not inserted:
        lines.append(replace_line(""))
    return "\n".join(lines) + ("\n" if lines else "")


def save_all(body):
    with _save_lock:
        pending, dataset_changed = pending_texts(body, saving=True)
        saved = []
        try:
            for path, old, new in pending:
                write_text_atomic(path, new)
                saved.append((path, old))
        except OSError:
            for path, old in reversed(saved):
                if old is None:
                    os.remove(path)
                else:
                    write_text_atomic(path, old)
            raise
    return {"saved": [path for path, _old in saved], "dataset_changed": dataset_changed}


def pending_texts(body, *, saving=False):
    """저장 또는 점검용으로 바뀐 파일의 글과 학습 데이터셋 변경 여부를 계산한다."""
    config_path = body["config_path"]
    if not is_config_path(config_path):
        raise ValueError("설정 파일은 도구 폴더의 .toml 만 저장할 수 있습니다")
    config_text = read_text(config_path)
    old_cfg = flatten_config(tomllib.loads(config_text))
    config_lines = config_text.split("\n")
    for key, value in body.get("config", {}).items():
        if not re.match(r"^[A-Za-z0-9_]+$", key):
            raise ValueError(f"잘못된 키: {key}")
        config_lines = toml_set(config_lines, None, 0, key, value)
    cfg = flatten_config(tomllib.loads("\n".join(config_lines)))

    ds_path = resolve_path(cfg.get("dataset_config"))
    ds_file_changed = norm(ds_path) != norm(resolve_path(old_cfg.get("dataset_config")))
    ds_update = None
    new_dataset = None
    dataset_changed = not cfg.get("slider") and ds_file_changed
    ds_body = body.get("dataset")
    if ds_body and ds_path and (ds_body.get("general") or ds_body.get("dataset")
                               or ds_body.get("remove_subsets") or any(ds_body.get("subsets", []))):
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
            ds_update = (ds_path, text, new)
            if not cfg.get("slider"):
                new_dataset = parse_dataset(new)

                def folders(subsets):
                    return sorted(norm(resolve_path(sub["image_dir"])) for sub in subsets if sub.get("image_dir"))

                dataset_changed = dataset_changed or folders(parse_dataset(text)[3]) != folders(new_dataset[3])

    if dataset_changed:
        for key in DATASET_TRIGGER_KEYS:
            config_lines = toml_set(config_lines, None, 0, key, None)
        if "," in str(cfg.get("dop_class") or ""):
            config_lines = toml_set(config_lines, None, 0, "dop_class", None)

    new_config = "\n".join(config_lines)
    validate_toml(new_config)

    out = []
    if new_config != config_text:
        out.append((config_path, config_text, new_config))
    if ds_update:
        out.append(ds_update)
    prompts = body.get("prompts")
    refresh_sample = dataset_changed and saving
    if cfg.get("sample_prompts") and (prompts is not None or refresh_sample):
        sp = resolve_path(cfg["sample_prompts"])
        if refresh_sample and norm(sp) in (norm(config_path), norm(ds_path)):
            raise ValueError("샘플 프롬프트는 학습 설정·데이터셋과 다른 파일이어야 합니다")
        old = read_text(sp) if os.path.exists(sp) else None
        if refresh_sample:
            if new_dataset is None and ds_path and os.path.isfile(ds_path):
                new_dataset = load_dataset(ds_path)
            caption = random_dataset_caption(new_dataset) if new_dataset else ""
            prompts = replace_sample_prompt(sp, prompts if prompts is not None else old, caption)
        if prompts is not None and old != prompts:
            out.append((sp, old, prompts))
    return out, dataset_changed


def triton_info():
    if importlib.util.find_spec("triton") is None:
        return {"installed": False, "version": ""}
    for dist in ("triton-windows", "triton"):
        try:
            return {"installed": True, "version": importlib.metadata.version(dist)}
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"installed": True, "version": ""}


def find_vcvars64():
    """Visual Studio(Build Tools 포함)의 vcvars64.bat. compile(inductor)이 Windows 에서 cl.exe 를 쓴다."""
    roots = []
    vswhere = os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "Microsoft Visual Studio", "Installer", "vswhere.exe")
    if os.path.isfile(vswhere):
        try:
            out = subprocess.run([vswhere, "-latest", "-products", "*", "-property", "installationPath"], capture_output=True,
                                 text=True, timeout=30, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            roots += [x.strip() for x in out.stdout.splitlines() if x.strip()]
        except (OSError, subprocess.SubprocessError):
            pass
    for pf in (os.environ.get("ProgramFiles", r"C:\Program Files"), os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")):
        for year in ("2022", "2019"):
            for ed in ("BuildTools", "Community", "Professional", "Enterprise"):
                roots.append(os.path.join(pf, "Microsoft Visual Studio", year, ed))
    for r in roots:
        p = os.path.join(r, "VC", "Auxiliary", "Build", "vcvars64.bat")
        if os.path.isfile(p):
            return p
    return None


def msvc_available():
    return bool(shutil.which("cl")) or find_vcvars64() is not None


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
            if not old or (key == "dataset_config" and not os.path.isfile(old)):
                continue
            copy = os.path.join(dest, f"{name}_{stem}{os.path.splitext(old)[1]}")
            if os.path.isfile(old):
                shutil.copy2(old, copy)
            lines = toml_set(lines, None, 0, key, copy.replace("\\", "/"))
        lines = toml_set(lines, None, 0, "output_name", name)
        new = "\n".join(lines)
        validate_toml(new)
        config_path = os.path.join(dest, name + ".toml")
        write_text_atomic(config_path, new)
        result = save_all({**body, "config_path": config_path})
    except Exception:
        shutil.rmtree(dest, ignore_errors=True)
        raise
    return {"config_path": config_path, "dataset_changed": result["dataset_changed"]}


def new_slider_profile(body):
    """새 슬라이더 설정(profiles/<이름>/)을 만든다(2026-11-13). 학습 설정(모델·메모리·옵티마이저 등)은 base 설정에서 가져오고,
    일반 LoRA 의 것은 따라오지 않게 한다: 데이터셋 설정은 해상도·버킷 값만(학습 폴더 없음), 샘플 프롬프트는 쌍 폴더 캡션으로 새로,
    DOP·마스크 학습은 끔, 출력 이름 = 이름."""
    name = str(body.get("name", "")).strip()
    if not PROFILE_NAME_RE.match(name) or name.endswith("."):
        raise ValueError("이름에는 글자·숫자·공백·. - _ 만 쓸 수 있습니다")
    base = body.get("base") or DEFAULT_CONFIG
    if not is_config_path(base) or not os.path.isfile(base):
        raise ValueError("가져올 설정은 도구 폴더의 .toml 만 됩니다")
    pdir = str(body.get("pair_dir") or "").strip().replace("\\", "/")
    if not pdir or not os.path.isdir(pdir):
        raise ValueError(f"쌍 폴더가 없습니다: {pdir or '(비어 있음)'}")
    dest = os.path.join(PROFILE_DIR, name)
    if os.path.exists(dest):
        raise ValueError(f"이미 있는 이름입니다: {name}")
    os.makedirs(dest)
    try:
        text = read_text(base)
        cfg = flatten_config(tomllib.loads(text))
        lines = text.split("\n")
        # 데이터셋 설정: [general]·첫 [[datasets]] 값만(그림 쌍 학습은 이 값만 이어받는다, moss_slider.prepare_pair_args)
        old_ds = resolve_path(cfg.get("dataset_config"))
        data = tomllib.loads(read_text(old_ds)) if old_ds and os.path.isfile(old_ds) else {}
        first = dict((data.get("datasets") or [{}])[0])
        first.pop("subsets", None)
        first.setdefault("resolution", 1024)
        ds_text = ("# 슬라이더 설정의 데이터셋 설정: 해상도·버킷 값만 쓴다(그림은 쌍 폴더에서). "
                   "프롬프트 방식으로 바꾸면 바탕 그림 폴더를 더한다\n")
        ds_text += moss_masks.dataset_toml({"general": data.get("general") or {}, "datasets": [first]})
        ds_path = os.path.join(dest, f"{name}_dataset.toml")
        validate_toml(ds_text)
        write_text_atomic(ds_path, ds_text)
        # 샘플 프롬프트: 쌍 폴더 캡션(중복 빼고 3개까지)으로 새로
        reso = first.get("resolution", 1024)
        w, h = (reso[0], reso[1]) if isinstance(reso, list) else (reso, reso)
        caps = []
        for it in slider_pairs.match_pairs(pdir, cfg.get("slider_pair_first") or "neg")["pairs"]:
            c = re.sub(r"\s+", " ", str(it.get("caption") or "")).strip().lstrip("#").strip().replace(" --", " —")
            if c and c not in caps:
                caps.append(c)
        if not caps:
            caps = [str(cfg.get("slider_target_class") or "1girl, solo, standing, full body")]
        prompts = "# 슬라이더 샘플: 바꿀 속성이 잘 보이는 장면으로 고치세요. 배율마다(slider_sample_multipliers) 같은 시드로 그립니다\n"
        prompts += "".join(f"{c} --w {w} --h {h} --d {42 + i}\n" for i, c in enumerate(caps[:3]))
        pr_path = os.path.join(dest, f"{name}_prompts.txt")
        write_text_atomic(pr_path, prompts)
        sets = {"dataset_config": ds_path.replace("\\", "/"), "sample_prompts": pr_path.replace("\\", "/"), "output_name": name,
                "slider": True, "slider_method": "pair", "slider_pair_dir": pdir}
        if cfg.get("dop"):
            sets["dop"] = False
        if cfg.get("masked_loss"):
            sets["masked_loss"] = False
        if cfg.get("masked_prior_weight"):
            sets["masked_prior_weight"] = None
        if cfg.get("clean_share"):
            sets["clean_share"] = None  # 일반 LoRA 용(슬라이더는 노이즈 구간으로 정함)
        for k, v in sets.items():
            lines = toml_set(lines, None, 0, k, v)
        new = "\n".join(lines)
        validate_toml(new)
        config_path = os.path.join(dest, name + ".toml")
        write_text_atomic(config_path, new)
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
    if os.path.splitext(path)[1].lower() not in IMAGE_EXTENSIONS and os.path.splitext(path)[1].lower() != moss_masks.MASK_EXT:
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
                        if s.get("conditioning_data_dir"):
                            roots.append(legacy_mask_dir(s))  # 예전 방식 마스크 폴더(이미지 처리 탭의 마스크 보기)
    except Exception:
        pass
    for extra in _image_roots:
        roots.append(extra)
    return any(p.startswith(norm(r) + os.sep) for r in roots) or ref_allowed(path)


_image_roots = set()  # 현재 화면에서 연 설정의 폴더들 (/api/load 때 등록)

# 비교 그림(2026-11-20 사용자 요청): 모니터는 학습 이름 × 프롬프트 번호마다, 에포크 비교·블록 보정은 하나를 함께 쓴다.
# 사용자가 고른 그림 파일은 학습·데이터셋 폴더 밖일 수 있어 고른 파일만 /api/image 로 내보낸다(최근 고른 순, 최대 REFS_RECENT 개).
REFS_FILE = os.path.join(HERE, ".moss_refs.json")
REFS_RECENT = 40
_refs_lock = threading.Lock()


def refs_load():
    try:
        with open(REFS_FILE, encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict):
            return {"recent": [str(x) for x in d.get("recent") or [] if isinstance(x, str)],
                    "monitor": {str(k): {str(p): str(v) for p, v in (m or {}).items()} for k, m in (d.get("monitor") or {}).items() if isinstance(m, dict)},
                    "sx": str(d.get("sx") or "")}
    except (OSError, ValueError):
        pass
    return {"recent": [], "monitor": {}, "sx": ""}


def refs_save(d):
    tmp = REFS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)
    for i in range(10):  # FS-001: 다른 요청이 읽는 중이면 잠깐 기다린다
        try:
            os.replace(tmp, REFS_FILE)
            return
        except PermissionError:
            time.sleep(0.05 * (i + 1))
    os.replace(tmp, REFS_FILE)


def refs_view(output):
    d = refs_load()
    exists = lambda p: bool(p) and os.path.isfile(p)
    return {"monitor": {k: v for k, v in (d["monitor"].get(output) or {}).items() if exists(v)}, "sx": d["sx"] if exists(d["sx"]) else "",
            "recent": [x for x in d["recent"] if exists(x)][:24]}


def refs_set(body):
    """scope 가 recent 면 다음 비교에 쓸 그림을 최근 목록에만 등록한다."""
    scope, path = body.get("scope"), str(body.get("path") or "").strip().strip('"')
    if scope not in ("monitor", "sx", "recent"):
        raise ValueError("비교 그림을 저장할 위치가 올바르지 않습니다")
    if path:
        path = os.path.abspath(path)
        if not os.path.isfile(path) or os.path.splitext(path)[1].lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"그림 파일이 아닙니다: {path}")
    output = str(body.get("output") or "")
    with _refs_lock:
        d = refs_load()
        if scope == "sx":
            d["sx"] = path
        elif scope == "monitor":
            if not output:
                raise ValueError("학습 이름(output)이 없습니다")
            key = str(int(body.get("p")))
            m = d["monitor"].setdefault(output, {})
            if path:
                m[key] = path
            else:
                m.pop(key, None)
        if path:
            d["recent"] = [path] + [x for x in d["recent"] if norm(x) != norm(path)][:REFS_RECENT - 1]
        refs_save(d)
    return refs_view(output)


def ref_allowed(path):
    p = norm(path)
    return any(norm(x) == p for x in refs_load()["recent"])
preview = PreviewWorker()


# ---------------------------------------------------------------- 마스크 학습용 마스크 (이미지 처리 탭, 2026-11-12)
def _image_subset(config_path, image):
    """그림이 들어 있는 저장된 데이터셋의 학습 폴더(subset). 학습 폴더 바로 아래의 그림만 받는다."""
    cfg = flatten_config(tomllib.loads(read_text(config_path)))
    ds = resolve_path(cfg.get("dataset_config"))
    if image and os.path.isfile(image) and os.path.splitext(image)[1].lower() in IMAGE_EXTENSIONS and ds and os.path.isfile(ds):
        for i, s in enumerate(load_dataset(ds)[3]):
            if s.get("image_dir") and norm(os.path.dirname(image)) == norm(resolve_path(s["image_dir"])):
                return i, s
    raise LookupError("저장된 데이터셋의 학습 폴더에 있는 그림이 아닙니다")


def _image_masks(sub, image):
    """그 그림의 마스크 파일들: .msk 와 예전 마스크 폴더의 같은 이름 그림."""
    out = [moss_masks.msk_path(image)] if os.path.isfile(moss_masks.msk_path(image)) else []
    legacy = legacy_mask_dir(sub)
    stem = os.path.splitext(os.path.basename(image))[0].lower()
    if legacy and os.path.isdir(legacy):
        out += [p for p in moss_masks.list_images(legacy) if os.path.splitext(os.path.basename(p))[0].lower() == stem]
    return out


def _decode_png(data_url):
    import base64
    if not isinstance(data_url, str) or not data_url.startswith("data:image/png;base64,"):
        raise ValueError("PNG 그림만 저장할 수 있습니다")
    return base64.b64decode(data_url.split(",", 1)[1], validate=True)


def dataset_mask_save(config_path, image, data_url):
    """그린 마스크(PNG data URL, 흰색 = 배울 곳)를 그림 원래 크기로 그림 옆 '<이름>.msk' 에 저장한다(내용은 PNG).
    data_url 이 None 이면 그 그림의 마스크(.msk, 예전 마스크 폴더의 것도)를 휴지통으로."""
    import io
    from PIL import Image
    _i, sub = _image_subset(config_path, image)
    old = _image_masks(sub, image)
    if data_url is None:
        if not old:
            raise FileNotFoundError("지울 마스크 파일이 없습니다")
        recycle(old)
        return {"mask": None, "removed": [p.replace("\\", "/") for p in old]}
    with Image.open(io.BytesIO(_decode_png(data_url))) as m:
        m = m.convert("L")
    with Image.open(image) as im:
        size = im.size
    if m.size != size:
        m = m.resize(size, Image.BILINEAR)
    dest = moss_masks.msk_path(image)
    moss_masks.save_mask(m, dest)
    others = [p for p in old if norm(p) != norm(dest)]  # 예전 마스크 폴더의 것(그림 하나에 마스크 하나)
    if others:
        recycle(others)
    return {"mask": dest.replace("\\", "/")}


def dataset_mask_clear(config_path, folders, images, confirm):
    """학습 폴더(folders) 또는 고른 그림(images)의 마스크(.msk)를 휴지통으로. confirm 이 아니면 개수만 센다."""
    targets = _mask_targets(config_path, folders, images)
    paths = [moss_masks.msk_path(img) for img, _s in targets if os.path.isfile(moss_masks.msk_path(img))]
    if confirm and paths:
        recycle(paths)
    return {"count": len(paths), "removed": bool(confirm and paths)}


UPSCALE_BACKUP_DIR = "ORIGINAL_FORMAT"  # 2배 확대 전 원본을 옮겨 두는 학습 폴더 안 폴더(EXCLUDED_DIRS 라 학습·폴더 선택에서 빠짐)


def _unique_path(path):
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    k = 1
    while os.path.exists(f"{stem}_{k}{ext}"):
        k += 1
    return f"{stem}_{k}{ext}"


def upscale2x_image(img):
    """원본 유지 2배 확대(사용자의 pixel_preserving_upscaler 와 같은 방식): 최근접 정수 2배 = 픽셀 하나를 같은 값의 2×2 로 복제.
    보간·선명화·색 보정 없음. 팔레트·투명도·ICC·DPI 를 그대로 두고, 저장한 파일을 다시 읽어 모든 픽셀을 비교한다.
    결과는 '<이름>.png'(JPEG 등도 재압축 손실이 없게 PNG), 원본은 학습 폴더 안 ORIGINAL_FORMAT/ 으로 옮긴다.
    그림 옆 .msk 도 같은 방식으로 2배(원본 마스크는 ORIGINAL_FORMAT 에 사본), 예전 크기의 latent 캐시는 지운다."""
    import numpy as np
    from PIL import Image
    d, name = os.path.split(img)
    stem, ext = os.path.splitext(name)
    out = os.path.join(d, stem + ".png")
    if ext.lower() != ".png" and os.path.exists(out):
        raise FileExistsError(f"{stem}.png 가 이미 있어 {name} 을 PNG 로 바꿔 저장할 수 없습니다")
    with Image.open(img) as im:
        im.load()
        keep = {k: im.info[k] for k in ("icc_profile", "dpi", "transparency", "gamma") if k in im.info}
        src = im if im.mode in ("1", "L", "LA", "P", "RGB", "RGBA", "I", "I;16") else im.convert("RGBA" if "A" in im.getbands() else "RGB")
        w, h = src.size
        up = src.resize((w * 2, h * 2), Image.NEAREST)
        a = np.asarray(src)
    tmp = out + ".tmp"
    up.save(tmp, format="PNG", **keep)
    with Image.open(tmp) as chk:  # 저장한 파일을 다시 읽어 2×2 블록마다 원본 픽셀과 같은지
        b = np.asarray(chk)
        same_palette = src.mode != "P" or chk.getpalette() == src.getpalette()
    if b.shape[:2] != (h * 2, w * 2) or not same_palette or not all(np.array_equal(b[i::2, j::2], a) for i in (0, 1) for j in (0, 1)):
        os.remove(tmp)
        raise RuntimeError(f"{name}: 확대 결과가 원본 픽셀과 달라 저장하지 않았습니다")
    bdir = os.path.join(d, UPSCALE_BACKUP_DIR)
    os.makedirs(bdir, exist_ok=True)
    backup = _unique_path(os.path.join(bdir, name))
    os.replace(img, backup)
    os.replace(tmp, out)
    msk = moss_masks.msk_path(img)
    if os.path.isfile(msk):
        shutil.copy2(msk, _unique_path(os.path.join(bdir, os.path.basename(msk))))
        with Image.open(msk) as m:
            m2 = m.convert("L").resize((m.width * 2, m.height * 2), Image.NEAREST)
        moss_masks.save_mask(m2, msk)
    for f in os.listdir(d):  # 예전 크기의 latent 캐시(이름에 원본 크기가 들어 있어 더는 쓰이지 않음)
        mm = moss_cache.LATENT_RE.match(f)
        if mm and mm.group("stem").lower() == stem.lower():
            try_remove(os.path.join(d, f))
    return {"from": img.replace("\\", "/"), "to": out.replace("\\", "/"), "size": [w * 2, h * 2], "backup": backup.replace("\\", "/")}


def upscale2x(config_path, images):
    if runner.running():
        raise RuntimeError("학습 중에는 학습 그림을 바꿀 수 없습니다")
    targets = _mask_targets(config_path, None, images)
    if not targets:
        raise ValueError("확대할 그림이 없습니다(저장된 데이터셋의 학습 폴더 그림만)")
    done, errors = [], []
    for img, _s in targets:
        try:
            done.append(upscale2x_image(img))
        except (OSError, RuntimeError, ValueError) as e:
            errors.append(str(e))
    return {"done": done, "errors": errors}


def _mask_targets(config_path, folders, images):
    """[(그림 경로, subset)]: 저장된 데이터셋의 학습 폴더(folders) 전체 또는 그 안의 고른 그림(images)."""
    if not is_config_path(config_path):
        raise ValueError("도구 폴더의 설정만 다룰 수 있습니다")
    cfg = flatten_config(tomllib.loads(read_text(config_path)))
    ds = resolve_path(cfg.get("dataset_config"))
    subs = load_dataset(ds)[3] if ds and os.path.isfile(ds) else []
    want_dirs = {norm(p) for p in folders or []}
    want_imgs = {norm(p) for p in images or []}
    out = []
    for s in subs:
        idir = resolve_path(s.get("image_dir", ""))
        if not idir or not os.path.isdir(idir):
            continue
        whole = norm(idir) in want_dirs
        for img in list_images(idir):
            if whole or norm(img) in want_imgs:
                out.append((img, s))
    return out


AUTOMASK_SCRIPT = os.path.join(HERE, "moss_automask.py")
AUTOMASK_LOG = os.path.join(RUN_LOG_DIR, "automask.log")
AUTOMASK_MODEL = "selfie_multiclass_256x256.tflite"
AUTOMASK_SHA256 = "c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0"
AUTOMASK_KINDS = ("person", "head", "body_clothes", "clothes", "not_faces", "background")
FIZGIG_DIR = os.path.join(os.path.dirname(HERE), "Fizgig")  # 같은 폴더에 설치된 Fizgig 의 venv·모델을 빌려 쓴다(받지 않음)


def other_python_env():
    """다른 venv 의 파이썬을 띄울 때의 환경. uv 런처가 PYTHONHOME 을 이 venv 의 바탕 파이썬(3.11)으로 정해 두어, 그대로 물려주면
    Fizgig venv(3.12) 파이썬이 3.11 표준 라이브러리를 읽다 죽는다(실측: import 가 0.1초 만에 실패)."""
    drop = {"PYTHONHOME", "PYTHONPATH", "UV_INTERNAL__PYTHONHOME", "__PYVENV_LAUNCHER__", "VIRTUAL_ENV"}
    return {**{k: v for k, v in os.environ.items() if k.upper() not in drop}, "PYTHONUTF8": "1"}


class AutoMasker:
    """마스크 자동 만들기(moss_automask.py). 이 venv 에는 ai-edge-litert 가 없어, 그것이 있는 파이썬(Fizgig venv 등)과
    이미 받아 둔 모델 파일을 찾아 쓴다. 작업은 하나씩이고, 접수만 하고 바로 답한다(진행은 status 로 조회, NET-001)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.rt = None
        self.proc = None
        self.state = {"running": False}

    def runtime(self, refresh=False):
        if self.rt and not refresh:
            return self.rt
        import hashlib
        model = None
        for p in (os.path.join(HERE, "models", AUTOMASK_MODEL), os.path.join(FIZGIG_DIR, "user_data", "models", AUTOMASK_MODEL)):
            if os.path.isfile(p):
                with open(p, "rb") as f:
                    if hashlib.sha256(f.read()).hexdigest() == AUTOMASK_SHA256:
                        model = p
                        break
        python = None
        for py in (PYTHON, os.path.join(FIZGIG_DIR, "venv", "Scripts", "python.exe")):
            if not os.path.isfile(py):
                continue
            try:
                r = subprocess.run([py, "-c", "import ai_edge_litert.interpreter, numpy, PIL"], capture_output=True, timeout=90,
                                   env=other_python_env(), creationflags=subprocess.CREATE_NO_WINDOW)
            except (OSError, subprocess.TimeoutExpired):
                continue
            if r.returncode == 0:
                python = py
                break
        note = "" if model and python else ("사람 찾기 모델(selfie_multiclass_256x256.tflite, 16MB)이 없습니다" if not model else
                                            "ai-edge-litert 가 설치된 파이썬이 없습니다(이 venv 또는 Fizgig venv)")
        self.rt = {"ok": bool(model and python), "model": model, "python": python, "note": note}
        return self.rt

    def _spawn(self):
        rt = self.runtime()
        if not rt["ok"]:
            raise RuntimeError(rt["note"])
        os.makedirs(RUN_LOG_DIR, exist_ok=True)
        log = open(AUTOMASK_LOG, "a", encoding="utf-8")
        p = subprocess.Popen([rt["python"], "-X", "utf8", "-u", AUTOMASK_SCRIPT], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                             text=True, encoding="utf-8", env=other_python_env(), creationflags=subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS)
        log.close()
        return p

    def start(self, jobs, kind, overwrite, label):
        with self.lock:
            if self.state.get("running"):
                raise RuntimeError("마스크를 만드는 중입니다")
            p = self._spawn()
            self.proc = p
            self.state = {"running": True, "total": len(jobs), "done": 0, "counts": {}, "errors": [], "label": label,
                          "kind": kind, "started": time.time(), "seq": self.state.get("seq", 0) + 1}
        p.stdin.write(json.dumps({"model": self.rt["model"], "kind": kind, "overwrite": bool(overwrite), "jobs": jobs}, ensure_ascii=False) + "\n")
        p.stdin.close()
        threading.Thread(target=self._read, args=(p,), daemon=True).start()

    def _read(self, p):
        st = self.state
        try:
            for line in p.stdout:
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                if msg.get("done"):
                    st["counts"] = msg.get("counts", st["counts"])
                    continue
                st["done"] = msg["i"]
                st["counts"][msg["result"]] = st["counts"].get(msg["result"], 0) + 1
                if msg["result"] == "error" and len(st["errors"]) < 20:
                    st["errors"].append(f"{os.path.basename(msg['image'])}: {msg['msg']}")
            p.wait()
            if p.returncode and not st.get("cancelled"):
                st["errors"].append(f"작업 프로세스가 끝났습니다(코드 {p.returncode}, {os.path.basename(AUTOMASK_LOG)})")
        finally:
            st["running"] = False
            st["finished"] = time.time()

    def cancel(self):
        p = self.proc
        if p and p.poll() is None and self.state.get("running"):
            self.state["cancelled"] = True
            p.kill()
            return True
        return False

    def one(self, image, kind):
        """그림 한 장의 마스크를 만들어 PNG 바이트로(저장하지 않음, 마스크 그리기의 '사람 찾기로 채우기')."""
        out = os.path.join(HERE, ".preview_cache", f"automask_{threading.get_ident()}_{time.time_ns()}.png")
        p = self._spawn()
        try:
            res, _ = p.communicate(json.dumps({"model": self.rt["model"], "kind": kind, "one": {"image": image, "out": out}}, ensure_ascii=False) + "\n", timeout=120)
        except subprocess.TimeoutExpired:
            p.kill()
            raise RuntimeError("시간이 초과되었습니다")
        msg = json.loads(res.strip().splitlines()[-1]) if res.strip() else {"result": "error", "msg": "응답이 없습니다"}
        if msg["result"] != "made":
            try_remove(out)
            raise LookupError(msg["msg"])
        with open(out, "rb") as f:
            data = f.read()
        try_remove(out)
        return data

    def status(self):
        return {k: v for k, v in self.state.items()}


def try_remove(path):
    try:
        os.remove(path)
    except OSError:
        pass


automasker = AutoMasker()


def automask_start(body):
    """마스크 자동 만들기 접수. folders(학습 폴더 전체) 또는 images(고른 그림만). 마스크는 그림 옆 <이름>.msk."""
    cp = body.get("config_path", DEFAULT_CONFIG)
    kind = body.get("kind")
    if kind not in AUTOMASK_KINDS:
        raise ValueError("마스크 종류가 올바르지 않습니다")
    targets = _mask_targets(cp, body.get("folders"), body.get("images"))
    jobs = []
    for img, s in targets:
        old = _image_masks(s, img)
        jobs.append({"image": img, "out": moss_masks.msk_path(img), "existing": old[0] if old else None})
    if not jobs:
        raise ValueError("마스크를 만들 그림이 없습니다(저장된 데이터셋의 학습 폴더만)")
    names = sorted({os.path.basename(os.path.dirname(i)) for i, _s in targets})
    automasker.start(jobs, kind, body.get("overwrite"), ", ".join(names))
    return {"ok": True, "total": len(jobs)}


# ---------------------------------------------------------------- 스튜디오·에포크 비교 (moss_studio.py, 2026-11-10)
STUDIO_SCRIPT = os.path.join(HERE, "moss_studio.py")
STUDIO_LOG = os.path.join(RUN_LOG_DIR, "studio_worker.log")
STUDIO_OUT = os.path.join(HERE, ".studio_out")
STUDIO_FILE_RE = re.compile(r"^[0-9a-f]{24}\.png$")
STUDIO_BLOCKS = 42
STUDIO_MAX_ITEMS = 120
STUDIO_MAX_LORAS = 10  # 한 항목에 덧붙일 LoRA 수 상한(지금은 주 + 기증 2개. 에포크 섞기(최대 8개)는 2026-11-15 지웠다)
STUDIO_NAME_RE = re.compile(r"^[^\\/:*?\"<>|\x00-\x1f]{1,120}$")
STUDIO_META_KEYS = ("ss_output_name", "ss_network_dim", "ss_network_alpha", "ss_network_args", "ss_steps", "ss_epoch",
                    "ss_max_train_steps", "ss_num_train_images", "ss_learning_rate", "ss_optimizer", "ss_slider",
                    "ss_training_started_at", "ss_studio", "ss_block_scale", "ss_lora_ema_decay", "ss_clean_share")


class StudioWorker:
    """스튜디오·에포크 비교의 렌더 작업 프로세스(moss_studio.py, GPU). 처음 쓸 때 띄우고, 학습을 시작하면 끈다.
    렌더 요청은 접수만 하고 바로 응답한다(NET-001). 진행은 /api/studio/events 로 짧게 조회한다.
    새 렌더를 받으면 진행 중인 렌더는 취소한다(슬라이더를 움직일 때마다 새로 그리므로 마지막 요청만 의미가 있다)."""
    MAX_EVENTS = 500

    def __init__(self):
        self.lock = threading.Lock()
        self.proc = None
        self.events = []
        self.seq = 0
        self.job_seq = 0
        self.active = None
        self.status = ""
        self.waiters = {}

    def _push(self, ev):
        if ev.get("ev") != "error":
            ev.pop("trace", None)
        with self.lock:
            self.seq += 1
            ev["seq"] = self.seq
            self.events.append(ev)
            del self.events[:-self.MAX_EVENTS]
            if ev.get("ev") == "status":
                self.status = ev.get("text", "")
            if ev.get("ev") in ("done", "error", "cancelled") and ev.get("id") == self.active:
                self.active = None
            w = self.waiters.get(ev.get("id"))
        if w and ev.get("ev") in ("saved", "error"):
            w["result"] = ev
            w["done"].set()

    def _start(self):
        os.makedirs(RUN_LOG_DIR, exist_ok=True)
        env = {**os.environ, "PYTHONUTF8": "1", "TRITON_CACHE_AUTOTUNING": "1"}
        with open(STUDIO_LOG, "w", encoding="utf-8") as log:
            self.proc = subprocess.Popen([PYTHON, "-u", STUDIO_SCRIPT], cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=log, text=True, encoding="utf-8", creationflags=subprocess.CREATE_NO_WINDOW)
        threading.Thread(target=self._read, args=(self.proc,), daemon=True).start()

    def _read(self, proc):
        for line in proc.stdout:
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            self._push(ev)
        proc.wait()
        with self.lock:
            if self.proc is proc:
                self.proc = None
            active, self.active, self.status = self.active, None, ""
            waiters = list(self.waiters.values())
        if active:
            self._push({"ev": "error", "id": active, "msg": f"렌더 프로세스가 끝났습니다 ({os.path.basename(STUDIO_LOG)} 참고)"})
        self._push({"ev": "stopped"})
        for w in waiters:
            w["done"].set()

    def _send(self, msg):
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                self._start()
            self.proc.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()

    def alive(self):
        with self.lock:
            return self.proc is not None and self.proc.poll() is None

    def render(self, job):
        with self.lock:
            self.job_seq += 1
            jid = f"r{self.job_seq}"
            prev, self.active = self.active, jid
        if prev:
            self._send({"op": "cancel", "target": prev})
        self._send({**job, "op": "render", "id": jid})
        return jid

    def map(self, job):
        """히트맵(LoRA 작용 지도). 렌더 대기열에 들어가지만 '진행 중인 렌더'로 치지 않아 렌더를 취소하지 않는다. 결과는 map 이벤트."""
        with self.lock:
            self.job_seq += 1
            mid = f"m{self.job_seq}"
        self._send({**job, "op": "map", "id": mid})
        return mid

    def cancel(self):
        with self.lock:
            target = self.active
        if target and self.alive():
            self._send({"op": "cancel", "target": target})
        return target

    def stop(self):
        """모델을 내린다(프로세스 종료). 학습을 시작할 때도 부른다."""
        with self.lock:
            proc = self.proc
        if proc is None or proc.poll() is not None:
            return False
        try:
            proc.stdin.write('{"op": "unload"}\n')
            proc.stdin.flush()
            proc.wait(15)
        except Exception:
            pass
        if proc.poll() is None:
            proc.kill()
        return True

    def save(self, job, timeout=300):
        w = {"done": threading.Event(), "result": None}
        with self.lock:
            self.job_seq += 1
            sid = f"s{self.job_seq}"
            self.waiters[sid] = w
        try:
            self._send({**job, "op": "save", "id": sid})
            if not w["done"].wait(timeout):
                raise RuntimeError("저장 시간이 초과되었습니다")
        finally:
            with self.lock:
                self.waiters.pop(sid, None)
        r = w["result"]
        if not r:
            raise RuntimeError(f"렌더 프로세스가 끝났습니다 ({os.path.basename(STUDIO_LOG)} 참고)")
        if r["ev"] == "error":
            raise RuntimeError(r["msg"])
        return r

    def snapshot(self, since):
        with self.lock:
            return {"events": [e for e in self.events if e["seq"] > since], "seq": self.seq,
                    "alive": self.proc is not None and self.proc.poll() is None, "active": self.active, "status": self.status}


def lora_header(path):
    """LoRA 파일 머리만 읽어 블록마다 모듈 수·rank·메타데이터를 준다(torch 없이)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if not 0 < n < 100_000_000:
            raise ValueError("safetensors 파일이 아닙니다")
        h = json.loads(f.read(n))
    meta = h.pop("__metadata__", None) or {}
    blocks, other, ranks, dtypes = [0] * STUDIO_BLOCKS, 0, set(), set()
    for k, v in h.items():
        if not k.endswith(".lora_down.weight"):
            continue
        ranks.add(int(v["shape"][0]))
        dtypes.add(v.get("dtype"))
        m = re.match(r"^lora_unet_layers_(\d+)_", k)
        if m and int(m[1]) < STUDIO_BLOCKS:
            blocks[int(m[1])] += 1
        else:
            other += 1
    st = os.stat(path)
    return {"path": path, "name": os.path.basename(path), "size": st.st_size, "mtime": st.st_mtime, "blocks": blocks, "other": other,
            "modules": sum(blocks) + other, "rank_min": min(ranks) if ranks else 0, "rank_max": max(ranks) if ranks else 0,
            "dtypes": sorted(d for d in dtypes if d), "meta": {k: meta[k][:400] for k in STUDIO_META_KEYS if k in meta}}


def lora_sort_key(name):
    """같은 학습의 파일 순서: -000001(에포크)·-step00000100(스텝) 순, 번호 없는 최종 파일은 맨 뒤."""
    m = re.search(r"-(?:step)?(\d{6,8})$", os.path.splitext(name)[0])
    return (0, int(m[1])) if m else (1, 0)


def studio_config(config_path):
    if not is_config_path(config_path) or not os.path.isfile(config_path):
        config_path = DEFAULT_CONFIG
    return flatten_config(tomllib.loads(read_text(config_path)))


def studio_models(cfg):
    """고를 수 있는 DiT: 샘플 모델(sample_dit, 없으면 학습 모델 옆의 SFT 파일)과 학습 모델."""
    out = []
    train = resolve_path(cfg.get("pretrained_model_name_or_path") or "")
    sample = resolve_path(cfg.get("sample_dit") or "")
    if not sample and train:
        guess = os.path.join(os.path.dirname(train), "moss_image_sft_bf16.safetensors")
        sample = guess if os.path.isfile(guess) else ""
    if sample:
        out.append({"key": "sample", "label": f"샘플 모델 · {os.path.basename(sample)}", "path": sample, "exists": os.path.isfile(sample)})
    if train:
        out.append({"key": "train", "label": f"학습 모델 · {os.path.basename(train)}", "path": train, "exists": os.path.isfile(train)})
    return out


def studio_info(config_path):
    cfg = studio_config(config_path)
    out = resolve_path(cfg.get("output_dir") or "")
    groups = []
    for g in output_groups(out) if out else []:
        models = sorted((m for m in g["models"] if m["name"].lower().endswith(".safetensors")), key=lambda m: lora_sort_key(m["name"]))
        if models:
            groups.append({"name": g["name"], "mtime": g["mtime"],
                           "models": [{"name": m["name"], "path": m["path"], "size": m["size"], "mtime": m["mtime"]} for m in models]})
    return {"models": studio_models(cfg), "output_dir": out, "output_name": cfg.get("output_name") or "", "groups": groups,
            "preset": cfg.get("sample_preset") or "user", "negative": cfg.get("sample_negative") or "",
            "training": runner.running(), "worker": studio.snapshot(10 ** 12)}


def studio_loras(lst):
    out = []
    for sp in lst or []:
        p = resolve_path(str(sp.get("path") or "").strip().strip('"'))
        if not p.lower().endswith(".safetensors") or not os.path.isfile(p):
            raise ValueError(f"LoRA 파일이 없습니다: {sp.get('path')}")
        sc = [float(v) for v in (sp.get("scales") or [])][:STUDIO_BLOCKS]
        sc += [sc[-1] if sc else 1.0] * (STUDIO_BLOCKS - len(sc))
        other = float(sp.get("other", 1.0))
        if any(not (abs(v) <= 20) for v in sc + [other]):  # NaN 도 걸러진다
            raise ValueError("블록 배율은 −20 ~ 20 이어야 합니다")
        out.append({"path": p, "scales": sc, "other": other})
    if len(out) > STUDIO_MAX_LORAS:
        raise ValueError(f"LoRA 는 한 번에 {STUDIO_MAX_LORAS}개까지 걸 수 있습니다")
    return out


def lora_name(name):
    """저장할 LoRA 파일 이름(확장자 .safetensors 붙임). 경로 구분자·윈도우에서 못 쓰는 글자는 거부한다."""
    name = str(name or "").strip()
    if name.lower().endswith(".safetensors"):
        name = name[: -len(".safetensors")]
    if not STUDIO_NAME_RE.match(name) or name.strip(". ") == "":
        raise ValueError("파일 이름에 쓸 수 없는 글자가 있습니다")
    return name + ".safetensors"


def dest_folder(body, cfg):
    """저장할 폴더: dest_dir 이 있으면 그 폴더(ComfyUI LoRA 폴더 등, 이미 있는 폴더만), 없으면 출력 폴더."""
    d = str(body.get("dest_dir") or "").strip().strip('"')
    if d:
        d = resolve_path(d)
        if not os.path.isabs(d) or not os.path.isdir(d):
            raise ValueError(f"보낼 폴더가 없습니다: {body.get('dest_dir')}")
        return d
    out_dir = resolve_path(cfg.get("output_dir") or "")
    if not out_dir or not os.path.isdir(out_dir):
        raise ValueError("출력 폴더가 없습니다")
    return out_dir


def lora_export(body):
    """출력 폴더의 LoRA 파일 하나를 다른 폴더(ComfyUI LoRA 폴더)로 복사한다. 같은 이름이 있으면 overwrite 없이는 409."""
    cfg = studio_config(body.get("config_path", DEFAULT_CONFIG))
    out_dir = resolve_path(cfg.get("output_dir") or "")
    src = resolve_path(str(body.get("src") or "").strip().strip('"'))
    if not (out_dir and src.lower().endswith(".safetensors") and os.path.isfile(src) and is_inside(src, out_dir)):
        raise ValueError("출력 폴더 안의 LoRA 파일만 보낼 수 있습니다")
    if not str(body.get("dest_dir") or "").strip():
        raise ValueError("보낼 폴더를 입력하세요")
    dest = os.path.join(dest_folder(body, cfg), lora_name(body.get("name") or os.path.basename(src)))
    if norm(dest) == norm(src):
        return 400, {"error": "원본과 같은 파일입니다"}
    if os.path.exists(dest) and not body.get("overwrite"):
        return 409, {"error": "같은 이름의 파일이 있습니다", "conflict": dest}
    tmp = dest + ".tmp"
    try:
        shutil.copy2(src, tmp)
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return 200, {"file": dest, "size": os.path.getsize(dest)}


def studio_ref(v):
    """저장할 때의 세기와 같은 기준(주 LoRA 세기). 미리보기를 저장 파일과 같은 순서로 계산하는 데 쓴다(moss_studio.set_loras)."""
    v = float(v if v not in (None, "") else 1.0)
    if not abs(v) <= 20:
        raise ValueError("세기는 −20 ~ 20 이어야 합니다")
    return v if v != 0 else 1.0


def studio_map_job(body):
    """히트맵 요청: 렌더한 그림(.studio_out 의 해시 이름)과 그 그림을 그린 LoRA·프롬프트. 모델 검사는 렌더와 같다."""
    name = os.path.basename(str(body.get("file") or ""))
    path = studio_file(name) if STUDIO_FILE_RE.match(name) else None
    if not path:
        raise ValueError("그림을 찾을 수 없습니다")
    job = studio_render_job({**body, "items": [{"key": "map", "loras": body.get("loras"), "ref": body.get("ref")}]})
    return {"model": job["model"], "prompt": job["prompt"], "file": path, "loras": job["items"][0]["loras"], "ref": job["items"][0]["ref"]}


def studio_render_job(body):
    cfg = studio_config(body.get("config_path", DEFAULT_CONFIG))
    models = {m["key"]: m for m in studio_models(cfg)}
    m = models.get(body.get("model") or "sample") or next(iter(models.values()), None)
    if not m or not m["exists"]:
        raise ValueError("DiT 파일을 찾을 수 없습니다 — 출력 탭의 샘플 모델 또는 학습 모델 경로를 확인하세요")
    model = {"dit": m["path"], "int8": bool(body.get("int8")), "vae": resolve_path(cfg.get("vae") or ""),
             "text_encoder": resolve_path(cfg.get("text_encoder") or ""), "tokenizer_cache_dir": resolve_path(cfg.get("tokenizer_cache_dir") or "")}
    for k, label in (("vae", "VAE"), ("text_encoder", "텍스트 인코더")):
        if not model[k] or not os.path.isfile(model[k]):
            raise ValueError(f"{label} 파일을 찾을 수 없습니다: {model[k]}")
    prompt = str(body.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("프롬프트를 입력하세요")
    w, h = int(body.get("width") or 1024), int(body.get("height") or 1024)
    if not (256 <= w <= 2048 and 256 <= h <= 2048):
        raise ValueError("크기는 가로·세로 256~2048 이어야 합니다")
    preset = body.get("preset") if body.get("preset") in ("user", "fizgig", "official") else "user"
    steps = int(body.get("steps") or 0)
    if not 0 <= steps <= 150:
        raise ValueError("스텝은 1~150 이어야 합니다(0 = 방식 기본값)")
    items = body.get("items") or []
    if not 0 < len(items) <= STUDIO_MAX_ITEMS:
        raise ValueError("그릴 그림이 없습니다")
    job = {"model": model, "prompt": prompt, "negative": str(body.get("negative") or ""), "seed": int(body.get("seed") or 0) & 0x7FFFFFFF,
           "width": w, "height": h, "preset": preset, "steps": steps,
           "items": [{"key": str(it.get("key", i))[:1000], "loras": studio_loras(it.get("loras")), "ref": studio_ref(it.get("ref"))}
                     for i, it in enumerate(items)]}
    if body.get("cfg") not in (None, ""):
        cfgv = float(body["cfg"])
        if not 0 < cfgv <= 20:
            raise ValueError("CFG 는 0~20 이어야 합니다")
        job["cfg"] = cfgv
    return job


# 에포크 비교 기록(2026-11-12 사용자 요청): '모두 그리기'가 끝날 때마다 학습(묶음 이름)별로 저절로 남긴다.
# .studio_out/history/<묶음>/history.json + 그 기록이 쓰는 그림의 하드 링크(렌더 프로세스가 .studio_out 바로 아래를 600장으로
# 줄여도 기록의 그림은 남는다). 같은 조건(그리기 설정·세기)의 기록은 하나로 합친다(그림을 더 그리면 그 기록에 더해짐).
STUDIO_HIST = os.path.join(STUDIO_OUT, "history")
STUDIO_HIST_MAX = 60
STUDIO_REF_RE = re.compile(r"^ref-[0-9a-f]{24}\.(?:png|jpe?g|webp|bmp)$")
_hist_lock = threading.Lock()


def hist_dir(group):
    if not isinstance(group, str) or not group.strip() or len(group) > 200:
        raise ValueError("학습 이름이 없습니다")
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", group).strip(" .") or "_"
    return os.path.join(STUDIO_HIST, safe)


def hist_read(group):
    try:
        with open(os.path.join(hist_dir(group), "history.json"), encoding="utf-8") as f:
            data = json.load(f)
        return data.get("entries", []) if data.get("group") == group else []
    except (OSError, ValueError):
        return []


def _hist_write(group, entries):
    d = hist_dir(group)
    os.makedirs(d, exist_ok=True)
    write_text_atomic(os.path.join(d, "history.json"), json.dumps({"group": group, "entries": entries}, ensure_ascii=False))
    used = {f for e in entries for f in e.get("imgs", {}).values()}
    used.update(e["reference"]["file"] for e in entries if e.get("reference"))
    for e in os.scandir(d):  # 어느 기록도 쓰지 않는 그림은 지운다
        if e.is_file() and (STUDIO_FILE_RE.fullmatch(e.name) or STUDIO_REF_RE.fullmatch(e.name)) and e.name not in used:
            try:
                os.remove(e.path)
            except OSError:
                pass


def _hist_reference(directory, reference):
    """비교 기록에 원본의 사본을 보관하고, 다시 그릴 때는 보관된 그림도 받는다."""
    if not reference:
        return None
    if isinstance(reference, str):
        source = os.path.abspath(reference)
        name = os.path.basename(source)
        ext = os.path.splitext(source)[1].lower()
        if ext not in IMAGE_EXTENSIONS or not os.path.isfile(source):
            raise ValueError("비교 그림 파일을 찾을 수 없습니다")
        with open(source, "rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()[:24]
        file = f"ref-{digest}{ext}"
    elif isinstance(reference, dict):
        file = str(reference.get("file") or "")
        source = studio_file(file) if STUDIO_REF_RE.fullmatch(file) else None
        if not source:
            raise ValueError("기록에 보관된 비교 그림 파일을 찾을 수 없습니다")
        name = os.path.basename(str(reference.get("name") or file))
    else:
        raise ValueError("비교 그림을 읽을 수 없습니다")

    dest = os.path.join(directory, file)
    if not os.path.isfile(dest):
        tmp = dest + ".tmp"
        try:
            shutil.copyfile(source, tmp)
            os.replace(tmp, dest)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    return {"file": file, "name": name}


def _hist_signature(sg, strength, reference):
    settings = [sg, strength]
    if reference:
        settings.append(reference["file"])
    return json.dumps(settings, sort_keys=True, ensure_ascii=False)


def hist_add(body):
    group, entry = body.get("group"), body.get("entry") or {}
    d = hist_dir(group)
    # 렌더 프로세스는 그림을 전체 경로로 알린다(moss_studio emit file=dest) → 파일 이름만 받는다(2026-11-12: 이것 때문에 기록이 하나도 안 남았다)
    imgs = {str(k)[:1000]: os.path.basename(v) for k, v in (entry.get("imgs") or {}).items()
            if isinstance(v, str) and STUDIO_FILE_RE.match(os.path.basename(v))}
    if not imgs:
        raise ValueError("그린 그림이 없습니다")
    if len(imgs) > STUDIO_MAX_ITEMS:
        raise ValueError("항목이 너무 많습니다")
    sg = {k: entry.get("sg", {}).get(k) for k in ("model", "preset", "steps", "cfg", "prompt", "negative", "seed", "w", "h", "inc", "exc")}
    strength = str(entry.get("str", ""))[:20]
    warning = ""
    with _hist_lock:
        os.makedirs(d, exist_ok=True)
        for f in set(imgs.values()):
            src, dst = os.path.join(STUDIO_OUT, f), os.path.join(d, f)
            if os.path.isfile(dst) or not os.path.isfile(src):
                continue
            try:
                os.link(src, dst)  # 같은 드라이브: 하드 링크(공간을 더 쓰지 않음)
            except OSError:
                shutil.copy2(src, dst)
        imgs = {k: v for k, v in imgs.items() if os.path.isfile(os.path.join(d, v))}
        try:
            reference = _hist_reference(d, entry.get("reference"))
        except (OSError, ValueError):
            reference, warning = None, "비교 그림 파일을 읽을 수 없어 그림 없이 기록을 저장했습니다"
        sig = _hist_signature(sg, strength, reference)
        entries = hist_read(group)
        old = next((e for e in entries if e.get("sig") == sig), None)
        rate = {}
        if old:
            entries.remove(old)
            imgs = {**old.get("imgs", {}), **imgs}
            rate = old.get("rate", {})  # 합쳐도 평가는 남긴다
        new = {"id": old["id"] if old else uuid.uuid4().hex, "time": time.time(), "sig": sig, "sg": sg, "str": strength,
               "imgs": imgs, "rate": rate, "reference": reference}
        entries = [new] + entries[: STUDIO_HIST_MAX - 1]
        _hist_write(group, entries)
    return {"id": new["id"], "entries": entries, "warning": warning}


def hist_ref_set(body):
    group, hid = body.get("group"), str(body.get("id", ""))
    directory = hist_dir(group)
    with _hist_lock:
        entries = hist_read(group)
        entry = next((e for e in entries if e.get("id") == hid), None)
        if entry is None:
            raise ValueError("기록이 없습니다")
        reference = _hist_reference(directory, body.get("reference"))
        if "reference" not in entry or entry["reference"] != reference:
            entry["reference"] = reference
            entry["sig"] = _hist_signature(entry["sg"], entry["str"], reference)
            _hist_write(group, entries)
    return {"entries": entries}


def hist_rate(body):
    """기록 하나의 항목 평가: 1 = 합격(별), -1 = 불합격, 0 = 지움. 한 기록(모두 그리기)의 한 에포크는 평가 하나."""
    group, hid, key = body.get("group"), str(body.get("id", "")), str(body.get("key", ""))[:1000]
    value = int(body.get("value", 0))
    if value not in (-1, 0, 1):
        raise ValueError("평가 값은 1, -1, 0 입니다")
    hist_dir(group)
    with _hist_lock:
        entries = hist_read(group)
        e = next((x for x in entries if x.get("id") == hid), None)
        if not e:
            raise ValueError("기록이 없습니다")
        if key not in e.get("imgs", {}):
            raise ValueError("이 기록에 그 그림이 없습니다")
        rate = e.setdefault("rate", {})
        if value:
            rate[key] = value
        else:
            rate.pop(key, None)
        _hist_write(group, entries)
    return {"entries": entries}


def hist_delete(body):
    group, hid = body.get("group"), str(body.get("id", ""))
    hist_dir(group)
    with _hist_lock:
        entries = [e for e in hist_read(group) if e.get("id") != hid]
        _hist_write(group, entries)
    return {"entries": entries}


def studio_file(name):
    """렌더 결과 그림 경로. .studio_out 에서 지워졌으면 비교 기록 폴더의 사본."""
    p = os.path.join(STUDIO_OUT, name)
    if os.path.isfile(p):
        return p
    try:
        for d in os.scandir(STUDIO_HIST):
            if d.is_dir() and os.path.isfile(os.path.join(d.path, name)):
                return os.path.join(d.path, name)
    except OSError:
        pass
    return None


def studio_save(body):
    cfg = studio_config(body.get("config_path", DEFAULT_CONFIG))
    fname = lora_name(body.get("name"))
    name = fname[: -len(".safetensors")]
    loras = studio_loras(body.get("loras"))
    if not loras:
        raise ValueError("저장할 LoRA 가 없습니다")
    dest = os.path.join(dest_folder(body, cfg), fname)
    if any(norm(sp["path"]) == norm(dest) for sp in loras):
        return 400, {"error": "원본 LoRA 에 덮어쓸 수 없습니다 — 다른 이름을 쓰세요"}
    if os.path.exists(dest) and not body.get("overwrite"):
        return 409, {"error": "같은 이름의 파일이 있습니다", "conflict": dest}
    strength = studio_ref(body.get("strength"))
    info = {"strength": strength, "loras": [{"name": os.path.basename(sp["path"]), "scales": [round(v, 4) for v in sp["scales"]]} for sp in loras]}
    meta = {"ss_studio": json.dumps(info, ensure_ascii=False), "ss_output_name": name}
    r = studio.save({"loras": loras, "strength": strength, "out": dest, "meta": meta})
    return 200, {"file": r["file"], "modules": r["modules"], "rank_max": r["rank_max"], "rank_min": r["rank_min"], "size": r["size"]}


studio = StudioWorker()


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
                return self._send(200, {**runner.status(int(q.get("since", ["0"])[0])), "system": sysmon.read(),
                                        "queue": queue.snapshot() if queue else None})
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
            if url.path == "/api/train_runs":
                return self._send(200, {"runs": list_train_runs(), "running": runner.running()})
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
                                            "sets": find_training_sets(root, ext, q.get("force", ["0"])[0] == "1", q.get("kind", ["training"])[0])})
                path = q.get("path", [root])[0] or root
                if not os.path.isdir(path):
                    return self._send(404, {"error": f"폴더가 없습니다: {path}"})
                _image_roots.add(os.path.abspath(path))
                detail = None
                if q.get("detail", ["0"])[0] == "1":
                    detail = {"multiline": q.get("multiline", ["0"])[0] == "1", "low_area": reso_area(q.get("reso", ["1024"])[0])}
                return self._send(200, browse_dir(path, ext, detail))
            if url.path == "/api/slider_pairs":
                return self._send(200, slider_pair_list(q.get("config", [DEFAULT_CONFIG])[0], q.get("excluded", ["0"])[0] == "1"))
            if url.path in ("/api/slider_pair_image", "/api/slider_pair_info"):
                side = "info" if url.path.endswith("_info") else q.get("side", [""])[0]
                if side not in ("pos", "neg", "mask", "info", "rawpos", "rawneg", "automask", "filemask"):
                    return self._send(400, {"error": "bad side"})
                try:
                    data = slider_pair_image(q.get("config", [DEFAULT_CONFIG])[0], int(q.get("index", ["0"])[0]), side)
                except (FileNotFoundError, ValueError) as e:
                    return self._send(404, {"error": str(e)})
                if side == "info":
                    return self._send(200, data)
                return self._send(200, data, "image/png", {"Cache-Control": "no-store"})
            if url.path == "/api/dataset_images":
                return self._send(200, dataset_images(q.get("config", [DEFAULT_CONFIG])[0]))
            if url.path == "/api/automask/info":
                return self._send(200, {**automasker.runtime(q.get("refresh", ["0"])[0] == "1"), "status": automasker.status(),
                                        "kinds": AUTOMASK_KINDS})
            if url.path == "/api/automask/status":
                return self._send(200, automasker.status())
            if url.path == "/api/thumb":
                p = q.get("path", [""])[0]
                if not p or not os.path.isfile(p) or not allowed_image(p):
                    return self._send(404, {"error": "not found"})
                try:
                    size = int(q.get("s", ["220"])[0] or 220)
                except ValueError:
                    size = 220
                try:
                    data = image_thumb(p, max(64, min(512, size)))  # s: 긴 변(스튜디오 그림 칸 '큼'은 480)
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
                if kind not in ("processed", "vae", "mask", "up2x"):
                    return self._send(400, {"error": "bad kind"})
                extra = {}
                if kind == "mask":  # 그 그림의 마스크를 학습처럼 크기 처리한 것(같은 이름의 마스크 파일만, alpha=1 이면 투명도를 곱함)
                    m = q.get("mask", [""])[0]
                    alpha = q.get("alpha", [""])[0] == "1" and moss_masks.is_transparent(p)
                    if m and (not os.path.isfile(m) or not allowed_image(m) or
                              os.path.splitext(os.path.basename(m))[0].lower() != os.path.splitext(os.path.basename(p))[0].lower()):
                        return self._send(404, {"error": "마스크 파일이 없습니다"})
                    if not m and not alpha:
                        return self._send(404, {"error": "마스크 파일이 없습니다"})
                    extra["mask"], extra["alpha"] = m, alpha
                with open(preview.call("image", path=p, kind=kind, opts=preview_options(q, p), **extra)["file"], "rb") as f:
                    data = f.read()
                return self._send(200, data, "image/png")
            if url.path == "/api/studio/info":
                return self._send(200, studio_info(q.get("config", [DEFAULT_CONFIG])[0]))
            if url.path == "/api/refs":
                return self._send(200, refs_view(q.get("output", [""])[0]))
            if url.path in ("/api/tags/complete", "/api/tags/help"):  # 프롬프트 태그 편집기(moss_tags, HWtagger 자료)
                try:
                    if url.path.endswith("complete"):
                        return self._send(200, {"items": moss_tags.complete(q.get("q", [""])[0][:200], min(30, int(q.get("n", ["15"])[0] or 15)))})
                    return self._send(200, moss_tags.help_of(q.get("tag", [""])[0][:300]))
                except FileNotFoundError:
                    return self._send(404, {"error": "태그 자료(tagdata/)가 없습니다"})
            if url.path == "/api/studio/history":
                try:
                    return self._send(200, {"entries": hist_read(q.get("group", [""])[0])})
                except ValueError as e:
                    return self._send(400, {"error": str(e)})
            if url.path == "/api/studio/events":
                return self._send(200, {**studio.snapshot(int(q.get("since", ["0"])[0] or 0)), "training": runner.running()})
            if url.path == "/api/studio/lora":
                p = resolve_path(q.get("path", [""])[0].strip().strip('"'))
                if not p.lower().endswith(".safetensors") or not os.path.isfile(p):
                    return self._send(404, {"error": "LoRA 파일이 없습니다"})
                return self._send(200, lora_header(p))
            if url.path == "/api/studio/image":
                name = q.get("file", [""])[0]
                p = studio_file(name) if STUDIO_FILE_RE.fullmatch(name) or STUDIO_REF_RE.fullmatch(name) else None
                if not p:
                    return self._send(404, {"error": "not found"})
                with open(p, "rb") as f:
                    data = f.read()
                return self._send(200, data, mimetypes.guess_type(p)[0] or "application/octet-stream",
                                  {"Cache-Control": "private, max-age=86400"})
            if url.path == "/api/image":
                p = q.get("path", [""])[0]
                if not p or not os.path.isfile(p) or not allowed_image(p):
                    return self._send(404, {"error": "not found"})
                with open(p, "rb") as f:
                    data = f.read()
                mime = "image/png" if p.lower().endswith(moss_masks.MASK_EXT) else mimetypes.guess_type(p)[0]  # .msk 는 내용이 PNG
                return self._send(200, data, mime or "application/octet-stream")
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
                return self._send(200, save_all(body))
            if url.path == "/api/check":  # 화면의 저장 안 된 값까지 넣어 점검한다(파일은 쓰지 않음)
                cp = body.get("config_path", DEFAULT_CONFIG)
                if not is_config_path(cp):
                    return self._send(400, {"error": "도구 폴더의 설정만 점검할 수 있습니다"})
                note = None
                try:
                    pend, _dataset_changed = pending_texts(body)
                except Exception as e:  # 입력 중인 값의 형식 오류 등: 저장된 값으로 점검하고 알린다
                    pend, note = [], f"저장 안 된 값을 읽을 수 없어 저장된 설정으로 점검했습니다: {e}"
                res = analyze(cp, {p: new for p, _old, new in pend})
                if note:
                    res["issues"].insert(0, {"level": "error", "msg": note, "tab": "run", "key": None})
                res["unsaved"] = len(pend)
                if res.get("dataset"):
                    for s in res["dataset"]["subsets"]:
                        if s.get("ok"):
                            _image_roots.add(resolve_path(s["image_dir"]))
                return self._send(200, res)
            if url.path == "/api/save_profile":
                return self._send(200, save_profile(body))
            if url.path == "/api/new_slider_profile":
                return self._send(200, {"config_path": new_slider_profile(body)})
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
                # 이어하기는 같은 이름에 이어 쓰는 것이 정상이라 묻지 않는다
                conflict = None if resume or body.get("overwrite") else existing_outputs(cp)
                if conflict:
                    return self._send(409, {"error": "같은 이름의 학습 결과가 있습니다", "conflict": conflict})
                studio.stop()  # 스튜디오 모델이 올라가 있으면 VRAM 을 비운다
                runner.start(cp, body.get("extra_args", ""), resume)
                return self._send(200, {"ok": True, "settings_dir": runner.settings_dir})
            if url.path in ("/api/slider_pair_override", "/api/slider_pair_mask"):
                cp = body.get("config_path", DEFAULT_CONFIG)
                if not is_config_path(cp):
                    return self._send(400, {"error": "도구 폴더의 설정만 다룰 수 있습니다"})
                names = body.get("names") or []
                if not isinstance(names, list) or len(names) != 2:
                    return self._send(400, {"error": "두 그림 이름이 필요합니다"})
                try:
                    if url.path.endswith("_override"):
                        res = {"override": slider_pair_override(cp, body.get("index", 0), names, body.get("override"))}
                    else:
                        res = {"mask": slider_pair_mask_save(cp, body.get("index", 0), names, body.get("png"))}
                except (LookupError, FileNotFoundError) as e:
                    return self._send(409, {"error": str(e)})
                except ValueError as e:
                    return self._send(400, {"error": str(e)})
                except OSError as e:
                    return self._send(500, {"error": f"저장하지 못했습니다: {e}"})
                return self._send(200, {"ok": True, **res})
            if url.path == "/api/dataset_mask":
                cp = body.get("config_path", DEFAULT_CONFIG)
                if not is_config_path(cp):
                    return self._send(400, {"error": "도구 폴더의 설정만 다룰 수 있습니다"})
                try:
                    return self._send(200, {"ok": True, **dataset_mask_save(cp, body.get("image", ""), body.get("png"))})
                except (LookupError, FileNotFoundError) as e:
                    return self._send(409, {"error": str(e)})
                except ValueError as e:
                    return self._send(400, {"error": str(e)})
                except OSError as e:
                    return self._send(500, {"error": f"저장하지 못했습니다: {e}"})
            if url.path == "/api/upscale2x":
                try:
                    return self._send(200, upscale2x(body.get("config_path", DEFAULT_CONFIG), body.get("images") or []))
                except RuntimeError as e:
                    return self._send(409, {"error": str(e)})
                except ValueError as e:
                    return self._send(400, {"error": str(e)})
            if url.path == "/api/dataset_mask_clear":
                try:
                    return self._send(200, dataset_mask_clear(body.get("config_path", DEFAULT_CONFIG), body.get("folders"),
                                                              body.get("images"), bool(body.get("confirm"))))
                except ValueError as e:
                    return self._send(400, {"error": str(e)})
            if url.path in ("/api/automask/start", "/api/automask/cancel", "/api/automask/one"):
                try:
                    if url.path.endswith("/start"):
                        return self._send(200, automask_start(body))
                    if url.path.endswith("/cancel"):
                        return self._send(200, {"cancelled": automasker.cancel()})
                    cp = body.get("config_path", DEFAULT_CONFIG)
                    if not is_config_path(cp) or body.get("kind") not in AUTOMASK_KINDS:
                        return self._send(400, {"error": "설정 또는 마스크 종류가 올바르지 않습니다"})
                    _image_subset(cp, body.get("image", ""))  # 학습 폴더의 그림만
                    import base64
                    data = automasker.one(body["image"], body["kind"])
                    return self._send(200, {"png": "data:image/png;base64," + base64.b64encode(data).decode()})
                except LookupError as e:
                    return self._send(409, {"error": str(e)})
                except RuntimeError as e:
                    return self._send(409, {"error": str(e)})
                except (ValueError, OSError) as e:
                    return self._send(400, {"error": str(e)})
            if url.path == "/api/slider_pair_move":
                cp = body.get("config_path", DEFAULT_CONFIG)
                if not is_config_path(cp):
                    return self._send(400, {"error": "도구 폴더의 설정만 다룰 수 있습니다"})
                names = body.get("names") or []
                if not isinstance(names, list) or len(names) != 2:
                    return self._send(400, {"error": "두 그림 이름이 필요합니다"})
                try:
                    moved = slider_pair_move(cp, body.get("index", 0), names, body.get("restore"))
                except (LookupError, FileNotFoundError) as e:
                    return self._send(409, {"error": str(e)})
                except FileExistsError as e:
                    return self._send(409, {"error": str(e)})
                except OSError as e:
                    return self._send(500, {"error": f"옮기지 못했습니다: {e}"})
                return self._send(200, {"ok": True, "moved": moved})
            if url.path in ("/api/train_runs/notes", "/api/train_runs/rating", "/api/train_runs/settings", "/api/train_runs/delete"):
                rid = str(body.get("id", ""))
                try:
                    if url.path == "/api/train_runs/notes":
                        run_notes_save(rid, body.get("text", ""))
                        return self._send(200, {"ok": True})
                    if url.path == "/api/train_runs/rating":
                        return self._send(200, run_rating_save(rid, body.get("rating")))
                    if url.path == "/api/train_runs/settings":
                        return self._send(200, run_settings_compare(rid, bool(body.get("restore"))))
                    return self._send(200, run_delete(rid, bool(body.get("confirm"))))
                except FileNotFoundError as e:
                    return self._send(404, {"error": str(e)})
                except RuntimeError as e:
                    return self._send(409, {"error": str(e)})
                except (OSError, ValueError) as e:
                    return self._send(400, {"error": str(e)})
            if url.path == "/api/delete_state":
                cp, path = body.get("config_path", DEFAULT_CONFIG), body.get("path", "")
                if not is_config_path(cp):
                    return self._send(400, {"error": "도구 폴더의 설정만 다룰 수 있습니다"})
                if not any(norm(r["path"]) == norm(path) for r in resume_states(cp)):  # 목록에 보인 폴더만 지운다
                    return self._send(404, {"error": "상태 폴더를 찾을 수 없습니다"})
                if runner.running() and runner.resume_path and norm(runner.resume_path) == norm(path):
                    return self._send(409, {"error": "지금 이 지점에서 이어 학습하는 중입니다"})
                delete_state_dir(path)
                return self._send(200, {"ok": True})
            if url.path == "/api/studio/render":
                if runner.running():
                    return self._send(409, {"error": "학습 중에는 쓸 수 없습니다 — GPU 를 함께 쓰면 학습이 느려지고 메모리가 모자랄 수 있습니다"})
                try:
                    job = studio_render_job(body)
                except (ValueError, TypeError) as e:
                    return self._send(400, {"error": str(e)})
                return self._send(200, {"id": studio.render(job)})
            if url.path == "/api/tags/meta":
                tags = [str(t)[:300] for t in (body.get("tags") or []) if isinstance(t, str)]
                try:
                    return self._send(200, {"meta": moss_tags.metas(tags)})
                except FileNotFoundError:
                    return self._send(404, {"error": "태그 자료(tagdata/)가 없습니다"})
            if url.path in ("/api/studio/history/add", "/api/studio/history/delete", "/api/studio/history/rate", "/api/studio/history/reference"):
                try:
                    fn = {"add": hist_add, "delete": hist_delete, "rate": hist_rate, "reference": hist_ref_set}[url.path.rsplit("/", 1)[1]]
                    return self._send(200, fn(body))
                except (ValueError, TypeError, AttributeError) as e:
                    return self._send(400, {"error": str(e)})
            if url.path == "/api/studio/map":
                if runner.running():
                    return self._send(409, {"error": "학습 중에는 쓸 수 없습니다 — 히트맵도 GPU 로 계산합니다"})
                try:
                    job = studio_map_job(body)
                except (ValueError, TypeError) as e:
                    return self._send(400, {"error": str(e)})
                return self._send(200, {"id": studio.map(job)})
            if url.path == "/api/refs":
                try:
                    return self._send(200, refs_set(body))
                except (ValueError, TypeError) as e:
                    return self._send(400, {"error": str(e)})
            if url.path == "/api/refs/pick":
                return self._send(200, {"path": browse_dialog("image", body.get("initial", ""))})
            if url.path == "/api/studio/cancel":
                return self._send(200, {"cancelled": studio.cancel()})
            if url.path == "/api/studio/unload":
                return self._send(200, {"stopped": studio.stop()})
            if url.path == "/api/studio/save":
                try:
                    code, res = studio_save(body)
                except (ValueError, TypeError) as e:
                    return self._send(400, {"error": str(e)})
                except RuntimeError as e:
                    return self._send(500, {"error": str(e)})
                return self._send(code, res)
            if url.path == "/api/lora_export":
                try:
                    code, res = lora_export(body)
                except (ValueError, TypeError, OSError) as e:
                    return self._send(400, {"error": str(e)})
                return self._send(code, res)
            if url.path == "/api/stop":
                return self._send(200, {"stopped": runner.stop(bool(body.get("force")))})
            if url.path == "/api/queue":
                # 연속 학습(2026-11-15): set = 목록 바꾸기, start = 시작(점검 → 같은 이름이면 overwrite 로 다시), pause = 지금 학습 뒤 멈춤
                act = body.get("action")
                if act == "set":
                    return self._send(200, queue.set_items(body.get("items")))
                if act == "start":
                    res = queue.start(bool(body.get("overwrite")))
                    return self._send(409 if "problems" in res or "conflicts" in res else 200, {**res, **queue.snapshot()})
                if act == "pause":
                    return self._send(200, queue.pause())
                return self._send(400, {"error": "알 수 없는 동작"})
            if url.path == "/api/shutdown_after":
                # 학습이 끝나면 PC 끄기 스위치(파일 하나, 학습 파일이 완료 때 보고 지운다)
                if body.get("on"):
                    write_text_atomic(moss_notify.SHUTDOWN_FLAG, json.dumps({"set": time.time()}))
                elif os.path.exists(moss_notify.SHUTDOWN_FLAG):
                    os.remove(moss_notify.SHUTDOWN_FLAG)
                return self._send(200, {"on": moss_notify.shutdown_requested()})
            if url.path == "/api/shutdown_cancel":
                return self._send(200, {"ok": runner.cancel_shutdown()})
            if url.path == "/api/ack_error":
                return self._send(200, {"ok": runner.ack_error()})
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
    global runner, queue, sysmon, PORT
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
    queue = TrainQueue(runner)
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
