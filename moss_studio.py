# 스튜디오·에포크 비교 렌더 작업 프로세스 (GPU, 2026-11-10, Fizgig 의 Studio·LoRA Royale 참고).
# moss_ui.py 가 처음 쓸 때 venv 파이썬으로 띄운다. 표준 입력 한 줄 = 요청 JSON, 표준 출력 한 줄 = 이벤트 JSON.
# - 모델(DiT·VAE)은 한 번만 올리고, LoRA 는 합치지 않고 Linear 마다 덧붙여 계산한다(블록마다 배율을 바로 바꿈).
#   여러 LoRA(주 LoRA + 기증 LoRA)를 함께 걸 수 있다. 샘플링은 moss_sampling(ComfyUI 와 같은 식).
# - 텍스트 인코더는 CPU 에 올려 새 문장만 인코딩하고(.studio_out/te_cache.pt 에 보관) 잠시 안 쓰면 내린다.
# - 요청: render {id, model, items:[{key?, loras:[{path, scales[42], other}]}], prompt, negative, seed, width, height,
#   preset, steps} → 그림마다 image 이벤트, 끝에 done. cancel {target}. save {id, loras, strength, out, meta}(CPU, 바로). unload.
#   map {id, model, prompt, file, loras, ref} → map 이벤트(히트맵, 2026-11-20): 그린 그림을 VAE 로 다시 인코딩하고 세 노이즈 단계에서
#   LoRA 를 켠 예측과 끈 예측의 차이(Fizgig 의 'Where the LoRA acts'와 같은 방법). 같은 조건은 .studio_out/maps/ 에 기억.
# - 같은 조건의 그림은 다시 그리지 않는다(조건 해시가 파일 이름).
import hashlib
import json
import os
import queue
import re
import sys
import threading
import time
import traceback

reply_stream = os.fdopen(os.dup(1), "w", encoding="utf-8")
os.dup2(2, 1)  # 라이브러리의 print 가 이벤트 줄을 깨뜨리지 않게 표준 출력을 로그로 돌린다

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "sd-scripts")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

OUT_DIR = os.path.join(HERE, ".studio_out")
TE_CACHE = os.path.join(OUT_DIR, "te_cache.pt")
MAX_OUT_FILES = 600
IDLE_EXIT_SECONDS = 900   # 안 쓰면 끝내서 VRAM 을 돌려준다. 다음 요청 때 다시 뜬다
TE_IDLE_SECONDS = 300     # 텍스트 인코더(CPU, 약 16GB)는 더 일찍 내린다
LORA_CACHE = 8            # GPU 에 올려 둘 LoRA 파일 수
BLOCKS = 42
LAYER_RE = re.compile(r"^layers\.(\d+)\.")
MAP_DIR = os.path.join(OUT_DIR, "maps")
MAP_LEVELS = (0.25, 0.5, 0.75)   # 노이즈 단계: 세부(낮음) ~ 구도(높음). Fizgig 학습 지도·스튜디오 지도와 같은 값
MAP_SEED = 1234
MAP_MAX_SIDE = 64                 # 격자의 긴 변(1024 그림이면 잠재 격자 64 그대로)

write_lock = threading.Lock()
jobs = queue.Queue()
cancel_ids = set()
cancel_lock = threading.Lock()
current = {"id": None}
last_used = time.time()

S = {"dit": None, "dit_key": None, "vae": None, "vae_path": None, "te": None, "tok": None, "te_key": None,
     "te_used": 0.0, "conds": None, "hooks": None, "loras": {}, "args": None}


class Cancelled(Exception):
    pass


def emit(**d):
    with write_lock:
        reply_stream.write(json.dumps(d, ensure_ascii=False) + "\n")
        reply_stream.flush()


def cancelled(job_id):
    with cancel_lock:
        return job_id in cancel_ids or "*" in cancel_ids


# ---------------------------------------------------------------- 모델
def _args(model):
    import moss_image_minimal_inference as mi
    attn = "flash"
    try:
        import flash_attn  # noqa: F401
    except Exception:
        attn = "torch"
    argv = sys.argv
    sys.argv = ["x", "--dit", model["dit"], "--vae", model["vae"], "--text_encoder", model["text_encoder"],
                "--tokenizer_cache_dir", model["tokenizer_cache_dir"], "--save_path", OUT_DIR, "--prompt", "x",
                "--attn_mode", attn] + (["--convrot_int8"] if model.get("int8") else [])
    try:
        return mi.parse_args()
    finally:
        sys.argv = argv


def ensure_dit(model):
    import moss_image_minimal_inference as mi
    key = (os.path.abspath(model["dit"]), bool(model.get("int8")))
    if S["dit_key"] == key:
        return S["dit"]
    if S["dit"] is not None:
        unhook()
        S["dit"] = None
        S["loras"].clear()
        torch.cuda.empty_cache()
    emit(ev="status", text=f"모델을 올리는 중: {os.path.basename(model['dit'])}{' (INT8)' if model.get('int8') else ''}")
    t = time.time()
    S["args"] = _args(model)
    S["dit"] = mi.load_dit_model(S["args"], torch.device("cuda"))
    S["dit_key"] = key
    hook_all(S["dit"])
    emit(ev="status", text=f"모델을 올렸습니다 ({time.time() - t:.0f}초)")
    return S["dit"]


def ensure_vae(model):
    import moss_image_minimal_inference as mi
    if S["vae_path"] != model["vae"]:
        S["vae"] = mi.load_vae(_args(model), torch.device("cuda"))
        S["vae_path"] = model["vae"]
    return S["vae"]


def te_conds():
    if S["conds"] is None:
        try:
            S["conds"] = torch.load(TE_CACHE) if os.path.isfile(TE_CACHE) else {}
        except Exception:
            S["conds"] = {}
    return S["conds"]


def encode(model, texts):
    """[L, C] bf16(CPU) 를 돌려준다. 새 문장만 텍스트 인코더로 인코딩한다."""
    import moss_image_minimal_inference as mi
    from library import moss_image_utils
    conds = te_conds()
    key = lambda t: f"{os.path.abspath(model['text_encoder'])}\n{t}"
    need = [t for t in dict.fromkeys(texts) if key(t) not in conds]
    if need:
        if S["te"] is None or S["te_key"] != model["text_encoder"]:
            emit(ev="status", text="텍스트 인코더를 올리는 중(CPU, 처음 한 번 약 30초)")
            args = _args(model)
            S["tok"] = moss_image_utils.load_tokenizer(args.tokenizer_cache_dir)
            S["te"] = mi.load_text_encoder(args, torch.device("cpu"))
            S["te_key"] = model["text_encoder"]
        emit(ev="status", text=f"프롬프트 {len(need)}개 인코딩 중")
        embeds = mi.encode_prompts(S["tok"], S["te"], need, S["args"].max_token_length if S["args"] else 512, torch.device("cpu"))
        for t, e in zip(need, embeds):
            conds[key(t)] = e
        if len(conds) > 400:  # 오래된 것부터 버린다(dict 는 넣은 순서)
            for k in list(conds)[: len(conds) - 400]:
                del conds[k]
        tmp = TE_CACHE + ".tmp"
        torch.save(conds, tmp)
        os.replace(tmp, TE_CACHE)
    S["te_used"] = time.time()
    return [conds[key(t)] for t in texts]


# ---------------------------------------------------------------- LoRA (합치지 않고 덧붙임)
class Hook:
    __slots__ = ("org", "adapters", "own")

    def __init__(self, org):
        self.org = org
        self.adapters = []  # (down, up, scale)

    def __call__(self, x, *a, **k):
        y = self.org(x, *a, **k)
        for d, u, s in self.adapters:
            y = y + F.linear(F.linear(x.to(d.dtype), d), u).to(y.dtype) * s
        return y


def hook_all(dit):
    hooks = {}
    for name, m in dit.named_modules():
        if isinstance(m, torch.nn.Linear):  # INT8(ConvRot)도 nn.Linear 의 forward 를 바꿔 끼운 것이라 같다
            h = Hook(m.forward)
            h.own = "forward" in m.__dict__  # INT8 처럼 인스턴스에 forward 를 이미 바꿔 끼운 경우 되돌릴 때 그것을 살린다
            m.forward = h
            hooks["lora_unet_" + name.replace(".", "_")] = (h, name)
    S["hooks"] = hooks


def unhook():
    if S["dit"] is None:
        return
    for m in S["dit"].modules():
        h = m.__dict__.get("forward")
        if isinstance(h, Hook):
            if h.own:
                m.forward = h.org
            else:
                del m.forward
    S["hooks"] = None


def block_of(module_name):
    m = LAYER_RE.match(module_name)
    return int(m.group(1)) if m else None


def load_lora(path):
    """{lora 이름: (down, up, alpha, rank, 블록)} 를 fp32 로 GPU 에 올려 기억한다. 파일이 바뀌면 다시 읽는다."""
    from safetensors.torch import load_file
    st = os.stat(path)
    sig = (st.st_mtime_ns, st.st_size)
    hit = S["loras"].get(path)
    if hit and hit["sig"] == sig:
        S["loras"][path] = S["loras"].pop(path)  # 최근에 쓴 것으로
        return hit
    sd = load_file(path)
    mods, unknown = {}, 0
    for k in sd:
        if not k.endswith(".lora_down.weight"):
            continue
        name = k[: -len(".lora_down.weight")]
        if name not in S["hooks"] or name + ".lora_up.weight" not in sd:
            unknown += 1
            continue
        d, u = sd[k], sd[name + ".lora_up.weight"]
        r = d.shape[0]
        alpha = float(sd[name + ".alpha"]) if name + ".alpha" in sd else float(r)
        mods[name] = (d.to("cuda", torch.float32), u.to("cuda", torch.float32), alpha, r, block_of(S["hooks"][name][1]))
    if not mods:
        raise ValueError(f"{os.path.basename(path)}: 이 모델에 맞는 LoRA 모듈이 없습니다(키 형식이 다름)")
    entry = {"sig": sig, "mods": mods, "unknown": unknown}
    S["loras"][path] = entry
    while len(S["loras"]) > LORA_CACHE:
        S["loras"].pop(next(iter(S["loras"])))
    return entry


def set_loras(specs, ref=1.0):
    """저장한 파일(run_save)과 똑같은 방식으로 계산한다: 모듈마다 블록 배율 × alpha / rank / ref 를 fp32 의 lora_up 에 곱하고,
    여러 LoRA 는 rank 를 이어 붙인 뒤 bf16 으로 바꿔 한 번에 계산하고 끝에 ref(주 LoRA 세기)를 곱한다.
    LoRA 마다 따로 bf16 으로 계산해 더하면 반올림 순서가 달라 확산 샘플링에서 그림이 갈라진다(평균 차 약 9/255)."""
    for h, _n in S["hooks"].values():
        h.adapters = []
    ref = float(ref) or 1.0
    parts = {}
    for sp in specs or []:
        entry = load_lora(sp["path"])
        scales = sp.get("scales") or [1.0] * BLOCKS
        other = float(sp.get("other", 1.0))
        for name, (d, u, alpha, r, blk) in entry["mods"].items():
            s = float(scales[blk]) if blk is not None and blk < len(scales) else other
            if s != 0.0:
                parts.setdefault(name, []).append((d, u * (s * alpha / r / ref)))
    for name, ps in parts.items():
        d = (ps[0][0] if len(ps) == 1 else torch.cat([p[0] for p in ps], 0)).to(torch.bfloat16)
        u = (ps[0][1] if len(ps) == 1 else torch.cat([p[1] for p in ps], 1)).to(torch.bfloat16)
        S["hooks"][name][0].adapters.append((d, u, ref))


# ---------------------------------------------------------------- 렌더
def file_sig(path):
    st = os.stat(path)
    return [os.path.abspath(path), st.st_mtime_ns, st.st_size]


def render_key(job, item):
    loras = [[*file_sig(sp["path"]), [round(float(v), 4) for v in (sp.get("scales") or [])], round(float(sp.get("other", 1.0)), 4)]
             for sp in item.get("loras") or []]
    m = job["model"]
    blob = [m["dit"], bool(m.get("int8")), m["vae"], job["prompt"], job["negative"], int(job["seed"]), int(job["width"]),
            int(job["height"]), job["preset"], int(job.get("steps") or 0), loras, round(float(item.get("ref") or 1.0), 4), 3,
            None if job.get("cfg") in (None, "") else round(float(job["cfg"]), 4)]  # 3: CFG 를 넣음(2026-11-12, 전에는 CFG 만 바꾸면 예전 그림이 그대로 왔다)
    return hashlib.sha1(json.dumps(blob, ensure_ascii=False).encode()).hexdigest()[:24]


def run_render(job):
    from PIL import Image
    import moss_sampling as ms
    import moss_image_minimal_inference as mi
    from library import moss_image_train_utils as tu, moss_image_utils
    jid = job["id"]
    os.makedirs(OUT_DIR, exist_ok=True)
    items = job["items"]
    todo, same = [], {}  # same: 같은 조건(같은 파일 이름)의 다른 항목은 한 번만 그리고 함께 알린다
    for i, it in enumerate(items):
        dest = os.path.join(OUT_DIR, render_key(job, it) + ".png")
        if os.path.isfile(dest):
            os.utime(dest)
            emit(ev="image", id=jid, i=i, key=it.get("key"), file=dest, cached=True, secs=0)
        elif dest in same:
            same[dest].append((i, it))
        else:
            same[dest] = []
            todo.append((i, it, dest))
    if not todo:
        return
    P = dict(ms.PRESETS.get(job["preset"]) or ms.PRESETS["user"])
    if int(job.get("steps") or 0) > 0:
        P["steps"] = int(job["steps"])
    if "cfg" in job and job["cfg"] not in (None, ""):
        P["cfg"] = float(job["cfg"])
    negative = ms.strip_weights(job["negative"] or "")
    pos_e, neg_e = encode(job["model"], [job["prompt"], negative])
    if cancelled(jid):
        raise Cancelled
    dit = ensure_dit(job["model"])
    vae = ensure_vae(job["model"])
    dev = torch.device("cuda")
    args = S["args"]
    pos, pm = moss_image_utils.pad_text_embeds([pos_e.to(dev)])
    neg, nm = moss_image_utils.pad_text_embeds([neg_e.to(dev)])
    pos, neg = pos.to(dit.dtype), neg.to(dit.dtype)
    official_skip = tu.parse_skip_layers(args.uncond_skip_layers) if job["preset"] == "official" else ()
    sig = ms.get_sigmas(P["scheduler"], P["steps"], args.flow_shift)
    n = len(sig) - 1
    W, H = int(job["width"]) // 16 * 16, int(job["height"]) // 16 * 16
    for k, (i, it, dest) in enumerate(todo):
        set_loras(it.get("loras"), it.get("ref") or 1.0)
        den = ms.make_denoiser(dit, pos, pm, neg if negative or P["cfg"] != 1.0 else None, nm, P["cfg"], args.flow_shift,
                               slg_layers=(22,) if P["slg"] else (), slg_start=0.0, slg_end=0.5, official_skip=official_skip)
        t0, last = time.time(), [0.0]

        def denoise(x, s, step, den=den, i=i, t0=t0):
            if cancelled(jid):
                raise Cancelled
            now = time.time()
            if now - last[0] > 0.4:
                last[0] = now
                emit(ev="progress", id=jid, i=i, step=step, steps=n, done=k, todo=len(todo), secs=round(now - t0, 1))
            return den(x, s, step)

        g = torch.Generator(device="cpu").manual_seed(int(job["seed"]))
        lat = torch.randn((1, dit.latent_channels, H // 16, W // 16), generator=g, dtype=torch.float32).to(dev)
        with torch.no_grad():
            lat = ms.sample(denoise, lat, sig, P["sampler"])
        px = mi.decode_latents(vae, lat, dev)
        img = Image.fromarray(((px.permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype("uint8"))
        tmp = dest + ".tmp.png"
        img.save(tmp, compress_level=1)
        os.replace(tmp, dest)
        emit(ev="image", id=jid, i=i, key=it.get("key"), file=dest, cached=False, secs=round(time.time() - t0, 1))
        for j, other in same.get(dest, []):
            emit(ev="image", id=jid, i=j, key=other.get("key"), file=dest, cached=True, secs=0)
    set_loras([])
    torch.cuda.empty_cache()


def run_save(job):
    """미리보기와 똑같이 그려지는 LoRA 를 저장한다. 블록 배율·세기를 lora_up 에 곱하고, 두 LoRA 가 같은 모듈을 쓰면
    rank 를 이어 붙인다(alpha = rank, 즉 배율 1). 이 파일을 `strength` 로 쓰면 미리보기와 같다."""
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    strength = float(job.get("strength") or 1.0)
    if strength == 0:
        raise ValueError("세기가 0 이면 저장할 수 없습니다")
    parts = {}
    dtype = None
    for sp in job["loras"]:
        sd = load_file(sp["path"])
        scales = sp.get("scales") or [1.0] * BLOCKS
        other = float(sp.get("other", 1.0))
        for k in sd:
            if not k.endswith(".lora_down.weight"):
                continue
            name = k[: -len(".lora_down.weight")]
            d, u = sd[k].float(), sd[name + ".lora_up.weight"].float()
            dtype = dtype or sd[k].dtype
            r = d.shape[0]
            alpha = float(sd[name + ".alpha"]) if name + ".alpha" in sd else float(r)
            m = re.match(r"^lora_unet_layers_(\d+)_", name)
            s = float(scales[int(m.group(1))]) if m and int(m.group(1)) < len(scales) else other
            if s == 0.0:
                continue
            parts.setdefault(name, []).append((d, u * (s * alpha / r / strength)))  # set_loras 와 같은 순서로 계산
    if not parts:
        raise ValueError("남는 모듈이 없습니다(모든 블록이 0)")
    out = {}
    ranks = []
    for name, ps in parts.items():
        d = torch.cat([p[0] for p in ps], 0)
        u = torch.cat([p[1] for p in ps], 1)
        out[name + ".lora_down.weight"] = d.to(dtype).contiguous()
        out[name + ".lora_up.weight"] = u.to(dtype).contiguous()
        out[name + ".alpha"] = torch.tensor(float(d.shape[0]))
        ranks.append(d.shape[0])
    with safe_open(job["loras"][0]["path"], "pt") as f:
        meta = dict(f.metadata() or {})
    meta.update({k: str(v) for k, v in (job.get("meta") or {}).items()})
    meta["ss_network_dim"] = str(max(ranks))
    meta["ss_network_alpha"] = str(max(ranks))
    tmp = job["out"] + ".tmp"
    save_file(out, tmp, metadata=meta)
    os.replace(tmp, job["out"])
    emit(ev="saved", id=job["id"], file=job["out"], modules=len(parts), rank_max=max(ranks), rank_min=min(ranks),
         size=os.path.getsize(job["out"]))


def save_thread(job):
    """저장은 CPU 만 쓰므로 렌더를 기다리지 않고 바로 한다."""
    try:
        run_save(job)
    except Exception as e:
        emit(ev="error", id=job["id"], msg=str(e) or type(e).__name__, trace=traceback.format_exc()[-1500:])


def prune():
    try:
        files = sorted((e for e in os.scandir(OUT_DIR) if e.is_file() and e.name.endswith(".png")), key=lambda e: e.stat().st_mtime)
        for e in files[: max(0, len(files) - MAX_OUT_FILES)]:
            os.remove(e.path)
    except OSError:
        pass


def _grid(m):
    """(h, w) → {w, h, v}: 긴 변이 MAP_MAX_SIDE 를 넘으면 평균으로 줄인다. 값은 유효숫자 4자리."""
    m = m.float()
    h, w = m.shape
    k = max(1, -(-max(h, w) // MAP_MAX_SIDE))
    if k > 1:
        m = F.avg_pool2d(m[None, None], k, ceil_mode=True)[0, 0]
    h, w = m.shape
    return {"w": int(w), "h": int(h), "v": [float(f"{x:.4g}") for x in m.flatten().tolist()]}


def map_key(job):
    p = job["file"]  # 서버가 확인한 전체 경로(.studio_out 또는 비교 기록 폴더)
    loras = [[*file_sig(sp["path"]), [round(float(v), 4) for v in (sp.get("scales") or [])], round(float(sp.get("other", 1.0)), 4)]
             for sp in job.get("loras") or []]
    m = job["model"]
    blob = [m["dit"], bool(m.get("int8")), m["vae"], job["prompt"], os.path.basename(p), os.path.getsize(p), loras,
            round(float(job.get("ref") or 1.0), 4), list(MAP_LEVELS), MAP_SEED, 1]
    return hashlib.sha1(json.dumps(blob, ensure_ascii=False).encode()).hexdigest()[:24]


def run_map(job):
    """그린 그림의 'LoRA 가 작용하는 곳' 지도. 그림 → VAE 잠재 → 세 노이즈 단계(같은 시드의 노이즈)에서 LoRA 를 켠 예측과 끈 예측의
    차이 |v_LoRA − v_0| 를 채널 평균 → 잠재 격자(그림 16px 당 한 칸). LoRA 가 없으면 지도도 없다."""
    import numpy as np
    from PIL import Image
    from library import moss_image_train_utils as tu, moss_image_utils
    jid = job["id"]
    if not job.get("loras"):
        emit(ev="map", id=jid, file=job["file"], error="LoRA 가 없는 그림에는 작용 지도가 없습니다")
        return
    os.makedirs(MAP_DIR, exist_ok=True)
    cache = os.path.join(MAP_DIR, map_key(job) + ".json")
    if os.path.isfile(cache):
        try:
            with open(cache, encoding="utf-8") as f:
                emit(ev="map", id=jid, file=job["file"], cached=True, **json.load(f))
            return
        except (OSError, ValueError):
            pass
    t0 = time.time()
    pos_e, = encode(job["model"], [job["prompt"]])
    if cancelled(jid):
        raise Cancelled
    dit = ensure_dit(job["model"])
    vae = ensure_vae(job["model"])
    dev = torch.device("cuda")
    emit(ev="status", text="히트맵을 만드는 중")
    with Image.open(job["file"]) as im:
        im = im.convert("RGB")
        W, H = im.size[0] // 16 * 16, im.size[1] // 16 * 16
        if (W, H) != im.size:
            im = im.crop((0, 0, W, H))
        px = torch.from_numpy(np.asarray(im, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None]
    with torch.no_grad():
        z = vae.encode_for_dit(px.to(dev, vae.dtype)).float()
        pos, pm = moss_image_utils.pad_text_embeds([pos_e.to(dev)])
        pos = pos.to(dit.dtype)
        maps = {}
        for i, s in enumerate(MAP_LEVELS):
            if cancelled(jid):
                raise Cancelled
            g = torch.Generator(device="cpu").manual_seed(MAP_SEED * 1000003 + i)
            eps = torch.randn(z.shape, generator=g, dtype=torch.float32).to(dev)
            xt = ((1.0 - s) * z + s * eps).to(dit.dtype)
            ts = torch.full((1,), s * tu.NUM_TRAIN_TIMESTEPS, device=dev, dtype=torch.float32)
            set_loras(job["loras"], job.get("ref") or 1.0)
            v1 = dit(xt, ts, pos, pm).float()
            set_loras([])
            v0 = dit(xt, ts, pos, pm).float()
            maps[f"{s:g}"] = _grid((v1 - v0).abs().mean(dim=1)[0])
            del v1, v0, xt, eps
    set_loras([])
    res = {"levels": [f"{s:g}" for s in MAP_LEVELS], "maps": maps, "size": [W, H]}
    tmp = cache + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(res, f)
    os.replace(tmp, cache)
    emit(ev="map", id=jid, file=job["file"], secs=round(time.time() - t0, 1), **res)


def prune_maps():
    try:
        files = sorted((e for e in os.scandir(MAP_DIR) if e.is_file() and e.name.endswith(".json")), key=lambda e: e.stat().st_mtime)
        for e in files[: max(0, len(files) - MAX_OUT_FILES)]:
            os.remove(e.path)
    except OSError:
        pass


def work_loop():
    global last_used
    while True:
        job = jobs.get()
        current["id"] = job["id"]
        try:
            if cancelled(job["id"]):
                raise Cancelled
            if job["op"] == "render":
                run_render(job)
                emit(ev="done", id=job["id"])
            elif job["op"] == "map":
                run_map(job)
        except Cancelled:
            set_loras([]) if S["hooks"] else None
            emit(ev="map" if job["op"] == "map" else "cancelled", id=job["id"], **({"file": job.get("file"), "error": "취소됨"} if job["op"] == "map" else {}))
        except Exception as e:
            msg = str(e) or type(e).__name__
            if "out of memory" in msg.lower():
                msg = "GPU 메모리가 모자랍니다 — 크기를 줄이거나 다른 프로그램(ComfyUI·Fizgig 등)의 GPU 사용을 줄이세요"
                torch.cuda.empty_cache()
            if job["op"] == "map":
                set_loras([]) if S["hooks"] else None
                emit(ev="map", id=job["id"], file=job.get("file"), error=msg)
            else:
                emit(ev="error", id=job["id"], msg=msg, trace=traceback.format_exc()[-1500:])
        finally:
            with cancel_lock:
                cancel_ids.discard(job["id"])
                if jobs.empty():
                    cancel_ids.discard("*")
            current["id"] = None
            last_used = time.time()
            jobs.task_done()


def idle_loop():
    while True:
        time.sleep(20)
        if S["te"] is not None and current["id"] is None and time.time() - S["te_used"] > TE_IDLE_SECONDS:
            S["te"] = S["tok"] = None
            S["te_key"] = None
            emit(ev="status", text="텍스트 인코더를 내렸습니다(잠시 안 씀)")
        if current["id"] is None and jobs.empty() and time.time() - last_used > IDLE_EXIT_SECONDS:
            emit(ev="exit", reason="idle")
            os._exit(0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    prune()
    prune_maps()
    threading.Thread(target=work_loop, daemon=True).start()
    threading.Thread(target=idle_loop, daemon=True).start()
    emit(ev="ready", pid=os.getpid())
    for line in sys.stdin:
        try:
            job = json.loads(line)
        except ValueError:
            continue
        op = job.get("op")
        if op == "cancel":
            with cancel_lock:
                cancel_ids.add(job.get("target") or "*")
        elif op == "unload":
            emit(ev="exit", reason="unload")
            os._exit(0)
        elif op in ("render", "map"):
            jobs.put(job)
        elif op == "save":
            threading.Thread(target=save_thread, args=(job,), daemon=True).start()
    os._exit(0)


if __name__ == "__main__":
    main()
