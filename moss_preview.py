# MOSS-Image 자료 미리보기 작업 프로세스. moss_ui.py 가 처음 쓸 때 띄운다.
# 학습과 같은 코드로 리사이즈·VAE 를 처리해 PNG 로 남긴다. GPU 는 쓰지 않는다(CUDA_VISIBLE_DEVICES 비움).
# 표준 입력 한 줄에 요청 JSON 하나, 표준 출력 한 줄에 응답 JSON 하나. 입력이 닫히면(UI 종료) 끝난다.
import hashlib
import json
import os
import queue
import sys
import threading
import time

reply_stream = os.fdopen(os.dup(1), "w", encoding="utf-8")
os.dup2(2, 1)  # 라이브러리의 print 가 응답 줄을 깨뜨리지 않게 표준 출력을 로그로 돌린다

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "sd-scripts"))
from library import flux2_vae
from library.dataset import BaseDataset, BucketManager
from library.strategy_moss_image import MossImageLatentsCachingStrategy
from library.utils import IMAGE_TRANSFORMS, load_image, trim_and_resize_if_required

sys.path.insert(0, HERE)
import moss_cache  # noqa: E402

CACHE_DIR = os.path.join(HERE, ".preview_cache")
MAX_CACHE_FILES = 300
IDLE_EXIT_SECONDS = 600  # 안 쓰면 끝내서 RAM 을 돌려준다. 다음 요청 때 다시 뜬다
VAE_THREADS = max(2, min(8, (os.cpu_count() or 4) // 2))  # 학습과 CPU 를 나눠 쓴다

latents = MossImageLatentsCachingStrategy(True, 1, False)
vae_jobs = queue.Queue()
write_lock = threading.Lock()
vae, vae_path = None, None
last_used = time.time()


def reply(msg):
    with write_lock:
        reply_stream.write(json.dumps(msg, ensure_ascii=False) + "\n")
        reply_stream.flush()


def bucket_manager(o):
    """dataset.py 의 make_buckets 와 같은 순서로 BucketManager 를 만든다."""
    res = tuple(o["resolution"])
    if not o["enable_bucket"]:
        m = BucketManager(False, res, None, None, None)
        m.set_predefined_resos([res])
        return m
    if o["steps"] % 16:
        raise ValueError(f"bucket_reso_steps 는 16 의 배수여야 합니다 ({o['steps']})")
    lo, hi = BaseDataset.adjust_min_max_bucket_reso_by_steps(None, res, o["min"], o["max"], o["steps"])
    m = BucketManager(o["no_upscale"], res, lo, hi, o["steps"])
    if not o["no_upscale"]:
        m.make_buckets()
    return m


def resize_step(w, h, o):
    """그림 크기 (w, h) 를 학습 코드와 같은 버킷·리사이즈로: (버킷, 리사이즈 크기, 가운데 자르기, 보간 방식)."""
    bucket, resized, _ = bucket_manager(o).select_bucket(w, h)
    bucket, resized = [int(v) for v in bucket], [int(v) for v in resized]
    method = None
    if (w, h) != tuple(resized):
        method = o["interp"] or ("area" if w >= resized[0] and h >= resized[1] else "lanczos")  # library.utils.resize_image 와 같은 기준
    return bucket, resized, [(resized[0] - bucket[0]) // 2, (resized[1] - bucket[1]) // 2], method


def make_plan(path, o):
    with Image.open(path) as im:
        w, h = im.size
    bucket, resized, _crop, method = resize_step(w, h, o)
    stat = os.stat(path)
    key = hashlib.sha1(json.dumps([os.path.abspath(path), stat.st_mtime_ns, stat.st_size, bucket, resized, o["interp"]]).encode()).hexdigest()
    # 원본 유지 2배 확대(이미지 처리 탭, 최근접 정수 2배)를 했다면 학습이 그 그림을 어떻게 다시 맞추는지
    ub, ur, uc, um = resize_step(w * 2, h * 2, o)
    up = {"orig": [w * 2, h * 2], "bucket": ub, "resized": ur, "crop": uc, "method": um}
    npz = latents.get_latents_npz_path(path, (w, h))
    return {
        "orig": [w, h],
        "resized": resized,
        "bucket": bucket,
        "crop": [(resized[0] - bucket[0]) // 2, (resized[1] - bucket[1]) // 2],  # trim_and_resize_if_required 의 가운데 자르기
        "method": method,
        "key": key,
        # 그림이 캐시보다 새로우면(그림을 고침) 캐시는 학습 시작 때 지워지므로 미리보기도 쓰지 않는다(moss_cache.py)
        "npz": npz if latents.is_disk_cached_latents_expected(tuple(bucket), npz, False, False) and not moss_cache.is_stale(npz, path) else None,
        "vae_done": os.path.isfile(cache_file(key, "vae")),
        "up": up,
    }


def cache_file(key, kind):
    return os.path.join(CACHE_DIR, f"{key}_{kind}.png")


def save_png(arr, dest):
    tmp = dest + ".tmp"
    Image.fromarray(arr).save(tmp, "PNG", compress_level=1)
    os.replace(tmp, dest)


def processed_array(path, plan, interp):
    image = load_image(path)
    return trim_and_resize_if_required(False, image, tuple(plan["bucket"]), tuple(plan["resized"]), interp)[0]


def load_vae(path):
    global vae, vae_path
    if vae is None or vae_path != path:
        if not path or not os.path.isfile(path):
            raise ValueError(f"VAE 파일이 없습니다: {path}")
        vae = flux2_vae.load_flux2_vae(path, dtype=torch.float32, device="cpu", disable_mmap=False)
        vae.eval()
        vae_path = path
    return vae


def vae_array(path, plan, o):
    """학습 캐시가 있으면 그 latent 를, 없으면 캐시를 만들 때와 같은 방식으로 인코딩한 latent 를 복원한다."""
    model = load_vae(o["vae"])
    bw, bh = plan["bucket"]
    z = None
    if plan["npz"]:
        z = torch.from_numpy(latents.load_latents_from_disk(plan["npz"], (bw, bh))[0]).float()
        if z.shape[-2:] != (bh // flux2_vae.LATENT_SCALE_FACTOR, bw // flux2_vae.LATENT_SCALE_FACTOR):
            z = None
    with torch.no_grad():
        if z is None:
            z = model.encode(IMAGE_TRANSFORMS(processed_array(path, plan, o["interp"]))[None].float())[0]
        px = model.decode(z[None])[0]
    return ((px.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()


def mask_array(path, mask, plan, interp, alpha=False):
    """마스크를 학습(controlnet_dataset)처럼 처리한다: 그림과 같은 버킷·리사이즈·가운데 자르기. 학습이 쓰는 빨강 채널을 회색으로.
    크기가 그림과 다르면 먼저 그림 크기로 맞춘다(moss_masks.prepare_dataset 이 학습 전에 같은 일을 한다).
    mask 가 없으면 흰색(그림 전체), alpha 면 그림의 투명도를 곱한다(투명한 곳은 학습하지 않음, moss_masks 와 같음)."""
    w, h = plan["orig"]
    if mask:
        m = load_image(mask)
        if m.shape[:2] != (h, w):
            m = np.asarray(Image.fromarray(m).resize((w, h), Image.BILINEAR))
    else:
        m = np.full((h, w, 3), 255, np.uint8)
    if alpha:
        a = load_image(path, alpha=True)[:, :, 3:4].astype(np.uint16)
        m = (m[:, :, :1].astype(np.uint16) * a // 255).astype(np.uint8).repeat(3, axis=2)
    m = trim_and_resize_if_required(False, m, tuple(plan["bucket"]), tuple(plan["resized"]), interp)[0]
    return np.repeat(m[:, :, :1], 3, axis=2)


def up2x_array(path, plan, interp):
    """원본 유지 2배 확대(픽셀 하나 → 같은 값 2×2) 뒤 학습이 하는 버킷·리사이즈·가운데 자르기까지 한 최종 결과."""
    image = load_image(path)
    up = np.repeat(np.repeat(image, 2, axis=0), 2, axis=1)
    u = plan["up"]
    return trim_and_resize_if_required(False, up, tuple(u["bucket"]), tuple(u["resized"]), interp)[0]


def run_image(job):
    plan = make_plan(job["path"], job["opts"])
    if job["kind"] == "up2x":
        dest = cache_file(plan["key"], "up2x_" + "x".join(map(str, plan["up"]["bucket"] + plan["up"]["resized"])))  # 설정마다 버킷이 달라짐
        if not os.path.isfile(dest):
            save_png(up2x_array(job["path"], plan, job["opts"]["interp"]), dest)
        return {"file": dest}
    if job["kind"] == "mask":
        mk, alpha = job.get("mask") or "", bool(job.get("alpha"))
        st = os.stat(mk) if mk else None
        key = hashlib.sha1(json.dumps([plan["key"], os.path.abspath(mk) if mk else "", st.st_mtime_ns if st else 0, st.st_size if st else 0, alpha]).encode()).hexdigest()
        dest = cache_file(key, "mask")
        if not os.path.isfile(dest):
            save_png(mask_array(job["path"], mk, plan, job["opts"]["interp"], alpha), dest)
        return {"file": dest}
    dest = cache_file(plan["key"], job["kind"])
    if not os.path.isfile(dest):
        if job["kind"] == "vae":
            save_png(vae_array(job["path"], plan, job["opts"]), dest)
        else:
            save_png(processed_array(job["path"], plan, job["opts"]["interp"]), dest)
    return {"file": dest}


def answer(job, work):
    global last_used
    try:
        reply({"id": job["id"], **work(job)})
    except Exception as e:
        reply({"id": job["id"], "error": str(e) or type(e).__name__})
    last_used = time.time()


def vae_loop():
    while True:
        answer(vae_jobs.get(), run_image)
        vae_jobs.task_done()


def idle_loop():
    while True:
        time.sleep(30)
        if time.time() - last_used > IDLE_EXIT_SECONDS and vae_jobs.unfinished_tasks == 0:
            os._exit(0)


def prune_cache():
    files = sorted((e for e in os.scandir(CACHE_DIR) if e.is_file()), key=lambda e: e.stat().st_mtime)
    for e in files[:max(0, len(files) - MAX_CACHE_FILES)]:
        os.remove(e.path)


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    prune_cache()
    torch.set_num_threads(VAE_THREADS)
    threading.Thread(target=vae_loop, daemon=True).start()
    threading.Thread(target=idle_loop, daemon=True).start()
    for line in sys.stdin:  # VAE 는 수십 초 걸리므로 따로 돌리고 그동안 다른 그림의 처리 결과는 바로 답한다
        job = json.loads(line)
        if job["op"] == "plan":
            answer(job, lambda j: make_plan(j["path"], j["opts"]))
        elif job["kind"] == "vae":
            vae_jobs.put(job)
        else:
            answer(job, run_image)


if __name__ == "__main__":
    main()
