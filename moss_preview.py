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


def make_plan(path, o):
    with Image.open(path) as im:
        w, h = im.size
    bucket, resized, _ = bucket_manager(o).select_bucket(w, h)
    bucket, resized = [int(v) for v in bucket], [int(v) for v in resized]
    stat = os.stat(path)
    key = hashlib.sha1(json.dumps([os.path.abspath(path), stat.st_mtime_ns, stat.st_size, bucket, resized, o["interp"]]).encode()).hexdigest()
    method = None
    if (w, h) != tuple(resized):
        method = o["interp"] or ("area" if w >= resized[0] and h >= resized[1] else "lanczos")  # library.utils.resize_image 와 같은 기준
    npz = latents.get_latents_npz_path(path, (w, h))
    return {
        "orig": [w, h],
        "resized": resized,
        "bucket": bucket,
        "crop": [(resized[0] - bucket[0]) // 2, (resized[1] - bucket[1]) // 2],  # trim_and_resize_if_required 의 가운데 자르기
        "method": method,
        "key": key,
        "npz": npz if latents.is_disk_cached_latents_expected(tuple(bucket), npz, False, False) else None,
        "vae_done": os.path.isfile(cache_file(key, "vae")),
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


def run_image(job):
    plan = make_plan(job["path"], job["opts"])
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
