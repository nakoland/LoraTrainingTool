# latent 캐시 자동 갱신 (2026-11-12). UI 서버(moss_ui.py)·미리보기(moss_preview.py)·학습(moss_extras.prepare_args)이 함께 쓴다.
# sd-scripts 의 latent 디스크 캐시(<그림 이름>_<원본 가로>x<원본 세로>_moss.npz)는 파일 이름(원본 크기)과 버킷 크기 키만 보고
# 그림 내용·수정 시각은 보지 않는다 → 크기를 그대로 두고 그림을 고치면 예전 latent 로 오류 없이 학습한다.
# 여기서는 그림 파일이 캐시보다 새로우면(수정 시각) 그 캐시를 낡은 것으로 보고, 학습 직전에 지워 다시 만들게 한다.
# TE 캐시(_moss_te.npz)는 sd-scripts 가 캡션 글을 비교해 다시 만들므로 건드리지 않는다.
import os
import re

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
LATENT_RE = re.compile(r"^(?P<stem>.+)_\d{4,}x\d{4,}_moss\.npz$", re.IGNORECASE)
SLACK = 2.0  # 초. 파일 시스템 시각 단위 차이로 갓 만든 캐시를 낡았다고 보지 않게


def is_stale(npz_path, image_path):
    try:
        return os.path.getmtime(npz_path) + SLACK < os.path.getmtime(image_path)
    except OSError:
        return False


def stale_latents(image_dir):
    """폴더의 낡은 latent 캐시 [(npz 경로, 그림 경로)]. 그림이 없는 캐시(지운 그림)는 건드리지 않는다."""
    try:
        names = os.listdir(image_dir)
    except OSError:
        return []
    images = {}  # 이름(소문자) → (가장 최근 수정 시각, 경로). 같은 이름의 png·jpg 는 캐시를 함께 쓰므로 새 쪽 기준
    for n in names:
        if os.path.splitext(n)[1].lower() in IMAGE_EXTS:
            p = os.path.join(image_dir, n)
            try:
                t = os.path.getmtime(p)
            except OSError:
                continue
            k = os.path.splitext(n)[0].lower()
            if k not in images or t > images[k][0]:
                images[k] = (t, p)
    out = []
    for n in names:
        m = LATENT_RE.match(n)
        if not m or m.group("stem").lower() not in images:
            continue
        t, img = images[m.group("stem").lower()]
        p = os.path.join(image_dir, n)
        try:
            if os.path.getmtime(p) + SLACK < t:
                out.append((p, img))
        except OSError:
            pass
    return out


def remove_stale_latents(dataset_config, log=print):
    """학습 직전: 데이터셋 설정의 모든 학습 폴더에서 그림보다 오래된 latent 캐시를 지운다 → sd-scripts 가 그 그림만 다시 인코딩한다."""
    import tomllib
    with open(dataset_config, "rb") as f:
        data = tomllib.load(f)
    total = 0
    for ds in data.get("datasets", []):
        for s in ds.get("subsets", []):
            d = s.get("image_dir")
            if not d:
                continue
            d = d if os.path.isabs(d) else os.path.join(os.getcwd(), d)  # sd-scripts 처럼 작업 폴더 기준
            gone, files = [], 0
            for npz, img in stale_latents(d):
                try:
                    os.remove(npz)
                    files += 1
                    if os.path.basename(img) not in gone:
                        gone.append(os.path.basename(img))
                except OSError as e:
                    log(f"latent cache: {os.path.basename(npz)} 를 지우지 못했습니다 ({e})")
            if gone:
                total += files
                log(f"latent cache: {os.path.basename(os.path.normpath(d))}: 그림이 캐시보다 새로워 {len(gone)}장을 다시 만듭니다"
                    f"(캐시 {files}개 지움): {', '.join(gone[:5])}{' …' if len(gone) > 5 else ''}")
    return total
