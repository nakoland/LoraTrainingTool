# 에포크 LoRA 저장 범위(2026-11-14 사용자 요청). 표준 라이브러리만 씀(학습 파일·UI 서버 공통, 화면 moss_ui.html 의 saveEpochs 와 같은 식).
# - save_from_pct(기본 50): 전체 에포크의 이 % 지점부터 저장. 0 이면 처음부터.
# - save_max_count(기본 20): 최종 파일을 포함한 최대 개수. 넘으면 최신부터 이 개수만(앞의 % 설정은 무시). 0 이면 제한 없음.
# 예) 10에포크 기본값 → 에포크 5~9 + 최종(10) = 6개. 100에포크 → 81~99 + 최종(100) = 20개.
# 최종 파일(<이름>.safetensors)은 sd-scripts 가 학습 끝에 늘 저장한다. 스텝 주기 저장(-stepNNNNNNNN)·중지 상태는 건드리지 않는다.
import re

DEFAULT_FROM_PCT = 50
DEFAULT_MAX_COUNT = 20
EPOCH_CKPT_RE = re.compile(r"-(\d{6})\.[A-Za-z0-9]+$")  # sd-scripts get_epoch_ckpt_name: <이름>-000005.safetensors


def _num(v, default):
    try:
        return default if v is None or v == "" else float(v)
    except (TypeError, ValueError):
        return default


def save_epochs(total, every, from_pct=None, max_count=None):
    """저장할 에포크 번호 목록(최종 에포크 제외, 오름차순). total 을 모르면 None."""
    total = int(_num(total, 0))
    every = int(_num(every, 0))
    pct = _num(from_pct, DEFAULT_FROM_PCT)
    maxn = int(_num(max_count, DEFAULT_MAX_COUNT))
    if total <= 0:
        return None
    if every <= 0:
        return []
    cands = list(range(every, total, every))  # sd-scripts: (epoch+1) % every == 0 and epoch+1 < 전체
    if pct > 0:
        start = max(1, -int(-(total * pct) // 100))  # 올림
        cands = [e for e in cands if e >= start]
    if maxn > 0:
        cands = cands[-(maxn - 1):] if maxn > 1 else []
    return cands


def epoch_of_ckpt(name):
    m = EPOCH_CKPT_RE.search(name or "")
    return int(m.group(1)) if m else None
