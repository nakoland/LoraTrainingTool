# 캡션 셔플: 스타일 트리거 → 일반 트리거 → 나머지 태그 순서이며 각 그룹 안에서만 섞는다.
#
# sd-scripts 는 TE 출력 캐시를 쓰면 shuffle_caption 을 assert 로 막는다(캡션이 캐시에 고정). 대신 캡션 후보 여러 개는 캐시와
# 함께 된다. 에포크별 모드는 그림마다 에포크 수만큼 만들고 에포크 번호로 고른다. 고정 모드는 지정한 후보 중 스텝마다 고른다.
#   - 후보는 정해진 시드로 재현되므로 같은 설정에서 TE 디스크 캐시를 다시 쓴다.
#   - 첫 후보는 원래 순서(트리거만 앞으로). 같은 태그가 두 번 든 줄은 한 번만 남긴다. 같은 결과는 한 번만.
#   - 원래 캡션 후보(---===---)가 여럿이면 에포크별 모드는 번갈아 사용하며, 고정 모드는 후보마다 N 개를 만든다.
#   - 여러 줄(자연어) 후보는 섞지 않는다.
# 학습: install() 이 학습 프로세스 안에서 데이터셋을 만드는 함수를 감싼다(moss_extras.prepare_args, 아래 '학습 프로세스' 참고).
# 캡션 함수(shuffle_variants·keep_words)는 UI 서버도 쓰므로 이 파일 맨 위에서 torch·sd-scripts 를 가져오지 않는다.
import hashlib
import math
import os
import random
import re
from itertools import zip_longest

DEFAULT_COUNT = 4
MAX_COUNT = 32
VERSION = 1  # 바꾸면 후보가 달라져 TE 캐시를 다시 만든다
DEFAULT_MODE = "epoch"
FIXED_MODE = "fixed"
MODES = (DEFAULT_MODE, FIXED_MODE)


def shuffle_mode(value):
    mode = value or DEFAULT_MODE
    if mode not in MODES:
        raise ValueError("캡션 셔플 방식(caption_shuffle_mode)은 epoch 또는 fixed 여야 합니다")
    return mode


def fixed_count(value):
    return max(1, min(MAX_COUNT, int(value or DEFAULT_COUNT)))


def training_epochs(epochs, steps, batches, processes, accumulation):
    if epochs is not None:
        if epochs < 1:
            raise ValueError("캡션 셔플: 에포크 수는 1 이상이어야 합니다")
        return epochs
    steps_per_epoch = max(1, math.ceil(batches / (processes * accumulation)))
    return max(1, math.ceil(steps / steps_per_epoch))


def norm(tag):
    return re.sub(r"\s+", " ", str(tag).replace("_", " ")).strip().lower()


def keep_words(text):
    """'fmmi1, fmmi2' → ['fmmi1', 'fmmi2'](정규화, 순서 유지, 중복 제거)."""
    out = []
    for t in str(text or "").split(","):
        n = norm(t)
        if n and n not in out:
            out.append(n)
    return out


def split_tags(caption, sep=","):
    seen, tags = set(), []
    for t in str(caption).split(sep):
        t = t.strip()
        n = norm(t)
        if t and n not in seen:
            seen.add(n)
            tags.append(t)
    return tags


def tag_groups(caption, keep, style, sep):
    style_tags, triggers, rest = [], [], []
    for tag in split_tags(caption, sep):
        key = norm(tag)
        group = style_tags if key in style else triggers if key in keep else rest
        group.append(tag)
    return style_tags, triggers, rest


def shuffle_variants(caption, keep, count=DEFAULT_COUNT, sep=",", *, style=()):
    """고정 모드의 후보. 첫 후보는 각 그룹 안에서 원래 순서를 유지한다."""
    caption = str(caption or "")
    if not caption.strip() or "\n" in caption.strip():
        return [caption]
    ks, styles = set(keep), set(style)
    style_tags, triggers, rest = tag_groups(caption, ks, styles, sep)
    joiner = sep + " "
    first = joiner.join(style_tags + triggers + rest)
    count = max(1, min(MAX_COUNT, int(count or 1)))
    out = [first]
    possible = _fact(len(style_tags)) * _fact(len(triggers)) * _fact(len(rest))
    want = min(count, possible)
    seed_text = f"{VERSION}|{caption}|{','.join(sorted(ks))}|{count}"
    if styles:
        seed_text += "|" + ",".join(sorted(styles))
    seed = hashlib.md5(seed_text.encode("utf-8")).hexdigest()
    rng = random.Random(int(seed[:16], 16))
    tries = 0
    while len(out) < want and tries < want * 20:
        tries += 1
        a, b, c = style_tags[:], triggers[:], rest[:]
        rng.shuffle(a)
        rng.shuffle(b)
        rng.shuffle(c)
        v = joiner.join(a + b + c)
        if v not in out:
            out.append(v)
    return out


def _ranked_tags(tags, rank, orders):
    pool = list(tags)
    for size in range(len(pool), 0, -1):
        orders //= size
        position, rank = divmod(rank, orders)
        yield pool.pop(position)


def epoch_variant_count(caption, keep, style, sep):
    if not caption.strip() or "\n" in caption.strip():
        return 1
    return math.prod(math.factorial(len(group)) for group in tag_groups(caption, keep, style, sep))


def epoch_variants(caption, keep, style, sep, image):
    """가능한 순서를 중복 없이 섞어, 필요한 만큼 읽을 수 있게 한다."""
    if not caption.strip() or "\n" in caption.strip():
        yield caption
        return
    groups = tag_groups(caption, keep, style, sep)
    joiner = sep + " "
    orders = [math.factorial(len(group)) for group in groups]
    rng = random.Random(repr((VERSION, image, caption, sorted(keep), sorted(style), sep)))
    # 순서에 번호를 매겨 필요한 만큼만 Fisher–Yates 로 뽑는다. 전체 순열을 메모리에 두지 않는다.
    swaps = {}
    for last in range(math.prod(orders) - 1, -1, -1):
        pick = rng.randrange(last + 1)
        rank = swaps.get(pick, pick)
        replacement = swaps.pop(last, last)
        if pick != last:
            swaps[pick] = replacement
        tags = []
        for group, radix in zip(groups, orders):
            rank, group_rank = divmod(rank, radix)
            tags.extend(_ranked_tags(group, group_rank, radix))
        yield joiner.join(tags)


def combine_variants(captions, keep, count, sep=",", *, style=(), by_epoch=False, image=""):
    if by_epoch:
        # 원래 후보들을 번갈아 뽑고, 그림 전체의 후보 수를 에포크 수로 제한한다.
        keep, style = set(keep), set(style)
        image = os.path.basename(image)
        streams = [epoch_variants(caption, keep, style, sep, image) for caption in captions]
        out = {}
        for row in zip_longest(*streams):
            for variant in row:
                if variant is not None:
                    out[variant] = None
                    if len(out) >= count:
                        return list(out)
        return list(out) or [""]
    return list(dict.fromkeys(variant for c in captions for variant in shuffle_variants(c, keep, count, sep, style=style))) or [""]


def _fact(n):
    r = 1
    for i in range(2, n + 1):
        r *= i
        if r > MAX_COUNT:
            return r
    return r


def resolve_groups(style=None, general=None, legacy_keep="", dop_trigger=""):
    """예전 설정의 맨 앞 그룹은 일반 트리거로 읽는다. 새 항목의 빈 문자열은 그대로 쓴다."""
    if style is None and general is None:
        general = legacy_keep or dop_trigger
    styles = keep_words(style)
    return styles, [tag for tag in keep_words(general) if tag not in styles]


# ── 학습 프로세스 ──
# Windows 데이터로더 작업 프로세스(spawn)에는 캡션 후보 텍스트와 공유 에포크 값이 넘어간다.
# 고정 모드는 데이터셋을 만든 직후 확장하고, 에포크별 모드는 캐시 직전에 실제 학습 길이를 계산한 뒤 확장한다.
_STATE = {}


def install(keep, count, log=print, *, style=()):
    """config_util.generate_dataset_group_by_blueprint 를 감싸 만든 데이터셋의 캡션을 셔플 후보로 바꾼다."""
    from library import config_util

    _STATE.update(keep=list(keep), style=list(style), count=int(count), log=log)
    if getattr(config_util, "_moss_shuffle", False):
        return
    orig = config_util.generate_dataset_group_by_blueprint

    def generate(*a, **kw):
        res = orig(*a, **kw)
        groups = res if isinstance(res, tuple) else (res,)
        for g in groups:
            if g is not None:
                expand_group(g, _STATE["keep"], _STATE["count"], style=_STATE["style"], log=_STATE["log"])
        return res

    config_util.generate_dataset_group_by_blueprint = generate
    config_util._moss_shuffle = True


def _raw_candidates(subset, caption):
    """sd-scripts process_caption_candidates_for_cache 와 같은 후보 고르기(접두·접미·바꾸기 전, 원래 설정으로)."""
    from library import dataset as D
    if subset.enable_multiline_caption:
        return D.split_caption_candidates(caption, subset.multiline_caption_separator)
    if getattr(subset, "enable_wildcard", False):
        return [caption]
    return [caption.split("\n")[0]]


def caption_datasets(group):
    for d in group.datasets:
        yield d
        delegate = getattr(d, "dreambooth_dataset_delegate", None)
        if delegate is not None:  # 마스크 학습(ControlNet 데이터셋)
            yield delegate


def expand_group(group, keep, count, *, style=(), log=print, epoch=None):
    datasets = list(caption_datasets(group))
    done, subsets = set(), {}
    n_img = n_cand = 0
    for d in datasets:
        i2s = getattr(d, "image_to_subset", {}) or {}
        for key, info in d.image_data.items():
            if id(info) in done:
                continue
            sub = i2s.get(key)
            if sub is None:
                continue
            done.add(id(info))
            vs = combine_variants(_raw_candidates(sub, info.caption or ""), keep, count, getattr(sub, "caption_separator", None) or ",",
                                  style=style, by_epoch=epoch is not None, image=info.absolute_path)
            sep = sub.multiline_caption_separator
            info.caption = f"\n{sep}\n".join(vs) if len(vs) > 1 else vs[0]
            subsets[id(sub)] = sub
            n_img += 1
            n_cand += len(vs)
        for sub in list(getattr(d, "subsets", []) or []) + list(i2s.values()):
            subsets[id(sub)] = sub
    for sub in subsets.values():  # 모든 캡션을 바꾼 뒤에 켠다(원래 설정으로 후보를 골라야 하므로)
        sub.enable_multiline_caption = True
    for d in datasets:
        d.has_multiple_caption_candidates = True
        d.caption_candidate_epoch = epoch
    if n_img:
        log(f"caption shuffle: 그림 {n_img}장 → 캡션 후보 {n_cand}개(평균 {n_cand / n_img:.1f})")
