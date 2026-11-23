# MOSS-Image 학습 추가 기능 (2026-11-10, Fizgig 를 참고해 sd-scripts 트레이너에 더한다)
# 1) 학습 중 샘플을 ComfyUI 와 같게: 샘플 프리셋(샘플러·스케줄러·스텝·CFG·SLG), 기본 네거티브, 다른 DiT(SFT)로 샘플
#    --sample_preset user|fizgig|official, --sample_negative, --sample_dit <SFT 파일>
#    프롬프트 줄에 --n/--l/--s 가 있으면 그 값이 우선한다.
# 4) Clean-end share: σ < 0.5 인 스텝의 비율을 에포크마다 정한다(truncated logit-normal 의 평균을 맞춤).
#    --clean_share "40" | "30:7:60" | "30:7:60:8" | "30:7:60:8:3" (Fizgig 형식: 처음 비율:에포크:나중 비율[:끝 에포크[:R]])
#                  | "auto" | "auto:30:60" (2026-11-11: 학습 에포크 수 E 에 맞춰 Fizgig 의 70에포크 일정 30:7:60:8:3 을 같은 비율로
#                  옮긴다: 처음 비율 구간 = round(E × 0.1)에포크(최소 1), 다음 에포크부터 나중 비율, 그 뒤 3에포크마다 처음 비율 + 학습률 절반.
#                  E 70 → 30:7:60:8:3, 10 → 30:1:60:2:3, 20 → 30:2:60:3:3)
#    R: 끝 에포크 뒤에는 높은 비율, R 에포크마다 낮은 비율이고 그 에포크는 학습률을 반으로 한다(Fizgig 와 같음).
# 5) 마스크 밖 원본 유지(masked prior preservation, OneTrainer·Fizgig 방식): --masked_prior_weight w (> 0 이면 켬)
#    마스크 안은 학습 그림을, 마스크 밖은 LoRA 0 의 원본 예측을 목표로 한다(가중 w). 마스크는 sd-scripts 의 masked_loss 와
#    같은 것(데이터셋 conditioning_data_dir 의 마스크 그림 또는 alpha_mask). 슬라이더에는 쓰지 않는다.
# 6) LoRA EMA: --lora_ema_decay d (> 0 이면 켬, Fizgig 기본 0.98). 스텝마다 가중치 평균을 갱신하고, 저장하는 LoRA 파일에는
#    평균 가중치를 쓴다(학습 자체는 원래 가중치로 계속). 이어하기 상태에는 평균이 남지 않아 이어하면 그때부터 다시 시작한다.
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import moss_sampling as ms  # noqa: E402
import moss_save  # noqa: E402
import moss_shuffle  # noqa: E402

_EPOCH = {"value": None}  # clean share·캡션 셔플에서 쓰는 sd-scripts 공유 에포크 값


def add_extra_arguments(parser):
    parser.add_argument("--save_from_pct", type=float, default=moss_save.DEFAULT_FROM_PCT,
                        help="에포크 LoRA 를 전체 에포크의 이 %% 지점부터 저장(0 = 처음부터). save_max_count 가 우선한다")
    parser.add_argument("--save_max_count", type=int, default=moss_save.DEFAULT_MAX_COUNT,
                        help="최종 파일을 포함해 최대 이 개수만 저장(넘으면 최신부터, 0 = 제한 없음)")
    parser.add_argument("--sample_preset", type=str, default="", choices=["", "turbo", "user", "fizgig", "official"],
                        help="학습 중 샘플 방식. turbo = 터보 LoRA + Euler 8스텝 CFG 2(Fizgig 빠른 미리보기), "
                             "user = dpmpp_2m+beta57 50스텝 CFG 3.5, fizgig = res_multistep+simple 30스텝 CFG 4"
                             "+SLG 22, official = sd-scripts 기본(Euler). 비우면 sd-scripts 기본")
    parser.add_argument("--muon_unbatched", action="store_true",
                        help="Muon 을 torch.optim.Muon 그대로(행렬마다 따로) 계산한다. 기본은 moss_muon.BatchedMuon"
                             "(모양이 같은 LoRA 행렬을 묶어 계산, 같은 식, rank 32 에서 스텝당 96 → 9ms)")
    parser.add_argument("--sample_negative", type=str, default=None,
                        help="프롬프트에 --n 이 없을 때 쓸 네거티브(ComfyUI 가중치 문법은 글만 남김)")
    parser.add_argument("--sample_dit", type=str, default=None,
                        help="샘플만 이 DiT(예: SFT)로 만든다. 학습 중인 LoRA 를 복사해 걸어 쓴다(INT8 로 올림, VRAM 약 6GB 더)")
    parser.add_argument("--sample_turbo_lora", type=str, default=None,
                        help="sample_preset turbo 에서 쓸 8스텝 터보 LoRA(lodestones/Kromoss 8-steps-dmd-lora.safetensors)")
    parser.add_argument("--sample_turbo_strength", type=float, default=0.6, help="터보 LoRA 세기(Fizgig 0.6)")
    parser.add_argument("--sample_look_lora", type=str, default=None,
                        help="sample_preset turbo 에서 함께 쓸 SFT 룩 LoRA(Fizgig moss_image_sft_look_lora.safetensors, 없으면 안 씀)")
    parser.add_argument("--sample_look_strength", type=float, default=1.5, help="SFT 룩 LoRA 세기(Fizgig 1.5)")
    parser.add_argument("--clean_share", type=str, default=None,
                        help="σ < 0.5 스텝 비율(%%). '40', '30:7:60:8:3'(Fizgig Slide 형식) 또는 'auto'·'auto:30:60'(에포크 수에 맞춤). "
                             "timestep_sampling 을 truncated_logit_normal 로 바꾼다")
    parser.add_argument("--slider_clean_share", type=str, default=None,
                        help="슬라이더 학습 전용 clean_share(같은 형식). 슬라이더는 clean_share 를 무시하고 이 값만 쓴다"
                             "(일반 LoRA 설정에서 따라온 값과 구분, Fizgig 슬라이더는 30:7:60:8:3)")
    parser.add_argument("--caption_shuffle", action="store_true",
                        help="캡션 셔플(moss_shuffle): 스타일 트리거·일반 트리거·나머지 태그를 각각 섞어 TE 캐시와 함께 쓴다")
    parser.add_argument("--caption_shuffle_mode", choices=moss_shuffle.MODES, default=moss_shuffle.DEFAULT_MODE,
                        help="epoch(기본): 에포크별 후보. fixed: 고정 후보 중 스텝마다 선택")
    parser.add_argument("--caption_shuffle_count", type=int, default=moss_shuffle.DEFAULT_COUNT,
                        help="고정 후보 모드의 캡션 후보 수")
    parser.add_argument("--caption_shuffle_keep", type=str, default="",
                        help="예전 설정의 맨 앞 트리거. 새 셔플 그룹을 지정하지 않았으면 일반 트리거로 쓴다")
    parser.add_argument("--caption_shuffle_style", type=str, default=None, help="셔플할 스타일 트리거(쉼표로 구분)")
    parser.add_argument("--caption_shuffle_general", type=str, default=None, help="셔플할 일반 트리거(쉼표로 구분)")
    parser.add_argument("--transparent_mask", type=str, default="on",
                        help="투명 배경 그림(PNG 알파)은 투명한 곳을 학습하지 않는다(on, 기본). sd-scripts 는 투명도를 버리고 "
                             "투명한 곳 밑에 남은 픽셀을 학습하므로, 투명도를 마스크로 쓴다. off 면 sd-scripts 그대로")
    parser.add_argument("--lora_ema_decay", type=float, default=0.0,
                        help="LoRA 가중치 EMA(0 = 끔, 예 0.98). 저장 파일에 평균 가중치를 쓴다")
    parser.add_argument("--masked_prior_weight", type=float, default=0.0,
                        help="마스크 밖은 LoRA 0 원본 예측에 맞춘다(0 = 끔, 1 = 안쪽과 같은 무게). masked_loss 와 마스크가 필요하다")


def patch_collator():
    """collator 가 받은 current_epoch 를 기억한다."""
    from library import dataset as ds
    if getattr(ds.collator_class, "_moss_patched", False):
        return
    orig = ds.collator_class.__init__

    def init(self, epoch, step, dataset):
        _EPOCH["value"] = epoch
        orig(self, epoch, step, dataset)

    ds.collator_class.__init__ = init
    ds.collator_class._moss_patched = True


_SAMPLE = {"args": None}


def _prompt_defaults(args):
    """(프리셋, 기본 네거티브). 학습 중 설정 파일을 고치면 refresh_sample_settings 가 args 를 바꾸므로 부를 때마다 본다."""
    preset = ms.PRESETS.get(getattr(args, "sample_preset", None) or "")
    negative = getattr(args, "sample_negative", None)
    if negative is None and preset is not None:
        negative = ms.DEFAULT_NEGATIVE if args.sample_preset in ("fizgig", "turbo") else ""
    return preset, ms.strip_weights(negative or "")


def patch_prompt_defaults(args):
    """샘플 프롬프트에 기본 네거티브·CFG·스텝을 넣는다(TE 캐시와 샘플이 같은 값을 보도록 load_prompts 자체를 바꾼다)."""
    _SAMPLE["args"] = args
    from library import sampling
    orig = sampling.load_prompts
    if getattr(orig, "_moss_defaults", False):
        return

    def load_prompts(path):
        preset, negative = _prompt_defaults(_SAMPLE["args"])
        out = []
        for p in orig(path):
            p = dict(p)
            if negative and not p.get("negative_prompt"):
                p["negative_prompt"] = negative
            elif p.get("negative_prompt"):
                p["negative_prompt"] = ms.strip_weights(p["negative_prompt"])
            if preset is not None and preset.get("force"):  # 터보: 줄마다 쓴 스텝·CFG 보다 우선(적은 스텝 전용 LoRA)
                p["scale"], p["sample_steps"] = preset["cfg"], preset["steps"]
            elif preset is not None:
                p.setdefault("scale", preset["cfg"])
                p.setdefault("sample_steps", preset["steps"])
            out.append(p)
        return out

    load_prompts._moss_defaults = True
    sampling.load_prompts = load_prompts


LIVE_SAMPLE_KEYS = ("sample_negative", "sample_preset", "sample_turbo_lora", "sample_look_lora")


# ---------------------------------------------------------------- 터보 샘플: 샘플 동안만 LoRA 덧붙이기
class _SampleAdapter:
    """Linear 의 지금 forward(학습 LoRA·INT8 포함) 출력에 LoRA 를 더한다. 합치지 않으므로 학습 가중치는 그대로다."""
    __slots__ = ("org", "items", "own")

    def __init__(self, org, items, own):
        self.org, self.items, self.own = org, items, own

    def __call__(self, x, *a, **k):
        y = self.org(x, *a, **k)
        for d, u, s in self.items:
            y = y + torch.nn.functional.linear(torch.nn.functional.linear(x.to(d.dtype), d), u).to(y.dtype) * s
        return y


def load_sample_lora(path, linear_names):
    """LoRA 파일을 이 DiT 의 Linear 이름으로 맞춰 CPU bf16 으로 읽는다: {이름: (down, up, alpha/rank)}.
    키 형식 두 가지: ComfyUI 'diffusion_model.layers.0.proj.q' 와 kohya 'lora_unet_layers_0_proj_q'.
    q·k·v 가 합쳐진 'proj.qkv' / '_proj_qkv' 는 이 DiT 가 q·k·v 를 나눠 두므로 lora_up 을 세 조각으로 나눈다(down 은 같이 씀)."""
    from safetensors.torch import load_file
    sd = load_file(path)
    kohya = {"lora_unet_" + n.replace(".", "_"): n for n in linear_names}
    out, unknown = {}, 0
    for k in sd:
        if not k.endswith(".lora_down.weight"):
            continue
        base = k[: -len(".lora_down.weight")]
        if base + ".lora_up.weight" not in sd:
            continue
        d, u = sd[k], sd[base + ".lora_up.weight"]
        sc = float(sd[base + ".alpha"]) / d.shape[0] if base + ".alpha" in sd else 1.0
        name = base[len("diffusion_model."):] if base.startswith("diffusion_model.") else kohya.get(base)
        if name in linear_names:
            out[name] = (d.to(torch.bfloat16), u.to(torch.bfloat16), sc)
            continue
        fused = None
        if base.startswith("diffusion_model.") and base.endswith(".qkv"):
            fused = base[len("diffusion_model."):-4]
            fused = [f"{fused}.{x}" for x in "qkv"]
        elif base.endswith("_qkv") and kohya.get(base[:-4] + "_q"):
            fused = [kohya.get(base[:-4] + "_" + x) for x in "qkv"]
        if fused and all(n in linear_names for n in fused) and u.shape[0] % 3 == 0:
            for n, part in zip(fused, u.chunk(3, 0)):
                out[n] = (d.to(torch.bfloat16), part.contiguous().to(torch.bfloat16), sc)
        else:
            unknown += 1
    return out, unknown


def refresh_sample_settings(args):
    """학습 중 설정 파일(--config_file)의 기본 네거티브·샘플 방식을 고치면 다음 샘플부터 쓴다(2026-11-11).
    명령줄로 준 값은 그대로 둔다. 바뀐 것만 알린다."""
    path = getattr(args, "config_file", None)
    if not path:
        return
    path = path if path.endswith(".toml") else path + ".toml"
    try:
        import tomllib
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except (OSError, ValueError):
        return
    flat = {}
    for k, v in data.items():
        if isinstance(v, dict):
            flat.update(v)
        else:
            flat[k] = v
    cli = " ".join(sys.argv)
    for k in LIVE_SAMPLE_KEYS:
        if f"--{k}" in cli:
            continue
        v = flat.get(k)
        v = None if v in (None, "") else str(v)
        if v != getattr(args, k, None):
            print(f"sample settings: {k} {getattr(args, k, None)!r} -> {v!r} (설정 파일)", flush=True)
            setattr(args, k, v)


def will_sample(args, epoch, steps):
    """moss_image_train_utils.sample_images 의 첫 조건과 같다(이번 호출에 샘플을 만드는지)."""
    if not getattr(args, "sample_prompts", None):
        return False
    if steps == 0:
        return bool(args.sample_at_first)
    if args.sample_every_n_steps is None and args.sample_every_n_epochs is None:
        return False
    if args.sample_every_n_epochs is not None:
        return epoch is not None and epoch % args.sample_every_n_epochs == 0
    return steps % args.sample_every_n_steps == 0 and epoch is None


# ---------------------------------------------------------------- clean-end share
AUTO_WARMUP = 0.1   # Fizgig 7 / 70 에포크
AUTO_EVERY = 3      # Fizgig 의 R(3에포크마다 낮은 비율 + 학습률 절반)


def resolve_share(text, epochs):
    """'auto'·'auto:처음:나중' 을 에포크 수에 맞는 '처음:에포크:나중:끝:R' 로 바꾼다. 다른 값은 그대로.
    화면(moss_ui.html 의 cleanShareParts)도 같은 식으로 미리 보여 준다."""
    t = str(text or "").replace(" ", "").lower()
    if not t.startswith("auto"):
        return text
    rest = [float(x) for x in t[4:].lstrip(":").split(":") if x != ""]
    if len(rest) > 2 or any(not 0 < v < 100 for v in rest):
        raise ValueError(f"clean_share '{text}': 'auto' 또는 'auto:30:60' 형식(비율은 0~100)")
    first, later = (rest + [30.0, 60.0][len(rest):])[:2]
    until = max(1, int(int(epochs) * AUTO_WARMUP + 0.5))
    return f"{first:g}:{until}:{later:g}:{until + 1}:{AUTO_EVERY}"


def parse_share(text):
    parts = [float(x) for x in str(text).replace(" ", "").split(":") if x != ""]
    if len(parts) not in (1, 3, 4, 5):
        raise ValueError(f"clean_share '{text}': '40' 또는 '30:7:60[:8[:3]]' 형식")
    return parts


def share_half(parts, epoch):
    """이 에포크가 R 주기의 '낮은 비율' 에포크(학습률 절반)인지."""
    if len(parts) != 5:
        return False
    first, until, rest, end, every = parts[0], parts[1], parts[2], parts[3], int(parts[4])
    e = int(epoch)
    return every > 0 and e > end and (e - end) % every == 0 and first != rest


def share_for_epoch(parts, epoch):
    if len(parts) == 1:
        return parts[0]
    first, until, rest = parts[:3]
    end = parts[3] if len(parts) >= 4 else until
    every = int(parts[4]) if len(parts) == 5 else 0
    e = int(epoch)
    if e <= until:
        share = first
    elif e >= end:
        share = rest
    else:
        share = first + (rest - first) * (e - until) / max(1e-9, (end - until))
    if every > 0 and e > end:
        share = min(first, rest) if (e - end) % every == 0 else max(first, rest)
    return share


def mu_for_share(pct, std, cutoff, uniform_ratio):
    """truncated logit-normal(sd-scripts sample_truncated_logit_normal_indices)에서 σ < 0.5 비율이 pct% 가 되는 평균.
    σ = sigmoid(−Z)/cutoff, Z ≥ −logit(cutoff) 로 자름. σ < 0.5 ⇔ Z > −logit(0.5·cutoff). 균등 부분은 절반이 σ < 0.5."""
    nd = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    z_a = -math.log(0.5 * cutoff / (1 - 0.5 * cutoff))
    z_lo = -math.log(cutoff / (1 - cutoff)) if cutoff < 1.0 else -1e9
    want = (pct / 100.0 - uniform_ratio * 0.5) / max(1e-9, 1.0 - uniform_ratio)
    want = min(max(want, 1e-4), 1 - 1e-4)
    lo, hi = -12.0, 12.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        tail = 1 - nd((z_lo - mid) / std)
        share = (1 - nd((z_a - mid) / std)) / max(tail, 1e-300)
        if share < want:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def save_keep(args):
    """저장할 에포크 번호 집합(최종 제외). 에포크 수를 모르면 None = 거르지 않음."""
    if not hasattr(args, "_moss_save_keep"):
        keep = moss_save.save_epochs(getattr(args, "max_train_epochs", None), getattr(args, "save_every_n_epochs", None),
                                     getattr(args, "save_from_pct", None), getattr(args, "save_max_count", None))
        args._moss_save_keep = None if keep is None else set(keep)
    return args._moss_save_keep


def patch_epoch_state_save():
    """에포크 상태 폴더(save_state)도 저장 범위 안의 에포크만. train_network 가 호출 때 모듈 속성으로 찾으므로 바꿔 끼운다."""
    import library.checkpoint_io as cio
    if getattr(cio.save_and_remove_state_on_epoch_end, "_moss", False):
        return
    orig = cio.save_and_remove_state_on_epoch_end

    def wrapped(args, accelerator, epoch_no):
        keep = save_keep(args)
        if keep is not None and epoch_no < (getattr(args, "max_train_epochs", 0) or 0) and epoch_no not in keep:
            return
        return orig(args, accelerator, epoch_no)
    wrapped._moss = True
    cio.save_and_remove_state_on_epoch_end = wrapped


class ExtrasMixin:
    """moss_train 의 트레이너에 섞는다. 슬라이더 믹스인보다 뒤(MRO 상 아래)에 둔다."""

    def _save_model(self, *, args, ckpt_name, **kw):
        # 에포크 LoRA 저장 범위(moss_save, 2026-11-14): 범위 밖 에포크 파일은 쓰지 않는다. 최종·스텝 저장은 그대로
        e = moss_save.epoch_of_ckpt(ckpt_name)
        keep = save_keep(args)
        if e is not None and keep is not None and e not in keep:
            print(f"save: 에포크 {e} 는 저장 범위 밖이라 건너뜁니다", flush=True)
            return
        return super()._save_model(args=args, ckpt_name=ckpt_name, **kw)

    _extras_net = None
    _extras_epoch = None
    _sample_dit = None
    _caption_shuffle_epochs = None

    def assert_extra_args(self, args, train_dataset_group, val_dataset_group):
        if getattr(args, "caption_shuffle", False) and args.caption_shuffle_mode == moss_shuffle.DEFAULT_MODE:
            # 에포크별 후보의 세 그룹은 MOSS 가 섞는다. sd-scripts 셔플이 그 순서를 다시 바꾸지 않게 한다.
            for group in (train_dataset_group, val_dataset_group):
                if hasattr(group, "datasets"):
                    for dataset in moss_shuffle.caption_datasets(group):
                        for subset in dataset.subsets:
                            subset.shuffle_caption = False
        return super().assert_extra_args(args, train_dataset_group, val_dataset_group)

    def cache_text_encoder_outputs_if_needed(self, args, accelerator, unet, vae, text_encoders, dataset, weight_dtype):
        if (getattr(args, "caption_shuffle", False) and args.caption_shuffle_mode == moss_shuffle.DEFAULT_MODE
                and hasattr(dataset, "datasets")):
            if self._caption_shuffle_epochs is None:
                self._caption_shuffle_epochs = moss_shuffle.training_epochs(
                    args.max_train_epochs, args.max_train_steps, len(dataset), accelerator.num_processes,
                    args.gradient_accumulation_steps,
                )
                if args.cache_text_encoder_outputs_to_disk and args.skip_cache_check:
                    from library import strategy_base
                    strategy = strategy_base.TextEncoderOutputsCachingStrategy.get_strategy()
                    strategy.skip_disk_cache_validity_check = False
                    print("caption shuffle: TE 디스크 캐시의 후보 수와 캡션 순서를 확인합니다", flush=True)
            count = self._caption_shuffle_epochs
            style, keep = caption_shuffle_groups(args)
            moss_shuffle.expand_group(dataset, keep, count, style=style, log=lambda m: print(m, flush=True), epoch=_EPOCH["value"])
            args.caption_shuffle_info = f"mode=epoch, count={count}, style={','.join(style)}, keep={','.join(keep)}"
        return super().cache_text_encoder_outputs_if_needed(args, accelerator, unet, vae, text_encoders, dataset, weight_dtype)

    def update_metadata(self, metadata, args):
        super().update_metadata(metadata, args)
        for k in ("clean_share", "clean_share_auto", "masked_prior_weight", "lora_ema_decay", "sample_preset",
                  "caption_shuffle_info"):
            v = getattr(args, k, None)
            if v not in (None, "", 0, 0.0):
                metadata[f"ss_{k}"] = str(v)

    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        self._extras_net = network  # 학습 전 샘플(sample_at_first)도 이 네트워크를 쓴다
        return super().post_process_network(args, accelerator, network, text_encoders, unet)

    # ── compile: 시퀀스 축 동적 + 샘플 때는 컴파일 안 한 블록 ──
    def prepare_unet_with_accelerator(self, args, accelerator, unet):
        model = super().prepare_unet_with_accelerator(args, accelerator, unet)
        if getattr(args, "compile", False):
            dit = accelerator.unwrap_model(model)
            n = 0
            for layer in dit.layers:
                if hasattr(layer, "_orig_mod"):  # sd-scripts compile_transformer 가 OptimizedModule 로 바꿔 둔 블록
                    layer.register_forward_pre_hook(_seq_dynamic)
                    n += 1
            print(f"compile: 블록 {n}개의 시퀀스 축을 동적으로 표시", flush=True)
        return model

    @staticmethod
    def _uncompiled_layers(dit):
        """샘플 동안 컴파일 안 한 블록으로 바꾼다(추론 모양·grad 모드·터보 LoRA 로 다시 컴파일하지 않게). 되돌릴 목록을 준다."""
        layers = getattr(dit, "layers", None)
        if layers is None or not any(hasattr(l, "_orig_mod") for l in layers):
            return None
        saved = list(layers)
        for i, l in enumerate(saved):
            if hasattr(l, "_orig_mod"):
                layers[i] = l._orig_mod
        return saved

    # ── 에포크마다 clean share ──
    def on_step_start(self, args, accelerator, network, text_encoders, unet, batch, weight_dtype, is_train=True):
        self._extras_net = network
        if getattr(args, "clean_share", None) and _EPOCH["value"] is not None:
            e = int(_EPOCH["value"].value)
            if e != self._extras_epoch:
                self._extras_epoch = e
                pct = share_for_epoch(parse_share(args.clean_share), e)
                args.logit_mean = mu_for_share(pct, float(args.logit_std), float(args.timestep_cutoff),
                                               float(args.timestep_uniform_ratio))
                self._lr_half = share_half(parse_share(args.clean_share), e)
                print(f"clean share: epoch {e} -> {pct:g}% of steps at sigma < 0.5 (logit_mean {args.logit_mean:.3f})"
                      + (" at half the learning rate" if self._lr_half else ""), flush=True)
            if getattr(self, "_lr_half", False):
                # 스케줄러가 스텝마다 학습률을 다시 정하므로(LambdaLR), 이번 스텝 직전에 절반으로 낮춘다(쌓이지 않음)
                for opt in getattr(accelerator, "_optimizers", []):
                    for g in getattr(opt, "optimizer", opt).param_groups:
                        g["lr"] = g["lr"] * 0.5
        return super().on_step_start(args, accelerator, network, text_encoders, unet, batch, weight_dtype, is_train)

    # ── 마스크 밖 원본 유지 ──
    def get_noise_pred_and_target(self, args, accelerator, noise_scheduler, latents, batch, text_encoder_conds, unet, network,
                                  weight_dtype, train_unet, is_train=True):
        wp = float(getattr(args, "masked_prior_weight", 0.0) or 0.0)
        mask = None
        if wp > 0 and not getattr(args, "slider", False):
            if batch.get("conditioning_images") is not None:
                mask = batch["conditioning_images"][:, 0:1].float() / 2 + 0.5
            elif batch.get("alpha_masks") is not None:
                mask = batch["alpha_masks"].float().unsqueeze(1)
        if mask is None:
            return super().get_noise_pred_and_target(args, accelerator, noise_scheduler, latents, batch, text_encoder_conds,
                                                     unet, network, weight_dtype, train_unet, is_train)
        calls = []

        class Rec:  # 학습 순전파의 입력을 그대로 기억한다(같은 노이즈·σ 로 원본 예측을 다시 구하려고)
            def __init__(self, m):
                self._m = m

            def __call__(self, *a, **k):
                calls.append((a, k))
                return self._m(*a, **k)

            def __getattr__(self, name):
                return getattr(self._m, name)

        pred, target, timesteps, weighting = super().get_noise_pred_and_target(
            args, accelerator, noise_scheduler, latents, batch, text_encoder_conds, Rec(unet), network, weight_dtype, train_unet,
            is_train)
        net = accelerator.unwrap_model(network)
        a, k = calls[0]
        a = tuple(x.detach() if torch.is_tensor(x) else x for x in a)
        net.set_multiplier(0.0)
        try:
            with torch.no_grad(), accelerator.autocast():
                base = unet(*a, **k)
        finally:
            net.set_multiplier(1.0)
        base = base.float() if base.shape == target.shape else base.float().reshape(target.shape)
        m = torch.nn.functional.interpolate(mask.to(target.device), size=target.shape[-2:], mode="area")
        new_target = m * target.float() + (1 - m) * base
        w = m + wp * (1 - m)
        sw = w.sqrt()
        batch["alpha_masks"] = None  # 학습 루프가 마스크를 한 번 더 곱하지 않게
        self._extras_skip_mask = True
        return pred * sw.to(pred.dtype), new_target * sw, timesteps, weighting

    def process_batch(self, batch, text_encoders, unet, network, vae, noise_scheduler, vae_dtype, weight_dtype, accelerator, args,
                      *rest, **kw):
        self._extras_skip_mask = False
        prior = float(getattr(args, "masked_prior_weight", 0.0) or 0.0) > 0 and not getattr(args, "slider", False)
        if not prior:
            return super().process_batch(batch, text_encoders, unet, network, vae, noise_scheduler, vae_dtype, weight_dtype,
                                         accelerator, args, *rest, **kw)
        saved = args.masked_loss
        args.masked_loss = False  # 마스크는 get_noise_pred_and_target 에서 이미 반영한다
        try:
            return super().process_batch(batch, text_encoders, unet, network, vae, noise_scheduler, vae_dtype, weight_dtype,
                                         accelerator, args, *rest, **kw)
        finally:
            args.masked_loss = saved

    # ── LoRA EMA ──
    _ema = None

    def _ema_update(self, accelerator, args):
        d = float(getattr(args, "lora_ema_decay", 0.0) or 0.0)
        if d <= 0 or self._extras_net is None:
            return
        net = accelerator.unwrap_model(self._extras_net)
        params = [(n, p) for n, p in net.named_parameters() if p.requires_grad]
        if self._ema is None:
            self._ema = {n: p.detach().float().clone() for n, p in params}
            orig = net.save_weights
            ema = self._ema

            def save_weights(file, dtype, metadata, *a, **k):
                backup = {n: p.detach().clone() for n, p in params}
                with torch.no_grad():
                    for n, p in params:
                        p.copy_(ema[n].to(p.dtype))
                try:
                    md = dict(metadata or {})
                    md["ss_lora_ema_decay"] = str(d)
                    return orig(file, dtype, md, *a, **k)
                finally:
                    with torch.no_grad():
                        for n, p in params:
                            p.copy_(backup[n])

            net.save_weights = save_weights
            print(f"lora ema: decay {d}, saved files use the averaged weights", flush=True)
            return
        with torch.no_grad():
            for n, p in params:
                self._ema[n].mul_(d).add_(p.detach().float(), alpha=1 - d)

    # ── 샘플: 프리셋 샘플러, SFT DiT ──
    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):
        if epoch is None:  # 스텝이 끝날 때마다 불린다
            self._ema_update(accelerator, args)
        sampling_now = will_sample(args, epoch, global_step)
        if sampling_now:
            refresh_sample_settings(args)
            self._encode_new_prompts(args, accelerator)
        preset = ms.PRESETS.get(getattr(args, "sample_preset", "") or "")
        sdit_path = getattr(args, "sample_dit", None)
        if preset is not None and preset.get("turbo"):
            sdit_path = None  # 터보 LoRA 는 Base 에서 증류한 것이라 학습 모델로 그린다(Fizgig: SFT 에는 쓰지 않음)
        compiled = self._uncompiled_layers(accelerator.unwrap_model(unet)) if sampling_now else None
        turbo_hooks = self._attach_sample_loras(args, accelerator, unet) if sampling_now and preset and preset.get("turbo") else None
        try:
            return self._sample_images_preset(accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet,
                                              preset, sdit_path)
        finally:
            if turbo_hooks:
                self._detach_sample_loras(turbo_hooks)
            if compiled is not None:
                layers = accelerator.unwrap_model(unet).layers
                for i, l in enumerate(compiled):
                    layers[i] = l

    def _attach_sample_loras(self, args, accelerator, unet):
        """터보 LoRA(+ SFT 룩 LoRA)를 학습 DiT 의 Linear 에 샘플 동안만 덧붙인다. 파일은 CPU 에 기억하고 샘플 때만 GPU 로."""
        import time
        dit = accelerator.unwrap_model(unet)
        names = {n for n, m in dit.named_modules() if isinstance(m, torch.nn.Linear)}
        specs = [(getattr(args, "sample_turbo_lora", None), float(getattr(args, "sample_turbo_strength", 0.6) or 0.0), "turbo"),
                 (getattr(args, "sample_look_lora", None), float(getattr(args, "sample_look_strength", 1.5) or 0.0), "look")]
        cache = getattr(self, "_sample_lora_cache", None)
        if cache is None:
            cache = self._sample_lora_cache = {}
        dev = accelerator.device
        per_module = {}
        for path, strength, label in specs:
            if not path or strength == 0.0:
                continue
            if not os.path.isfile(path):
                print(f"sample turbo: {label} LoRA 파일이 없습니다 — 빼고 그립니다: {path}", flush=True)
                continue
            st = os.stat(path)
            key = (os.path.abspath(path), st.st_mtime_ns, st.st_size)
            if key not in cache:
                t = time.time()
                cache[key] = load_sample_lora(path, names)
                print(f"sample turbo: {label} {os.path.basename(path)} {len(cache[key][0])} Linears"
                      f"{f', 맞지 않는 모듈 {cache[key][1]}개' if cache[key][1] else ''} ({time.time() - t:.0f}s, CPU 에 기억)", flush=True)
            for n, (d, u, sc) in cache[key][0].items():
                per_module.setdefault(n, []).append((d.to(dev, non_blocking=True), u.to(dev, non_blocking=True), sc * strength))
        hooks = []
        for n, items in per_module.items():
            m = dit.get_submodule(n)
            a = _SampleAdapter(m.forward, items, "forward" in m.__dict__)
            m.forward = a
            hooks.append((m, a))
        if hooks:
            print(f"sample turbo: LoRA {len(hooks)} Linears 에 덧붙임(샘플 동안만)", flush=True)
        return hooks

    @staticmethod
    def _detach_sample_loras(hooks):
        for m, a in hooks:
            if m.__dict__.get("forward") is a:
                if a.own:
                    m.forward = a.org
                else:
                    del m.forward
            a.items = []
        torch.cuda.empty_cache()

    def _sample_images_preset(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet, preset, sdit_path):
        if preset is None and not sdit_path:
            return super().sample_images(accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet)
        from library import moss_image_train_utils as utils

        orig_gen = utils.generate_latents
        train_net = accelerator.unwrap_model(self._extras_net) if self._extras_net is not None else getattr(self, "slider_network", None)
        sample = {"dit": None, "net": None, "baseline": None}

        def gen(dit, pos, pos_mask, neg, neg_mask, height, width, steps, scale, flow_shift, skip_layers, seed, dev, show_progress=True):
            use = dit
            if sdit_path:
                use = self._get_sample_dit(args, dev)
                if sample["net"] is None:
                    sample["net"], sample["baseline"] = self._attach_lora(use, train_net, dev)
                if train_net is not None and getattr(train_net, "unet_loras", None):
                    sample["net"].set_multiplier(float(train_net.unet_loras[0].multiplier))
            g = torch.Generator(device="cpu")
            if seed is not None:
                g.manual_seed(seed)
            lat = torch.randn((1, use.latent_channels, height // 16, width // 16), generator=g, dtype=torch.float32).to(dev)
            sched = preset["scheduler"] if preset else "official"
            smp = preset["sampler"] if preset else "euler"
            slg = (22,) if (preset and preset["slg"]) else ()
            official_skip = tuple(skip_layers) if (not preset or getattr(args, "sample_preset", "") == "official") else ()  # 터보도 건너뛰기 없음
            pos = pos.to(dev, dtype=use.dtype)
            neg_ = neg.to(dev, dtype=use.dtype) if neg is not None else None
            den = ms.make_denoiser(use, pos, pos_mask.to(dev), neg_, neg_mask.to(dev) if neg_mask is not None else None, scale,
                                   flow_shift, slg_layers=slg, slg_start=0.0, slg_end=0.5, official_skip=official_skip)
            with torch.no_grad():
                return ms.sample(den, lat, ms.get_sigmas(sched, steps, flow_shift), smp)

        utils.generate_latents = gen
        try:
            return super().sample_images(accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet)
        finally:
            utils.generate_latents = orig_gen
            if sample["dit"] is not None or sample["net"] is not None:
                self._detach_lora(self._sample_dit, sample["baseline"])
                sample["net"] = None
            if self._sample_dit is not None:
                self._sample_dit.to("cpu")
                torch.cuda.empty_cache()

    def _encode_new_prompts(self, args, accelerator):
        """학습 중 샘플 프롬프트 파일에 새 문장(긍정·부정)이 생기면 샘플 직전에 텍스트 인코더를 CPU 에 잠시 올려 인코딩한다
        (2026-11-11). 학습 시작 때의 캐시(sample_prompts_te_outputs)와 같은 전략으로 인코딩해 캐시에 더한다. 텍스트 인코더는
        학습 중에 지워져 있어(train_network: is_text_encoder_not_needed_for_training) 이것이 없으면 새 문장은 건너뛴다."""
        if not getattr(args, "cache_text_encoder_outputs", False) or not os.path.isfile(args.sample_prompts or ""):
            return
        from library import sampling, strategy_base, moss_image_utils
        cache = getattr(self, "sample_prompts_te_outputs", None)
        if cache is None:
            cache = self.sample_prompts_te_outputs = {}
        try:
            prompts = sampling.load_prompts(args.sample_prompts)
        except Exception as e:
            print(f"sample prompts: 파일을 읽지 못했습니다 ({e})", flush=True)
            return
        need = []
        for d in prompts:
            for p in (d.get("prompt", ""), d.get("negative_prompt", "")):
                if p not in cache and p not in need and (p or d.get("prompt")):
                    need.append(p)
        if not need:
            return
        import gc
        import time
        t = time.time()
        print(f"sample prompts: 새 문장 {len(need)}개를 인코딩합니다(텍스트 인코더를 CPU 에 잠시 올림)", flush=True)
        te = moss_image_utils.load_text_encoder(
            args.text_encoder, dtype=torch.bfloat16, device="cpu", disable_mmap=args.disable_mmap_load_safetensors,
            convrot_int8=args.convrot_int8_text_encoder, convrot_int8_bwd=args.convrot_int8_bwd, calc_device=accelerator.device)
        te.eval()
        tok = strategy_base.TokenizeStrategy.get_strategy()
        enc = strategy_base.TextEncodingStrategy.get_strategy()
        try:
            with torch.no_grad():
                for p in need:
                    cache[p] = enc.encode_tokens(tok, [te], tok.tokenize(p))
        finally:
            del te
            gc.collect()
            torch.cuda.empty_cache()
        print(f"sample prompts: 인코딩 {time.time() - t:.0f}초", flush=True)

    def _get_sample_dit(self, args, dev):
        if self._sample_dit is None:
            from library import moss_image_utils
            print(f"sample DiT: loading {args.sample_dit} (INT8, CPU)", flush=True)
            self._sample_dit = moss_image_utils.load_moss_image_model(
                dev, args.sample_dit, args.attn_mode, True, "cpu", torch.bfloat16, convrot_int8=True)
            self._sample_dit.eval().requires_grad_(False)
        self._sample_dit.to(dev)
        return self._sample_dit

    @staticmethod
    def _attach_lora(dit, train_net, dev):
        from networks import lora_generic
        baseline = {id(m) for m in dit.modules() if "forward" in m.__dict__}
        sd = {k: v.detach().cpu().clone() for k, v in train_net.state_dict().items()}  # alpha 는 CPU 여야 한다(lora_generic 이 numpy 로 읽음)
        net, sd = lora_generic.create_network_from_weights(1.0, None, None, [], dit, weights_sd=sd)
        net.apply_to([], dit, apply_text_encoder=False, apply_unet=True)
        net.load_state_dict(sd, strict=False)
        net.to(dev, dtype=torch.bfloat16).eval()
        return net, baseline

    @staticmethod
    def _detach_lora(dit, baseline):
        if dit is None or baseline is None:
            return
        for m in dit.modules():
            if "forward" in m.__dict__ and id(m) not in baseline:
                del m.forward


def _patch_sympy_mod():
    """torch 의 sympy Mod 가 음수 피제수에서 assert 로 죽는 버그(GPU-005, inductor 타일링이 음수 시험값을 넣음)를 피한다.
    Fizgig(modules/compile_util.py)와 같은 방법: torch 함수를 감싸, 두 값이 모두 숫자이고 법이 1 이상인데 왼쪽이 음수일
    때만 답을 준다. 다른 assert 는 그대로 올린다. torch 가 고치면 감싼 쪽은 불리지 않는다."""
    try:
        from torch.utils._sympy.functions import Mod
    except Exception:
        return False
    if getattr(Mod, "_moss_patched", False):
        return True
    original = Mod.eval.__func__ if hasattr(Mod.eval, "__func__") else Mod.eval

    def eval_allowing_negatives(cls, p, q):
        try:
            return original(cls, p, q)
        except AssertionError:
            if getattr(q, "is_Number", False) and getattr(p, "is_Number", False) and q >= 1 and p < 0:
                return p % q
            raise

    Mod.eval = classmethod(eval_allowing_negatives)
    Mod._moss_patched = True
    return True


def patch_flash_attention_for_compile():
    """compile 할 때 MOSS 어텐션(flash, split 아님)을 항상 varlen 한 경로로, 길이는 텐서 모양에서 읽게 한다.
    sd-scripts library.attention 은 ① 배치의 캡션 길이가 모두 같으면 잘라내는 다른 경로를 타고 ② max_seqlen 을 파이썬
    정수로 넘겨, 컴파일한 블록이 이 둘 때문에 한 번씩 더(42블록, 각 약 2분) 컴파일했다(2026-11-11 측정). 계산은 같다:
    varlen 은 패딩 칸을 따로 짧은 구간으로 계산하고(길이 0 구간도 됨), 잘라내기 경로와 결과가 같다(유효 토큰)."""
    from library import attention as A
    from library import moss_image_models as M
    if getattr(M, "_moss_attn_patched", False):
        return
    orig = A.attention

    def attention(qkv_or_q, k=None, v=None, attn_params=None, drop_rate=0.0):
        p = attn_params
        if (p is None or p.attn_mode != "flash" or p.split_attn or p.cu_seqlens is None or A.flash_attn_varlen_func is None
                or not isinstance(qkv_or_q, list)):
            return orig(qkv_or_q, k, v, attn_params=attn_params, drop_rate=drop_rate)
        q, k, v = qkv_or_q
        qkv_or_q.clear()
        b, n = q.shape[0], q.shape[1]
        x = A.flash_attn_varlen_func(q.reshape(b * n, *q.shape[2:]), k.reshape(b * n, *k.shape[2:]),
                                     v.reshape(b * n, *v.shape[2:]), p.cu_seqlens, p.cu_seqlens, n, n, drop_rate)
        return x.reshape(b, n, -1)

    M.attention = attention  # moss_attention 이 부르는 이름만 바꾼다(텍스트 블록도 같은 함수)
    M._moss_attn_patched = True


def _ensure_msvc():
    """inductor 는 Windows 에서 cl.exe 로 작은 C++ 코드를 만든다. PATH 에 없으면 vcvars64.bat 이 만드는 환경을 가져온다
    (개발자 프롬프트가 하는 일, Fizgig families/compile.py 와 같은 방법). 실패해도 학습은 계속하고 알리기만 한다."""
    import shutil
    import subprocess
    if os.name != "nt" or shutil.which("cl"):
        return
    vcvars = None
    for pf in (os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), os.environ.get("ProgramFiles", r"C:\Program Files")):
        for year in ("2022", "2019"):
            for ed in ("BuildTools", "Community", "Professional", "Enterprise"):
                p = os.path.join(pf, "Microsoft Visual Studio", year, ed, "VC", "Auxiliary", "Build", "vcvars64.bat")
                if vcvars is None and os.path.isfile(p):
                    vcvars = p
    if not vcvars:
        print("compile: MSVC(cl.exe)를 찾지 못했습니다 — Visual Studio Build Tools 2022 의 C++ 개발 도구가 필요합니다", flush=True)
        return
    try:
        out = subprocess.run(f'"{vcvars}" >nul && set', shell=True, capture_output=True, text=True, timeout=120,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        for line in out.stdout.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                os.environ[k] = v
        print(f"compile: MSVC 환경을 불러왔습니다({vcvars})" if shutil.which("cl") else "compile: vcvars64 를 실행했지만 cl.exe 가 없습니다",
              flush=True)
    except Exception as e:
        print(f"compile: vcvars64 실행 실패: {e}", flush=True)


COMPILE_CACHE_LIMIT = 8192  # 블록 42개가 모양·모드마다 칸을 하나씩 쓴다. 기본 8(누적 256)이면 몇 모양 뒤 조용히 eager 로 돌아간다


def prepare_compile(args):
    """--compile 을 켤 때(2026-11-11, Fizgig 방식): 캐시 한도를 올리고 sympy 버그를 피한다. 블록 입력의 시퀀스 축은
    ExtrasMixin.prepare_unet_with_accelerator 가 동적으로 표시한다(새 캡션 길이·그림 크기마다 다시 컴파일하지 않게)."""
    if not getattr(args, "compile", False):
        return
    import torch._dynamo
    lim = max(COMPILE_CACHE_LIMIT, int(getattr(args, "compile_cache_size_limit", None) or 0))
    args.compile_cache_size_limit = lim  # sd-scripts compile_transformer 가 이 값으로 cache_size_limit 을 정한다
    torch._dynamo.config.cache_size_limit = lim
    if hasattr(torch._dynamo.config, "accumulated_cache_size_limit"):
        torch._dynamo.config.accumulated_cache_size_limit = max(lim, torch._dynamo.config.accumulated_cache_size_limit)
    # LoRA 배율(module.multiplier, 파이썬 float)을 바꿔 가며 같은 블록을 부르면(DOP·마스크 밖 원본 유지·슬라이더) dynamo 가 그 float 을
    # CPU 0차원 텐서 입력으로 바꿔 다시 컴파일하고, inductor 가 CPU 코드를 만들며 cl.exe 언어를 검사하다(한국어 MSVC)
    # UnicodeDecodeError 로 죽는다(2026-11-12 실측, DOP). float 을 값마다 고정하면 배율 0·1 그래프가 하나씩 생길 뿐이다
    torch._dynamo.config.specialize_float = True
    _ensure_msvc()
    patched = _patch_sympy_mod()
    patch_flash_attention_for_compile()
    print(f"compile: cache_size_limit {lim}, 시퀀스 축 동적, sympy Mod 음수 assert 우회 {'적용' if patched else '실패'}, "
          "flash 어텐션은 항상 varlen", flush=True)


def _seq_dynamic(_module, args):
    """컴파일한 블록의 forward pre-hook(그래프 밖에서 실행): [B, L, D] 입력의 길이 L 과 같은 축(RoPE 등)을 동적으로 표시한다.
    dynamo 의 자동 동적 모양은 한 번 다시 컴파일한 뒤에야 배우므로 처음부터 표시한다(Fizgig families/compile.py 와 같음)."""
    x = args[0] if args else None
    if not (torch.is_tensor(x) and x.dim() == 3):
        return
    b, n = x.shape[0], x.shape[1]
    for t in args:
        if torch.is_tensor(t):
            if t.dim() >= 1 and t.shape[0] == b:  # 배치 축(버킷 끝의 남는 배치)
                torch._dynamo.maybe_mark_dynamic(t, 0)
            for d in range(1, t.dim()):
                if t.shape[d] == n:
                    torch._dynamo.maybe_mark_dynamic(t, d)
        elif torch.is_tensor(getattr(t, "cu_seqlens", None)):  # AttentionParams: 길이 2B + 1
            torch._dynamo.maybe_mark_dynamic(t.cu_seqlens, 0)


def transparent_mask_on(args):
    """transparent_mask: 키가 없으면 켬. 끄려면 "off"(화면 스위치)·false."""
    v = getattr(args, "transparent_mask", "on")
    return not (v is False or str(v).strip().lower() in ("off", "false", "0", "no"))


def caption_shuffle_groups(args):
    return moss_shuffle.resolve_groups(args.caption_shuffle_style, args.caption_shuffle_general,
                                       args.caption_shuffle_keep, args.dop_trigger)


def prepare_args(args):
    """학습 전에 인자를 고친다."""
    prepare_compile(args)
    keep = save_keep(args)
    if keep is not None and getattr(args, "save_every_n_epochs", None):
        patch_epoch_state_save()
        k = sorted(keep)
        print(f"save: 에포크 LoRA {f'{k[0]}~{k[-1]}' if k else '없음'} + 최종({args.max_train_epochs}) = {len(k) + 1}개 "
              f"(시작 {args.save_from_pct:g}%, 최대 {args.save_max_count})", flush=True)
    elif getattr(args, "save_every_n_epochs", None):
        print("save: max_train_epochs 가 없어 에포크 저장 범위를 적용하지 않습니다", flush=True)
    if getattr(args, "slider", False):
        # 기본 설정 파일에 clean_share 가 들어 있어(2026-11-10) 슬라이더로 바꾸면 따라온다 → 슬라이더는 clean_share 를 쓰지 않고
        # 슬라이더 전용 slider_clean_share 만 쓴다(2026-11-15, Fizgig 방식 프리셋: Fizgig 슬라이더도 Slide 30:7:60:8:3)
        if getattr(args, "clean_share", None):
            print(f"clean share: slider 학습이라 clean_share 는 무시합니다 ({args.clean_share})", flush=True)
        args.clean_share = getattr(args, "slider_clean_share", None) or None
        if args.clean_share:
            print(f"clean share: slider_clean_share {args.clean_share}", flush=True)
    if getattr(args, "slider", False) and (args.masked_loss or float(getattr(args, "masked_prior_weight", 0.0) or 0.0) > 0):
        # 마스크 학습은 일반 LoRA 용(학습 폴더의 .msk). 그림 쌍은 쌍 마스크로 묶음 그림을 합성한다(2026-11-13)
        print("masked loss: slider 학습이라 끕니다", flush=True)
        args.masked_loss = False
        args.masked_prior_weight = 0.0
    if getattr(args, "clean_share", None):
        if str(args.clean_share).strip().lower().startswith("auto"):
            epochs = getattr(args, "max_train_epochs", None)
            if not epochs:
                print("clean share: max_train_epochs 가 없어 auto 를 10에포크 기준으로 계산합니다", flush=True)
                epochs = 10
            resolved = resolve_share(args.clean_share, epochs)
            print(f"clean share: {args.clean_share} -> {resolved} ({epochs} epochs)", flush=True)
            args.clean_share_auto = f"{args.clean_share} ({epochs} epochs)"
            args.clean_share = resolved
        parse_share(args.clean_share)
        if args.timestep_sampling != "truncated_logit_normal":
            print(f"clean share: timestep_sampling {args.timestep_sampling} -> truncated_logit_normal", flush=True)
            args.timestep_sampling = "truncated_logit_normal"
        patch_collator()
    if float(getattr(args, "masked_prior_weight", 0.0) or 0.0) > 0 and not args.masked_loss:
        print("masked prior: masked_loss 를 켭니다(마스크가 필요)", flush=True)
        args.masked_loss = True
    if getattr(args, "dataset_config", None) and getattr(args, "cache_latents_to_disk", False):  # 그림 쌍은 캐시를 끄므로 해당 없음
        # 그림을 고쳐도 sd-scripts 는 크기가 같으면 예전 latent 캐시를 쓴다 → 그림보다 오래된 캐시를 지워 다시 만들게 한다(moss_cache.py)
        import moss_cache
        moss_cache.remove_stale_latents(args.dataset_config, log=lambda m: print(m, flush=True))
    if getattr(args, "dataset_config", None) and not getattr(args, "slider", False):
        # 마스크 폴더(conditioning_data_dir): 마스크가 없는 그림은 그림 전체, 크기가 다른 마스크는 맞춰서, 마스크 학습이
        # 꺼져 있으면 마스크 폴더를 빼고 쓰도록 .mask_fill/ 에 학습용 마스크 폴더와 데이터셋 설정을 만든다(moss_masks.py)
        import moss_masks
        new, n_alpha = moss_masks.prepare_dataset(args.dataset_config, bool(args.masked_loss), args.output_name,
                                                  log=lambda m: print(m, flush=True), alpha=transparent_mask_on(args))
        if new:
            print(f"masks: 데이터셋 설정 {args.dataset_config} -> {new}", flush=True)
            args.dataset_config = new
        if n_alpha and not args.masked_loss:  # 투명 배경 그림: 투명한 곳을 빼는 마스크만으로 마스크 학습을 켠다(2026-11-20)
            print(f"masks: 투명 배경 그림 {n_alpha}장이 있어 마스크 학습을 켭니다(투명한 곳은 학습하지 않음)", flush=True)
            args.masked_loss = True
    if getattr(args, "caption_shuffle", False):
        if getattr(args, "slider", False):
            print("caption shuffle: 슬라이더 학습이라 쓰지 않습니다", flush=True)
            args.caption_shuffle = False
        else:
            mode = args.caption_shuffle_mode = moss_shuffle.shuffle_mode(args.caption_shuffle_mode)
            style, keep = caption_shuffle_groups(args)
            if mode == moss_shuffle.FIXED_MODE:
                count = moss_shuffle.fixed_count(args.caption_shuffle_count)
                moss_shuffle.install(keep, count, log=lambda m: print(m, flush=True), style=style)
                args.caption_shuffle_info = f"mode=fixed, count={count}, style={','.join(style)}, keep={','.join(keep)}"
                label = f"고정 후보 {count}개"
            else:
                from library import dataset

                if not hasattr(dataset, "_select_moss_caption"):
                    raise SystemExit("에포크별 캡션 셔플을 쓰려면 install.bat 을 다시 실행하세요")
                patch_collator()
                label = "에포크별 셔플"
            print(f"caption shuffle: {label}, 스타일 {', '.join(style) or '없음'}, "
                  f"일반 {', '.join(keep) or '없음'}", flush=True)
    patch_prompt_defaults(args)
    if str(getattr(args, "optimizer_type", "") or "") == "Muon" and not getattr(args, "muon_unbatched", False):
        # sd-scripts get_optimizer 는 "Muon" 을 getattr(torch.optim, "Muon") 으로 찾는다 → 학습 프로세스 안에서만 바꿔 끼운다
        import moss_muon
        if moss_muon.install():
            print("muon: 모양이 같은 LoRA 행렬을 묶어 계산합니다(moss_muon.BatchedMuon, Fizgig v9.2.1 방식). "
                  "끄려면 --muon_unbatched", flush=True)
