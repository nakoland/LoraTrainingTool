# 슬라이더 LoRA 학습. 방식은 둘이다 (slider_method, 기본 pair).
# text: ai-toolkit concept_slider 방식. 데이터셋 그림에 노이즈를 넣고, LoRA 를 끈 원본 모델의 예측을 세 프롬프트로 받는다.
#   목표(+1) = 중립 + g × (긍정 − 부정),  목표(−1) = 중립 − g × (긍정 − 부정)   (중립 예측의 평균·표준편차로 맞춤)
# LoRA 를 +1 로 켠 예측은 목표(+1) 에, −1 로 켠 예측은 목표(−1) 에 맞춘다. 캡션은 쓰지 않는다(중립 프롬프트를 쓴다).
# 기준 프롬프트(anchor)를 주면 그 프롬프트에서는 LoRA 를 켜도 원본과 같게 학습해 다른 것이 바뀌지 않게 한다.
# pair: ai-toolkit image_reference_slider 방식. + 그림과 − 그림 쌍에 같은 노이즈·타임스텝·캡션을 쓰고, + 는 LoRA +1,
#   − 는 LoRA −1 로 학습한다. 목표(slider_pair_loss):
#   diff(기본, ADDifT 와 같은 생각): 원본 모델(LoRA 0)의 예측과의 차이만 목표로 한다. 같은 노이즈라 v+ − v− = x− − x+ 이고
#     D = f0(x−ₜ) − f0(x+ₜ) + (x− − x+),  목표(+1) = f0(x+ₜ) + s·D,  목표(−1) = f0(x−ₜ) − s·D   (s = slider_pair_scale)
#     → 깨끗한 그림 추정이 원본보다 s·(x+ − x−) 만큼 옮겨 간다(노이즈가 많을수록 그대로, 적으면 원본이 이미 알아 0 에 가까움).
#     원본이 그 그림들에서 틀리는 공통 부분(얼굴·화풍)은 D 에서 빠진다. 원본 순전파가 2번(기울기 없음) 더 든다.
#   s 는 세기를 거의 정하지 못한다: Muon·AdamW 가 기울기 크기를 정규화해 한 스텝의 이동량은 학습률이 정한다
#     배율 1의 효과는 학습률과 학습 스텝 수의 영향도 받는다.
#   recon(예전 방식): 각 그림을 그대로 복원(목표 = ε − x). 공통 오차까지 배워 얼굴·질감이 바뀌고 그림이 무너질 수 있다.
#   마스크(slider_pair_mask): 마스크 밖을 기준 그림으로 채워 바꾸려는 곳만 다르게 만든다(slider_pairs.py 머리 주석).
#   쌍 폴더 하나의 그림을 이름순으로 2장씩 묶는다: (1,2) (3,4) …, 앞 그림이 −(원본 → 바뀐 그림 순, slider_pair_first="pos" 면 앞이 +). prepare_pair_args 가 학습 전에 묶음 그림([+ | −])과 데이터셋 설정을 만들어 dataset_config 를 바꾼다(slider_pairs.py).
#   묶음은 반으로 나눠 따로 VAE 에 넣으므로(인코더의 어텐션이 두 그림을 섞지 않게) latent 캐시는 끈다.
# slider_sigma_min/max: 학습할 노이즈 구간(σ). 뽑은 σ 를 이 구간으로 옮긴다(분포 모양은 유지). 노이즈 많은 쪽만 쓰면
#   형태·자세만 배우고 세부·질감은 덜 외운다(ADDifT 의 timestep 500~1000 ≈ σ 0.62~1.0, SNR 기준).
# slider_layers: attn(기본)이면 FFN 을 빼고 어텐션(q·k·v·o)만 학습한다(network_args 의 exclude_patterns 에 더함).
#   참고한 MOSS 슬라이더(Waist·Thighs 등)가 어텐션만 학습했고, FFN 은 질감·세부를 주로 바꾼다.
# slider_skip_last_blocks: 뒤에서 N 개 블록을 통째로 뺀다(같은 exclude_patterns). 참고 슬라이더는 변화가 앞 블록에서 가장 크고
#   뒤로 갈수록 작은데, 우리 학습은 마지막 블록에 몰려 그림체가 바뀌었다.
# moss_train.py 가 설정의 `slider = true` 를 보고 이 믹스인을 붙인다. train_moss.bat(원래 학습 파일)은 이 키를 무시한다.
import json
import os
import tomllib

import torch

from library import flux_train_utils, strategy_base

import slider_pairs

HERE = os.path.dirname(os.path.abspath(__file__))
PAIRS_ROOT = os.path.join(HERE, ".slider_pairs")  # 묶음 그림 (output_name 마다 폴더 하나)


def add_slider_arguments(parser):
    parser.add_argument("--slider", action="store_true", help="슬라이더 LoRA 학습 (moss_train.py 전용)")
    parser.add_argument("--slider_method", type=str, default="pair", choices=["text", "pair"], help="text: 프롬프트 방식, pair: 그림 쌍 방식")
    parser.add_argument("--slider_target_class", type=str, default="", help="중립 프롬프트 (예: 1girl, standing). 그림 쌍 방식은 캡션이 없을 때 쓴다")
    parser.add_argument("--slider_positive", type=str, default="", help="+ 방향 프롬프트 (중립 + 속성)")
    parser.add_argument("--slider_negative", type=str, default="", help="- 방향 프롬프트 (중립 + 반대 속성)")
    parser.add_argument("--slider_anchor", type=str, default=None, help="LoRA 를 켜도 바뀌지 않게 할 프롬프트 (선택)")
    parser.add_argument("--slider_guidance", type=float, default=3.0, help="목표를 만들 때 (긍정 − 부정) 배수")
    parser.add_argument("--slider_anchor_strength", type=float, default=1.0, help="기준 프롬프트 loss 가중치")
    parser.add_argument("--slider_no_normalize", action="store_true", help="목표를 중립 예측의 평균·표준편차로 맞추지 않음")
    parser.add_argument("--slider_pair_dir", type=str, default=None, help="그림 쌍: 쌍 폴더 (이름순으로 2장씩 묶음: (1,2) (3,4) ...)")
    parser.add_argument("--slider_pair_first", type=str, default="neg", choices=["pos", "neg"], help="그림 쌍: 묶음의 앞 그림 방향 (pos: 앞이 +, neg: 앞이 −)")
    parser.add_argument("--slider_pair_loss", type=str, default="diff", choices=["diff", "cross", "recon", "wrecon"],
                        help="그림 쌍: diff = 원본 예측과의 차이만 학습(ADDifT 식), cross = 반대쪽 그림에서 출발해 이쪽으로 옮기기"
                             "(구도를 바꾸는 속성용), recon = 각 그림을 그대로 복원(예전 방식), wrecon = 각 그림을 복원하되 두 그림이 "
                             "다른 곳에 가중(Fizgig·Krea 2 방식)")
    parser.add_argument("--slider_pair_diff_weight", type=float, default=1.0,
                        help="wrecon: 0 = 일반 복원, 1 = 두 그림이 다른 곳에 집중(토큰별 |x+ − x−| ÷ 평균, 최대 8배)")
    parser.add_argument("--slider_pair_hold", type=float, default=0.0,
                        help="그림 쌍 각자 복원(recon·wrecon): 두 그림이 같은 곳은 원본(LoRA 0) 예측에 붙잡아 둔다(0 = 끔, 1 = 권장 시작값). "
                             "차이 가중 1 은 같은 곳의 가중이 0 이라 그곳에서 LoRA 가 무엇을 해도 벌이 없다(머리·배경까지 퍼짐). "
                             "토큰별 유지 정도 k = clamp(1 − |x+ − x−| ÷ 평균, 0, 1), 가중 = 이 값 × k. 원본 예측 2번이 더 든다")
    parser.add_argument("--slider_pair_noise", type=str, default="same", choices=["same", "separate"],
                        help="그림 쌍 각자 복원(recon·wrecon): same = 두 그림에 같은 노이즈·σ(Concept Sliders 논문·ai-toolkit), "
                             "separate = 그림마다 따로(Fizgig). diff·cross 는 늘 같은 노이즈")
    parser.add_argument("--slider_pair_scale", type=float, default=1.0,
                        help="그림 쌍(diff·cross): 배율 1 에서 옮길 양(1 = 쌍의 차이 하나). Muon·AdamW 는 기울기 크기를 정규화하므로 "
                             "이 값을 키워도 학습되는 변화량은 거의 그대로다(0.5/1/2 → ‖ΔW‖ 6.33/6.73/7.18). 세기는 학습률·스텝이 정한다")
    parser.add_argument("--slider_pair_center", action="store_true",
                        help="그림 쌍(diff): 두 그림이 다른 곳 안에서 차이의 채널별 평균을 뺀다. 바뀐 곳을 통째로 밝게(배경처럼) 만드는 "
                             "성분을 없애고 윤곽 이동만 남긴다(다리 사이를 지우는 대신 옮기게)")
    parser.add_argument("--slider_layers", type=str, default="attn", choices=["attn", "all"],
                        help="attn: 어텐션(q·k·v·o)만 학습, all: FFN 까지 전부")
    parser.add_argument("--slider_skip_last_blocks", type=int, default=0,
                        help="뒤에서 이 수만큼의 블록은 학습하지 않음 (MOSS 42블록: 8 이면 34~41 제외). 뒤 블록은 세부·질감을 주로 바꾼다")
    parser.add_argument("--slider_pair_mask", type=str, default="off", choices=["off", "file", "auto"],
                        help="그림 쌍: 마스크 밖을 기준 그림으로 채워 두 그림을 같게 만듦. file = <이름>_mask 그림만, auto = 없으면 두 그림이 형태로 다른 곳(세로 범위 안)")
    parser.add_argument("--slider_pair_mask_keep", type=str, default="neg", choices=["neg", "pos"],
                        help="그림 쌍 마스크: 마스크 밖에 쓸 기준 그림 (neg: − 그림, pos: + 그림)")
    parser.add_argument("--slider_pair_mask_top", type=float, default=0.45, help="그림 쌍 auto 마스크: 세로 범위 위 끝(그림 높이 비율, 0 = 맨 위)")
    parser.add_argument("--slider_pair_mask_bottom", type=float, default=1.0, help="그림 쌍 auto 마스크: 세로 범위 아래 끝(1 = 맨 아래)")
    parser.add_argument("--slider_pair_repeats", type=int, default=1, help="그림 쌍: 에포크당 반복 수")
    parser.add_argument("--slider_sigma_min", type=float, default=0.0, help="학습할 노이즈(σ) 구간의 아래 끝 (0~1, 1 = 완전한 노이즈)")
    parser.add_argument("--slider_sigma_max", type=float, default=1.0, help="학습할 노이즈(σ) 구간의 위 끝")
    parser.add_argument(
        "--slider_sample_multipliers", type=str, default="-1,1", help="샘플을 만들 LoRA 배율 (쉼표로 구분, 프롬프트마다 같은 시드)"
    )


def parse_multipliers(text):
    values = [float(x) for x in str(text or "").replace(" ", "").split(",") if x]
    if not values:
        raise ValueError("slider_sample_multipliers 가 비어 있습니다")
    return values


def is_pair(args):
    return bool(getattr(args, "slider", False)) and getattr(args, "slider_method", "pair") == "pair"


def pair_mask_options(args):
    if not 0 <= args.slider_pair_mask_top < args.slider_pair_mask_bottom <= 1:
        raise ValueError("slider_pair_mask_top 은 0 이상이고 slider_pair_mask_bottom(1 이하)보다 작아야 합니다")
    return {"mode": args.slider_pair_mask, "keep": args.slider_pair_mask_keep,
            "top": args.slider_pair_mask_top, "bottom": args.slider_pair_mask_bottom}


def prepare_pair_args(args):
    """그림 쌍 방식: 묶음 그림과 데이터셋 설정을 만들고 args 를 그쪽으로 바꾼다 (학습 시작 전 한 번)."""
    if not args.slider_pair_dir:
        raise ValueError("그림 쌍 방식에는 slider_pair_dir (쌍 폴더)가 필요합니다")
    with open(args.dataset_config, "rb") as f:
        user = tomllib.load(f)
    general = user.get("general", {})
    ds = (user.get("datasets") or [{}])[0]
    reso = ds.get("resolution", general.get("resolution", 1024))
    reso = int(max(reso) if isinstance(reso, (list, tuple)) else reso)
    steps = int(ds.get("bucket_reso_steps", general.get("bucket_reso_steps", 64)))
    caption_ext = general.get("caption_extension", ds.get("caption_extension", ".txt"))
    out_dir = os.path.join(PAIRS_ROOT, args.output_name or "slider")
    count, max_area, max_side = slider_pairs.build_pairs(
        args.slider_pair_dir, args.slider_pair_first, out_dir, reso, steps, caption_ext,
        args.slider_target_class or "", mask=pair_mask_options(args),
    )
    ds_path = os.path.join(out_dir, "dataset.toml")
    with open(ds_path, "w", encoding="utf-8") as f:
        f.write(slider_pairs.dataset_toml(user, out_dir, max_area, max_side, steps, args.slider_pair_repeats))
    print(f"slider pairs: {count} pairs, dataset {ds_path}", flush=True)
    args.dataset_config = ds_path
    # 묶음 전체를 한 번에 인코딩하면 VAE 인코더의 어텐션이 두 그림을 섞는다 → 캐시를 끄고 반씩 인코딩한다
    if args.cache_latents or args.cache_latents_to_disk:
        print("slider pairs: cache_latents is turned off (each half is encoded separately)", flush=True)
    args.cache_latents = args.cache_latents_to_disk = False


FFN_PATTERN = r".*\.ffn_.*"  # MOSS DiT 의 layers.N.proj.ffn_up_gate / ffn_down (전체 일치)


MOSS_BLOCKS = 42  # MOSS-Image DiT 블록 수 (layers.0 ~ layers.41)


def skip_blocks_pattern(n):
    """뒤에서 n 개 블록(layers.{42-n} ~ layers.41)의 모든 모듈을 빼는 정규식. n = 0 이면 None."""
    if n <= 0:
        return None
    first = MOSS_BLOCKS - n
    return r"layers\.(" + "|".join(str(i) for i in range(first, MOSS_BLOCKS)) + r")\..*"


def apply_layer_args(args):
    """slider_layers = attn 이면 FFN 을, slider_skip_last_blocks > 0 이면 뒤 블록을 network_args 의 exclude_patterns 에
    더한다 (학습 시작 전 한 번). 사용자가 적은 exclude_patterns 가 있으면 합친다."""
    import ast

    extra = []
    if getattr(args, "slider_layers", "attn") == "attn":
        extra.append(FFN_PATTERN)
    skip = int(getattr(args, "slider_skip_last_blocks", 0) or 0)
    if not 0 <= skip < MOSS_BLOCKS:
        raise ValueError(f"slider_skip_last_blocks 는 0 ~ {MOSS_BLOCKS - 1} 이어야 합니다 ({skip})")
    if skip:
        extra.append(skip_blocks_pattern(skip))
    if not extra:
        return
    net_args = list(args.network_args or [])
    for i, a in enumerate(net_args):
        key, _, value = a.partition("=")
        if key.strip() == "exclude_patterns":
            pats = ast.literal_eval(value.strip())
            pats = pats if isinstance(pats, list) else [pats]
            pats += [p for p in extra if p not in pats]
            net_args[i] = f"exclude_patterns={pats!r}"
            break
    else:
        net_args.append(f"exclude_patterns={extra!r}")
    args.network_args = net_args
    print(f"slider: exclude {extra} (network_args {net_args})", flush=True)


def norm_like(tensor, ref):
    """샘플마다 ref 의 평균·표준편차로 맞춘다 (ai-toolkit 은 배치 전체로 맞춘다)."""
    dims = list(range(1, tensor.ndim))
    t_mean, t_std = tensor.mean(dim=dims, keepdim=True), tensor.std(dim=dims, keepdim=True)
    r_mean, r_std = ref.mean(dim=dims, keepdim=True), ref.std(dim=dims, keepdim=True)
    return (tensor - t_mean) / (t_std + 1e-8) * r_std + r_mean


class SliderMixin:
    """ResumableTrainer 앞에 붙는 믹스인: class SliderTrainer(SliderMixin, ResumableTrainer)."""

    def assert_extra_args(self, args, train_dataset_group, val_dataset_group):
        super().assert_extra_args(args, train_dataset_group, val_dataset_group)
        self.slider_pair = is_pair(args)
        if not self.slider_pair:  # 그림 쌍은 프롬프트 대신 캡션을 쓴다
            if not (args.slider_positive or args.slider_negative):
                raise ValueError("슬라이더 학습에는 slider_positive 나 slider_negative 중 하나가 필요합니다")
            if args.slider_positive == args.slider_negative:
                raise ValueError("slider_positive 와 slider_negative 가 같습니다")
        if getattr(args, "blocks_to_swap", None):
            raise ValueError("슬라이더 학습은 blocks_to_swap 을 지원하지 않습니다 (한 스텝에 순전파를 여러 번 함)")
        self.slider_multipliers = parse_multipliers(args.slider_sample_multipliers)
        lo, hi = args.slider_sigma_min, args.slider_sigma_max
        if not (0.0 <= lo < hi <= 1.0):
            raise ValueError(f"slider_sigma_min/max 는 0 ≤ 아래 < 위 ≤ 1 이어야 합니다 ({lo}, {hi})")

    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        super().post_process_network(args, accelerator, network, text_encoders, unet)
        self.slider_network = network

    # ── 프롬프트 인코딩: 텍스트 인코더가 아직 있을 때 한 번 ──
    def cache_text_encoder_outputs_if_needed(self, args, accelerator, unet, vae, text_encoders, dataset, weight_dtype):
        if self.slider_pair:
            return super().cache_text_encoder_outputs_if_needed(args, accelerator, unet, vae, text_encoders, dataset, weight_dtype)
        names = {"target": args.slider_target_class, "positive": args.slider_positive, "negative": args.slider_negative}
        if args.slider_anchor:  # 빈 칸은 기준 프롬프트 없음
            names["anchor"] = args.slider_anchor
        te = text_encoders[0]
        org_device = next(te.parameters()).device
        te.to(accelerator.device)
        tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
        text_encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()
        self.slider_conds = {}
        with torch.no_grad():
            for key, prompt in names.items():
                tokens = tokenize_strategy.tokenize(prompt or "")
                embeds, mask = text_encoding_strategy.encode_tokens(tokenize_strategy, [te], tokens)[:2]
                self.slider_conds[key] = (embeds.cpu(), mask.cpu())
                print(f"slider prompt {key}: {prompt!r} ({int(mask.sum())} tokens)", flush=True)
        te.to(org_device)
        super().cache_text_encoder_outputs_if_needed(args, accelerator, unet, vae, text_encoders, dataset, weight_dtype)

    def _cond(self, key, batch_size, device, dtype, grad):
        embeds, mask = self.slider_conds[key]
        embeds = embeds.to(device, dtype=dtype).repeat(batch_size, 1, 1)
        mask = mask.to(device).repeat(batch_size, 1)
        if grad:
            embeds.requires_grad_(True)  # 체크포인팅(non-reentrant)은 기울기가 필요한 입력이 있어야 한다
        return embeds, mask

    def encode_images_to_latents(self, args, vae, images):
        if not self.slider_pair:
            return super().encode_images_to_latents(args, vae, images)
        half = images.shape[-1] // 2  # [+ | −] 묶음을 나눠 따로 인코딩하고 다시 붙인다
        pos = super().encode_images_to_latents(args, vae, images[..., :half])
        neg = super().encode_images_to_latents(args, vae, images[..., half:])
        return torch.cat([pos, neg], dim=-1)

    def _noisy(self, args, noise_scheduler, latents, noise, device):
        """일반 학습과 같은 분포로 σ 를 뽑고, 노이즈 구간이 정해져 있으면 σ 를 그 구간으로 옮긴다."""
        noisy, timesteps, sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
            args, noise_scheduler, latents, noise, device, dtype=torch.float32
        )
        lo, hi = args.slider_sigma_min, args.slider_sigma_max
        if lo > 0.0 or hi < 1.0:
            sigmas = lo + sigmas * (hi - lo)
            timesteps = (sigmas.view(-1) * noise_scheduler.config.num_train_timesteps).to(timesteps.dtype)
            noisy = ((1.0 - sigmas) * latents + sigmas * noise).to(noisy.dtype)
        return noisy, timesteps, sigmas

    def _pair_pred_and_target(self, args, accelerator, noise_scheduler, latents, text_encoder_conds, unet, network,
                              weight_dtype, is_train):
        net = accelerator.unwrap_model(network)
        half = latents.shape[-1] // 2
        pos, neg = latents[..., :half].contiguous(), latents[..., half:].contiguous()
        noise = torch.randn_like(pos)
        noisy_pos, timesteps, sigmas = self._noisy(args, noise_scheduler, pos, noise, accelerator.device)
        noisy_neg = ((1.0 - sigmas) * neg + sigmas * noise).to(noisy_pos.dtype)  # 같은 노이즈·타임스텝
        weighting = flux_train_utils.compute_loss_weighting_for_sd3(args.weighting_scheme, sigmas)
        embeds, mask = text_encoder_conds[:2]
        embeds, mask = embeds.to(accelerator.device, dtype=weight_dtype), mask.to(accelerator.device)

        def run(x, grad=is_train, ts=None):
            e = embeds
            if grad and args.gradient_checkpointing:
                x = x.detach().requires_grad_(True)
                e = embeds.detach().clone().requires_grad_(True)
            with torch.set_grad_enabled(grad), accelerator.autocast():
                return unet(x, timesteps if ts is None else ts, e, mask)

        if args.slider_pair_loss == "cross":
            # +1 은 − 그림(x−)의 노이즈 입력을 받아 예측을 x+ 쪽으로, −1 은 x+ 입력을 받아 x− 쪽으로 옮긴다.
            # 같은 노이즈에서 두 그림의 이상적 속도 차이는 v+ − v− = x− − x+ 이므로 목표 = 원본 예측 ± s·(x− − x+).
            # diff 는 + 입력에 이미 + 구도가 들어 있어 원본이 아는 큰 구도 차이가 빠지고 세부만 남는다(노이즈가 많을 때
            # 자리를 모르는 세부 → '하얗게 바래기'). cross 는 생성 때와 같이 '붙은 다리 구도 → 벌린 구도'를 배운다.
            net.set_multiplier(0.0)
            with torch.no_grad():
                base_pos = run(noisy_pos, False).float()
                base_neg = run(noisy_neg, False).float()
            d = (neg - pos).float()
            edit = args.slider_pair_scale * d
            if getattr(args, "slider_pair_center", False):
                wmap = d.abs().mean(dim=1, keepdim=True)
                m = (wmap > 0.1 * wmap.amax(dim=(2, 3), keepdim=True)).float()
                mean = (edit * m).sum(dim=(2, 3), keepdim=True) / m.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
                edit = edit - mean * m
            target_on_neg, target_on_pos = base_neg + edit, base_pos - edit
            del base_pos, base_neg, edit
            net.set_multiplier(1.0)
            loss_pos = torch.nn.functional.mse_loss(run(noisy_neg).float(), target_on_neg, reduction="none")
            if weighting is not None:
                loss_pos = loss_pos * weighting
            loss_pos = loss_pos.mean()
            if is_train:
                accelerator.backward(loss_pos)
            self._slider_pos_loss = loss_pos.detach()
            net.set_multiplier(-1.0)
            return run(noisy_pos), target_on_pos, timesteps, weighting

        if args.slider_pair_loss == "diff":
            # 원본(LoRA 0) 예측과의 차이만 목표로 한다: 원본이 두 그림에서 함께 틀리는 부분(얼굴·화풍)은 빠진다
            net.set_multiplier(0.0)
            with torch.no_grad():
                base_pos = run(noisy_pos, False).float()
                base_neg = run(noisy_neg, False).float()
            d = (neg - pos).float()
            edit = args.slider_pair_scale * (base_neg - base_pos + d)
            if getattr(args, "slider_pair_center", False):
                # 두 그림이 다른 칸(차이 크기가 그 그림 최대의 10% 이상) 안에서 채널별 평균을 빼 '통째로 밝게' 성분을 없앤다
                wmap = d.abs().mean(dim=1, keepdim=True)
                m = (wmap > 0.1 * wmap.amax(dim=(2, 3), keepdim=True)).float()
                mean = (edit * m).sum(dim=(2, 3), keepdim=True) / m.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
                edit = edit - mean * m
            target_pos, target_neg = base_pos + edit, base_neg - edit
            del base_pos, base_neg, edit
        else:
            target_pos, target_neg = (noise - pos).float(), noise - neg
        # 각자 복원(recon·wrecon): slider_pair_noise = separate 면 − 쪽에 노이즈·σ 를 새로 뽑는다(Fizgig 는 극마다 따로,
        # Concept Sliders 논문·ai-toolkit 은 같은 노이즈). diff·cross 는 같은 노이즈가 식의 전제라 늘 같음.
        ts_neg, weighting_neg = None, weighting
        if args.slider_pair_loss in ("recon", "wrecon") and getattr(args, "slider_pair_noise", "same") == "separate":
            noise_neg = torch.randn_like(neg)
            noisy_neg, ts_neg, sigmas_neg = self._noisy(args, noise_scheduler, neg, noise_neg, accelerator.device)
            noisy_neg = noisy_neg.to(noisy_pos.dtype)
            weighting_neg = flux_train_utils.compute_loss_weighting_for_sd3(args.weighting_scheme, sigmas_neg)
            target_neg = noise_neg - neg

        # wrecon: 토큰별 가중 = (1 − dw) + dw·(|x+ − x−| ÷ 그 그림의 평균, 최대 8), 평균 1 로 맞춤(두 그림이 같으면 1).
        # Fizgig(moss_image/driver.py training_loss)의 그림 쌍 슬라이더 가중과 같은 식. 노이즈는 slider_pair_noise 를 따른다.
        tok_w = None
        if args.slider_pair_loss == "wrecon" and float(getattr(args, "slider_pair_diff_weight", 1.0)) > 0:
            dw = float(args.slider_pair_diff_weight)
            d = (pos - neg).float().abs().mean(dim=1, keepdim=True)
            dm = d.mean(dim=(2, 3), keepdim=True)
            r = (d / dm.clamp_min(1e-8)).clamp(max=8.0)
            tok_w = (1.0 - dw) + dw * r
            tok_w = tok_w / tok_w.mean(dim=(2, 3), keepdim=True).clamp_min(1e-8)
            tok_w = torch.where(dm <= 1e-6, torch.ones_like(tok_w), tok_w)

        # 붙잡기(hold, 2026-11-17): 두 그림이 같은 곳은 원본(LoRA 0) 예측을 목표로 한다. 가중 w 의 복원 목표 t 와 가중 h 의
        # 원본 예측 b 를 함께 맞추는 것은 w(p − t)² + h(p − b)² = (w + h)(p − (w·t + h·b)/(w + h))² + 상수 이므로,
        # 목표를 섞고 가중을 w + h 로 바꾸면 아래의 한 목표·√가중 방식 그대로 된다.
        hold = float(getattr(args, "slider_pair_hold", 0.0) or 0.0)
        if hold > 0 and args.slider_pair_loss in ("recon", "wrecon"):
            d = (pos - neg).float().abs().mean(dim=1, keepdim=True)
            keep = (1.0 - d / d.mean(dim=(2, 3), keepdim=True).clamp_min(1e-8)).clamp(0.0, 1.0)  # 같은 곳 1, 평균 이상 다른 곳 0
            h = hold * keep
            w = tok_w if tok_w is not None else torch.ones_like(h)
            net.set_multiplier(0.0)
            with torch.no_grad():
                base_pos = run(noisy_pos, False).float()
                base_neg = run(noisy_neg, False, ts=ts_neg).float()
            target_pos = (w * target_pos.float() + h * base_pos) / (w + h)
            target_neg = (w * target_neg.float() + h * base_neg) / (w + h)
            tok_w = w + h
            del base_pos, base_neg

        net.set_multiplier(1.0)
        loss_pos = torch.nn.functional.mse_loss(run(noisy_pos).float(), target_pos, reduction="none")
        if tok_w is not None:
            loss_pos = loss_pos * tok_w
        if weighting is not None:
            loss_pos = loss_pos * weighting
        loss_pos = loss_pos.mean()
        if is_train:
            accelerator.backward(loss_pos)  # 배율을 바꾸기 전에 (체크포인팅이 역전파 때 순전파를 다시 한다)
        self._slider_pos_loss = loss_pos.detach()
        net.set_multiplier(-1.0)
        pred_neg = run(noisy_neg, ts=ts_neg)
        ts_out = timesteps if ts_neg is None else ts_neg
        if tok_w is not None:  # 학습 루프의 MSE 가 가중 MSE 가 되도록 √w 를 양쪽에 곱한다: (√w·p − √w·t)² = w·(p − t)²
            sw = tok_w.sqrt()
            return pred_neg * sw.to(pred_neg.dtype), target_neg.float() * sw, ts_out, weighting_neg
        return pred_neg, target_neg, ts_out, weighting_neg

    def get_noise_pred_and_target(
        self, args, accelerator, noise_scheduler, latents, batch, text_encoder_conds, unet, network, weight_dtype, train_unet,
        is_train=True,
    ):
        self._slider_pos_loss = None
        if self.slider_pair:
            return self._pair_pred_and_target(
                args, accelerator, noise_scheduler, latents, text_encoder_conds, unet, network, weight_dtype, is_train
            )
        if not is_train:  # 검증 loss 는 일반 학습과 같은 방식으로
            return super().get_noise_pred_and_target(
                args, accelerator, noise_scheduler, latents, batch, text_encoder_conds, unet, network, weight_dtype, train_unet,
                is_train,
            )
        net = accelerator.unwrap_model(network)
        device = accelerator.device
        batch_size = latents.shape[0]
        noise = torch.randn_like(latents)
        noisy, timesteps, sigmas = self._noisy(args, noise_scheduler, latents, noise, device)
        weighting = flux_train_utils.compute_loss_weighting_for_sd3(args.weighting_scheme, sigmas)
        has_anchor = "anchor" in self.slider_conds

        def run(key, grad):
            x = noisy
            if grad and args.gradient_checkpointing:
                x = noisy.detach().requires_grad_(True)
            embeds, mask = self._cond(key, batch_size, device, weight_dtype, grad and args.gradient_checkpointing)
            with torch.set_grad_enabled(grad), accelerator.autocast():
                return unet(x, timesteps, embeds, mask)

        def mse(pred, target):
            loss = torch.nn.functional.mse_loss(pred.float(), target, reduction="none")
            if weighting is not None:
                loss = loss * weighting
            return loss.mean()

        # 1) LoRA 를 끈 원본 예측
        net.set_multiplier(0.0)
        with torch.no_grad():
            neutral = run("target", False).float()
            positive = run("positive", False).float()
            negative = run("negative", False).float()
            anchor_target = run("anchor", False).float() if has_anchor else None
        direction = args.slider_guidance * (positive - negative)
        target_pos, target_neg = neutral + direction, neutral - direction
        if not args.slider_no_normalize:
            target_pos, target_neg = norm_like(target_pos, neutral), norm_like(target_neg, neutral)
        del positive, negative, direction

        # 2) +1: 여기서 바로 역전파한다. 체크포인팅은 역전파 때 순전파를 다시 하므로 배율을 바꾸기 전에 끝내야 한다
        net.set_multiplier(1.0)
        loss_pos = mse(run("target", True), target_pos)
        if has_anchor:
            loss_pos = loss_pos + args.slider_anchor_strength * mse(run("anchor", True), anchor_target)
        accelerator.backward(loss_pos)
        self._slider_pos_loss = loss_pos.detach()

        # 3) −1: 기준 프롬프트는 여기서 역전파하고, 중립 예측은 학습 루프가 loss 를 계산해 역전파한다(배율 −1 유지)
        net.set_multiplier(-1.0)
        if has_anchor:
            accelerator.backward(args.slider_anchor_strength * mse(run("anchor", True), anchor_target))
        pred_neg = run("target", True)
        return pred_neg, target_neg, timesteps, weighting

    def post_process_loss(self, loss, args, timesteps, noise_scheduler):
        loss = super().post_process_loss(loss, args, timesteps, noise_scheduler)
        pos = getattr(self, "_slider_pos_loss", None)
        if pos is not None:
            # 기록되는 값은 +1·−1 의 평균, 기울기는 −1 쪽 그대로 (+1 쪽은 이미 역전파함)
            loss = loss + (pos - loss.detach()) / 2
            self._slider_pos_loss = None
        return loss

    # ── 샘플: 프롬프트마다 배율별로 같은 시드로 만든다 ──
    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):
        from library import moss_image_train_utils as utils

        multipliers = self.slider_multipliers
        net = getattr(self, "slider_network", None)
        orig_load, orig_infer = utils.sampling.load_prompts, utils._sample_image_inference

        def load_prompts(path):
            out = []
            for i, p in enumerate(orig_load(path)):
                seed = p.get("seed")
                if seed is None:
                    seed = (args.seed or 0) + i
                for j, m in enumerate(multipliers):
                    q = dict(p, seed=seed, enum=i * len(multipliers) + j)
                    q["slider_multiplier"] = m
                    out.append(q)
            return out

        def infer(accelerator_, args_, dit, te, vae_, ts, tes, save_dir, prompt_dict, *rest, **kw):
            m = prompt_dict.get("slider_multiplier", 1.0)
            if net is not None:
                accelerator_.unwrap_model(net).set_multiplier(m)
            print(f"slider sample multiplier {m:g}", flush=True)
            return orig_infer(accelerator_, args_, dit, te, vae_, ts, tes, save_dir, prompt_dict, *rest, **kw)

        def load_prompts_and_note(path):
            out = load_prompts(path)
            # 결과 탭이 샘플 번호를 (프롬프트, 배율)로 읽도록 남긴다
            try:
                save_dir = os.path.join(args.output_dir, "sample")
                os.makedirs(save_dir, exist_ok=True)
                with open(os.path.join(save_dir, f"{args.output_name or ''}_slider.json"), "w", encoding="utf-8") as f:
                    json.dump({"multipliers": multipliers, "method": args.slider_method}, f)
            except OSError as e:
                print(f"slider: cannot write sample info: {e}", flush=True)
            return out

        utils.sampling.load_prompts, utils._sample_image_inference = load_prompts_and_note, infer
        try:
            super().sample_images(accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet)
        finally:
            utils.sampling.load_prompts, utils._sample_image_inference = orig_load, orig_infer
            if net is not None:
                accelerator.unwrap_model(net).set_multiplier(1.0)

    def update_metadata(self, metadata, args):
        super().update_metadata(metadata, args)
        metadata["ss_slider"] = "image_pair" if self.slider_pair else "concept_slider"
        metadata["ss_slider_target_class"] = args.slider_target_class
        metadata["ss_slider_positive"] = args.slider_positive
        metadata["ss_slider_negative"] = args.slider_negative
        metadata["ss_slider_anchor"] = args.slider_anchor or ""
        metadata["ss_slider_guidance"] = args.slider_guidance
        metadata["ss_slider_sigma_range"] = f"{args.slider_sigma_min},{args.slider_sigma_max}"
        metadata["ss_slider_layers"] = args.slider_layers
        metadata["ss_slider_skip_last_blocks"] = args.slider_skip_last_blocks
        if self.slider_pair:
            metadata["ss_slider_pair_dir"] = args.slider_pair_dir
            metadata["ss_slider_pair_first"] = args.slider_pair_first
            metadata["ss_slider_pair_loss"] = args.slider_pair_loss
            metadata["ss_slider_pair_scale"] = args.slider_pair_scale
            metadata["ss_slider_pair_noise"] = getattr(args, "slider_pair_noise", "same")
            metadata["ss_slider_pair_diff_weight"] = getattr(args, "slider_pair_diff_weight", 1.0)
            metadata["ss_slider_pair_hold"] = getattr(args, "slider_pair_hold", 0.0)
            metadata["ss_slider_pair_center"] = str(bool(getattr(args, "slider_pair_center", False)))
            metadata["ss_slider_pair_mask"] = (args.slider_pair_mask if args.slider_pair_mask == "off" else
                                               f"{args.slider_pair_mask},keep={args.slider_pair_mask_keep},"
                                               f"rows={args.slider_pair_mask_top}~{args.slider_pair_mask_bottom}")
