# DOP(Differential Output Preservation, Ostris AI-Toolkit 방식, Fizgig v8.4 "Leave the rest of the model alone") (2026-11-12)
# 트리거 단어를 뺀 캡션으로 같은 그림을 한 번 더 학습하고, 목표는 같은 노이즈에서 LoRA 를 끈(배율 0) 원래 모델의 예측이다.
# → 트리거가 없는 프롬프트에서는 LoRA 가 아무것도 바꾸지 않도록 배운다(캐릭터·화풍이 다른 사람·그림으로 번지는 것을 줄임).
#
#   --dop                켬
#   --dop_trigger "a, b" 뺄 트리거(쉼표로 여럿). 캡션에 하나라도 든 그림만 DOP 를 한다(대소문자 무시, 단어 경계)
#   --dop_class "x"      트리거 자리에 넣을 말(비우면 그냥 뺌, 하나면 모든 트리거에, 여럿이면 트리거 순서대로)
#   --dop_weight 2       DOP 손실의 무게 / --dop_every 2  N 스텝마다(Fizgig 기본: 두 스텝마다 무게 2 = 평균은 매 스텝 무게 1)
#
# 제약(사용자 결정 2026-11-12: 일단 배치 1 고정):
#  - 배치 1 로 학습한다. prepare_args 가 데이터셋 설정의 batch_size 를 1 로 바꾼 사본(.dop_fill/<출력 이름>/)을 쓴다.
#  - TE 출력 캐시 + DiT 만 학습이 필요하다(배치의 캡션이 캐시 후보 그대로라야 미리 인코딩한 트리거 없는 캡션을 찾는다).
#  - 슬라이더 학습·블록 스왑에서는 쓰지 않는다.
# 트리거를 뺀 캡션의 텍스트 인코딩은 TE 캐시를 만든 직후(텍스트 인코더가 아직 있을 때) 만들고 .dop_cache/<출력 이름>.pt 에 기억한다.
# 캡션 함수(dop_pairs·dop_caption)는 UI 서버(moss_ui.analyze)도 쓰므로 이 파일 맨 위에서 torch 를 가져오지 않는다.
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
FILL_ROOT = os.path.join(HERE, ".dop_fill")
CACHE_ROOT = os.path.join(HERE, ".dop_cache")
DEFAULT_EVERY = 2
DEFAULT_WEIGHT = 2.0
LOG_EVERY = int(os.environ.get("MOSS_DOP_LOG_EVERY") or 50)  # DOP 이만큼마다 평균 손실을 로그에 적는다


def add_arguments(parser):
    parser.add_argument("--dop", action="store_true",
                        help="DOP: 트리거를 뺀 캡션에서는 LoRA 0 원래 모델 예측에 맞춘다(배치 1, TE 캐시 필요)")
    parser.add_argument("--dop_trigger", type=str, default="", help="DOP 로 뺄 트리거(쉼표로 여럿)")
    parser.add_argument("--dop_class", type=str, default="", help="트리거 자리에 넣을 말(비우면 뺌, 하나 또는 트리거 수만큼)")
    parser.add_argument("--dop_weight", type=float, default=DEFAULT_WEIGHT, help="DOP 손실 무게(기본 2)")
    parser.add_argument("--dop_every", type=int, default=DEFAULT_EVERY, help="N 스텝마다 DOP(기본 2)")


# ── 캡션 (UI 서버와 공통) ──
def dop_pairs(triggers, classes=""):
    """"a, b" + "x" → [(a, x), (b, x)]. 바꿀 말이 비면 트리거를 뺀다. 수가 안 맞으면 ValueError."""
    trig = [t.strip() for t in str(triggers or "").split(",") if t.strip()]
    cls = [c.strip() for c in str(classes or "").split(",")]
    if len(cls) == 1:
        cls = cls * len(trig)
    if len(cls) != len(trig):
        raise ValueError(f"DOP 트리거 {len(trig)}개({', '.join(trig)})에 바꿀 말이 {len(cls)}개입니다 — "
                         "하나만 쓰거나(모든 트리거에) 트리거와 같은 수로 순서대로 쓰세요")
    return list(zip(trig, cls))


def dop_caption(caption, pairs):
    """트리거를 바꾸거나 뺀 캡션(Fizgig multi_person.dop_caption 과 같음). 트리거가 하나도 없으면 None."""
    if not caption:
        return None
    out, hit = caption, False
    for trigger, cls in pairs:
        pat = re.compile(r"(?<!\w)" + re.escape(trigger) + r"(?!\w)", re.I)
        if pat.search(out):
            hit = True
            out = pat.sub(lambda _m, c=cls: c, out)
    if not hit:
        return None
    out = re.sub(r"\s*,\s*(?:,\s*)+", ", ", out)  # 뺀 자리에 남은 ", ," 와 앞뒤 쉼표
    out = re.sub(r"^\s*,\s*|\s*,\s*$", "", out)
    return re.sub(r"[ \t]{2,}", " ", out).strip()


# ── 학습 준비 ──
def enabled(args):
    return bool(getattr(args, "dop", False)) and not getattr(args, "slider", False)


def prepare_dataset(dataset_config, output_name, log=print):
    """batch_size 를 모두 1 로 바꾼 데이터셋 설정을 만든다. 이미 모두 1 이면 None."""
    import tomllib
    import moss_masks
    with open(dataset_config, "rb") as f:
        data = tomllib.load(f)
    general = data.get("general", {})
    old = sorted({int(ds.get("batch_size", general.get("batch_size", 1))) for ds in data.get("datasets", [])} or {1})
    if old == [1] and int(general.get("batch_size", 1)) == 1:
        return None
    if "batch_size" in general:
        general["batch_size"] = 1
    base = os.getcwd()  # sd-scripts 는 데이터셋 설정의 상대 경로를 작업 폴더 기준으로 읽는다
    for ds in data.get("datasets", []):
        ds["batch_size"] = 1
        for s in ds.get("subsets", []):
            for k in ("image_dir", "conditioning_data_dir", "metadata_file"):
                if s.get(k):
                    s[k] = moss_masks.resolve(s[k], base).replace("\\", "/")
    out_root = os.path.join(FILL_ROOT, output_name or "default")
    os.makedirs(out_root, exist_ok=True)
    path = os.path.join(out_root, "dataset.toml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(moss_masks.dataset_toml(data))
    log(f"dop: 배치 {'/'.join(map(str, old))} → 1 (DOP 는 그림 한 장씩 학습합니다. 총 스텝이 그만큼 늘어납니다)")
    return path


def prepare_args(args):
    """moss_train 이 학습 전에 부른다."""
    if not getattr(args, "dop", False):
        return
    if getattr(args, "slider", False):
        print("dop: 슬라이더 학습이라 쓰지 않습니다", flush=True)
        args.dop = False
        return
    args.dop_pairs = dop_pairs(args.dop_trigger, args.dop_class)
    if not args.dop_pairs:
        raise SystemExit("dop: 뺄 트리거(dop_trigger)가 비어 있습니다")
    if not args.cache_text_encoder_outputs and not getattr(args, "cache_text_encoder_outputs_to_disk", False):
        raise SystemExit("dop: TE 출력 캐시(cache_text_encoder_outputs)가 필요합니다")
    if not args.network_train_unet_only:
        raise SystemExit("dop: 'DiT 만 학습'(network_train_unet_only)이 필요합니다")
    if int(getattr(args, "blocks_to_swap", 0) or 0) > 0:
        raise SystemExit("dop: 블록 스왑과 함께 쓸 수 없습니다")
    if int(args.dop_every or 1) < 1:
        args.dop_every = 1
    args.train_batch_size = 1
    if getattr(args, "dataset_config", None):
        new = prepare_dataset(args.dataset_config, args.output_name, log=lambda m: print(m, flush=True))
        if new:
            print(f"dop: 데이터셋 설정 {args.dataset_config} -> {new}", flush=True)
            args.dataset_config = new
    shown = ", ".join(f"{t} → {c or '(뺌)'}" for t, c in args.dop_pairs)
    print(f"dop: {shown}, {args.dop_every}스텝마다 무게 {args.dop_weight:g}", flush=True)


def _te_key(args):
    return f"{os.path.abspath(str(args.text_encoder or ''))}|int8={bool(getattr(args, 'convrot_int8_text_encoder', False))}"


def _cache_path(output_name):
    return os.path.join(CACHE_ROOT, f"{output_name or 'default'}.pt")


class DopMixin:
    """moss_train 의 일반 LoRA 트레이너에 섞는다(ExtrasMixin 보다 앞)."""

    _dop_cond = None   # {캐시 캡션: 트리거를 뺀 캡션의 텍스트 인코딩 [L, C] bf16 CPU}
    _dop_calls = 0
    _dop_sum = 0.0
    _dop_n = 0
    _dop_total = 0

    def update_metadata(self, metadata, args):
        super().update_metadata(metadata, args)
        if enabled(args):
            metadata["ss_dop"] = (f"trigger={args.dop_trigger}; class={args.dop_class}; weight={args.dop_weight:g}; "
                                  f"every={args.dop_every}")

    def cache_text_encoder_outputs_if_needed(self, args, accelerator, unet, vae, text_encoders, dataset, weight_dtype):
        super().cache_text_encoder_outputs_if_needed(args, accelerator, unet, vae, text_encoders, dataset, weight_dtype)
        if enabled(args):
            self._dop_prepare(args, accelerator, text_encoders, dataset)

    def _dop_prepare(self, args, accelerator, text_encoders, dataset):
        import torch
        from library import strategy_base
        from library.device_utils import clean_memory_on_device
        need, images, hit = {}, 0, 0
        for ds in dataset.datasets:
            d = getattr(ds, "dreambooth_dataset_delegate", ds)  # 마스크 학습(ControlNet 데이터셋)이면 안쪽 데이터셋
            for info in d.image_data.values():
                images += 1
                caps = d.process_caption_candidates_for_cache(d.image_to_subset[info.image_key], info.caption)
                got = False
                for c in caps:
                    dc = dop_caption(c, args.dop_pairs)
                    if dc is not None:
                        need[c] = dc
                        got = True
                hit += got
        if not need:
            raise RuntimeError(f"dop: 그림 {images}장의 캡션에 트리거({args.dop_trigger})가 하나도 없습니다")
        path = _cache_path(args.output_name)
        key = _te_key(args)
        store = {}
        try:
            saved = torch.load(path, map_location="cpu", weights_only=True)
            if saved.get("key") == key:
                store = saved.get("items", {})
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"dop: 기억해 둔 인코딩을 읽지 못해 다시 만듭니다 ({type(e).__name__}: {e})", flush=True)
        missing = sorted({dc for dc in need.values() if dc not in store})
        if missing:
            te = text_encoders[0]
            org = next(te.parameters()).device
            print(f"dop: 트리거를 뺀 캡션 {len(missing)}개를 인코딩합니다", flush=True)
            tok = strategy_base.TokenizeStrategy.get_strategy()
            enc = strategy_base.TextEncodingStrategy.get_strategy()
            te.to(accelerator.device)
            try:
                with torch.no_grad():
                    for i in range(0, len(missing), 8):
                        chunk = missing[i:i + 8]
                        emb, mask = enc.encode_tokens(tok, [te], tok.tokenize(chunk))
                        for j, text in enumerate(chunk):
                            n = int(mask[j].sum().item())
                            store[text] = emb[j, :n].to(torch.bfloat16).cpu()
            finally:
                te.to(org)
                clean_memory_on_device(accelerator.device)
            os.makedirs(CACHE_ROOT, exist_ok=True)
            keep = {dc: store[dc] for dc in set(need.values())}
            tmp = path + ".tmp"
            torch.save({"key": key, "items": keep}, tmp)
            os.replace(tmp, path)
        self._dop_cond = {c: store[dc] for c, dc in need.items()}
        ex_c, ex_d = next(iter(need.items()))
        print(f"dop: 트리거가 든 그림 {hit}/{images}장(나머지는 DOP 없이 학습). 예: '{ex_c[:120]}' → '{ex_d[:120]}'", flush=True)

    def process_batch(self, batch, text_encoders, unet, network, vae, noise_scheduler, vae_dtype, weight_dtype, accelerator, args,
                      text_encoding_strategy, tokenize_strategy, is_train=True, *rest, **kw):
        if is_train and enabled(args) and self._dop_cond:
            self._dop_calls += 1
            if (self._dop_calls - 1) % int(args.dop_every) == 0:
                caps = batch.get("captions") or []
                cond = self._dop_cond.get(caps[0]) if len(caps) == 1 else None
                if cond is not None:
                    self._dop_step(args, accelerator, batch, cond, unet, network, vae, vae_dtype, weight_dtype, noise_scheduler)
        return super().process_batch(batch, text_encoders, unet, network, vae, noise_scheduler, vae_dtype, weight_dtype,
                                     accelerator, args, text_encoding_strategy, tokenize_strategy, is_train, *rest, **kw)

    def _dop_step(self, args, accelerator, batch, cond, unet, network, vae, vae_dtype, weight_dtype, noise_scheduler):
        """같은 그림을 새 노이즈로, 트리거를 뺀 캡션으로: 목표 = LoRA 0 예측. 여기서 역전파한다(학습 순전파보다 먼저라 그래프가 겹치지 않음)."""
        import torch
        from library import flux_train_utils
        dev = accelerator.device
        with torch.no_grad():
            if batch.get("latents") is not None:
                latents = batch["latents"].to(dev)
            else:
                latents = self.encode_images_to_latents(args, vae, batch["images"].to(dev, dtype=vae_dtype))
            latents = self.shift_scale_latents(args, latents)
            noise = torch.randn_like(latents)
            noisy, timesteps, _sig = flux_train_utils.get_noisy_model_input_and_timesteps(
                args, noise_scheduler, latents, noise, dev, torch.float32)
        emb = cond[None].to(dev, dtype=weight_dtype)
        # 학습 배치의 마스크는 collate 가 float32 로 쌓는다(none_or_stack_elements(..., torch.FloatTensor)). 같은 dtype 이라야
        # compile 이 학습과 같은 그래프를 쓴다
        mask = torch.ones((1, emb.shape[1]), dtype=torch.float32, device=dev)
        net = accelerator.unwrap_model(network)

        def inputs():
            if args.gradient_checkpointing:  # non-reentrant 체크포인팅은 기울기가 필요한 입력이 있어야 한다(학습 순전파와 같음)
                return noisy.clone().requires_grad_(True), emb.clone().requires_grad_(True)
            return noisy, emb

        # compile 이면 no_grad 순전파는 추론용 그래프를 따로 컴파일한다. 학습 그래프를 그대로 쓰도록 기울기를 켠 채 계산하고 떼어 낸다
        # (값은 같고, 체크포인팅이라 남는 메모리는 블록 입력뿐이며 바로 풀린다). 배율 0·1 을 오가며 죽던 문제는 GPU-008
        # (moss_extras.prepare_compile 의 specialize_float)
        compiled = bool(getattr(args, "compile", False))
        net.set_multiplier(0.0)
        try:
            with (torch.enable_grad() if compiled else torch.no_grad()), accelerator.autocast():
                x, e = inputs() if compiled else (noisy, emb)
                target = unet(x, timesteps, e, mask).detach().float()
                del x, e
        finally:
            net.set_multiplier(1.0)
        x, e = inputs()
        with accelerator.autocast():
            pred = unet(x, timesteps, e, mask)
        loss = torch.nn.functional.mse_loss(pred.float(), target)
        accelerator.backward(loss * float(args.dop_weight))
        v = float(loss.detach().item())
        self._dop_sum += v
        self._dop_n += 1
        self._dop_total += 1
        if self._dop_n >= LOG_EVERY or self._dop_total == 1:
            print(f"dop: 손실 평균 {self._dop_sum / self._dop_n:.5f} (최근 {self._dop_n}번, 지금까지 {self._dop_total}번)", flush=True)
            self._dop_sum, self._dop_n = 0.0, 0
