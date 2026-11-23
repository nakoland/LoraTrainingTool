# ComfyUI 와 같은 방식의 MOSS-Image 샘플링 (2026-11-10)
# 학습 샘플과 스튜디오 렌더에서 사용한다.
# - 스케줄러: ComfyUI 의 simple / beta / normal 과 RES4LYF 의 beta57(= beta α 0.5, β 0.7). MOSS 의 shift(기본 5)를
#   ComfyUI ModelSamplingDiscreteFlow 와 같이 시그마 표(1000칸)에 적용한다. official = sd-scripts 공식 일정.
# - 샘플러: euler, dpmpp_2m, res_multistep(η 0) — comfy/k_diffusion/sampling.py 와 같은 식.
# - SLG: ComfyUI SkipLayerGuidanceDiTSimple 처럼 정해진 구간(% )에서만 네거티브 쪽 계산에서 블록을 건너뛴다.
# - 네거티브의 ComfyUI 가중치 문법 "(text:2)" 는 가중치를 빼고 글만 쓴다(MOSS 텍스트 인코더에는 가중치 적용이 없다).
import math
import re

import torch

SCHEDULERS = ("simple", "beta57", "beta", "normal", "official")
SAMPLERS = ("euler", "dpmpp_2m", "res_multistep")
DEFAULT_NEGATIVE = "worst quality, bad anatomy, watermark, jpeg artifact"  # Fizgig 의 MOSS 기본 네거티브


def strip_weights(text):
    """ComfyUI 가중치 문법 '(word:1.5)' → 'word', 남는 괄호는 지운다."""
    text = re.sub(r"\(([^():]+):\s*[0-9.]+\)", r"\1", text or "")
    return text.replace("(", "").replace(")", "").strip()


def _sigma_table(shift, total=1000):
    """ComfyUI ModelSamplingDiscreteFlow.sigmas (오름차순): t = 1..1000 / 1000 에 time_snr_shift."""
    t = torch.arange(1, total + 1, dtype=torch.float64) / total
    return shift * t / (1 + (shift - 1) * t)


def _beta_ppf(q, a, b, n=200001):
    """정규화 불완전 베타 함수의 역함수(scipy.stats.beta.ppf 대신). 밀도를 촘촘히 적분해 보간한다."""
    x = torch.linspace(0, 1, n, dtype=torch.float64)
    xm = x.clamp(1e-12, 1 - 1e-12)
    pdf = torch.exp((a - 1) * torch.log(xm) + (b - 1) * torch.log1p(-xm))
    cdf = torch.cumulative_trapezoid(pdf, x)
    cdf = torch.cat([cdf.new_zeros(1), cdf]) / cdf[-1]
    q = torch.as_tensor(q, dtype=torch.float64)
    idx = torch.searchsorted(cdf, q).clamp(1, n - 1)
    c0, c1 = cdf[idx - 1], cdf[idx]
    w = ((q - c0) / (c1 - c0).clamp_min(1e-300)).clamp(0, 1)
    return x[idx - 1] + w * (x[idx] - x[idx - 1])


def get_sigmas(scheduler, steps, shift=5.0):
    """내림차순 시그마 [n+1], 마지막은 0. ComfyUI 와 같은 값."""
    if scheduler == "official":
        from library import moss_image_train_utils as tu
        return tu.get_sampling_sigmas(steps, shift).double()
    table = _sigma_table(shift)
    total = len(table) - 1
    if scheduler == "simple":
        ss = len(table) / steps
        sigs = [float(table[-(1 + int(x * ss))]) for x in range(steps)]
    elif scheduler in ("beta", "beta57"):
        a, b = (0.5, 0.7) if scheduler == "beta57" else (0.6, 0.6)
        ts = 1 - torch.linspace(0, 1, steps + 1, dtype=torch.float64)[:-1]
        ts = torch.round(_beta_ppf(ts, a, b) * total).long().tolist()
        sigs, last = [], -1
        for t in ts:
            if t != last:
                sigs.append(float(table[t]))
            last = t
    elif scheduler == "normal":
        # ComfyUI normal_scheduler: timestep(σ) = σ (multiplier 1) 의 양끝 사이를 고르게 나누고 σ(t) = shift 함수로 되돌린다
        tt = torch.linspace(float(table[-1]), float(table[0]), steps, dtype=torch.float64)
        sigs = [float(shift * t / (1 + (shift - 1) * t)) for t in tt]
    else:
        raise ValueError(f"알 수 없는 스케줄러: {scheduler}")
    return torch.tensor(sigs + [0.0], dtype=torch.float64)


def percent_to_sigma(p, shift=5.0):
    """ComfyUI ModelSamplingDiscreteFlow.percent_to_sigma."""
    if p <= 0:
        return 1.0
    if p >= 1:
        return 0.0
    t = 1.0 - p
    return shift * t / (1 + (shift - 1) * t)


def sample(denoise, x, sigmas, sampler="euler"):
    """denoise(x, sigma, i) → 깨끗한 그림 추정(x0). x 는 σ_max 의 노이즈 잠재. ComfyUI k-diffusion 과 같은 갱신식."""
    sigmas = sigmas.to(x.device, torch.float32)
    old_denoised = old_sigma_down = None
    n = len(sigmas) - 1
    for i in range(n):
        s, s_next = sigmas[i], sigmas[i + 1]
        den = denoise(x, s, i)
        if sampler == "euler":
            x = x + (x - den) / s * (s_next - s)
        elif sampler == "dpmpp_2m":
            t, t_next = -torch.log(s), (-torch.log(s_next) if s_next > 0 else None)
            if s_next == 0:
                x = den
            else:
                h = t_next - t
                if old_denoised is None:
                    x = (s_next / s) * x - torch.expm1(-h) * den
                else:
                    h_last = t - (-torch.log(sigmas[i - 1]))
                    r = h_last / h
                    d = (1 + 1 / (2 * r)) * den - (1 / (2 * r)) * old_denoised
                    x = (s_next / s) * x - torch.expm1(-h) * d
        elif sampler == "res_multistep":
            sigma_down = s_next  # η 0
            if sigma_down == 0 or old_denoised is None:
                x = x + (x - den) / s * (sigma_down - s)
            else:
                t, t_old, t_next, t_prev = -torch.log(s), -torch.log(old_sigma_down), -torch.log(sigma_down), -torch.log(sigmas[i - 1])
                h = t_next - t
                c2 = (t_prev - t_old) / h
                phi1 = torch.expm1(-h) / (-h)
                phi2 = (phi1 - 1.0) / (-h)
                b1 = torch.nan_to_num(phi1 - phi2 / c2, nan=0.0)
                b2 = torch.nan_to_num(phi2 / c2, nan=0.0)
                x = torch.exp(-h) * x + h * (b1 * den + b2 * old_denoised)
            old_sigma_down = sigma_down
        else:
            raise ValueError(f"알 수 없는 샘플러: {sampler}")
        old_denoised = den
    return x


def make_denoiser(dit, pos, pos_mask, neg, neg_mask, cfg, shift=5.0, slg_layers=(), slg_start=0.0, slg_end=0.0,
                  official_skip=()):
    """DiT 를 감싼 denoise 함수. CFG(cfg != 1 이고 네거티브가 있을 때), SLG 구간에서만 네거티브 쪽 블록 건너뛰기.
    official_skip: 모든 스텝에서 네거티브 쪽 블록 건너뛰기(sd-scripts 공식 기본 22)."""
    from library import moss_image_train_utils as tu
    hi_sig = percent_to_sigma(slg_start, shift)
    lo_sig = percent_to_sigma(slg_end, shift)

    def denoise(x, s, i):
        ts = (s * tu.NUM_TRAIN_TIMESTEPS).expand(x.shape[0])
        xin = x.to(dit.dtype)
        v = dit(xin, ts, pos, pos_mask).float()
        if cfg != 1.0 and neg is not None:
            skip = tuple(official_skip)
            if slg_layers and lo_sig <= float(s) <= hi_sig:
                skip = tuple(sorted(set(skip) | set(slg_layers)))
            vu = dit(xin, ts, neg, neg_mask, skip_layers=skip).float() if skip else dit(xin, ts, neg, neg_mask).float()
            v = vu + cfg * (v - vu)
        return x - s * v  # 흐름 모델: x0 = x_t − σ·v

    return denoise


PRESETS = {
    # ComfyUI 프리셋: sft bf16, dpmpp_2m + beta57, 50스텝, CFG 3.5, SLG 없음
    "user": dict(sampler="dpmpp_2m", scheduler="beta57", steps=50, cfg=3.5, slg=False),
    # Fizgig 의 ComfyUI 기준: res_multistep + simple, 30스텝, CFG 4, SLG 22번 블록 앞 절반
    "fizgig": dict(sampler="res_multistep", scheduler="simple", steps=30, cfg=4.0, slg=True),
    # Fizgig 의 빠른 미리보기(2026-11-11, families/moss_image.py 의 SpeedLoRA): 학습 모델(Base) + lodestones Kromoss 8스텝 DMD
    # 터보 LoRA 0.6 + SFT 룩 LoRA 1.5, Euler simple 8스텝, CFG 2, 블록 건너뛰기 없음. force = 프롬프트 줄의 --s·--l 보다 우선
    # (터보 LoRA 는 적은 스텝용). LoRA 두 개는 moss_extras 가 샘플 동안만 덧붙인다
    "turbo": dict(sampler="euler", scheduler="simple", steps=8, cfg=2.0, slg=False, force=True, turbo=True),
    # sd-scripts 공식 샘플러(지금까지의 학습 중 샘플·평가)
    "official": dict(sampler="euler", scheduler="official", steps=30, cfg=3.0, slg=False),
}
