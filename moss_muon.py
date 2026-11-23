"""Muon 을 LoRA 행렬 모양별로 묶어 계산하는 판 (2026-11-11, Fizgig v9.2.1 의 BatchedMuon 과 같은 생각).

torch.optim.Muon(torch 2.9 `_single_tensor_muon`)은 파라미터마다 Newton-Schulz 를 따로 돌린다. LoRA 는 작은 행렬이
수백 개라(MOSS rank 32: 504개) 작은 행렬곱 3 × 5 × 504 번이 커널 실행 비용으로 늘어진다. 여기서는 모양이 같은 행렬을
쌓아 한 번에 계산한다. 계산은 torch.optim.Muon 과 같다:
  buf ← lerp(buf, g, 1 − μ);  u = nesterov ? lerp(g, buf, μ) : buf;  O = NS(u)  (bf16, 행렬마다 자기 노름으로 나눔)
  θ ← θ · (1 − lr · wd);  θ ← θ − adjust_lr(lr, 모양) · O
상태 이름(`momentum_buffer`)과 param_groups 키가 torch.optim.Muon 과 같아 이어하기 상태 폴더를 서로 읽을 수 있다.

moss_extras.prepare_args 가 학습 프로세스 안에서만 `torch.optim.Muon` 을 이 클래스로 바꾼다(sd-scripts 의
get_optimizer 가 `getattr(torch.optim, "Muon")` 으로 찾음). `train_moss.bat`(원래 학습 파일)은 그대로 torch 것을 쓴다.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

EPS = 1e-7
DEFAULT_COEFFS = (3.4445, -4.7750, 2.0315)
DEFAULT_NS_STEPS = 5


def zeropower_batched(x: Tensor, coeffs: tuple[float, float, float], steps: int, eps: float) -> Tensor:
    """(N, m, n) 묶음의 각 행렬에 torch `_zeropower_via_newtonschulz` 와 같은 계산을 한다."""
    if steps >= 100:
        raise ValueError("Number of steps must be less than 100 for computational efficiency")
    a, b, c = coeffs
    x = x.bfloat16()
    tall = x.size(-2) > x.size(-1)
    if tall:
        x = x.transpose(-2, -1)
    x = x / x.norm(dim=(-2, -1), keepdim=True).clamp(min=eps)
    for _ in range(steps):
        gram = x @ x.transpose(-2, -1)
        upd = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        x = torch.baddbmm(x, upd, x, beta=a)
    return x.transpose(-2, -1) if tall else x


def adjust_lr(lr: float, fn: Optional[str], shape) -> float:
    A, B = shape[:2]
    if fn is None or fn == "original":
        return lr * math.sqrt(max(1, A / B))
    if fn == "match_rms_adamw":
        return lr * 0.2 * math.sqrt(max(A, B))
    return lr


class BatchedMuon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, weight_decay: float = 0.1, momentum: float = 0.95, nesterov: bool = True,
                 ns_coefficients: tuple[float, float, float] = DEFAULT_COEFFS, eps: float = EPS,
                 ns_steps: int = DEFAULT_NS_STEPS, adjust_lr_fn: Optional[str] = None) -> None:
        if not 0.0 <= float(lr):
            raise ValueError(f"Learning rate should be >= 0 but is: {lr}")
        if not 0.0 <= momentum:
            raise ValueError(f"momentum should be >= 0 but is: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"weight decay should be >= 0 but is: {weight_decay}")
        if adjust_lr_fn is not None and adjust_lr_fn not in ("original", "match_rms_adamw"):
            raise ValueError(f"Adjust learning rate function {adjust_lr_fn} is not supported")
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov,
                                      ns_coefficients=ns_coefficients, eps=eps, ns_steps=ns_steps, adjust_lr_fn=adjust_lr_fn))
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(f"Muon only supports 2D parameters whereas we found a parameter with size: {p.size()}")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, wd, mom = float(group["lr"]), float(group["weight_decay"]), float(group["momentum"])
            ps = [p for p in group["params"] if p.grad is not None]
            if not ps:
                continue
            grads = [p.grad for p in ps]
            if any(g.is_sparse for g in grads):
                raise RuntimeError("Muon does not support sparse gradients")
            bufs = []
            for p, g in zip(ps, grads):
                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(g, memory_format=torch.preserve_format)
                bufs.append(st["momentum_buffer"])
            torch._foreach_lerp_(bufs, grads, 1.0 - mom)
            ups = [g.lerp(b, mom) for g, b in zip(grads, bufs)] if group["nesterov"] else bufs
            if wd != 0.0:
                torch._foreach_mul_(ps, 1.0 - lr * wd)
            by_shape: dict = {}
            for i, p in enumerate(ps):
                by_shape.setdefault((tuple(p.shape), p.device, p.dtype), []).append(i)
            for (shape, _dev, dtype), idx in by_shape.items():
                ortho = zeropower_batched(torch.stack([ups[i] for i in idx]), group["ns_coefficients"],
                                          int(group["ns_steps"]), float(group["eps"]))
                torch._foreach_add_([ps[i] for i in idx], list(ortho.to(dtype).unbind(0)), alpha=-adjust_lr(lr, group["adjust_lr_fn"], shape))
        return loss


def install() -> bool:
    """학습 프로세스 안에서 torch.optim.Muon 을 BatchedMuon 으로 바꾼다. 이미 바꿨거나 torch 에 Muon 이 없으면 False."""
    cur = getattr(torch.optim, "Muon", None)
    if cur is None or cur is BatchedMuon:
        return False
    torch.optim._muon_original = cur  # 비교·되돌리기용
    torch.optim.Muon = BatchedMuon
    return True
