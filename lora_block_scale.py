# LoRA 블록별 강도 조절 (2026-11-10, Fizgig 의 Studio 를 참고)
# 이미 학습한 LoRA 의 블록마다 배율을 곱해 새 파일로 저장한다(재학습 없음). 배율 0 인 블록은 키를 뺀다.
# 실행: sd-scripts\venv\Scripts\python.exe lora_block_scale.py <입력.safetensors> --blocks "34-41:0" [--out 출력.safetensors]
#   --blocks 예: "34-41:0"(뒤 블록 끄기), "24-35:0.5,34-41:0", "0-33:1.2"
# MOSS 는 42블록(layers 0~41). 키 이름의 layers_N / layers.N 으로 블록을 찾는다(kohya·diffusers 형식 모두).
# 배율은 lora_up(또는 LoKr 의 w1·w2 중 하나)에 곱한다. alpha 는 그대로 둔다.
import argparse
import json
import os
import re
import sys

BLOCK_RE = re.compile(r"layers[._](\d+)[._]")


def parse_blocks(text):
    """'34-41:0,24-35:0.5' → {블록: 배율}. 뒤에 쓴 것이 앞을 덮는다."""
    out = {}
    for part in [p.strip() for p in (text or "").split(",") if p.strip()]:
        rng, _, val = part.partition(":")
        if not val:
            raise ValueError(f"'{part}': 블록:배율 형식이어야 합니다 (예: 34-41:0)")
        lo, _, hi = rng.partition("-")
        lo, hi = int(lo), int(hi or lo)
        for b in range(min(lo, hi), max(lo, hi) + 1):
            out[b] = float(val)
    return out


def module_of(key):
    return key.split(".")[0] if not key.startswith("diffusion_model.") else key.rsplit(".", 2)[0]


def scale_lora(sd, scales):
    """반환: (새 state dict, {블록: 바뀐 모듈 수}, 뺀 모듈 수)."""
    import torch
    out, changed, dropped = {}, {}, set()
    for k, v in sd.items():
        m = BLOCK_RE.search(k)
        f = scales.get(int(m.group(1))) if m else None
        if f is None or f == 1.0:
            out[k] = v
            continue
        if f == 0.0:
            dropped.add(module_of(k))
            continue
        # 한 모듈에서 한 행렬에만 곱한다: lora_up / lokr_w1 / (그 밖의 형식이면 down)
        if any(t in k for t in ("lora_up.weight", "lokr_w1", ".lora_B.")):
            out[k] = (v.float() * f).to(v.dtype)
            b = int(m.group(1))
            changed[b] = changed.get(b, 0) + 1
        else:
            out[k] = v
    return out, changed, len(dropped)


def main():
    ap = argparse.ArgumentParser(description="LoRA 블록별 강도 조절")
    ap.add_argument("input")
    ap.add_argument("--blocks", required=True, help='예: "34-41:0" 또는 "24-35:0.5,34-41:0"')
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    scales = parse_blocks(a.blocks)
    sd = load_file(a.input)
    with safe_open(a.input, "pt") as f:
        meta = dict(f.metadata() or {})
    new, changed, dropped = scale_lora(sd, scales)
    if not changed and not dropped:
        print("바뀐 블록이 없습니다 (키 이름에서 layers_N 을 찾지 못했거나 배율이 모두 1)")
        return 1
    desc = "_".join(f"{k}" for k in a.blocks.replace(" ", "").replace(":", "x").replace("-", "to").split(","))
    out = a.out or os.path.splitext(a.input)[0] + f"_bs_{desc}.safetensors"
    meta["ss_block_scale"] = json.dumps({str(k): v for k, v in sorted(scales.items())})
    meta["ss_block_scale_source"] = os.path.basename(a.input)
    save_file(new, out, metadata=meta)
    print(f"저장: {out}")
    print(f"배율을 곱한 블록 {len(changed)}개, 뺀 모듈 {dropped}개 (키 {len(sd)} → {len(new)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
