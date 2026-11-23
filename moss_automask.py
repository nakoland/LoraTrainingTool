# 마스크 자동 만들기 작업 프로세스 (2026-11-12, Fizgig 의 automask 와 같은 방식).
# Google MediaPipe 의 multiclass selfie 분할 모델(selfie_multiclass_256x256.tflite, Apache-2.0, 16MB)을 LiteRT(ai-edge-litert)로
# CPU 에서 돌려 픽셀마다 배경 / 머리카락 / 몸 피부 / 얼굴 피부 / 옷 / 기타(장신구)를 고르고, 그 조합으로 마스크를 만든다.
# 이 venv 에는 ai-edge-litert 가 없어 moss_ui.py 가 그것이 있는 파이썬(Fizgig venv 등)으로 실행한다 → numpy·PIL·ai_edge_litert 만 쓴다.
# 표준 입력: JSON 한 줄 {model, kind, overwrite, jobs: [{image, out, existing}]} 또는 {model, kind, one: {image, out}}
# 표준 출력: 그림마다 JSON 한 줄 {i, n, image, result: made|kept|no_person|error, msg}, 끝에 {done: true, counts}
import json
import os
import sys

import numpy as np
from PIL import Image

BACKGROUND, HAIR, BODY, FACE, CLOTHES, OTHER = range(6)
MIN_COVER = 0.01  # 흰 곳이 이보다 적으면(찾은 것이 없음) 마스크를 만들지 않는다 → 그 그림은 전체를 배운다
KINDS = ("person", "head", "body_clothes", "clothes", "not_faces", "background")


class Segmenter:
    def __init__(self, path):
        from ai_edge_litert.interpreter import Interpreter
        self.it = Interpreter(model_path=path)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        self.out = self.it.get_output_details()[0]
        self.h, self.w = int(self.inp["shape"][1]), int(self.inp["shape"][2])

    def classes(self, img):
        """(H, W, 6) 확률, 그림 원래 크기."""
        x = np.asarray(img.resize((self.w, self.h), Image.BILINEAR), np.float32) / 255.0
        self.it.set_tensor(self.inp["index"], x[None])
        self.it.invoke()
        y = self.it.get_tensor(self.out["index"])[0].astype(np.float32)
        if y.min() < 0 or y.max() > 1.0001:  # 로짓이면 softmax
            y = np.exp(y - y.max(-1, keepdims=True))
            y /= y.sum(-1, keepdims=True)
        return np.stack([np.asarray(Image.fromarray(y[..., c]).resize(img.size, Image.BILINEAR)) for c in range(y.shape[-1])], -1)


def mask_from(c, kind):
    if kind == "person":
        m = 1.0 - c[..., BACKGROUND]
    elif kind == "head":
        m = c[..., FACE] + c[..., HAIR]
    elif kind == "body_clothes":
        m = c[..., BODY] + c[..., CLOTHES] + c[..., OTHER]
    elif kind == "clothes":
        m = c[..., CLOTHES] + c[..., OTHER]
    elif kind == "not_faces":
        m = 1.0 - (c[..., FACE] + c[..., HAIR])
    elif kind == "background":
        m = c[..., BACKGROUND]
    else:
        raise ValueError(f"모르는 마스크 종류: {kind}")
    return np.clip(m, 0.0, 1.0)


def load_rgb(path):
    """학습(sd-scripts load_image)처럼 RGB 로. 투명한 그림은 흰 바탕에 올린다(투명 픽셀의 숨은 색 대신)."""
    with Image.open(path) as im:
        if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
            rgba = im.convert("RGBA")
            base = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            return Image.alpha_composite(base, rgba).convert("RGB")
        return im.convert("RGB")


def make_one(seg, image, out, kind):
    img = load_rgb(image)
    cls = seg.classes(img)
    if float((1.0 - cls[..., BACKGROUND]).mean()) < MIN_COVER and kind != "background":
        return "no_person", "사람을 찾지 못했습니다"
    m = mask_from(cls, kind)
    if float(m.mean()) < MIN_COVER:
        return "no_person", "그 종류를 찾지 못했습니다"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    Image.fromarray((m * 255).round().astype(np.uint8), "L").save(tmp, format="PNG")  # .msk 도 내용은 PNG
    os.replace(tmp, out)
    return "made", f"흰 곳 {m.mean() * 100:.0f}%"


def say(msg):
    sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main():
    req = json.loads(sys.stdin.readline().lstrip("﻿"))
    seg = Segmenter(req["model"])
    kind = req["kind"]
    if req.get("one"):
        j = req["one"]
        try:
            result, msg = make_one(seg, j["image"], j["out"], kind)
        except Exception as e:
            result, msg = "error", str(e) or type(e).__name__
        say({"done": True, "result": result, "msg": msg})
        return
    jobs = req["jobs"]
    counts = {"made": 0, "kept": 0, "no_person": 0, "error": 0}
    for i, j in enumerate(jobs, 1):
        if j.get("existing") and not req.get("overwrite"):
            result, msg = "kept", "이미 있는 마스크를 그대로 둡니다"
        else:
            try:
                result, msg = make_one(seg, j["image"], j["out"], kind)  # out = 그림 옆 <이름>.msk(예전 마스크 폴더의 것보다 우선)
            except Exception as e:
                result, msg = "error", str(e) or type(e).__name__
        counts[result] += 1
        say({"i": i, "n": len(jobs), "image": j["image"], "result": result, "msg": msg})
    say({"done": True, "counts": counts})


if __name__ == "__main__":
    main()
