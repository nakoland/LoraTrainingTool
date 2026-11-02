# LoRA Training Tool

[MOSS-Image](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base) LoRA 를 학습하는 Windows 용 로컬 웹 UI 입니다. 학습은 [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) 로 하고, 이 도구는 데이터셋 점검, 설정, 실행, 모니터링을 한 화면에서 다룹니다.

- 이미지 폴더 고르기와 캡션·해상도·캐시 점검
- 학습 입력으로 들어가는 처리 결과와 VAE 복원 미리보기
- 학습률, Rank/Alpha, 타임스텝, 메모리 설정과 도움말
- 학습 시작·중지·이어하기, 진행률과 loss 그래프, GPU·메모리 게이지
- 샘플 프롬프트 편집과 결과 샘플·LoRA 파일 모아 보기

## 필요한 것

| 항목 | 내용 |
|---|---|
| 운영체제 | Windows 10/11 64비트 |
| GPU | NVIDIA, VRAM 12GB 이상 권장, 최신 드라이버 |
| 디스크 | 설치 약 12GB + 모델 약 27GB + 학습 결과 |
| 인터넷 | 설치와 모델을 받을 때 필요 |

파이썬이나 git 은 미리 설치하지 않아도 됩니다.

## 설치

1. 이 저장소를 받습니다. 초록색 **Code → Download ZIP** 으로 받아 압축을 풀어도 됩니다.
2. **`install.bat`** 을 더블 클릭합니다.

설치는 10~20분 걸리고, 받은 파일은 모두 이 폴더 안에 둡니다. 시스템 파이썬과 PATH 는 바꾸지 않습니다. 중간에 실패하면 `install.bat` 을 다시 실행하세요. 끝난 단계는 건너뜁니다.

폴더를 옮기면 가상 환경과 설정 파일의 경로가 맞지 않게 됩니다. 옮긴 뒤에는 `tools`, `sd-scripts`, `moss_lora_32gb.toml`, `dataset.toml` 을 지우고 `install.bat` 을 다시 실행하세요.

## 모델 준비

모델은 용량이 커서 자동으로 받지 않습니다. [OpenMOSS-Team/MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base/tree/main) 에서 아래 파일을 받아 원하는 폴더에 두고, UI 의 **학습 설정** 탭에서 지정하세요.

| 항목 | 받을 파일 | 크기 | UI 에 지정할 파일 |
|---|---|---|---|
| DiT | `transformer/` 의 `.safetensors` 3개 | 약 10GB | `diffusion_pytorch_model-00001-of-00003.safetensors` |
| 텍스트 인코더 | `text_encoder/` 의 `.safetensors` 4개 | 약 17GB | `model-00001-of-00004.safetensors` |
| VAE | `vae/diffusion_pytorch_model.safetensors` | 약 250MB | 받은 파일 |

여러 조각으로 나뉜 모델은 첫 파일만 지정합니다. 나머지 조각은 같은 폴더에서 자동으로 읽습니다. ComfyUI 용 단일 파일(`moss_image_base_bf16.safetensors`, `qwen3.5_9b_int8_convrot.safetensors`, `flux2-vae.safetensors`)도 쓸 수 있습니다. 토크나이저는 처음 학습할 때 자동으로 받습니다(약 20MB).

모델 사용 조건은 각 모델의 라이선스를 따릅니다.

## 사용

1. **`run.bat`** 을 실행하면 브라우저에 `http://127.0.0.1:7865/` 가 열립니다.
2. **학습 설정** 탭에서 모델 파일을 지정합니다.
3. **데이터셋** 탭에서 이미지 폴더를 고릅니다. 기준 폴더는 도구 안의 `datasets` 이고 화면에서 바꿀 수 있습니다.
4. **실행** 탭에서 전체 점검을 확인하고 학습을 시작합니다. 진행은 **모니터** 탭에서, 결과는 **결과** 탭에서 봅니다.

이미지 폴더는 `<세트>/<반복_이름>` 형식을 권장합니다. 예를 들어 `datasets/my_set/10_style` 은 반복 수 10으로 읽습니다. 캡션은 이미지와 같은 이름의 `.txt` 입니다.

UI 창을 닫아도 진행 중인 학습은 멈추지 않고, 다시 켜면 이어서 표시합니다. 콘솔에서 바로 학습하려면 `train_moss.bat` 을 씁니다.

## VRAM

**메모리·속도** 탭의 프리셋을 고르면 됩니다.

| VRAM | 프리셋 내용 |
|---|---|
| 24GB 이상 | DiT INT8 |
| 16GB | DiT·텍스트 인코더 INT8 |
| 12GB | INT8 + 블록 스왑 28 |

해상도, 배치, Rank 를 올리면 VRAM 을 더 씁니다.

## 폴더 구조

| 경로 | 내용 |
|---|---|
| `moss_ui.py`, `moss_ui.html` | 로컬 웹 UI |
| `moss_train.py` | UI 가 실행하는 학습 파일. 중단 저장과 이어하기를 더합니다 |
| `moss_preview.py` | 자료 미리보기 작업 프로세스 |
| `templates/` | 처음 설치할 때 만드는 설정 파일의 원본 |
| `sd-scripts/` | 설치가 받은 sd-scripts 와 가상 환경 |
| `tools/` | 설치가 받은 uv 와 파이썬 |
| `datasets/`, `output/`, `logs/`, `profiles/` | 내 자료와 결과. git 에 올라가지 않습니다 |

지울 때는 폴더를 통째로 지우면 됩니다.

## 설치가 받는 것

버전을 고정해 받고, uv 와 flash_attn 은 SHA-256 을 확인합니다.

| 구성 요소 | 버전 |
|---|---|
| [sd-scripts](https://github.com/kohya-ss/sd-scripts) (Apache-2.0) | v0.12.3 |
| [uv](https://github.com/astral-sh/uv) | 0.13.6 |
| Python | 3.11 |
| PyTorch | 2.9.1 (CUDA 12.8) |
| [flash_attn 사전 빌드](https://github.com/mjun0812/flash-attention-prebuild-wheels) | 2.8.3 |
| 그 외 패키지 | `requirements-lock.txt` |

이 저장소에는 sd-scripts 코드와 모델이 들어 있지 않습니다.

## 문제 해결

- **학습이 시작되지 않음:** 실행 탭의 전체 점검에 나오는 오류를 먼저 확인하세요. 전체 로그는 `ui_logs/` 에 있습니다.
- **Flash Attention 오류:** 메모리·속도 탭에서 끌 수 있습니다.
- **실행 창이 바로 닫힘:** `install.bat` 이 끝까지 성공했는지 확인하세요.
