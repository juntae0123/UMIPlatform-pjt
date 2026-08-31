# RESEARCH — Tesla V100 32GB 에서 VLM 파인튜닝 (2026-08-31)

- 작성자: 김준태 (트랙 B) · 이슈: S15P21A103-36 · -42
- 확신도: 🔵 1차 문서·소스 확인 / 🟡 계산·설계 판단 / 🟢 우리 실측
- ⚠️ **아직 한 번도 실행하지 않았다.** 전부 문서 조사 결과다. §5 계측을 통과하기 전에는 확정이 아니다.

## 0. 용어 정정 2건

1. **vLLM 은 파인튜닝 도구가 아니다.** 추론·서빙 엔진이다. 학습은 PEFT/TRL · ms-swift · LLaMA-Factory 등으로 한다.
   (vLLM 은 추론 시 LoRA 어댑터 서빙은 지원한다 🔵)
2. **vLLM 은 V100 을 공식 지원하지 않는다.** v0.19.0 문서까지 "compute capability 7.0 or higher
   (e.g., **V100**, T4, …)" 였는데 **v0.20.0 부터 "7.5 or higher (e.g., T4, RTX20xx, …)"** 로 바뀌고 V100 이 삭제됐다 🔵.
   CUDA 12.7 이하 툴체인으로 소스 빌드하면 sm_70 이 CMake 에 남아 있으나 공식 지원·테스트가 아니다.

## 1. V100(sm_70) 제약 — 코드 규칙으로 봉인한다

| 제약 | 판정 | 근거 | 코드 규칙 |
|---|---|---|---|
| **bf16** | **불가** | bf16 Tensor Core 는 Ampere 3세대 도입 🔵. Volta 는 fp32 에뮬레이션이라 속도 이득 0. PyTorch SDPA 도 `major >= 8` 에서만 bf16 허용 🔵 | `bf16=False, fp16=True` + loss scaling. **bf16 경로 전면 금지** |
| **FlashAttention-2 / 3** | **불가** | 공식 README: Ampere/Ada/Hopper 🔵. FA3 는 Hopper 전용 | `attn_implementation="sdpa"`. **`flash_attention_2` 금지** |
| **LLM.int8() (8bit)** | **불가** | bitsandbytes 문서: LLM.int8() 은 **cc 7.5+** 🔵 | `load_in_8bit` 사용 금지 |
| NF4 / QLoRA (4bit) | 가능 | NF4/FP4 는 **cc 6.0+** 🔵 | 32GB 면 불필요. 쓸 경우 `bnb_4bit_compute_dtype=float16` |
| xformers | **최신 불가** | CHANGELOG 0.0.31: *"We will no longer support V100 or older GPUs"* 🔵 | `≤0.0.30` 또는 미사용 (PyTorch SDPA 로 대체) |
| PyTorch 휠 | **cu126 고정** | 2.11 부터 **CUDA 12.8 바이너리에서 Volta 제거** (cuDNN 이 Volta 를 버려서) 🔵. cu126 legacy 빌드에만 7.0 잔존 | `--index-url .../whl/cu126` |
| vLLM | 사실상 불가 | §0 | 평가·서빙은 transformers 또는 llama.cpp |

```python
# V100 학습 설정 요지
model = AutoModelForVision2Seq.from_pretrained(
    MODEL, torch_dtype=torch.float16,   # bf16 금지
    attn_implementation="sdpa",         # flash_attention_2 금지
)
TrainingArguments(fp16=True, bf16=False, gradient_checkpointing=True, ...)
assert not torch.cuda.is_bf16_supported(including_emulation=False)  # False 여야 정상
```

## 2. 최대 리스크 — 후보 모델이 전부 bf16 체크포인트다

V100 은 bf16 연산이 없어 fp16 으로 캐스팅해야 하는데, **Qwen-VL 계열에서 fp16 에서 inf/NaN·gibberish 가
나온 이슈가 문서화돼 있다** 🔵 (transformers #33294, #35151). Unsloth 는 bf16 체크포인트 + fp16 설정 충돌로
죽는 이슈가 있다 🔵 (#4082).

**"되겠지"로 넘기면 안 된다.** §5 M1 이 이 항목의 계측이다.

## 3. 모델 후보

| 모델 | 크기 | V100 LoRA | Orin Nano 8GB | 판정 |
|---|---|---|---|---|
| **Qwen2.5-VL-3B-Instruct** | 3B | 레퍼런스 최다 | NVIDIA 직접 권장 🔵 · 공식 GGUF | **1순위 (안전)** |
| **SmolVLM2-500M-Video** | 0.5B | 매우 가벼움 | 여유 큼 · **Apache-2.0** | **2순위 (경량)** |
| Cosmos-Reason2-2B | 2.4B | 카드가 **min 24GB / BF16 only tested / Blackwell·Hopper** 명시 🔵 | 실측치 존재 | 도메인 적합 최고 · **리스크 최고** |

NVIDIA 가이드: *"Jetson Orin Nano 8GB is suitable for VLMs and LLMs up to nearly 4B parameters,
such as Qwen2.5-VL-3B, VILA 1.5–3B, or Gemma-3/4B."* 🔵

## 4. 메모리 산정 🟡

우리 이미지 규격이 `224×224` 라 토큰이 매우 적다. patch 16 + 2×2 merge 기준
`224/16=14 → 196 patch → merge → 이미지 1장 = 49 토큰`, 카메라 2대 = **98 토큰**.

2B / batch 4 / seq 1024 / LoRA r=16 / grad checkpointing:

| 항목 | 크기 |
|---|---|
| Base 가중치 fp16 (frozen) | 4.9 GB |
| LoRA 파라미터 + gradient + AdamW | 0.36 GB |
| Activation 체크포인트 + recompute | 0.65 GB |
| **Logits + CE (vocab 151,936)** | **~3.7 GB ← 최대 항목** |
| CUDA context · 단편화 | ~1.5 GB |
| **합계** | **≈ 11 GB / 32 GB** |

여유가 크다. 우리 규격(seq 200~500)이면 batch 32~64 도 가능 🟡.
**병목은 메모리가 아니라 V100 연산량과 FA2 부재다.**

저비용 최적화: **응답 토큰에만 loss 계산**(프롬프트 `-100` 마스킹) → 최대 항목인 logits 메모리가 줄어
배치를 2~3배 키울 수 있다.

## 5. 착수 전 계측 — 판정 기준을 먼저 못박는다

| # | 계측 | 방법 | 판정 기준 |
|---|---|---|---|
| M1 | **bf16 체크포인트의 fp16 안전성** | fp32/fp16 각각 로드, 동일 입력 20개 forward → logit 최대차, inf/NaN, argmax 일치율 | inf/NaN **0건** AND argmax 일치율 **≥99%** |
| M2 | V100 실효 bf16 | `is_bf16_supported(including_emulation=False)` | `False` 확인 → 코드에서 bf16 봉인 |
| M3 | SDPA backend 실제 선택 | 디버그 로그 | `EFFICIENT_ATTENTION` (`MATH` fallback 아님) |
| M4 | LoRA 스텝 메모리·속도 | 랜덤 텐서 batch/seq 스윕 + `max_memory_allocated` | OOM 없이 목표 batch, loss scale overflow <5% |
| M5 | **Jetson 동시 적재 (이슈 42)** | VLM+정책 동시 로드, `tegrastats` peak | 합계 **≤6.0GB** (OS·ROS2 여유 2GB 🟡) |
| M6 | Jetson VLM 지연 | 224×224 ×2 + 짧은 프롬프트, TTFT/TPS median of 5, cold/warm 구분 | 제어 루프와 **분리된 비동기 경로**에서 TTFT ≤500ms (warm) |

## 6. Jetson 배포 — vLLM 은 여기서도 탈락

Orin Nano 8GB 실측 (MAXN_SUPER · JetPack 6.2.2 · median of 5) 🔵:

| 런타임 | 양자화 | TPS | Peak 메모리 | 정책과 공존 |
|---|---|---|---|---|
| llama.cpp | Q4_K_M | 38 | — | **가능** |
| vLLM (W4A16 port) | AWQ | 56 | **6.9 GB** | **불가 — 정책 자리 없음** |
| TRT Edge-LLM | W4A16 AWQ | 60 | **4.3 GB** | **가능** |

우리 정책이 1,307,974 파라미터 ≈ 5MB 라 🟢 4.3GB VLM 이면 여유가 남는다.
**이슈 42 는 이 3종을 같은 조건으로 비교하면 된다.**

주의: llama.cpp CUDA 빌드가 Jetson iGPU 에서 `cudaMalloc failed` 로 죽는 사례가 있고,
`GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` 이 워크어라운드다 🔵.

## 7. 이 문서가 만드는 결정 후보

1. **환경 고정을 `chore:` 커밋으로** — torch cu126 / xformers ≤0.0.30 또는 미사용 / bnb cu126.
   cu128 이상으로 올리면 V100 이 죽는다 → LIMITS 등재
2. **`flash_attention_2` 금지 · `sdpa` 강제**를 코드 규칙으로 명문화.
   후보 모델 카드들이 FA2 를 권장하므로 **기본값 복사 붙여넣기가 곧 버그다**
3. **bf16 금지 / fp16 + loss scaling** 을 학습 설정 기본값으로
4. **Jetson 8GB 예산에서 vLLM 배제**를 이슈 42 의 사전 판단으로 기록 (peak 6.9GB 근거)
5. VLM 후보 3종으로 축소. M1 을 통과한 것만 남긴다

⚠️ 다만 VLM 자체는 `REJECTED.md` R-AI-4 로 **MVP 제외** 상태다.
이 문서는 되살릴 때를 위한 사전 조사이고, **4주 임계경로에 넣지 않는다.**

## 8. 출처

- NVIDIA Ampere Tuning Guide — https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html
- PyTorch Forums, bfloat16 on V100 — https://discuss.pytorch.org/t/bfloat16-on-nvidia-v100-gpu/201629
- PyTorch dev-discuss, Dropping Volta from CUDA 12.8 — https://dev-discuss.pytorch.org/t/dropping-volta-support-from-cuda-12-8-binaries-for-release-2-11/3290
- Dao-AILab/flash-attention — https://github.com/Dao-AILab/flash-attention
- xformers CHANGELOG — https://github.com/facebookresearch/xformers/blob/main/CHANGELOG.md
- HF bitsandbytes installation (하드웨어 표) — https://huggingface.co/docs/bitsandbytes/installation
- vLLM GPU 요구사항 — https://github.com/vllm-project/vllm/blob/main/docs/getting_started/installation/gpu.cuda.inc.md
- vLLM LoRA — https://github.com/vllm-project/vllm/blob/main/docs/features/lora.md
- NVIDIA blog, Edge AI on Jetson — https://developer.nvidia.com/blog/getting-started-with-edge-ai-on-nvidia-jetson-llms-vlms-and-foundation-models-for-robotics/
- jetson-orin-nano-benchmarks — https://github.com/hokwangchoi/jetson-orin-nano-benchmarks
- ms-swift (V100 명시) — https://github.com/modelscope/ms-swift
- transformers #33294 / #35151 (Qwen2-VL fp16) — https://github.com/huggingface/transformers/issues/33294
