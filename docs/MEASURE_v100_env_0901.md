# MEASURE — V100 ×10 학습 환경 검증 (2026-09-01)

- 작성자: 김준태 (트랙 B) · 이슈: S15P21A103-111
- 확신도: 🟢 실행·출력 확인
- **판정: 통과.** 환경 고정 완료, M2·M3 완료

## 조건

| 항목 | 값 |
|---|---|
| 서버 | TLJH JupyterHub (`jupyter01`), 공유. sudo 불가(`no new privileges`) |
| GPU | **Tesla V100-PCIE-32GB × 10** (총 320GB) · compute capability **(7, 0)** |
| 드라이버 | 570.211.01 (`nvidia-smi` 표기 CUDA 12.8) |
| env | `~/envs/aiot_v100` (conda prefix, 홈에 격리) |
| Python | **3.11.16** (기존 EXP_LOG 기록 3.11.15 와 동일 계열) |
| torch | **2.13.0+cu126** · torchvision 0.28.0+cu126 |
| cuDNN | **9.10.2** (`torch.backends.cudnn.version() = 91002`) |
| 설치 경로 | `--index-url https://download.pytorch.org/whl/cu126` |

## 결과

```
arch      ['sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90']
bf16 실지원 False
GPU       10 장 | Tesla V100-PCIE-32GB (7, 0)
matmul fp16 OK True
conv2d fp16 fwd+bwd OK (4, 32, 112, 112) True
  SDPA flash          불가 — RuntimeError
  SDPA mem_efficient  사용가능
  SDPA math           사용가능
```

| 계측 | 기준 (결과 보기 전 확정) | 실측 | 판정 |
|---|---|---|---|
| **M2** bf16 실지원 | `False` 여야 한다 | `False` | **통과** |
| sm_70 커널 존재 | `arch` 에 `sm_70` 포함 | 포함 | **통과** |
| **M3** SDPA backend | `EFFICIENT_ATTENTION` 가능, `MATH` fallback 아님 | mem_efficient 사용가능 | **통과** |
| cuBLAS fp16 | 유한값 | True | 통과 |
| **cuDNN conv2d fp16 fwd+bwd** | 유한값 | True | **통과** |

FlashAttention 거부 사유가 로그에 그대로 찍혔다 —
*"Flash attention only supports gpu architectures in the range [sm80, sm121]. Attempting to run on a sm 7.0 gpu."*
문서 조사와 실측이 일치한다.

## 내가 제기한 위험 하나가 해소됐다

설치 직전에 이렇게 경고했었다:

> PyTorch 가 cu128 에서 Volta 를 버린 이유가 *"cuDNN 이 최신 바이너리에서 Volta 지원을 버렸다"* 였다 🔵.
> cu126 빌드는 sm_70 을 유지한다지만, **torch 자체 커널은 되는데 cuDNN 경로(Conv2d)만 죽는 조합**이 가능하다.
> 우리 BC 정책은 Conv2d 가 본체다.

**실측 결과 그 일은 일어나지 않았다.** cuDNN 9.10.2 에서 Conv2d fp16 forward+backward 가 정상 동작한다.
즉 cu126 빌드가 묶어 배포하는 cuDNN 은 Volta 경로를 유지하고 있다.

기록해두는 이유: 이 위험은 **문서 조사만으로는 판별할 수 없었고 실행으로만 답이 나왔다.**
같은 형태의 질문(A 는 되는데 B 는 되나)이 나오면 조사보다 1분짜리 실행이 빠르다.

## 확정된 환경 규칙

```bash
# 설치
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 금지
#   cu128 / cu13x 휠   → sm_70 커널 없음. nvidia-smi 의 "CUDA 12.8" 을 보고 깔면 즉사한다
#   xformers ≥0.0.31   → V100 지원 제거됨
#   bf16               → 하드웨어 미지원 (M2 로 확인)
#   flash_attention_2  → sm80+ 전용 (실측 확인)
```

```python
# 학습 코드 기본값
torch_dtype=torch.float16          # bf16 금지
attn_implementation="sdpa"         # flash_attention_2 금지
TrainingArguments(fp16=True, bf16=False, gradient_checkpointing=True)
```

## 아직 안 잰 것

| # | 항목 | 왜 못 쟀나 |
|---|---|---|
| M1 | bf16 체크포인트의 fp16 안전성 (inf/NaN, argmax 일치율 ≥99%) | **VLM 후보 모델을 아직 안 받았다.** 다음 단계 |
| M4 | LoRA 스텝 메모리·속도 스윕 | 모델 필요 |
| M8 | 10잡 동시 실행 시 처리량 저하 | 잡 러너 필요 (S15P21A103-114) |
| M9 | DDP 스케일링 (NVLink 없음) | 모델 필요 |
| M10 | 공유 서버에서 타 사용자 간섭 | 관측 기간 필요 |

## 부수 효과 — LIMITS L17 이 풀릴 수 있다

지금까지 EXP_LOG 의 **모든 기록이 `git_rev: unknown`** 이었다. 계측이 MuJoCo 깔린 환경에서 도는데
그곳이 git 체크아웃이 아니어서, 트래커가 수치를 코드에 묶는 자기 임무를 한 번도 수행하지 못했다.

이 서버에 저장소를 clone 하면 **계측이 git 안에서 돌게 되어 `git_rev` 가 정상 기록된다.**
서버 세팅의 예상 못 한 이득이다.
