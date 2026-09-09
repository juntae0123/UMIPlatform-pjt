# HANDOVER — V100 GPU 서버 사용 (2026-09-09)

작성 김준태(트랙 B) · 인수자: 트랙 A(황도경)
근거: `MEASURE_v100_env_0901.md` 🟢 · `runtime_limits.py` · L51

> **먼저 읽어야 하는 이유.** 이 서버는 **공유**다. 2026-09-07 하루 오후에
> 규약이 네 가지로 깨졌고 넷 다 "규칙을 몰라서"가 아니라 **손으로 명령을
> 다시 타이핑해서** 났다 🟢. 그래서 규약이 문서가 아니라 코드에 들어 있다.

## 1. 접속·환경

| 항목 | 값 |
|---|---|
| 서버 | TLJH JupyterHub (`jupyter01`) · **sudo 불가** (`no new privileges`) |
| 계정 | `j15a103` |
| 저장소 | `~/S15P21A103` |
| conda env | `~/envs/aiot_v100` |
| Python | 3.11.16 · torch **2.13.0+cu126** · torchvision 0.28.0+cu126 · cuDNN 9.10.2 |
| GPU | Tesla V100-PCIE-32GB × 10 · compute capability **(7,0)** · 드라이버 570.211.01 |

로컬(`[로컬]` `aiot_ai`)과 헷갈리지 않게 명령 블록마다 `# [서버]` 를 첫 줄에 쓴다.

## 2. GPU 할당 — 이게 제일 중요하다

```
할당분  1, 2, 3, 4, 6        우리 계정 것
우리 장  2                   기본값. 항상 쓸 수 있다
남의 것  0, 5, 7, 8, 9       잡으면 남의 작업이 죽는다
```

⚠️ **`CUDA_VISIBLE_DEVICES` 기본값은 0 이고 0 은 남의 장이다.**
아무것도 지정하지 않으면 남의 장을 잡는다. `runtime_limits.py` 의 첫 판이
기본값을 `"0"` 으로 뒀다 — 막으려고 쓴 파일에서 막으려던 실수를 냈다 🟢.

**규약: 잡 하나에 한 장.** 여러 장이 필요하면 잡을 나눠 각각 한 장씩 잡는다.
PCIe 이고 NVLink 가 없어서 DDP 보다 **GPU 당 독립 잡**이 맞다.

"한가함" 판정은 **메모리 우선**이다 — 사용률 0% 여도 메모리가 잡혀 있으면 남의
프로세스가 올라가 있다. 실측 🟢: 1.3M 파라미터 잡이 사용률 1% 인데 메모리 1GB.
기준은 `IDLE_MEM_MB=300` · `IDLE_UTIL_PCT=20`.

## 3. 규약은 코드에 있다 — 손으로 쓰지 마라

```python
import runtime_limits          # noqa: F401  — 반드시 numpy/torch 앞
runtime_limits.claim("작업이름")   # 중복 실행이면 여기서 종료
```

이 한 줄이 처리하는 것: BLAS 스레드 5종을 `AI_THREADS`(기본 2)로 · `MUJOCO_GL=egl`
(EGL 있는 곳에서만) · `PYTHONUNBUFFERED=1` · `CUDA_VISIBLE_DEVICES` 를 할당분 중
한가한 한 장으로. 명시 지정은 존중하되 **남의 장이면 경고를 찍는다.**

**`numpy`·`torch` 보다 먼저** 임포트해야 한다. 나중이면 스레드 설정이 **조용히
무시되고** 프로세스가 코어를 다 먹는다. 실측 🟢: 6개 프로세스에서 load average
165/80.

`claim()` 은 락파일(`out/`)이다. 바쁜 터미널에 큐잉된 입력으로 같은 명령이
**4번** 떠서 넷이 같은 로그에 쓴 사고가 있었다 🟢 — 그래서 있다.

### ⚠️ 아직 안 들어간 도구가 있다 (L51 미해소)

임포트하는 것 (보호됨) 🟢:
`policy/bc.py` · `policy/train_bc.py`(→ `tools/train_bc.py` 도 경유해서 보호) ·
`data/dataset.py` · `tools/audit_demos.py` · `audit_demo_lift.py` ·
`lift_height_sweep.py` · `probe_contract.py` · `probe_freeze.py` ·
`probe_recovery.py` · `repeat_runs.py`

**임포트하지 않는 것 (보호 안 됨)** 🟢 2026-09-09 확인:
`tools/collect_sim.py` · `umi_dump_from_dataset.py` · `convert_umi.py` ·
`verify_dataset.py` · `diff_datasets.py` · `eval_rollout.py`

→ 이 6개를 손으로 직접 돌리면 스레드 80개 + GPU 0(남의 장)으로 간다.
`tools/run_umi_layer2.sh` 는 **환경변수를 python 앞에 export 해서** 7단계 전체를
덮었다 (2026-09-09). 그러나 **낱개로 돌릴 때는 안 덮인다.**
근본 조치는 6개 파일 첫 줄에 `import runtime_limits` 를 넣는 것이고,
`collect_sim`·`eval_rollout`·`verify_dataset` 은 **양 트랙 공용이라 합의가 필요**하다.
그래서 내가 고치지 않았다.

## 4. 설치 금지 목록 — 하나라도 어기면 즉사한다

```bash
# 금지
#   cu128 / cu13x 휠     → sm_70 커널 없음.
#                          nvidia-smi 가 "CUDA 12.8" 이라고 찍는 것을 보고 깔면 즉사
#   xformers >= 0.0.31   → V100 지원 제거됨
#   bf16                 → 하드웨어 미지원 (실측 False 🟢)
#   flash_attention_2    → sm80+ 전용 (실측 거부 로그 확인 🟢)
#   Isaac Sim            → Volta 는 RT 코어 0개 (D-AI-16)

# 설치는 이것만
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

```python
# 학습 코드 기본값
torch_dtype=torch.float16          # bf16 금지
attn_implementation="sdpa"         # flash_attention_2 금지
TrainingArguments(fp16=True, bf16=False, gradient_checkpointing=True)
```

SDPA 는 `mem_efficient` 와 `math` 만 가능하다 (`flash` 는 RuntimeError) 🟢.

## 5. 시뮬·렌더

- **`MUJOCO_GL=egl`** 이 없으면 `mujoco.MjrContext` 안에서 크래시하고 메시지가
  **어느 변수가 없는지 말하지 않는다** 🟢
- 반대로 **EGL 없는 기계에서 무조건 설정하면** 렌더를 안 켜도 깨진다 🟢 —
  `runtime_limits` 가 `find_library("EGL")` 로 확인 후에만 넣는다
- 저장소 경로에 한글이 있으면 MuJoCo 가 씬을 못 연다 → `paths.resolve_for_mujoco()`
  가 ASCII 임시 경로로 복사한다 (서버 경로는 ASCII 라 해당 없음)
- 오프스크린 MSAA(`cameras.offsamples`)는 **0 이어야 한다** — 아니면 롤아웃이
  재현되지 않는다 (L37 해소 경위)

## 6. 첫날 순서

```bash
# [서버]
( set -e
  cd ~/S15P21A103/AI
  source ~/envs/aiot_v100/bin/activate 2>/dev/null || conda activate ~/envs/aiot_v100
  python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
  python -c "import runtime_limits as r; print('추천 장:', r.pick_gpu(), '/ 할당분', r.ALLOWED_GPUS)"
)
```

그다음 UMI 파이프라인 관통 (렌더 필요, VM 에서는 못 돈다):

```bash
# [서버]
( set -e
  cd ~/S15P21A103/AI
  bash tools/run_umi_layer2.sh v2 20     # 태그는 항상 다음 번호. 기존 이름 재사용 금지
)
```

7단계 = collect → dump → convert → verify → diff → train → rollout.
0908 실측 총 4511초 (학습 2596 + 롤아웃 1497 = 91%). 데이터 준비는 7분이다.

## 7. 수치를 낼 때의 함정

- **학습 실행 간 성공률이 12~25%p 갈린다** 🟢. 단일 학습의 단일 롤아웃으로
  체크포인트를 비교하지 마라. **3회 반복 + 평균·범위 보고**가 기본
- **`git_rev`·`code_sha` 가 로그 시점에 계산된다** (L45). 몇 시간 도는 잡에서는
  그 트리가 수치를 만든 트리가 아니다. `eval/repeat.py` 만 고쳐져 있고
  `eval/rollout.py`·`tools/*` 는 그대로다
- **추론 장치가 롤아웃 결과를 바꾼다** (L57) — cpu 11/100 vs cuda 12/100 🟢.
  기본값은 `cpu` 로 유지돼 있다(비교선 보존). `--policy-device` 로 명시하고
  **조건에 적어라**
- 평가 지표는 **롤아웃 성공률**이다. val loss 가 아니다 —
  0.00547 → 0.00517 로 낮췄는데 롤아웃은 0% 그대로였다 🟢

## 8. 아직 안 잰 것

| # | 항목 | 왜 |
|---|---|---|
| M8 | 10잡 동시 실행 시 처리량 저하 | 잡 러너 필요 (S15P21A103-114) |
| M9 | DDP 스케일링 (NVLink 없음) | 모델 필요 |
| M10 | 공유 서버에서 타 사용자 간섭 | 관측 기간 필요 |
| — | 우리 UMI 학습을 서버에서 돌린 적이 없다 | 지금까지 **CPU 86.2s/epoch** 으로 했다 |

마지막 줄이 인수자의 첫 이득이다 — 학습이 병목(2596초)이고 그게 CPU 다.
