# TS — Isaac Sim 을 GPU 서버에서 돌릴 수 있는가 (2026-09-01)

- 작성자: 김준태 (트랙 B)
- 이슈: S15P21A103-65 관련 · 확신도: 🔵 공식 문서·벤더 답변 확인
- **판정: 불가.** 우회 방법 없음

## 질문

GPU 서버를 받았다 (Tesla V100-PCIE-32GB × 10, 총 320GB). **여기서 Isaac Sim 을 돌릴 수 있나?**

"메모리가 320GB나 되니 되지 않을까"가 자연스러운 기대였다. 답은 아니다.

## 어떻게 좁혔나

### 1. 먼저 하드웨어 사실을 확인했다

`nvidia-smi` 출력:

```
NVIDIA-SMI 570.211.01   Driver 570.211.01   CUDA Version: 12.8
GPU 0~9  Tesla V100-PCIE-32GB   32768MiB   250W
```

**Tesla V100 = Volta (2017).** RT 코어는 **Turing (2018) 부터** 들어갔다. 즉 V100 에는 하드웨어
레이트레이싱 유닛이 **0개**다.

### 2. Isaac Sim 이 RT 코어를 요구하는지 확인했다

Isaac Sim 5.1 공식 요구사항 문서에 직접 적혀 있다 🔵:

> **"GPUs without RT Cores (A100, H100) are not supported."**

NVIDIA 개발자 포럼의 벤더 답변도 같다 🔵:

> **"Isaac Sim is not supported on V100. It requires an RTX capable GPU."**

### 3. 다른 요구사항도 대조했다

| 항목 | Isaac Sim 5.1 요구 | 우리 서버 | 판정 |
|---|---|---|---|
| GPU | RT 코어 필수, 최소 **RTX 4080 16GB** | V100-PCIE-32GB (RT 코어 없음) | **불가** |
| Linux 드라이버 | **580.65.06** 이상 | **570.211.01** | 미달 |
| VRAM | 16GB | 32GB | 충족 (무의미) |

**드라이버는 올릴 수 있지만 RT 코어는 못 만든다.** 그래서 이건 설정 문제가 아니라 하드웨어 문제다.

## 왜 320GB 가 도움이 안 되는가

Isaac Sim 의 병목은 메모리가 아니라 **Omniverse RTX 렌더러가 요구하는 하드웨어 레이트레이싱**이다.
V100 을 10장 붙여도 RT 코어의 합계는 여전히 0이다. 이건 GPU 를 늘려서 푸는 종류의 제약이 아니다.

같은 이유로 A100·H100 도 지원 목록에서 제외돼 있다 — **데이터센터 GPU 는 대체로 RT 코어가 없다.**

## 이미 알고 있던 제약이었다

프로젝트 지침에 이렇게 적혀 있었다:

> Isaac Sim 은 RT 코어 필요 → H200 에서 렌더링 불가. 주력 시뮬레이터는 MuJoCo

GPU 가 H200 에서 V100 으로 바뀌었지만 **RT 코어가 없다는 사실은 그대로**라 결론이 바뀌지 않는다.
즉 이 TS 는 새 발견이 아니라 **새 하드웨어에 대한 재확인**이다.

## 어디서는 되나

**RTX 5070 (Blackwell, 개인 장비).** RT 코어가 있다.

단서 두 가지 🟡:
- 공식 최소가 RTX 4080 **16GB** 인데 5070 은 12GB → **VRAM 미달**. 씬 규모 제약 가능
- 드라이버를 **580 이상**으로 올려야 한다

## 그래서 어떻게 하나

**MuJoCo 를 유지한다.** 양팔 데모도 MuJoCo 로 된다 — 공식 MJCF 를 두 번 include 하면 되고,
우리 씬이 config 주입 구조라 구조적으로 막힘이 없다 🟢 (D-AI-13).

Isaac 은 **"왜 Isaac 인가"가 숫자로 정해지기 전에는 착수하지 않는다.** 현재 우리 병목은
UMI 수용률과 실물 수집이지 시뮬 렌더 품질이 아니다.

개인 장비에서 시도할 경우의 게이트 (결과 보기 전에 확정):

```
타임박스 1일
- 5070 12GB 에서 SO-101 씬 + 물체 + 카메라 2대가 뜨는가
- 224×224 렌더가 MuJoCo 대비 몇 배 느린가
→ 둘 중 하나라도 실패하면 그 자리에서 접고 REJECTED.md 에 기록한다
```

## 배운 것

**"메모리가 크다"가 "무엇이든 돌아간다"를 뜻하지 않는다.** 가속기마다 없는 유닛이 있고,
그 유닛을 요구하는 소프트웨어는 GPU 수로 우회되지 않는다. 새 하드웨어를 받으면
**용량이 아니라 아키텍처 세대와 유닛 구성**을 먼저 확인해야 한다.

같은 방식으로 이번에 함께 확인된 것: V100 은 **PCIe 모델이라 NVLink 도 없다**(SXM2 가 아님).
그래서 멀티 GPU 는 DDP 가 아니라 **GPU 당 독립 잡**이 맞다.

## 출처

- Isaac Sim 5.1 Requirements — https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html
- NVIDIA Developer Forums, "Can isaac sim install on V100?" — https://forums.developer.nvidia.com/t/can-isaac-sim-install-on-v100/220970
