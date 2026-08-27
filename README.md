# AI 파트 — SO-101 조작 로봇

SSAFY 특화 프로젝트 (Physical AI 조작 로봇) · FLY ASIA 출품
사람 시연으로 로봇팔이 조작 태스크를 학습하는 시스템의 **AI 파트**.

| | |
|---|---|
| Jira | `S15P21A103` · 보드 15347 |
| 접두어 | `[AI]` (시뮬 환경 구축 자체는 `[ROS]`) |
| 브랜치 | `main ← dev ← ai ← <타입>/ai/<이슈키>-<슬러그>` |
| 트랙 | A = 현실·기하 (데이터) / B = 학습·모델 (정책) |
| 두 트랙 접점 | **`so101_ai.contract` 하나뿐** |

---

## 지금 무엇이 되고 무엇이 안 되는가

**실물 로봇팔·그리퍼·카메라가 아직 없다.** 그래서 지금까지의 모든 수치는 **시뮬**이다.
시뮬 성공률은 실물 성공률을 보장하지 않는다 — 마찰·센서 노이즈·조명·캘리브레이션 오차는
시뮬이 재현하지 못한다.

| | 상태 |
|---|---|
| MuJoCo 씬 (로봇·작업대·물체·카메라 2대) | 됨 |
| 자세구속 IK + 파지 시퀀스 | 됨 |
| 작업공간 도달성 지도 | 됨 |
| 데이터 계약 (코드 + 검증기 + 왕복 검증) | **코드는 됨. 결정 3건 트랙 A 합의 대기** |
| 시뮬 수집 파이프라인 | 됨 (단, 지금 데이터는 학습 부적합) |
| baseline 4종 + 게이트 4종 | 됨 |
| sim↔실물 정책 인터페이스 | 됨 (시뮬레이터 없는 더미 환경에서 증명) |
| **학습 정책 (BC / ACT / Diffusion)** | **없음** — S15P21A103-34 |
| **경량 VLM 파인튜닝** | **없음** — S15P21A103-36 |
| **실물 `RobotEnv` 구현** | **없음** — S15P21A103-46 |
| **실물 성공률** | **해당 없음** |
| H200 서버 | 미할당. 학습·파인튜닝은 할당 후 이관 |

---

## 반복해서 틀리는 지점

**평가 지표는 롤아웃 성공률이다. validation loss 가 아니다.**
모방학습에서 val loss 와 실제 성공률의 상관은 약하다. 손실은 정답 궤적과의 유사도를 재는데,
로봇은 다른 경로로도 성공하고 손실이 낮아도 오차 누적으로 실패한다.
val loss 는 "학습이 망가지지 않았나" 확인용으로만 쓴다.

**게이트 기준은 결과를 보기 전에 확정한다.** 기준은 `so101_ai/eval/rollout.py` 의 `GATES` 에
코드로 박혀 있다. 결과를 보고 고치면 판정이 아니라 사후 합리화다.

**baseline 없이 고급 모델로 넘어가지 않는다.** BC 성능을 재기 전에 ACT/Diffusion 을 돌리면,
성능이 나와도 이유를 설명 못 하고 안 나와도 원인(데이터 vs 모델)을 좁힐 수 없다.

**수집은 단계식으로.** 20개 → 검증 → 프로토콜 확정 → 100개 → 게이트 → 나머지.

**환경은 다양하게, 시연 방식은 동일하게.** 물체 위치·조명·배경은 다양할수록 좋지만,
접근 궤적·속도·그립 위치가 매번 다르면 그것은 다양성이 아니라 노이즈다.

---

## 폴더 구조

```
AI/
├── README.md                  ← 이 파일. 수치 절은 스크립트가 생성한다
├── pyproject.toml             pip install -e . → 임포트 경로 고정
├── configs/
│   └── so101.yaml             ★ 하드웨어 의존 값 전부. 코드에 하드코딩 금지
├── scenes/
│   └── pick_place.xml         공식 MJCF 를 include 만 하고 수정하지 않는다
├── src/so101_ai/
│   ├── paths.py               파일시스템 기준점 (한 곳에서만 계산)
│   ├── contract/              ★ 데이터 계약 — 트랙 A 와의 유일한 접점
│   │   └── episode.py
│   ├── sim/                   시뮬 전부 (MuJoCo 고유한 것은 여기 밖으로 나가지 않는다)
│   │   ├── build_scene.py     씬 조립, 카메라·패드 주입, 정규화
│   │   ├── kinematics.py      자세구속 IK
│   │   ├── env.py             RobotEnv 프로토콜 + MujocoPickEnv
│   │   ├── grasp.py           파지 시퀀스 계측
│   │   ├── workspace.py       도달성 스캔
│   │   ├── gripper.py         턱 간격·볼록껍질 프로브
│   │   └── render_check.py    카메라 렌더 확인
│   ├── policy/
│   │   ├── base.py            Policy 프로토콜
│   │   └── baselines.py       hold / zero / replay / scripted
│   ├── eval/
│   │   ├── rollout.py         롤아웃 성공률 + GATES
│   │   └── interface_check.py 시뮬레이터 없는 더미 환경 관통 증명
│   ├── data/
│   │   ├── collect.py         에피소드 수집
│   │   └── verify.py          데이터셋 전체 계약 게이트
│   ├── vlm/                   경량 VLM 파인튜닝 자리 (비어 있음, 이슈 36)
│   └── tracking/
│       └── exp_log.py         EXP_LOG.jsonl 추가 전용 트래커
├── tools/                     실행 진입점 (얇은 셸)
├── third_party/so101_mujoco/  ★ 공식 SO-101 MJCF·URDF. 수정 금지, 교체만
├── docs/                      MEASURE_* 계측 기록
├── EXP_LOG.jsonl              실행 1건 = 1줄
├── out/                       렌더·격자 산출물 (git 제외)
└── data/                      수집 데이터셋 (git 제외, 에피소드당 약 9MB)
```

### 어디에 무엇을 넣는가

| 넣을 것 | 위치 |
|---|---|
| 새 학습 정책 (BC/ACT/Diffusion) | `src/so101_ai/policy/bc.py` 등. `Policy` 프로토콜만 만족시킨다 |
| VLM 파인튜닝 | `src/so101_ai/vlm/` |
| 새 평가 지표·게이트 | `src/so101_ai/eval/` |
| 실물 로봇 구동 | **여기 아니다.** `RobotEnv` 를 채우는 것은 S15P21A103-46 |
| 관절 범위·링크 길이·카메라 위치 | `configs/so101.yaml`. **코드에 쓰지 마라** |
| 새 실행 명령 | `tools/` 에 3줄 셸 + 패키지 안의 `main()` |

---

## 설치와 실행

```bash
cd AI
pip install -e .                    # 임포트 경로 고정 (sys.path 조작 불필요)
pip install -r requirements-sim.txt
export MUJOCO_GL=egl                # 헤드리스 렌더링
```

```bash
python tools/render_check.py                       # 씬·카메라 확인 → out/*.png
python tools/grasp_check.py --trials 20 --seed 0 --log
python tools/reach_scan.py --x-range 0.05 0.40 --y-range -0.20 0.20 --step 0.025 \
                           --wrist-roll 0.0 --log  # 도달성 지도
python tools/gripper_probe.py --log                # 턱 간격·껍질 관통
python tools/collect_sim.py --episodes 20          # 수집 → data/sim_pick_v0
python tools/verify_dataset.py data/sim_pick_v0 --log
python tools/check_interface.py                    # 시뮬 없는 환경 관통 증명
python tools/eval_rollout.py --replay-from data/sim_pick_v0 --replay-tolerance --log
python tools/update_readme.py                      # 아래 수치 절 재생성
```

`--log` 를 준 실행만 `EXP_LOG.jsonl` 에 남고, README 수치는 그 로그에서 생성된다.
**로그에 없는 실험은 "미측정"으로 표시된다.**

⚠️ 계측은 MuJoCo 가 깔린 곳에서 돌기 때문에 git 체크아웃 밖일 수 있다. 그럴 때
`git_rev` 는 `unknown` 이 되고, 대신 `code_sha`(소스 트리 해시)로 코드를 특정한다.
`git_rev: unknown` 인 기록은 커밋으로 되짚어갈 수 없다는 뜻이다.

---

## 측정 수치

<!-- MEASURED:BEGIN — tools/update_readme.py 가 생성한다. 손으로 고치지 마라 -->

모든 수치는 **시뮬**이며, 각 실험의 **가장 최근 로그**에서 자동 생성된다.
직접 고치지 마라 — `python tools/update_readme.py` 로 다시 만든다.

### 스크립트 파지 성공률

**20/20 = 100.0%**

- 조건: 물체 xy 무작위 ±30mm, 시드 0, 제어 30Hz, 물체 반치수 [0.01, 0.01, 0.01]m, close_cmd 0.06
- 성공 판정: 들어올린 높이 >= 0.05m 이고 종료 시 접촉 유지

<sub>git `없음` (계측이 git 체크아웃 밖에서 돌았다) · code `68f9c2990c6e` · MuJoCo 3.12.0 · Python 3.11.15 · config `d9f997eadd6f` · 2026-08-27T06:40:17+00:00</sub>

### 작업공간 도달성

**163/255 = 63.9%**

- 최대 연속 가능 영역: x[0.100, 0.250] y[-0.175, 0.175] = **15cm x 35cm**
- 조건: x[0.05, 0.4] y[-0.2, 0.2] 격자 2.5cm, 파지 높이 0.018m, wrist_roll 0.0
- 판정: pos_err<5mm AND axis_err<5deg AND 반복 중 관절한계에 걸리지 않음
- ⚠️ 기구학만 본다. 충돌은 검사하지 않는다

<sub>git `없음` (계측이 git 체크아웃 밖에서 돌았다) · code `68f9c2990c6e` · MuJoCo 3.12.0 · Python 3.11.15 · config `d9f997eadd6f` · 2026-08-27T06:40:26+00:00</sub>

### baseline 성공률과 게이트

지표는 **롤아웃 성공률 (validation loss 아님)**

| 정책 | 성공률 | 실물 배포 |
|---|---|---|
| hold | 0.0% | 가능 |
| zero | 0.0% | 가능 |
| replay | 0.0% | 가능 |
| scripted | 95.0% | **불가 — 특권정보** |

- 조건: 20 에피소드, 시드 1000~1019, 물체 xy ±50mm, 모든 정책이 동일 시드, 렌더 없음

**게이트 판정** (기준은 결과 확인 전에 `so101_ai/eval/rollout.py` 의 `GATES` 에 확정):

- `task_validity` replay 0.0% < 30% → 통과
- `ceiling` scripted 95.0% >= 80% → 통과
- `floor`/`chance` → **학습 정책은 20.0% 를 넘어야 의미가 있다**

⚠️ scripted 는 성능이 아니라 **상한선**이다. 물체의 정답 위치를 시뮬에서 직접 읽는다.

**replay 위치 허용오차** — 태스크가 요구하는 정밀도

원본 `ep_00000.npz`, 기록 위치 (0.2337, -0.0230) → 성공

| 물체 이동량 | 성공 (4방향) |
|---|---|
| ±5mm | 4/4 |
| ±10mm | 3/4 |
| ±15mm | 1/4 |
| ±20mm | 1/4 |
| ±30mm | 1/4 |
| ±50mm | 0/4 |

<sub>git `없음` (계측이 git 체크아웃 밖에서 돌았다) · code `68f9c2990c6e` · MuJoCo 3.12.0 · Python 3.11.15 · config `d9f997eadd6f` · 2026-08-27T06:40:34+00:00</sub>

### 데이터 계약 왕복 검증

에피소드 **17건**, 계약 위반 **0건**

- 데이터셋: `data/sim_pick_v0`
- 파지 성공 17/17
- state 값 범위 실측 [-0.8182, 0.9939] (계약 [-1, 1])
- 용량 평균 9.20MB/에피소드 → 100건 약 0.92GB, 1000건 약 9.2GB (10진 GB. BE 파트에 전달한 수치와 같은 단위)

<sub>git `없음` (계측이 git 체크아웃 밖에서 돌았다) · code `68f9c2990c6e` · MuJoCo 3.12.0 · Python 3.11.15 · 2026-08-27T06:40:14+00:00</sub>

### 그리퍼 접촉 형상 (MuJoCo 볼록껍질 문제)

MuJoCo 는 메시를 **볼록껍질**로 충돌시킨다. 두 갈래 그리퍼의 껍질은 턱 사이 공간을 메우므로,
공식 MJCF 그대로는 완전히 벌려도 물체가 턱에 들어가지 못한다. 이것이 파지 0/24 의 원인이었다.

| 설정 | 파지점 관통 | 주변 격자 (24점) |
|---|---|---|
| 공식 MJCF 그대로 | 1.46cm | 20/24점 막힘, 최대 3.13cm |
| 접촉패드 주입 (현재) | 0.30cm | 16/24점 막힘, 최대 1.30cm |

- 조건: 물체 반치수 [0.01, 0.01, 0.01] m, open_cmd 0.6
- 격자: dz -0.045..-0.095 x dx -0.010..0.020, gripper body local
- 턱 간격: 닫힘 0.53cm (-0.175 rad) → 완전개방 7.94cm (1.745 rad), 13점 실측
- 패드는 `configs/so101.yaml` 의 `gripper_pads` 로 **로드 시점에 주입**한다. 공식 MJCF 는 수정하지 않는다
- ⚠️ 관통이 0 이 아니어도 파지는 성공한다. 이 수치는 껍질 형상 진단용이고, 실제 판정은 파지 성공률이다

<sub>git `없음` (계측이 git 체크아웃 밖에서 돌았다) · code `68f9c2990c6e` · MuJoCo 3.12.0 · Python 3.11.15 · config `d9f997eadd6f` · 2026-08-27T06:40:12+00:00</sub>

<!-- MEASURED:END -->

---

## 아직 결정되지 않은 것 — 트랙 A 확인 필요

`so101_ai.contract` 는 두 트랙의 유일한 접점이고, 바뀌면 이미 수집한 에피소드가 전부
무효가 된다. 아래 3건은 **합의와 D-AI 기록이 먼저** 있어야 한다. (S15P21A103-27)

1. **관절 정규화 규칙** — 6축 동일 선형 매핑, 그리퍼 특별 취급 없음 (제안)
2. **타임스탬프 필드 분리** — `state_timestamp` / `action_timestamp` (제안)
3. **정규화 범위 초과값 처리** — 클립 / 범위 확대 / 에피소드 폐기 (**제안 없음**)

3번은 실측 근거가 있다: 18건 수집 중 1건에서 정규화 state 최댓값이 **1.0062** 로
계약 `[-1, 1]` 을 벗어나 검증기가 저장을 거부했다. MuJoCo 관절 한계는 soft 이고,
실물 엔코더도 캘리브된 한계를 넘은 값을 보고할 수 있다 (백래시, 캘리브 오차, 역구동).

---

## 문서

| 문서 | 내용 |
|---|---|
| `docs/MEASURE_mujoco_scene_0827.md` | 씬 구축, 카메라 배치, config↔MJCF 대조 |
| `docs/MEASURE_grasp_0827.md` | 파지 가능성, 볼록껍질 문제, 도달성, 계약 왕복 |
| `docs/MEASURE_baseline_0827.md` | baseline 4종, 게이트, 허용오차, 인터페이스 증명 |

작성자 표기 규칙: 결정은 `D-AI-{n}`, 계측은 `MEASURE_*`, 트러블슈팅은 `TS_*`.
각 항목에 작성자와 되돌릴 조건을 적는다.
