# HANDOVER 2026-09-11 — 트랙 B (정책·평가) 인수인계

이슈 S15P21A103-34 · 브랜치 `ai` · **먼저 읽을 것: `AI/docs/DEVLOG_0910.md` 의 상태 보드**

> **이 문서의 목적은 시뮬 반복이 아니라 W6 MVP(실물 로봇 태스크 1종 자동 수행)로 가는
> 것이다.** 시뮬 성공률을 더 깎는 작업은 아래 §2 우선순위에서 뒤로 밀었다.

---

## 0. 확정 수치 (전부 🟢 실행·로그 확인)

조건: MuJoCo 3.12.0 · 30Hz · 2cm 20g 큐브 · 물체 xy 지터 ±50mm · seeds 3000~3099 ·
n=100 · render · policy-device cpu · 트리 `65370da` 이후

```
scripted                                84/100 · 미폐쇄  1 · 닫는순간  4.7mm · 최근접  0.3mm
bc v5 seed0 (joint_delta_gripper_abs)   11/100 · 미폐쇄 42 · 닫는순간 28.0mm · 최근접 14.3mm
hold 0/100 · zero 0/100
```

고정 폐쇄시각 개입(진단용, 배포 불가):

```
v5 seed0   learned 11 · fixed60 18 · fixed67 35 · fixed75 36 · fixed85 37 · oracle 3
DAgger     learned 1/0/7 · fixed 최고 86/49/44
→ v5 팔의 천장 37% · DAgger 팔은 같은 개입으로 86%. 차이 49%p 가 팔 정밀도의 몫
```

그리퍼 이진 헤드(`joint_delta_gripper_binary`, D-AI-38 초안):

```
B-dagger   6/69/25   평균 33.3%  범위 폭 63%p  구간 28.2~38.8%  rc=0 게이트 통과
           미폐쇄 67/20/40 · 닫는순간 3.7/4.9/5.9mm · val_loss 0.893/0.313/0.603
B-v5      21/23/21   평균 21.7%  범위 폭  2%p  구간 17.4~26.7%  rc=1 게이트 실패
           미폐쇄 27/4/8 · 닫는순간 25.6/25.5/23.6mm · 최근접 17.1/16.8/12.1mm
```

### ⚠️ 인용 금지

- `14/12/2` 는 트리 `1195c41` 의 값이다. 현재 트리 실측은 `11`
- `oracle` 수치 전부 (v5 3/100, DAgger 48/21/24) — 성능이 아니라 **tolerance 미충족률**.
  `never_closed = 91/100` 🟢. `_oracle_ready()` 가 3D 거리 5mm 조건인데 정책의 실패
  에피소드 최근접 수평거리 중앙이 14.3mm 라 조건이 거의 발동하지 않는다 (L72)
- `probe_gripper_schedule.py` 의 `verdict` 필드 — 체크포인트 1~2개 실행에서 무효 (L71)

### 확정된 결론

1. **L1 의 조건부 중앙값 붕괴 진단이 맞았고 개입으로 뒤집혔다.** 그리퍼 명령은 시간상
   다수가 "열림" 이라 L1 최적해가 "항상 열림" 이 된다. 이진 라벨 + BCE 로 빼자
   미폐쇄 42 → 4~27, dagger 닫는순간 4.9mm(scripted 4.7mm 수준)
2. **v5 에 남은 병목은 팔이다.** 라벨 닫힘 53.2%(불균형 없음)인데도 최근접 17.1mm.
   E1 의 천장 37% 와 일치
3. **dagger 는 게이트를 통과했지만 제품이 아니다.** 범위 6~69%. 어느 ckpt 를
   배포하느냐로 완전히 다른 물건이 된다

---

## 1. 지금 무엇이 실물로 가는 것을 막고 있나

정책 성공률이 아니다. **실물 실행 API 가 우리 궤적을 받아주는지가 미검증이다.**

- **L67 🔵** — 실물 API 가속 검사 `5.774*dq/dt^2 <= 3.0` 에서 `dq = v*dt` 이므로
  검사값 = `5.774*v/dt`. **`dt` 를 줄이면 검사값이 커진다.** 30Hz 에서 25.8배,
  1Hz 에서 0.9배. 속도 한계를 딱 지킨 시연(v=0.5333)도 `dt >= 1.026s` 를 요구한다.
  **정책 주기 하향도 시연 감속도 해결책이 아니다**
- `AI/configs/real/so101_ver1.json` 🟢 — `max_speed_rad_s 1.0` ·
  `max_accel_rad_s2 3.0` · `tracking_rad 0.08` · `max_gap_speed_m_s 0.08` ·
  `real_calibration_verified: false`
- `AI/tools/check_real_limits.py` 는 **이미 있다.** 그런데 **scripted 98편에만 돌렸고
  정책 롤아웃에는 한 번도 안 돌렸다.** 실물 서보에 도달할 궤적은 정책 것이다

**즉 "우리 정책이 실물 서보에 도달이나 하는가"에 숫자가 없다.** 로봇 없이 답할 수 있는데
아직 안 물었다.

---

## 2. 작업 순서 — 이 순서대로 한다

### S1. 정책 궤적 실물 실행가능성 관문 ★ 최우선

**질문: 우리 정책의 롤아웃 궤적이 실물 한계를 통과하는가. 통과율은 몇 %인가.**
로봇도 Jetson 도 필요 없다. 오늘 된다.

**만들 것:** `AI/tools/check_policy_real_limits.py`

- `tools/check_real_limits.py` 의 `denorm`·`gripper_rad_to_gap_m`·한계 판정 로직을
  **재사용한다.** 복제 금지 — 두 벌이 되면 어느 쪽으로 잰 건지 사후에 못 가린다
- 입력: `--policy-ckpt` (복수 가능) · `--episodes` · `--seed-base 3000`
- 동작: 롤아웃하며 **명령 행동열**을 에피소드별로 모으고, `so101_ver1.json` 의
  `max_speed_rad_s`·`max_accel_rad_s2`·`limits_rad`·`max_gap_speed_m_s`·
  `max_gap_accel_m_s2` 에 대해 프레임별 위반을 센다
- 출력 (반드시 전부):
  - 에피소드 통과율 (한 프레임도 위반 없는 비율)
  - **관절별·제약별 위반 프레임 비율** — 무엇이 구속하는지 갈려야 한다
  - 위반 크기 분포 (한계 대비 배수). "몇 배 초과" 가 처방을 정한다
  - `--log` 로 EXP_LOG append

**대상:** `dagger_..._gripper_binary_dagger_seed1.pt` (69/100, 현재 최고) +
비교군으로 `scripted`. scripted 는 전문가인데도 위반하면 문제는 정책이 아니라 태스크 설계다.

**착수 전에 `PREREG_policy_real_limits_0911.md` 를 쓰고 예측을 먼저 적는다.**
판정 기준 예시(직접 확정할 것): 통과율 >= 80% → 다음 단계 / 30~80% → 재샘플링 처방 /
< 30% → L67 구조 문제로 [ROS]·HW 와 합의 필요.

**예측 🟡 (전임자):** 통과율이 낮다. L67 의 산수가 30Hz 에서 25.8배를 말하므로
가속 제약이 지배적으로 걸릴 것이다. **그러면 결론은 "정책을 더 학습시켜라"가 아니라
"실행 계층에 리샘플링·시간 스케일링이 필요하다"** 가 된다. 이게 W6 로 가는 실제 경로다.

### S2. S1 결과에 따른 처방 계측

통과율이 낮으면 **정책을 다시 학습하지 말고** 실행 계층을 잰다:

- 궤적을 `dt` 를 늘려 리샘플링(예: 30Hz → 10Hz 웨이포인트 + 보간)했을 때 통과율 곡선
- 시간 스케일링(궤적 전체를 k 배 느리게)에 대한 통과율 곡선
- **그리고 그 느려진 궤적이 시뮬에서 여전히 성공하는가** — 이게 핵심이다.
  느리게 만들어 API 는 통과했는데 물체를 놓치면 아무 의미가 없다

산출물: "실물에서 실행 가능하면서 시뮬 성공률을 유지하는 최대 속도" 한 숫자.
이 숫자가 있어야 [ROS]·HW 와 합의할 것이 생긴다.

### S3. 폐쇄 시각 상한 (`PREREG_oracle_best_tick_0911.md`, 이미 작성됨)

`fixed` 는 **고정 시각**의 상한, 이건 **에피소드별 최적 시각**의 상한. 2패스 —
1패스에서 최근접 틱 t\* 를 찾고 2패스에서 t\* 에 닫는다. **둘의 차이가 "닫는 시각을
관측으로 맞히면 얼마나 버는가"** 다. 예측 🟡 40~55%. 60% 초과면 "v5 병목은 팔" 해석이 틀린 것.

### S4. dagger 범위 63%p 의 원인 (seed0 이 6/100 · 미폐쇄 67)

`val_loss` 가 0.893/0.313/0.603 으로 **롤아웃 이전에 학습에서 갈렸다.** 그런데 지금은
총손실만 기록해서 팔에서 갈렸는지 그리퍼에서 갈렸는지 모른다.

- `ArmL1GripperBCE.forward` 가 두 항을 따로 반환하게 하고 epoch 별 `arm_l1`·`grip_bce` 를
  `history` 와 EXP_LOG `result` 에 남긴다. **기존 총손실 값은 바꾸지 않는다** (기록 연속성)
- 이어서 `tools/probe_gripper_logits.py` — 롤아웃 중 프레임별 `sigmoid(logit)` 덤프.
  seed0 의 확률이 **항상 0.5 아래로 눌려 있으면** 다수 클래스 붕괴 재발,
  **요동치면** 관측이 순간을 못 정하는 것. 처방이 다르다

### S5. 실데이터 (트랙 A·MW 왕복, 병렬 진행)

UMI 번들 1편 수령·검사 완료 → `MEASURE_umi_bundle_intake_0910.md`.
앱 파이프라인은 SPEC 만족 🟢 (OIS/EIS off, AE·AWB lock, pose↔frame 141/141,
30.22Hz 지터 ±0.3ms, IMU skew 7.34ms, hfov 73.2°). 그러나 **시연이 아니다** —
손에 든 폰으로 사무실을 훑은 영상이고 그리퍼·마커·물체가 없다.

MW 에게 요청한 것:

1. 실제 시연 1편 (그리퍼 장착 + 16mm 마커 + 1.5~2.5cm 물체, 안정화 후 유효 20초 이상)
2. **`gripper.csv` (`frame_index, gap_m, status`)** — 지금 없다. 개폐 라벨을 만들 수 없다
3. `usable_segments`·`verdict` 를 `tracking_ok_segments` 로 좁히기 (지금 tracking jump 만
   재면서 "전체 구간 사용 가능" 이라 보고한다)
4. 진동 자동컷은 **불필요** — IMU·pose 를 이미 받으므로 후처리로 우리가 계산한다.
   대신 앱만 아는 것을 달라: `TrackingFailureReason` 컬럼 · 프레임별 특징점 수 ·
   "시연 시작" 버튼 이벤트 타임스탬프. **프레임은 자르지 말 것** (raw 무손실 보존)

---

## 3. 하지 말 것 (근거 포함)

- **5-run 확장으로 dagger 분산을 풀려는 것.** 폭 63%p 는 정밀도가 아니라 학습 불안정이다.
  n 을 올려도 안 줄어든다. D-AI-30 의 5회는 다른 문제를 푼다
- **`pos_weight`·그리퍼 손실 가중치 튜닝.** 사전등록에 1.0 고정으로 박았다.
  결과 보고 고치면 사후 합리화다. `pos_weight` 는 데이터 통계이지 손잡이가 아니다
- **oracle tolerance 스윕.** 반쪽 처방 — ±15mm 에서 재생 성공률이 1/4 이라(🟢)
  발동시켜도 대부분 놓친다. 정의를 바꾸는 S3 로 간다
- **E3 재개** (DAgger 학습 seed 3·4). 대상 공간 `joint_delta_gripper_abs` 는 이제 안 쓴다
- **게이트 옮기기.** n 을 올리는 것과 다른 문제다
- **큐가 도는 동안 `AI/policy`·`eval`·`sim` 수정.** `repeat_runs` 가 seed 마다
  `train_bc.py` 를 새 프로세스로 띄워서 같은 실험 안에서 코드가 갈린다.
  `run_queue` 는 시작 시 1회만 dirty 검사한다 (**HEAD 고정 가드 미구현**)

---

## 4. 파일 저장 위치

| 무엇 | 어디 |
|---|---|
| 사전등록 | `AI/docs/PREREG_<주제>_<MMDD>.md` |
| 측정 결과 | `AI/docs/MEASURE_<주제>_<MMDD>.md` |
| 트러블슈팅 | `AI/docs/TS_<주제>_<MMDD>.md` (「고쳤다」가 아니라 「어떻게 좁혔나」 구조) |
| 결정 초안 | `AI/docs/DECISIONS_<MMDD>_draft.md` (작성자·되돌릴 조건·D-AI 번호 포함) |
| 데브로그·회고 | `AI/docs/DEVLOG_<MMDD>.md` · `AI/docs/KPT_<MMDD>.md` |
| 미검증 | `AI/LIMITS.md` 표 끝에 추가 |
| 기각 | `AI/REJECTED.md` |
| 도구 | `AI/tools/*.py` — **반드시 `sys.path.insert(0, parents[1])` 를 넣는다** |
| 큐 항목 | `AI/queue/pending/<name>.yaml` (`name` = 파일명과 동일) |

**D-AI 번호는 세지 말고 조회한다.** 현재 `origin/ai` 기준 최대 **D-AI-37**, 최대 **L72**.
`DECISIONS_0910_draft.md` 가 **38 을 예약**했고 아직 푸시 전일 수 있다.

```bash
git fetch origin
git grep -oh 'D-AI-[0-9]\+' origin/ai -- AI/docs AI/LIMITS.md AI/REJECTED.md | sed 's/D-AI-//' | sort -n | tail -1
```

포트폴리오 사본 (`/c/Users/SSAFY/Desktop/pjt2자료`):
`02_측정기록/` ← MEASURE\_·PREREG\_·DECISIONS\_ · `05_트러블슈팅/` ← TS\_ ·
`06_회고/` ← DEVLOG\_·KPT\_·HANDOVER\_.
**"최신화했다"고 말하기 전에 해시로 대조한다.**

---

## 5. 명령 — 그대로 복붙

블록 첫 줄의 `# [로컬]` / `# [서버]` 를 확인한다. 프롬프트로 구분:
`(aiot_ai) /c/Users/...` = 로컬 · `(aiot_v100) j-j15a103@jupyter01` = 서버.

```bash
# [서버]  MuJoCo 도구는 항상 이 두 변수가 필요하다
cd ~/S15P21A103/AI && MUJOCO_GL=egl PYTHONPATH=$PWD python tools/<도구>.py --help
```

```bash
# [서버]  큐 — 사전등록 없으면 거부, 더러운 트리면 거부, 모르는 action_space 면 거부
cd ~/S15P21A103/AI && python tools/run_queue.py --dry-run
cd ~/S15P21A103/AI && nohup python tools/run_queue.py --parallel 2 > out/queue_console.log 2>&1 &
```

```bash
# [서버]  진행 확인 — 자식 출력은 콘솔이 아니라 항목 로그로 간다
cd ~/S15P21A103/AI && D=$(ls -td out/queue_*/ | head -1) && tail -n 20 "$D"*.log && cat queue/LEDGER.md
```

```bash
# [로컬]  커밋. 긴 메시지는 파일로 준다 (heredoc 붙여넣기는 빈 줄이 유실되고 절단된다)
cd /c/Users/SSAFY/Desktop/2nd_pjt_2/특화/S15P21A103
git add AI/docs AI/tools AI/queue AI/policy AI/eval AI/LIMITS.md
git commit -F out/msg.txt
```

```bash
# [로컬]  푸시 — 이것만 쓴다. `git push origin ai` 단독 금지 (개인 GitHub 미러가 빠진다)
cd /c/Users/SSAFY/Desktop/2nd_pjt_2/특화/S15P21A103 && bash AI/tools/push_both.sh
```

```bash
# [서버]  받기
cd ~/S15P21A103 && git pull origin ai
```

커밋 메시지: `<타입>: <무엇을 했는가> (S15P21A103-34)` ·
타입 `feat|fix|test|refactor|chore|docs` · **측정이 산출물이면 `test:`**. 끝에:

```
Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

---

## 6. 반복해서 틀린 것 (상세: `claude/규칙_정정_Git_MR.md`)

1. **`PYTHONPATH`·`MUJOCO_GL` 누락 — 하루 세 번.** 규칙 문서로 두 번 실패했고,
   도구에 `sys.path.insert` 를 넣어 **구조로** 막은 뒤에야 끝났다.
   새 도구도 같은 방식으로 만든다
2. **EXP_LOG 수치를 `git_rev` 대조 없이 인용** — 다른 트리의 값을 fixture 로 써서 실행이 죽었다
3. **`push_both.sh` 대신 `git push origin ai`** — 개인 미러가 빠진다
4. **터미널 혼동** — `# [로컬]` 블록을 서버에 붙였다. `cd` 실패는 셸을 안 멈춰서
   **뒤 명령 전부가 잘못된 저장소에서 실행된다**
5. **"계측기 수리"를 "진단 정교화"와 묶어 접었다** — 마감 압력에서 묶으면 계측기가 먼저 잘린다.
   분리해서 판단한다
6. **heredoc 종료자를 본문 줄 끝에 붙였다** — 스크립트 전체가 마크다운으로 먹혔다.
   파일로 낸 스크립트는 `bash -n` 으로 문법 확인 후 넘긴다
7. **DEVLOG 를 안 씀** — `KPT_0907` 과 `KPT_0910` 에 같은 Problem 이 두 번 올라왔다.
   `DEVLOG_<MMDD>.md` 상태 보드를 **매 턴** 갱신한다. 거기 없으면 안 한 것이다

---

## 7. 절대 원칙 (프로젝트 공통)

1. 예상을 결과로 쓰지 않는다 — 미실행 코드·미측정 수치는 "미검증" 명시
2. 계측 없이 결정하지 않는다 — 대책보다 계측기를 먼저 제안
3. 모든 수치에 측정 조건 병기 — 조건 없는 절대값 인용 금지
4. 소급 창작 금지
5. 한계를 먼저 말한다
6. 확신도 표기: 🟢 실행·로그 확인 / 🔵 문서·공개자료 / 🟡 설계 판단만
7. **평가 지표는 롤아웃 성공률이다. validation loss 가 아니다** (🟢 2026-09-01 실증:
   val_loss 를 낮췄는데 롤아웃은 0% 그대로, 독립 학습 2회 재현)
8. **게이트 기준은 결과를 보기 전에 확정한다.** 게이트를 옮기지 말고 n 을 올린다
