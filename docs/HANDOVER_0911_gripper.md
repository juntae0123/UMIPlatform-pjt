# HANDOVER 2026-09-11 — 그리퍼 이진 헤드 이후

작성: 김준태(트랙B) 세션 · 이슈 S15P21A103-34
**먼저 읽을 것: `AI/docs/DEVLOG_0910.md` 의 상태 보드.** 거기 없으면 안 한 것이다.

## 0. 지금 어디까지 왔나 (전부 🟢 실행·로그 확인)

트리 `65370da` 이후 · seeds 3000~3099 · n=100 · render · policy-device cpu · jitter ±50mm

```
scripted                                84/100 · 미폐쇄  1 · 닫는순간  4.7mm · 최근접  0.3mm
bc v5 seed0 (joint_delta_gripper_abs)   11/100 · 미폐쇄 42 · 닫는순간 28.0mm · 최근접 14.3mm
fixed schedule 상한   v5 seed0 37%   ·   dagger 86/49/44%
```

그리퍼 이진 헤드(`joint_delta_gripper_binary`) 결과:

```
B-dagger   6/69/25   평균 33.3%  범위 폭 63%p   구간 28.2~38.8%   rc=0 게이트 통과
           미폐쇄 67/20/40 · 닫는순간 3.7/4.9/5.9mm · val_loss 0.893/0.313/0.603
B-v5      21/23/21   평균 21.7%  범위 폭  2%p   구간 17.4~26.7%   rc=1 게이트 실패
           미폐쇄 27/4/8 · 닫는순간 25.6/25.5/23.6mm · 최근접 17.1/16.8/12.1mm
```

**⚠️ 인용 금지:** `14/12/2` 는 트리 `1195c41` 의 값이다. `verdict=` 필드는 체크포인트
1~2개 실행에서 무효다 (L71).

## 1. 확정된 결론

- **L1 의 조건부 중앙값 붕괴 진단이 맞았고 개입으로 뒤집혔다.** v5 미폐쇄 42 → 4~27,
  dagger 닫는순간 25.6 → 4.9mm (scripted 4.7mm 수준), 롤아웃 1/0/7 → 6/69/25
- **v5 에 남은 병목은 팔이다.** 라벨 닫힘 53.2%(불균형 없음)인데도 최근접 17.1mm.
  E1 의 천장 37% 와 일치한다
- **dagger 는 게이트를 통과했지만 제품이 아니다.** 범위 6~69%. 어느 ckpt 를
  배포하느냐로 완전히 다른 물건이 된다

## 2. 다음 계측 — 이것부터 한다

**질문: dagger seed0 은 왜 무너졌나.** (6/100 · 미폐쇄 67 · 닫은틱 113)

**5회로 늘리지 마라.** 폭 63%p 는 정밀도 문제가 아니라 학습 불안정이다. n 을 올려도
안 줄어든다. D-AI-30 의 5회는 다른 문제를 푼다.

단서: `val_loss` 가 0.893 / 0.313 / 0.603 으로 이미 3배 갈렸다. **롤아웃 이전에
학습에서 갈렸다.** 그런데 지금은 총손실만 기록해서 팔에서 갈렸는지 그리퍼에서
갈렸는지 알 수 없다.

### 스크립트 1 — 손실 항 분리 기록 (먼저, 작다)

`policy/train_bc.py` 의 `ArmL1GripperBCE.forward` 가 두 항을 합쳐서만 낸다.
epoch 별 `arm_l1` 과 `grip_bce` 를 따로 `history` 와 EXP_LOG `result` 에 남긴다.

- 기존 총손실 값은 **바꾸지 않는다** (기록 연속성). 항을 추가로 남기는 것뿐이다
- `run_epoch` 이 항별 합계를 누적하게 한다
- 이걸로 seed0 이 어느 항에서 무너졌는지 한 번에 갈린다

### 스크립트 2 — 롤아웃 중 그리퍼 확률 덤프

`tools/probe_gripper_logits.py` (신규). 체크포인트를 롤아웃하며 프레임별
`sigmoid(logit)` 을 기록하고 성공/실패별 분포를 낸다.

판정 대상: seed0 의 확률이 **항상 0.5 아래에 눌려 있는가**(다수 클래스 붕괴 재발)
아니면 **요동치는가**(관측이 순간을 못 정함). 두 가지는 처방이 다르다.
- 눌림 → `pos_weight`·임계값 문제. 손잡이는 사전등록 없이 건드리지 않는다
- 요동 → 관측 문제. 카메라 배치·시간 문맥(프레임 스택)으로 간다

**착수 전에 사전등록을 쓴다** (`docs/PREREG_gripper_variance_0911.md`),
예측을 먼저 적고, 판정 기준을 숫자로 박는다.

### 하지 말 것

- 5-run 확장 (위 이유)
- `pos_weight`·손실 가중치 튜닝 (사전등록에 1.0 고정으로 박았다. 결과 보고 고치면 사후 합리화)
- oracle 수치 사용 (L72, 고장)
- E2·E3 재개 (폐기함. 진단 정교화라 마감 대비 가치 없음)

## 3. 남은 기록 작업

- **D-AI-38** — 행동공간 기본값을 `joint_delta_gripper_binary` 로 바꿀지. 되돌릴 조건 포함.
  (`origin/ai` 기준 현재 최대 D-AI-37 · 최대 L72. **번호는 세지 말고 조회한다**)
- `KPT_0910.md` — 사용자가 사이클 끝을 알리면
- `TS_v5_fixture_drift_0910.md` — 미작성

## 4. 파일 저장 위치 (규칙)

| 무엇 | 어디 |
|---|---|
| 사전등록 | `AI/docs/PREREG_<주제>_<MMDD>.md` |
| 측정 결과 | `AI/docs/MEASURE_<주제>_<MMDD>.md` |
| 트러블슈팅 | `AI/docs/TS_<주제>_<MMDD>.md` (「어떻게 좁혔나」 구조) |
| 데브로그·회고 | `AI/docs/DEVLOG_<MMDD>.md` · `AI/docs/KPT_<MMDD>.md` |
| 미검증 | `AI/LIMITS.md` (표 끝에 추가) |
| 기각 | `AI/REJECTED.md` |
| 도구 | `AI/tools/*.py` — **반드시 `sys.path.insert(0, parents[1])` 를 넣는다** |
| 큐 항목 | `AI/queue/pending/<name>.yaml` (`name` = 파일명) |

포트폴리오 사본 (Git Bash 경로 `/c/Users/SSAFY/Desktop/pjt2자료`):
`02_측정기록/` ← MEASURE_·PREREG_ · `05_트러블슈팅/` ← TS_ · `06_회고/` ← DEVLOG_·KPT_
**"최신화했다"고 말하기 전에 해시로 대조한다.**

## 5. 명령 — 그대로 복붙

환경을 섞지 않는다. 블록 첫 줄의 `# [로컬]` / `# [서버]` 를 확인한다.

```bash
# [서버]  MuJoCo 도구는 항상 이 두 변수가 필요하다
cd ~/S15P21A103/AI && MUJOCO_GL=egl PYTHONPATH=$PWD python tools/<도구>.py --help
```

```bash
# [서버]  큐 — 사전등록 없으면 거부한다. 트리가 더러워도 거부한다
cd ~/S15P21A103/AI && python tools/run_queue.py --dry-run
cd ~/S15P21A103/AI && nohup python tools/run_queue.py --parallel 2 > out/queue_console.log 2>&1 &
```

```bash
# [서버]  진행 확인 (자식 출력은 콘솔이 아니라 항목 로그로 간다)
cd ~/S15P21A103/AI && D=$(ls -td out/queue_*/ | head -1) && tail -n 20 "$D"*.log && cat queue/LEDGER.md
```

```bash
# [로컬]  커밋
cd /c/Users/SSAFY/Desktop/2nd_pjt_2/특화/S15P21A103
git add AI/docs AI/tools AI/queue AI/policy AI/LIMITS.md
git commit -F out/msg.txt     # 긴 메시지는 파일로. heredoc 붙여넣기 금지
```

```bash
# [로컬]  푸시 — 이것만 쓴다. git push origin ai 단독 금지 (개인 미러 gh 가 빠진다)
cd /c/Users/SSAFY/Desktop/2nd_pjt_2/특화/S15P21A103 && bash AI/tools/push_both.sh
```

```bash
# [서버]  받기
cd ~/S15P21A103 && git pull origin ai
```

커밋 메시지: `<타입>: <무엇을 했는가> (S15P21A103-34)` · 타입은 `feat|fix|test|refactor|chore|docs`
측정이 산출물인 작업은 `test:` 를 쓴다. 끝에 붙인다:

```
Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

## 6. 반복해서 틀린 것 (같은 실수 금지)

`claude/규칙_정정_Git_MR.md` 에 전부 기록돼 있다. 요약:

1. **`PYTHONPATH`·`MUJOCO_GL` 누락** — 하루에 세 번 틀렸다. 도구에 `sys.path.insert` 를 넣어 구조로 막았다
2. **EXP_LOG 수치를 `git_rev` 대조 없이 인용** — 다른 트리의 값을 fixture 로 썼다가 실행이 죽었다
3. **`push_both.sh` 대신 `git push origin ai`** — 개인 미러가 빠진다
4. **터미널 혼동** — 프롬프트로 구분한다. `(aiot_ai) /c/Users/...` = 로컬, `(aiot_v100) j-j15a103@jupyter01` = 서버
5. **큐가 도는 동안 `policy/`·`eval/`·`sim/` 수정 금지** — `repeat_runs` 가 seed 마다
   `train_bc.py` 를 새 프로세스로 띄워서 같은 실험 안에서 코드가 갈린다.
   `run_queue` 는 시작 시 1회만 dirty 검사한다 (**HEAD 고정 가드 미구현**)
