# 사전등록 부칙 — 시드 확대 발동 규칙 · 공통 donor 세트 · 코드 동결 (2026-09-08)

- `docs/PREREG_ABC_0908.md` 의 부칙. 원본은 "실행 후 수정하지 않는다" 고 스스로
  박았고 Round 1 이 이 문서를 쓰는 동안 돌고 있으므로 원본을 고치지 않고 새 파일로 쓴다.
- **작성 시점에 Round 1 결과를 읽지 않았다.** V0-1(A = 98/98)만 확인했고
  `probe_abc` 의 B′·C 출력은 아직 보지 않았다. 이 문서는 여전히 결과 이전이다.
- 근거: 외부 자문 3차 회신에 대한 검토, 2026-09-08
- 작성 김준태(트랙B)

---

## 1. 시드 확대 발동 규칙 (결과 보기 전에 고정)

Round 1·2 는 v5 seed0·1·2 세 체크포인트다. 5시드로 늘리는 조건을 지금 박는다.
**결과를 보고 "애매하다" 고 판단하면 optional stopping 이다.**

```
확대하지 않는다:
    세 체크포인트가 모두 같은 사전등록 분기(J1|J2|J3)를 지지한다
    → 그 분기를 채택. 추가 학습 없음

확대한다 (seed3·seed4 학습 후 모든 조건 재실행):
    (a) 체크포인트 간 분기가 하나라도 다르다
    (b) 또는 B′_all 의 Wilson 구간이 A 와 겹치면서 동시에 게이트(20%)를 포함한다
    (c) 또는 B′_all 성공률이 분기 경계(게이트 20%)의 ±10%p 안에 있다
```

**확대는 모든 조건에 동일하게 적용한다.** 유리한 조건만 2회 더 돌리지 않는다.
`A`·`B′_all`·`B′_arm`·`B′_grip`·`C` 전부 같은 실행 수로 간다.

⚠️ 3체크포인트는 **분기 정찰**에 충분하고 **조건 순위 확정**에는 구조적으로 부족하다
(3 vs 3 순열검정 최소 단측 p = 1/C(6,3) = 0.05). Round 1·2 의 목적을 정찰로 명시한다.

---

## 2. 체크포인트 간 비교의 제한 — 결과에 붙일 꼬리표

실측 🟢 (`heldout_episode_ids`, 98편 · val_fraction 0.2):

```
seed0 홀드아웃 20편 · seed1 20편 · seed2 20편
교집합  0∩1 = 6편 · 0∩2 = 4편
```

즉 세 체크포인트는 **서로 다른 20편을 홀드아웃한다.** 체크포인트 간 성공률 차이에는
초기화 차이 · 분할 차이 · donor 난이도 차이가 섞여 있고 **분리 불가**다.

### 할 수 있는 판정

- 한 체크포인트 **안에서** A vs B′ vs C 의 차이
- 세 체크포인트에서 **같은 방향이 반복되는지**
- arm / gripper 중 어느 개입이 실패를 유발하는지

### 할 수 없는 판정

- "seed0 이 seed1 보다 좋은 모델이다"
- 세 체크포인트 성공 수를 합쳐 하나의 성공률로 쓰는 것 (**pooling 금지**)
- 서로 다른 20편에서 얻은 차이를 초기화 효과로 단정하는 것

### 결과 문서·MEASURE 에 반드시 붙일 문장

> 홀드아웃 집합은 체크포인트별로 다르다. 체크포인트 간 성공률은 unpaired 이며
> 합산하지 않는다. (checkpoint-specific held-out set; cross-checkpoint rates are unpaired)

⚠️ 주 판정의 분모는 **홀드아웃 20 이 아니라 유효 에피소드 98** 이다. 홀드아웃 20 은
"학습편에서만 좋은 것이 아닌가" 를 보는 보조 열이고, n=20 의 구간 반폭은 약 ±16%p 라
단독 판정에 쓸 수 없다.

---

## 3. 공통 외부 donor 세트 — J2/J3 진입 또는 정식 체크포인트 비교의 선행조건

체크포인트 간 unpaired 문제를 없애는 유일한 깨끗한 방법이다.

```
이름            datasets/sim_pick_donor_v1
                ⚠️ sim_pick_v6 을 쓰지 않는다 — v6 은 이미 증강 *조건* 이름이라
                   데이터셋 이름으로 재사용하면 로그에서 구분이 안 된다
수집 조건       기존 98편과 다른 scripted 시드. 그 외 조건은 v5 와 동일
                (씬·config·30Hz·2cm 20g 큐브·지터 ±50mm)
필수            모든 편이 notes.object_init_xy 보유
                직접 replay(조건 A) 성공 확인 — 실패편은 제외
                세 체크포인트 학습 데이터에서 전부 제외됨이 자명 (수집 시점이 나중)
                모든 체크포인트에 동일 에피소드 순서
기록            dataset SHA · 수집 시드 · code_sha · config_sha
```

**발동 조건**: 사전등록 §5(단계 2, O(α) vs C(α)) 에 들어가기 전, 또는 체크포인트 간
수치 비교를 결과로 쓰려 할 때. **Round 1 을 막지 않는다.**

⚠️ 수집 전 `ls -dlt datasets/*/` 로 기존 디렉터리를 확인한다 (2026-09-07 에
완주한 `sim_pick_v4` 98편을 같은 이름으로 재수집해 앞 69편을 덮어쓴 사고가 있었다).

---

## 4. 코드 동결 — Round 1·2 가 끝날 때까지

`tracking/exp_log.code_digest` 가 해싱하는 `CODE_GLOBS`:

```
*.py · contract/**/*.py · sim/**/*.py · policy/**/*.py · eval/**/*.py
data/**/*.py · vlm/**/*.py · tracking/**/*.py · configs/*.yaml · sim/**/scenes/*
```

**이 안의 파일을 지금 고치면 `code_sha` 가 바뀌어 Round 2 가 Round 1 과 비교 불가가 된다.**
Round 1·2 종료까지 동결한다. `docs/*.md` 는 이 목록 밖이므로 문서는 계속 쓴다.

동결 해제 후 처리할 것 (지금 하지 않는다):

| 항목 | 내용 |
|---|---|
| `CheckpointMeta` 확장 | initialization seed · split seed · **실제 train/val episode ID 목록** · dataset SHA · dataset episode-order hash |
| provenance 구멍 | `third_party/so101_mujoco/SO101/*` 가 `code_digest` 밖이다 (§5) |
| `lift_height_m` | 0.08 → 0.07 (`lift_height_sweep` n=100 근거, 여유 3.1mm → 13.1mm) |

`n_episodes` 일치 확인은 약한 대체물이다. 계약 위반 탈락이나 로딩 순서 변경이 생기면
에피소드 인덱스가 밀리고, 그때 파일명 기반 시드 복원은 **조용히 틀린 홀드아웃 집합**을
낸다. 실제 ID 목록을 저장하면 그 경로가 닫힌다.

---

## 5. 학습 시드의 출처 표기

`--seed` 가 `train_config` 에 되쓰이지 않아 세 체크포인트 모두 메타에 `seed: 0` 으로
기록돼 있다. 파일명 `_seed{N}` 에서 추론한다.

```
seed_source      = filename_inferred      (checkpoint_metadata 아님)
seed_confidence  = convention_only        (recorded 아님)
```

도구가 매 실행마다 "관례이지 기록이 아니다" 를 출력한다. **이 값을 근거로 쓰는 어떤
결론에도 이 꼬리표를 붙인다.**

---

## 6. preclip / postclip — 설계 논쟁이 아니라 측정으로 닫혔다 🟢

외부 자문은 `pred_action_rad_preclip` 과 `postclip` 을 따로 저장하라고 했으나,
pre-clip 은 `BCPolicy` 내부 훅 없이 꺼낼 수 없고 같은 자문이 훅 추가를 수용했다 —
서로 모순이다. 실측이 이 논쟁을 닫는다:

```
±1 경계에 앉은 성분   0.007%   (13,818틱 × 6관절 = 82,908 성분 중 약 6개)
```

**클리핑이 사실상 없다. 이 체크포인트에서 preclip ≈ postclip 이다.**
정규화가 affine 이므로 rad 값과 config 에서 normalized 값도 사후 복원된다.
훅을 넣지 않는 판단을 유지한다. 다른 체크포인트에서 `at_bound_frac` 이 올라가면
그때 재검토한다 — 도구가 매번 이 값을 출력한다.

---

## 7. 이슈 113 전달문 (127 대화 · 트랙 A 대행)

소유권은 127 이지만 최종 사슬의 의존성은 별개다. 그대로 전달한다.

```
이슈 113 의 frame IK acceptance 는 trajectory executability 를 보장하지 않는다.
모든 프레임이 개별적으로 IK 가능해도 그 관절 궤적을 30Hz 로 연속 실행하지
못할 수 있다.

확인된 것 🟢: configs/so101.yaml 에 joint velocity/acceleration limit 가 없다.
  → geometric feasibility  : 평가 가능
  → kinematic continuity   : Δq · branch jump 로 평가 가능
  → dynamic feasibility    : NOT_EVALUABLE

HW 담당자에게 요청할 값: 관절별 최대 속도 · 최대 가속도 · 명령 rate 제한.
값이 들어오기 전에는 추정값으로 PASS 를 내리지 않는다.

향후 필수 계측:
  qdot / qddot / jerk
  IK branch jump (프레임 간 해의 불연속)
  joint-limit margin (관절별 남은 여유)
  consecutive violation length
  self / table / object collision
  critical-span continuity

"어느 관절이 걸리는가" 에는 인덱스만으로 부족하다. 함께 기록할 것:
  joint index · lower|upper · raw unconstrained q · clamped q
  exceed amount · margin · first violation tick · consecutive length

수집 승인 게이트는 frame acceptance 하나가 아니다:
  kept_span (최장 연속 실행 가능 구간) + critical phase coverage 가 필요하다.
  전체 90% 가 가능해도 파지 순간 10% 가 끊기면 그 에피소드는 쓸 수 없다.

placement 는 세 값을 분리해 보고한다:
  existential (최적 배치의 상한) / fixed reproducible (본수집 게이트)
  / robust (설치·캘리브 오차 하한)
최적 배치 acceptance 를 본수집 게이트로 쓰면 탐색기 성능을 재게 된다.
```

---

## 8. 되돌릴 조건

- §1 의 확대 규칙을 결과를 본 뒤 바꾸면 이 문서 전체가 무효다. 새 파일에 취소선.
- §4 의 동결을 어겨 Round 1·2 사이에 `CODE_GLOBS` 파일이 바뀌면 Round 2 결과 폐기.
  `code_sha_at_launch` 를 두 라운드에서 대조해 확인한다.
- §3 의 donor 세트를 만들 때 기존 데이터셋 이름을 재사용하면 즉시 중단.
