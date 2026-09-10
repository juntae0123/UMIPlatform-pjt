# PREREG — E3 DAgger fixed schedule 학습 seed 3·4 추가 (2026-09-10)

작성자: 김준태(트랙B) · 이슈 S15P21A103-34

## 왜 하는가

fixed schedule 결과(seed0/1/2 fixed67 = 83/42/43)는 학습 3회다.
"같은 데이터·설정으로도 학습 실행 간 성공률이 25%p 갈린다" 🟢 는 이미 실증됐다.
D-AI-30 의 5회 기준을 채우려면 학습 seed 2개가 더 필요하다.

## 조건

- 데이터: 기존 seed0/1/2 와 **동일한** 최신 `out/dagger_merged_*`
- 학습 recipe 도 동일해야 한다 — epochs 를 EXP_LOG 의 seed0 train_bc 기록에서
  읽어 대조하고, 다르면 실행하지 않는다 (recipe 가 다르면 5회가 아니라 다른 실험이다)
- 학습: `--device cuda`, `--seed-base 3 --runs 2` → 학습 seed 3, 4
- 평가: policy-device=cpu · render=True · n=100 · evaluation seeds 3000~3099
- `repeat_runs` 의 배포 게이트 실패(exit 1)는 정상이다. 여기서는 체크포인트가 산출물이다
- probe 의 learned fixture 는 끄지 않는다 — `repeat_runs` 가 방금 측정한
  seed3/4 성공률을 `--expected` 로 넣어 probe 가 그 값을 재현하는지 확인한다
  (0909 의 `--expected 11 10 2` 오류가 정확히 이 대조를 건너뛴 결과였다)
- 단일 잡. E1 → E2 종료 후 실행한다 (LIMITS L51)

## 사전등록 판정

- fixed67 이 seed3·seed4 **모두** 성공률 > 20% 이면
  → 5개 학습 실행에서 재현된 후보로 유지한다
- 하나라도 <= 20% 이면
  → 학습 실행 간 분산이 커서 5-run 안정 후보로 인정하지 않는다
- 기존 seed0/1/2 fixed67: 83% / 42% / 43%

## 해석 제한

- fixed tick 은 배포 처방이 아니라 gripper phase 학습 실패의 진단이다.
- 학습 seed 를 늘린 것이지 평가 n 을 늘린 것이 아니다.
  n=100 의 95% 구간 반폭은 그대로다.
