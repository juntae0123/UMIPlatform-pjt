# MEASURE — v5 체크포인트 3개 롤아웃 게이트 (2026-09-09)

- 작성: 김준태(트랙B)
- 확신도: 🟢 실행·로그 확인
- 원본 로그: `out/eval_v5_seed012_20260909_132154/seed0.log` ~ `seed2.log`

## 측정 조건

```
dataset             datasets/sim_pick_v5
dataset manifest SHA 91ae8b02df48a331a65da7d55b5395654bc06a7d777f3690be2f08299ad016a3
git revision         22e2794de7b8238517d2a478601b4cbc24eceef0
checkpoints          sim_pick_v5_seed0/1/2.pt
evaluation           n=100/checkpoint, seeds=3000~3099
render               true, MUJOCO_GL=egl
replay               --replay-from datasets/sim_pick_v5
policy device         cpu
AI_THREADS            2
logging               --log
```

## 사전등록 게이트

- floor: policy > hold + 20%p
- chance: policy > zero + 20%p
- task_validity: replay < 30%
- ceiling: scripted >= 80%

## 계측기 검증

각 실행 로그에 네 게이트 라벨과 replay 결과가 모두 존재해야 유효하다.

## seed0 결과

- checkpoint: `checkpoints/bc/sim_pick_v5_seed0.pt`
- checkpoint SHA256: `ac16cfacc6ad7dc19660618d834776951607c2df80933034d308211a78070c5b`

```
게이트 기준 (결과 확인 전 확정):
  [floor] 학습 정책 성공률 > hold 성공률 + 20%p. 못 넘으면 정책이 무의미하다.
  [chance] 학습 정책 성공률 > zero 성공률 + 20%p. 못 넘으면 우연과 구분되지 않는다.
  [task_validity] replay 성공률 < 30%. 이보다 높으면 고정 궤적으로 풀리는 태스크이므로 시각 정책이 학습할 것이 없다. 태스크를 다시 설계해야 한다.
  [ceiling] scripted 성공률 >= 80%. 못 넘으면 태스크나 씬이 문제이지 정책 문제가 아니다.
학습 정책 로드: bc ckpt sim_pick_v5_seed0.pt · 행동공간 joint_delta_gripper_abs(표준화) · 파라미터 1,307,974 · 학습대상 datasets/sim_pick_v5 · 에피소드 98 · 샘플 13818 · epochs 30 · best val_loss 0.05314
hold         0/100 =   0.0%   평균 상승  0.00cm
zero         0/100 =   0.0%   평균 상승  0.00cm
replay      10/100 =  10.0%   평균 상승  0.51cm
scripted    84/100 =  84.0%   평균 상승  4.31cm (특권정보 사용 — 실물 배포 불가)
bc          11/100 =  11.0%   평균 상승  0.56cm
  [task_validity] replay 10.0% < 30% → 통과 — 이 태스크는 관측을 봐야 풀린다
  [ceiling]       scripted 84.0% >= 80% → 통과
```

## seed1 결과

- checkpoint: `checkpoints/bc/sim_pick_v5_seed1.pt`
- checkpoint SHA256: `f84d0d296d5607f1934f5293686c02d2e638e3223309f73b01a51184c0ccc83f`

```
게이트 기준 (결과 확인 전 확정):
  [floor] 학습 정책 성공률 > hold 성공률 + 20%p. 못 넘으면 정책이 무의미하다.
  [chance] 학습 정책 성공률 > zero 성공률 + 20%p. 못 넘으면 우연과 구분되지 않는다.
  [task_validity] replay 성공률 < 30%. 이보다 높으면 고정 궤적으로 풀리는 태스크이므로 시각 정책이 학습할 것이 없다. 태스크를 다시 설계해야 한다.
  [ceiling] scripted 성공률 >= 80%. 못 넘으면 태스크나 씬이 문제이지 정책 문제가 아니다.
학습 정책 로드: bc ckpt sim_pick_v5_seed1.pt · 행동공간 joint_delta_gripper_abs(표준화) · 파라미터 1,307,974 · 학습대상 datasets/sim_pick_v5 · 에피소드 98 · 샘플 13818 · epochs 30 · best val_loss 0.06203
hold         0/100 =   0.0%   평균 상승  0.00cm
zero         0/100 =   0.0%   평균 상승  0.00cm
replay      10/100 =  10.0%   평균 상승  0.51cm
scripted    84/100 =  84.0%   평균 상승  4.31cm (특권정보 사용 — 실물 배포 불가)
bc          10/100 =  10.0%   평균 상승  0.51cm
  [task_validity] replay 10.0% < 30% → 통과 — 이 태스크는 관측을 봐야 풀린다
  [ceiling]       scripted 84.0% >= 80% → 통과
```

## seed2 결과

- checkpoint: `checkpoints/bc/sim_pick_v5_seed2.pt`
- checkpoint SHA256: `c41e9fa434a9bba71ca15aa061554c3dd05aad9fc1fca62b7d5396972c9f4758`

```
게이트 기준 (결과 확인 전 확정):
  [floor] 학습 정책 성공률 > hold 성공률 + 20%p. 못 넘으면 정책이 무의미하다.
  [chance] 학습 정책 성공률 > zero 성공률 + 20%p. 못 넘으면 우연과 구분되지 않는다.
  [task_validity] replay 성공률 < 30%. 이보다 높으면 고정 궤적으로 풀리는 태스크이므로 시각 정책이 학습할 것이 없다. 태스크를 다시 설계해야 한다.
  [ceiling] scripted 성공률 >= 80%. 못 넘으면 태스크나 씬이 문제이지 정책 문제가 아니다.
학습 정책 로드: bc ckpt sim_pick_v5_seed2.pt · 행동공간 joint_delta_gripper_abs(표준화) · 파라미터 1,307,974 · 학습대상 datasets/sim_pick_v5 · 에피소드 98 · 샘플 13818 · epochs 30 · best val_loss 0.07559
hold         0/100 =   0.0%   평균 상승  0.00cm
zero         0/100 =   0.0%   평균 상승  0.00cm
replay      10/100 =  10.0%   평균 상승  0.51cm
scripted    84/100 =  84.0%   평균 상승  4.31cm (특권정보 사용 — 실물 배포 불가)
bc           2/100 =   2.0%   평균 상승  0.10cm
  [task_validity] replay 10.0% < 30% → 통과 — 이 태스크는 관측을 봐야 풀린다
  [ceiling]       scripted 84.0% >= 80% → 통과
```

## 판정 주의

- validation loss는 판정 지표가 아니다.
- 정책 판정은 롤아웃 성공률과 사전등록 게이트로만 한다.
- 세 실행은 같은 평가 시드 블록을 사용했으며 임의 합산하지 않는다.
