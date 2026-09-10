# PREREG — DAgger round 1 screening (2026-09-09)

## 근거

- P: seed0/1/2 = 11%/10%/2%
- H30: 세 체크포인트 모두 84%
- 정책 방문 상태는 tick 30 handoff에서 feedback expert로 복구 가능하다.

## 수집 조건

- behavior: sim_pick_v5_seed0/1/2 순환
- object seed: 4000~4099
- tick 0~29: BC 실행만
- tick 30~199: BC를 계속 실행하면서 feedback expert action을 라벨로 질의
- expert action은 환경에 실행하지 않는다.
- expert 질의 전후 MuJoCo qpos/qvel/ctrl/time 불변을 fixture로 검사한다.
- IK 실패가 한 번이라도 발생한 에피소드는 학습 데이터에서 제외한다.
- 원본 v5와 유효 DAgger 에피소드를 임시 out/ 데이터셋으로 병합한다.

## 결과 확인 전 게이트

- expert 유효 라벨 비율 >=95%
- IK 실패 없는 에피소드 유지율 >=80%
- 계약 검증 위반 0건

## 학습·평가

- 게이트 통과 시 30 epochs × 3회 고장검사
- 학습 CUDA, 롤아웃 정책 CPU
- 평가 n=100/run, seed 3000~3099
- 예측: 세 실행 중 하나 이상 strict 20% 게이트를 통과한다.
- 하나 이상 통과하면 D-AI-30에 따라 5회 정식 비교로 확대한다.
- 셋 다 실패하면 phase/temporal policy로 이동한다.
