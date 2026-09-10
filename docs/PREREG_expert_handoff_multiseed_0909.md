# PREREG — expert handoff 3-checkpoint screening (2026-09-09)

- 목적: 정책 방문 상태가 feedback expert로 복구 가능한지 검사
- 체크포인트: sim_pick_v5_seed0/1/2.pt
- 조건: P, H30, H60, H90, E
- n=100/조건/checkpoint, seed 3000~3099
- render=True, policy-device=cpu, jitter ±50mm, max_ticks=200
- 계측기 fixture: P가 각각 11/100, 10/100, 2/100을 정확히 재현
- DAgger 후보: H60 또는 H90이 P보다 20%p 이상 개선하고 절대 성공률 50% 이상
- expert ceiling: E >=80%
- IK 실패 에피소드: <5%
- 3체크포인트는 screening이다. 정식 조건 비교는 D-AI-30에 따라 5회가 필요하다.
