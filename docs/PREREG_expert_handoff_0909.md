# PREREG — BC→feedback expert handoff (2026-09-09)

- 체크포인트: sim_pick_v5_seed0.pt
- 조건: render=True, policy-device=cpu, jitter ±50mm
- 평가: n=100/조건, seed 3000~3099, 최대 200틱
- P: 끝까지 BC
- H30/H60/H90: 해당 틱 전까지 BC, 이후 ScriptedFeedbackPolicy
- E: 처음부터 ScriptedFeedbackPolicy
- 예측: H60 또는 H90이 P보다 20%p 이상 개선한다.
- 계측기 게이트: P가 직전 측정 11/100을 재현해야 한다.
- expert 게이트: E >= 80%
- DAgger 진행 게이트: H60 또는 H90이 P보다 >=20%p 개선하고 절대 성공률 >=50%
- IK 게이트: expert 사용 에피소드 중 IK 실패 에피소드 <5%
- 결과 확인 후 게이트를 변경하지 않는다.
