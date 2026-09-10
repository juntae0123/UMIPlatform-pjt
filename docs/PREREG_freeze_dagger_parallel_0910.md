# PREREG — DAgger post-close freeze probe (2026-09-10)

- 목적: DAgger 정책이 물체 근처에서 닫은 뒤에도 실패하는 이유를 측정
- checkpoints: 최신 dagger_segments seed0/1/2
- n=100/checkpoint
- 평가 seeds: 3100~3199
- policy device: cpu
- render: true
- 기록: freeze, jaw contact, lift, close 여부
- gripper schedule probe와 다른 시드 블록이므로 직접 paired 비교하지 않는다.
- mean_max_lift는 gripper probe가 200틱 전체를 실행하므로 기존 조기종료 rollout 수치와 직접 비교하지 않는다.
