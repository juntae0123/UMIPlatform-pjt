# PREREG — q[t+1] 재라벨 고장검사 (2026-09-09)

- 목적: ctrl 대신 다음 tick 도달값을 라벨로 썼을 때 정책 고장이 해소되는지 검사
- 예측: floor/chance 게이트에서 기존 v5와 구분되지 않는다
- 데이터: sim_pick_v5를 임시로 q[t+1] 재라벨, 마지막 observation 삭제
- 학습: 30 epochs, 초기화 seed 0~2
- 평가: n=100/run, 평가 seed 3000~3099
- 게이트: policy > hold+20%p, policy > zero+20%p
- 확대 규칙: 하나라도 strict 20%p 게이트를 넘으면 ctrl/qnext 모두 5회로 확대
- 중단 규칙: 세 실행 모두 게이트 실패면 배포 후보에서 제외한다
- 주의: 3회는 D-AI-30의 고장검사이며 정식 조건 비교가 아니다
