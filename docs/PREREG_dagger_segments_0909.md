# PREREG — DAgger Round 1b, valid segment (2026-09-09)

- 기존 whole-episode drop은 68편 시점에 최종 유지율 80% 달성이 불가능해져 실패했다.
- 새 조건은 기존 결과를 재해석하지 않고 데이터 표현을 바꾼 별도 실험이다.
- behavior: v5 seed0/1/2 순환
- object seeds: 4000~4099
- expert label: tick 30~199
- IK 실패 또는 state/action 범위 위반 tick에서 segment를 끊는다.
- state/action을 clip하지 않고 contract tolerance도 바꾸지 않는다.
- 길이 2틱 이상인 연속 valid segment만 저장한다.
- 게이트: 유효 label >=95%, 저장 label >=95%, 계약 위반 0건
- 통과 시 v5와 병합하고 30 epochs × 3회 고장검사
- 예측: 세 정책 중 하나 이상 롤아웃 strict 20% 게이트 통과
