# PREREG — DAgger 정책 gripper schedule probe (2026-09-09)

## 근거

DAgger 정책은 물체 최근접 1.5~1.6mm, 턱 접촉 83~94%까지 개선됐지만
45~54/100편에서 닫지 않았다.

## 조건

- checkpoints: DAgger segment seed0/1/2
- n=100/조건/checkpoint, 평가 seed 3000~3099
- render=True, policy-device=cpu, jitter ±50mm, max 200 ticks
- arm action은 BC 출력을 그대로 사용
- learned: gripper도 BC 출력
- fixed60/67/75/85: 해당 tick 전 open, 이후 close로 강제
- oracle: 파지점과 목표 파지 위치의 3D 거리가 5mm 이하가 되면 close
- oracle은 진단용 특권정보이며 배포 불가

## 계측기 게이트

learned 조건이 직전 결과 seed0/1/2 = 1/0/7을 정확히 재현해야 한다.

## 판정

- 같은 fixed tick이 3개 checkpoint 중 2개 이상에서
  성공률 >20%이고 learned 대비 >=20%p 개선하면 clock schedule 후보
- oracle이 2개 이상에서 같은 기준을 넘고 fixed는 못 넘으면
  시각 기반 close trigger가 필요
- 둘 다 못 넘으면 gripper 단독 문제가 아니며 lift/phase arm action을 검사
