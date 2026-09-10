# PREREG — E2 oracle close tolerance sweep (2026-09-10)

작성자: 김준태(트랙B) · 이슈 S15P21A103-34

## 왜 하는가

oracle 은 물체 실좌표(특권정보)를 쓰는데도 fixed 보다 나쁘다
(0909 결과: oracle 48/21/24 vs fixed60 81/49/38). 특권정보 조건이 멍청한 시계보다
나쁜 것은 정상이 아니다. **oracle 계측기가 고장 나 있을 가능성이 높다.**

이게 왜 중요한가: 0909 사전등록 판정 2번("oracle 이 통과하고 fixed 는 못 넘으면
시각 기반 trigger 가 필요하다")은 oracle 이 고장 나 있으면 애초에 검정 불가능했다.
그 분기는 영원히 안 켜진다.

유력 가설 🟡: `grasp_tol_m = 0.005` 조건이 늦게 만족되거나 아예 만족되지 않는다.
oracle 중앙 close tick 이 fixed 보다 뒤에 있는 것(70 / 92.5 / 81.5)이 그 증거다.

## 조건

- checkpoints: DAgger seed0/1/2 (`dagger_merged_*_dagger_segments_0909_seed{0,1,2}.pt`)
- n = 100 / 조건 / checkpoint · evaluation seeds 3000~3099
- render=True · policy-device=cpu · jitter ±50mm
- tolerance: 5mm, 10mm, 15mm
- **learned 재현 fixture 를 끄지 않는다** — `--expected 1 0 7` 로 매 스윕마다 검증한다
  (계측기를 쓰기 전에 계측기를 검증한다)
- 5mm oracle 결과가 48/21/24 를 재현하지 못하면 즉시 중단한다
- `never_closed` 와 중앙 close tick 을 반드시 결과 표에 찍는다.
  0909 실행은 never_closed 를 계산해놓고 출력하지 않았다 — oracle 에서 그게 핵심 숫자다
- 단일 잡. E1 종료 후 실행한다

## 사전등록 판정

- oracle 계측기 usable 조건:
  `never_closed <= 5/100` (세 checkpoint 모두) **그리고** 세 checkpoint 중 2개 이상 성공률 > 20%
- tolerance 선택: 위 조건을 만족하는 **가장 작은** tolerance
- 어느 tolerance 도 만족하지 못하면 원인은 tolerance 가 아니다.
  close 조건 자체(`grasp_point` 거리 정의)를 다시 봐야 한다 → 별도 TS 로 넘긴다

## 해석 제한

- oracle 은 물체 실좌표를 쓰는 **특권정보 진단**이다. 배포 정책이 아니다.
- tolerance 를 키워서 oracle 성공률을 올리는 것은 성능 개선이 아니라
  계측기 눈금 교정이다. 성능 수치로 인용하지 않는다.
