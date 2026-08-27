"""Track A — 현실·기하. UMI pose estimation, calibration, Sim2Real, real data.
트랙 A — 현실·기하. UMI 포즈 추정, 캘리브레이션, Sim2Real, 실데이터.

Track A's question is "is this data real, and can the robot reproduce it".
Its output is a trustworthy dataset. Everything here turns raw recordings from
the real world into episodes that satisfy `contract/`.
트랙 A 의 질문은 "이 데이터가 진짜인가, 로봇이 재현할 수 있나" 이고,
산출물은 신뢰 가능한 데이터셋이다. 여기 있는 것은 현실에서 찍힌 raw 기록을
`contract/` 를 만족하는 에피소드로 바꾸는 일이다.

NOT IMPLEMENTED. This is a scaffold — the folder shape and the boundary, not
the code. Track A writes what goes inside.
**구현 없음.** 폴더 형태와 경계만 잡아둔 골격이다. 안에 들어갈 코드는 트랙 A 가 쓴다.

## 경계 — 두 트랙의 접점은 `contract/` 하나뿐이다

    track_a/  →  contract/  ←  policy/ eval/ sim/ data/ (트랙 B)

- **트랙 B 는 `track_a/` 를 임포트하지 않는다.** 트랙 B 가 보는 것은 계약을 만족하는
  에피소드뿐이고, 그것이 어떻게 만들어졌는지 알 필요가 없다
- **트랙 A 는 `sim/` `policy/` `eval/` 을 임포트하지 않는다.** 필요한 것은 `contract/` 다
- 이 경계를 넘는 임포트가 생기면 접점이 둘이 된 것이다. 그때는 D-AI 기록이 필요하다

⚠️ 현재 AI/ 의 나머지(`sim/` `policy/` `eval/` `data/` `vlm/` `tracking/`)는 전부
   트랙 B 작업이다. **비대칭이다** — 트랙 B 코드가 `track_b/` 로 묶여 있지 않다.
   대칭으로 만들 이유가 생기면 그때 옮긴다. 지금 옮기면 임포트 경로만 또 바뀐다.

## 하위 폴더와 이슈

| 폴더 | 이슈 | 내용 |
| --- | --- | --- |
| `calibration/` | S15P21A103-28 | 카메라 캘리브레이션 (내부·외부 파라미터) |
| `pose/`        | S15P21A103-29 | pose → 로봇 좌표 변환 |
| `sync/`        | S15P21A103-30 | pose ↔ 영상 시간동기화 |
| `convert/`     | S15P21A103-31 | raw → dataset 포맷 변환 |

## 착수 전에 알아둘 것 (트랙 B 가 실측한 것)

1. **물체는 도달 가능 영역 안에만 놓는다.** 그 밖에서 찍은 에피소드는 로봇이 재현할 수
   없어 통째로 폐기 대상이다. 현재 실측 영역과 근거는 `AI/README.md` §8 참조
2. **위치 허용오차는 약 ±10mm 다.** 캘리브레이션·좌표변환 오차가 이 수준을 넘으면
   그 데이터로 학습한 정책은 물체를 못 집는다. 이게 정밀도 요구의 근거다
3. **시간동기화는 수집 전에 계측한다.** 어긋나면 모델이 "과거 화면 보고 미래 행동"을
   학습한다. 학습 손실과 시뮬 성공률은 정상이고 실물에서만 실패해서, 이 프로젝트에서
   추적이 가장 어려운 실패 형태다. `contract/` 가 `state_timestamp` /
   `action_timestamp` 를 분리해 둔 이유가 이 계측을 가능하게 하려는 것이다
4. **시뮬에서는 이 오차가 구조적으로 0 이다.** 시뮬로 S15P21A103-30 을 검증했다고
   볼 수 없다
"""
