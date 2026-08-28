# REVIEW — HW 블록 명령 런타임 (2026-08-28)

- 작성자: 김준태 (트랙 B)
- 대상: HW 가 전달한 `simulation.zip` — 우리 `AI/sim/` 복사본 + `runtime/` + `ROBOT_COMMAND_API.md v1.0`
- 관련 이슈: S15P21A103-64 (시뮬 수집 환경) · [ROS] 66 (Sim→Real 어댑터) · [ROS] 46 (정책 ckpt 로드→추론 노드)
- 확신도: 🟢 파일 대조·실행 확인 / 🔵 상대 문서 / 🟡 제안

## 1. 무엇을 받았나 🟢

zip 안은 **우리 `AI/sim/` 패키지 복사본 + 새 `runtime/`** 이다.

| 파일 | 대조 |
|---|---|
| `base.py` · `mujoco/{env,kinematics,grasp,gripper,workspace,render_check}.py` · `scenes/pick_place.xml` | **바이트 동일** |
| `mujoco/build_scene.py` | **2군데 다름** — 비ASCII 경로 패치(`resolve_for_mujoco`) 누락 |
| `sim/mujoco/{teleop,view}.py` | 그쪽에 없음 (당시 미푸시 브랜치) |
| 신규 `runtime/robot_runtime.py`(384) · `mujoco_backend.py`(152) · `send_job.py`(104) · `index.html`(324) | 블록 명령 HTTP 서버 + 웹 UI |
| `third_party/SO-ARM100/` (22MB, `.git` 포함) | 공식 저장소 통째. 우리는 `third_party/so101_mujoco/` — **경로 다름** |

런타임 자체는 잘 만들어졌다: 실행 전 전량 검증, 실행 스레드 분리, 정지 시 현재 위치 유지,
오류 응답 규약, `robot_id` 로 SIM/REAL 분기. **`robot_id` 분기는 우리 배포 매니페스트와 그대로 맞물린다.**

## 2. 각도 규격 — 계약을 깨지 않는다 🟢

`docs/MEASURE_hw_angles_0828.md` 참조. 초과 최대치 **1.370e-06**, 허용오차 1e-4 → 통과.
HW 는 지금 방식을 바꿀 필요가 없다.

## 3. 문제 1 — 런타임이 로드하는 씬에는 태스크가 없다 🟢

`mujoco_backend.py` 의 `MODEL_PATH` 가 공식 `Simulation/SO101/scene.xml` 이다.
그 파일은 `so101_new_calib.xml` include + 바닥 + 조명이 전부다. **작업대·물체·카메라·접촉 패드가 없다.**

접촉 패드가 없으면 파지가 물리적으로 성립하지 않는다. MuJoCo 는 메시를 **볼록 껍질**로
충돌 처리하므로 두 갈래 그리퍼의 껍질이 손가락 사이 틈을 메워버린다 —
스톡 상태 관통 **1.46cm**(격자 24점 중 20점), 최대 3.13cm.
패드 주입 후 0.30cm / 16점 / 최대 1.30cm 로 줄인 뒤에야 파지가 됐다
(`docs/TS_grasp_convex_hull_0827.md`, `docs/MEASURE_grasp_0827.md`).

즉 **팔은 움직이는데 물건은 못 잡는 상태**다. 두 세계를 그대로 두면,
블록 UI 에서 되던 동작이 데이터·정책 쪽에서 안 되는 이유를 아무도 설명하지 못하게 된다.

### 제안 🟡 — 모델 로드를 우리 빌더로 바꾼다 (한 곳)

```python
# runtime/mujoco_backend.py
- MODEL_PATH = (Path(__file__).resolve().parents[1]
-               / "third_party" / "SO-ARM100" / "Simulation" / "SO101" / "scene.xml")
  class MujocoBackend:
-     def __init__(self, model_path: Path = MODEL_PATH) -> None:
-         self.model = mujoco.MjModel.from_xml_path(str(model_path))
+     def __init__(self, scene: Path | None = None) -> None:
+         # build_model 이 configs/so101.yaml 을 읽어 카메라·접촉패드를 주입한다.
+         # 공식 MJCF 는 수정하지 않는다 — MjSpec 로 로드 시점에만 바꾼다.
+         from sim.mujoco.build_scene import build_model
+         self.model = build_model(scene)
          self.data = mujoco.MjData(self.model)
```

부수 효과로 비ASCII 경로 문제도 함께 해결된다 (`build_scene` 이 `resolve_for_mujoco` 를 탄다).

## 4. 문제 2 — 블록 잡 API 로는 정책을 돌릴 수 없다

| 이유 | 내용 |
|---|---|
| 미리 정할 수 없다 | 블록 잡은 **실행 전에 모든 명령이 확정**된다. 정책은 매 스텝 관측을 보고 정한다. 물체가 1cm 옆에 있으면 궤적 전체가 달라진다 |
| 주기가 안 맞는다 | 정책은 **33.3ms 마다** 관측→행동. 폴링 권장 간격 500ms — 15배 차이. HTTP 왕복을 30Hz 로 도는 설계가 아니다 |
| 관측 경로가 없다 | `GET /robot` 은 관절 각도만 준다. 정책 입력은 카메라 2대 이미지 + state + 타임스탬프 |

### 제안 🟡 — 프로토콜이 아니라 블록 타입 하나를 추가한다

HW 문서 §10 의 v1 제외 목록에 이미 "UMI 체크포인트 실행"이 있다. 답은 거기에 있다.

```json
{ "id": "block-9", "type": "policy",
  "checkpoint_id": 12,
  "max_steps": 300,
  "timeout_ms": 20000 }
```

이 블록을 만나면 런타임이 **자기 프로세스 안에서** 30Hz 루프를 돈다.
관측도 행동도 HTTP 를 타지 않는다. 프런트는 그대로 500ms 로 `current_block_id` 만 보면 되고,
진행 표시·정지·오류 처리는 이미 있는 것을 쓴다. 추가되는 것은 블록 타입 하나와
결과 필드 하나(`success`)뿐이다.

### 그 안에 우리가 넣어야 하는 것 (AI 쪽 작업)

| 필요 | 현재 |
|---|---|
| 카메라 2대 렌더 → `(3,224,224) uint8` | `sim/mujoco/env.py` 에 있음 |
| `state` 정규화 (deg → rad → [-1,1]) | `sim/mujoco/build_scene.normalize` 에 있음. **변환 지점을 한 곳으로 못박아야 한다** |
| `state_timestamp` / `action_timestamp` | 시뮬은 0 이지만 필드는 필수 |
| ckpt 로드 시 계약 버전·카메라 이름 대조 후 거부 | 없음 — 만들어야 함 |

## 5. HW 에 보낼 요약

1. 각도 규격 검증 완료. 초과 1.37e-06 < 허용오차 1e-4 → **지금 방식 유지**.
2. `build_scene.py` 에 비ASCII 경로 패치가 빠졌다. 한글 경로에서 `ParseXML` 로 죽는다. `ai` 브랜치에서 재동기화.
3. 런타임 모델 로드를 `build_model()` 로 바꾸면 블록 조작과 데이터 수집이 같은 세계에서 돈다.
4. 정책 실행은 `POST /jobs` 로 불가. `type: "policy"` 블록 하나만 추가하면 된다.
