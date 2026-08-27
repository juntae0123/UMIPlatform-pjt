# TS — 공식 MJCF 를 수정하지 않고 씬을 만들기까지 (2026-08-27)

- 작성자: 김준태 (트랙 B)
- 이슈: S15P21A103-63
- 제약: **공식 SO-101 MJCF 는 수정하지 않는다.** 상류가 갱신되면 통째로 교체만 한다
- 결론: 카메라·접촉패드는 XML 이 아니라 **`mujoco.MjSpec` 로 로드 시점에 주입**한다

공식 MJCF 에는 **카메라가 하나도 없고** 작업대·물체도 없다. 데이터 규격
(image (3,224,224) × 2대) 을 시뮬에서 재현할 방법이 없는 상태에서 시작했다.

## 걸린 것들, 순서대로

### 1. 중첩 include 가 경로를 두 번 붙였다

공식 `scene.xml` 을 include 했더니 메시 경로가 이렇게 됐다:

```
AI/Simulation_mujoco/SO101/AI/Simulation_mujoco/SO101/assets/...
```

`scene.xml` 이 이미 로봇 xml 을 include 하고 있어서, include 가 중첩되며 경로가 겹쳤다.

→ **공식 `scene.xml` 을 쓰지 않고 로봇 xml 을 직접 include 한다.**
조명·바닥은 우리 씬에서 직접 선언한다.

### 2. `<compiler>` 를 `<include>` 앞에 두면 meshdir 이 안 풀린다

include 되는 파일이 `meshdir="assets"` 를 선언하고 있다. 그 상대경로는
**포함하는 쪽 파일의 디렉터리**를 기준으로 풀린다.

```xml
<!-- 틀림 -->
<compiler meshdir="../../../third_party/so101_mujoco/SO101/assets"/>
<include file="../../../third_party/so101_mujoco/SO101/so101_new_calib.xml"/>

<!-- 맞음 — compiler 가 include 뒤에 와야 덮어쓴다 -->
<include file="../../../third_party/so101_mujoco/SO101/so101_new_calib.xml"/>
<compiler angle="radian" meshdir="../../../third_party/so101_mujoco/SO101/assets"/>
```

### 3. include 된 body 는 다시 열 수 없다 — 손목 카메라를 XML 에 못 쓴다

손목 카메라는 `gripper` body 안에 있어야 한다. 그런데 그 body 는 include 된 파일에 있고,
MJCF `include` 는 **포함된 파일의 body 를 다시 열지 못한다.**

공식 MJCF 를 수정하면 되지만 그건 금지다.

→ `mujoco.MjSpec` 으로 모델을 로드한 뒤 **파이썬에서 body 를 찾아 카메라를 붙인다**
(`sim/mujoco/build_scene.py` 의 `_set_camera`). 같은 방식으로 접촉 패드도 주입한다.

### 4. `MjsCamera` 에 `.orientation` 이 없다

MjSpec 카메라의 자세는 `alt` 를 통해 준다:

```python
cam.alt.xyaxes = [...]
cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
```

### 5. 카메라 값이 XML 과 config 두 곳에 있었다 — config 수정이 조용히 무시됐다

처음에는 고정 카메라를 씬 XML 에, 손목 카메라만 주입했다. 그래서 `configs/so101.yaml` 의
고정 카메라 값을 고쳐도 **아무 일도 일어나지 않았다.** XML 쪽이 이미 선언했기 때문이다.

이게 이 문서에서 가장 위험한 항목이다. **에러가 없고, 값이 반영 안 된 것만 조용히 남는다.**

→ **씬 XML 에서 카메라를 전부 제거했다.** 두 대 모두 config 에서만 정의하고 주입한다.
값의 출처가 하나면 이 실수가 구조적으로 불가능해진다.

### 6. 손목 카메라가 그리퍼 메시 껍질 안에 있었다

첫 배치에서 카메라 원점이 `gripper` 메시의 볼록껍질 내부였다 (거리 0.0799 < rbound 0.0843).
렌더가 메시 내부를 봐서 화면이 막혔다.

→ body 로컬 `(0.06, -0.07, 0.02)` 로 옮겼다. 실물 카메라 마운트 형상이 확정되면 재계측 대상.

## 검증

```
씬 컴파일: nq=13, nu=6, ncam=2
config ↔ MJCF 관절범위 불일치: 0건  (rtol=0.0, atol=1e-9)
카메라 2대가 각각 224x224 프레임 실제 렌더: 확인
```

## 남는 규칙

1. **하드웨어 의존 값의 출처는 하나다.** XML 과 config 양쪽에 두면 한쪽이 조용히 이긴다
2. **include 된 것은 수정 대상이 아니다.** 필요한 변경은 로드 시점 주입으로 한다
3. **`<compiler>` 는 `<include>` 뒤에.** meshdir 이 그 규칙에 의존한다
4. 렌더가 이상하면 **카메라가 메시 안에 있는지** 먼저 본다

## 재현

```bash
cd AI && export MUJOCO_GL=egl
python tools/render_check.py     # out/*.png 4장 + config 대조
```
