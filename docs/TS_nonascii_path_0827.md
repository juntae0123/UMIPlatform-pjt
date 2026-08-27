# TS — 한글 경로 때문에 MuJoCo 가 씬을 못 열었다 (2026-08-27)

- 작성자: 김준태 (트랙 B)
- 이슈: S15P21A103-63
- 증상: `ValueError: ParseXML: Error opening file '...\특화\...\pick_place.xml'`
- 결론: 경로의 `특화` 때문. MuJoCo 는 XML 을 C++ 에서 열고 **Windows 에서 비ASCII 경로를 처리하지 못한다**

## 왜 원인이 안 보였나

**파일은 정상이다.** 존재하고, 파이썬 `open()` 으로 읽히고, git 에도 들어 있다.
그런데 MuJoCo 만 "열 수 없다"고 한다. 에러 문구가 `Error opening file` 이라서
**파일이 없거나 깨진 것처럼 읽힌다.** 실제로는 파일이 아니라 경로의 문자 문제다.

이 함정은 환경에 따라 재현되지 않는다. 리눅스에서는 한글 경로가 그냥 동작한다.
"내 환경에서는 되는데" 가 성립하는 종류의 문제다.

## 좁혀간 순서

### 1. 파일이 없는 것 아닌가 ❌

```
ls -la sim/mujoco/scenes/pick_place.xml   → 존재, 2782 bytes
python -c "Path(...).read_text()"         → 읽힘, 2466자
```

→ 파일 문제가 아니다. 가설 기각.

### 2. 경로를 문자 단위로 봤다 (여기서 원인이 나왔다)

에러 메시지의 경로를 그대로 코드 포인트로 검사했다.

```python
p = r'C:\Users\SSAFY\Desktop\2nd_pjt_2\특화\S15P21A103\AI\...\pick_place.xml'
[(i, c) for i, c in enumerate(p) if ord(c) > 127]
→ [(33, '특'), (34, '화')]
```

**비ASCII 문자 2개.** MuJoCo 의 XML 파서는 C++ 이고, Windows 에서 좁은 문자
(ANSI) API 로 파일을 연다. 유니코드 경로가 그 변환에서 깨진다.

파이썬 레이어(`load_config` 의 YAML 읽기)는 정상 동작했다는 점이 단서였다 —
**같은 디렉터리의 파일을 한쪽은 읽고 한쪽은 못 읽는다면 파일이 아니라 읽는 쪽 문제다.**

## 조치 — `paths.resolve_for_mujoco()`

저장소를 옮기라고 하는 것이 가장 확실하지만, 팀 저장소 위치를 우리가 정할 수 없다.
그래서 코드에서 흡수한다. 세 단계로 시도한다.

| 단계 | 방법 | 대가 |
|---|---|---|
| 1 | 경로가 ASCII 면 그대로 | 없음. 대부분의 환경 |
| 2 | **Windows 8.3 단축 경로** (`GetShortPathNameW`) | 없음. 항상 ASCII, 복사 없음 |
| 3 | ASCII 임시 경로로 모델 트리 복사 | 첫 실행에 파일 복사. 이후 mtime 캐시 |

2단계가 통하지 않는 경우가 있다 — 볼륨별로 8.3 이름 생성을 꺼둘 수 있다
(`fsutil 8dot3name query`). 그때 Windows 는 긴 이름을 그대로 돌려주므로 3단계로 간다.

3단계는 씬의 `include` 와 `meshdir` 이 상대경로인 것을 이용해 **배치를 보존하며**
복사한다. 원본은 건드리지 않고, 복사가 일어나면 그 사실과 경로를 출력한다 —
조용히 다른 파일을 읽는 상황을 만들지 않기 위해서다.

MuJoCo 가 파일을 직접 여는 지점이 `build_scene.py` 한 곳뿐임을 확인한 뒤 그 자리에만 넣었다.

## 검증

| 항목 | 결과 |
|---|---|
| ASCII 경로 | 복사 없이 원본 경로 그대로 반환 |
| 강제 복사 (3단계) | 복사본에서 컴파일 성공, nq=13 nu=6 |
| **한글 경로 실전 재현** | `/tmp/특화테스트/AI` 에 트리를 두고 로드 → ASCII 로 해소, nq=13 nu=6 ncam=2, **관절범위 불일치 0건** |
| 전 도구 재실행 | 파지 20/20 등 **수치 불변** |
| **Windows 실기 동작** | **확인됨** — 사용자 환경(conda `aiot_ai`, Git Bash)에서 뷰어 정상 실행 |

### Windows 실측 — 2단계가 아니라 3단계로 해소됐다 🟢

```
ℹ️ 경로에 ASCII 가 아닌 문자가 있어 MuJoCo 가 파일을 열지 못한다.
   모델을 ASCII 경로로 복사해서 로드한다: C:\Users\SSAFY\AppData\Local\Temp\so101_mjcf_0ba894e51e
   (새로 복사한 파일 35개. 원본은 건드리지 않는다)
```

복사 파일 35개 = 씬 1 + `third_party` 34. 임시 경로는 `%TEMP%` 아래이고
사용자명이 ASCII(`SSAFY`)라서 조건을 만족했다.

**즉 2단계(Windows 8.3 단축 경로)는 이 환경에서 통하지 않았다.**
`resolve_for_mujoco` 를 단축 경로만으로 설계했다면 실패했을 것이고, 3단계를
넣어둔 것이 실제로 필요했다.

⚠️ 2단계가 왜 안 됐는지는 **구분하지 않았다.** 두 가능성이 있다:
   ① 볼륨에서 8.3 이름 생성이 꺼져 있다 (`fsutil 8dot3name query C:`)
   ② `GetShortPathNameW` 는 성공했지만 반환 경로에 여전히 비ASCII 가 남았다
      — 단축 이름은 컴포넌트별로 부여되므로 `특화` 만 단축되지 않았을 수 있다

   구분하려면 이 한 줄을 돌리면 된다:

```bash
python -c "from paths import AI_ROOT, _windows_short_path; print(repr(_windows_short_path(AI_ROOT)))"
```

   `None` 이면 ①(또는 호출 실패), 비ASCII 가 섞인 경로가 나오면 ②다.
   **어느 쪽이든 3단계가 받아내므로 동작에는 영향이 없다.** 기록만 비어 있다.

## 곁에서 잡은 것 — CRLF 노이즈

이 수정을 커밋할 때 `view.py` 190줄이 함께 변경으로 잡혔다.
`git show --ignore-cr-at-eol` 로 보니 **변경이 없었다 — 순수 CRLF 뒤집힘.**

Git for Windows 는 보통 `core.autocrlf=true` 로 설치되어 체크아웃 시 CRLF 로 바꾼다.
내용을 한 글자도 고치지 않았는데 파일 전체가 변경으로 잡히고, 커밋에 섞이면
**diff 로 무엇이 바뀌었는지 볼 수 없게 된다.**

조치: 그 커밋을 되돌리고 `AI/.gitattributes` 로 `eol=lf` 고정.
각자의 autocrlf 설정과 무관해진다. `third_party/**` 는 `-text` 로 제외했다 —
벤더링본은 Git 이 내용을 건드리면 안 된다.

## 남는 규칙

1. **`Error opening file` 을 "파일이 없다"로 읽지 마라.** 경로의 문자, 권한,
   읽는 쪽의 API 를 함께 본다
2. **같은 파일을 한쪽은 읽고 한쪽은 못 읽으면 파일이 아니라 읽는 쪽 문제다**
3. **환경 차이로 재현되지 않는 버그가 있다.** 리눅스에서 한글 경로는 그냥 된다.
   "내 환경에서는 됨" 은 검증이 아니다
4. **커밋 전에 `--stat` 을 본다.** 의도한 파일 수와 다르면 이유를 찾는다.
   줄바꿈 노이즈는 `--ignore-cr-at-eol` 로 가른다

## 재현

```bash
cd AI && unset MUJOCO_GL
python tools/view.py --policy scripted --episodes 3 --speed 0.5
```
