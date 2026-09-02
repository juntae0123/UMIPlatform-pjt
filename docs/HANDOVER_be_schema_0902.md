# BE 전달 — AI 파트가 확정한 스키마 (2026-09-02)

작성: 김준태(트랙B) → 최은찬(BE) · 관련 이슈 S15P21A103-18, -19, -78, -27, -109

## 먼저: 지금 못 박아도 되는 것 / 기다려야 하는 것

| | 상태 |
|---|---|
| 에피소드 저장 포맷·필드 | **확정** (계약 0.2.0). 스키마 잡아도 된다 |
| 스킬 레지스트리 구조 | **확정** (D-AI-22·25) |
| 배포 게이트 강제 규칙 | **확정** (D-AI-23). 이건 꼭 DB 레벨로 넣어달라 |
| VLM 출력 스키마 | **확정** (D-AI-24) |
| 스킬 5종의 `skill_id` 값 | **확정** — 다만 이건 튜토리얼 층이다 (아래 참조) |
| 데모 층 스킬 | **미정** — 페르소나 확정 대기. `tier='demo'` 행은 나중에 들어온다 |
| 정책 체크포인트 실물 | **없음** — 아직 게이트를 넘은 정책이 0개다 |

**중요**: 스킬이 다섯 개로 확정됐다는 말이 다섯 개가 동작한다는 말이 아니다.
`status` 컬럼을 반드시 두고, UI 는 `deployed` 만 실행 가능으로 보여야 한다.

---

## 1. 에피소드 — 파일은 파일로, DB 에는 메타만

BE 쪽에서 "에피소드를 파일 통째로 올리려 한다"고 했는데 **그게 맞다.** 배열을 DB 에
넣지 마라.

| 항목 | 실측값 🟢 |
|---|---|
| 에피소드 1건 | 평균 141스텝 / **9.04MB** (npz 압축 후) |
| 98편 | 885.6MB |
| 스킬 5종 × 150편 | **약 6.8GB** |

한 에피소드는 `이미지 (T,3,224,224) uint8 × 카메라 2대` 가 용량의 대부분이다.
141스텝 기준 카메라당 약 21MB 원본이 압축돼 9MB 가 된다. 이걸 행으로 쪼개면
한 시연이 141행 × 2 이미지가 되고, 학습은 어차피 파일 단위로 읽는다.

**저장 형태**: `<episode_id>.npz` (배열) + `<episode_id>.json` (메타) 두 파일.
AI 쪽 `contract/episode.py` 의 `write_episode()` 가 만드는 그대로다.

### 계약 0.2.0 필드

```
관측  image             (T, 3, 224, 224) uint8 CHW × 카메라 2대
      state             (T, 6) float32, [-1, 1] 정규화
      state_timestamp   (T,) float64 초
행동  action            (T, 6) float32, [-1, 1]   목표 관절각 5 + 그리퍼 1
      action_timestamp  (T,) float64 초
메타  episode_id, skill_id, task, source, success, n_steps, control_rate_hz,
      cameras, contract_version, collected_by, config_sha, git_rev, notes
```

`state_timestamp` 와 `action_timestamp` 를 따로 두는 이유: ARCore 는 pose 스트림과
RGB 스트림이 별도라 둘 사이에 오차가 생긴다. 이슈 30 이 그 오차를 계측해야 하는데,
필드가 없으면 계측 자체가 불가능하다. **시뮬에서는 값이 같아 보이지만 지우지 마라.**

`skill_id` 는 자유 문자열이 아니라 enum 이다 (아래 3절).

---

## 2. DDL 제안

PostgreSQL 기준. 타입·이름은 BE 관례에 맞춰 바꿔도 되지만
**CHECK 제약과 유니크 인덱스는 그대로 유지해달라** — 그게 이 스키마의 요점이다.

```sql
-- 스킬 카탈로그 -------------------------------------------------------------
CREATE TABLE skill (
  skill_id      TEXT PRIMARY KEY,          -- contract/ids.py 의 SKILL_IDS 와 일치
  tier          TEXT NOT NULL CHECK (tier IN ('tutorial','demo')),
  status        TEXT NOT NULL CHECK (status IN
                  ('planned','collecting','training','gate_failed','deployed')),
  display_name  TEXT NOT NULL,
  needs_vlm     BOOLEAN NOT NULL DEFAULT FALSE,
  persona       TEXT,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  -- 데모 층 스킬은 누구를 겨냥한 것인지 없이 존재할 수 없다
  CONSTRAINT demo_needs_persona CHECK (tier <> 'demo' OR persona IS NOT NULL)
);

-- 에피소드 (시연 1건) --------------------------------------------------------
CREATE TABLE episode (
  episode_id       TEXT PRIMARY KEY,
  skill_id         TEXT NOT NULL REFERENCES skill(skill_id),
  contract_version TEXT NOT NULL,           -- '0.2.0-provisional'
  source           TEXT NOT NULL CHECK (source IN ('sim','real')),
  success          BOOLEAN NOT NULL,
  n_steps          INT  NOT NULL CHECK (n_steps > 0),
  control_rate_hz  REAL NOT NULL,
  cameras          TEXT[] NOT NULL,
  collected_by     TEXT NOT NULL,
  config_sha       TEXT NOT NULL,           -- 어느 하드웨어 설정으로 찍었나
  git_rev          TEXT NOT NULL,           -- 어느 코드가 찍었나
  storage_uri      TEXT NOT NULL,           -- .npz
  meta_uri         TEXT NOT NULL,           -- .json
  bytes            BIGINT NOT NULL,
  created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ON episode (skill_id, success);
CREATE INDEX ON episode (contract_version);

-- 학습 잡 -------------------------------------------------------------------
CREATE TABLE training_job (
  job_id        UUID PRIMARY KEY,
  skill_id      TEXT NOT NULL REFERENCES skill(skill_id),
  -- 어떤 에피소드를 썼는지 재현 가능하게. 예:
  -- {"skill_id":"pick_place","source":"real","contract_version":"0.2.0-provisional"}
  dataset_query JSONB NOT NULL,
  status        TEXT NOT NULL CHECK (status IN ('queued','running','failed','done')),
  gpu_index     INT,
  git_rev       TEXT,
  started_at    TIMESTAMPTZ,
  finished_at   TIMESTAMPTZ
);

-- 체크포인트 + 게이트 --------------------------------------------------------
CREATE TABLE checkpoint (
  ckpt_id          UUID PRIMARY KEY,
  job_id           UUID NOT NULL REFERENCES training_job(job_id),
  skill_id         TEXT NOT NULL REFERENCES skill(skill_id),
  contract_version TEXT NOT NULL,
  action_space     TEXT NOT NULL CHECK (action_space IN
                     ('joint_absolute','joint_delta')),
  storage_uri      TEXT NOT NULL,
  n_params         BIGINT NOT NULL,

  -- 게이트. 전부 NULL 이면 "아직 안 재봤다"는 뜻이고, 그 상태로는 배포 불가다.
  gate_metric      TEXT,
  gate_n           INT,
  gate_runs        INT,
  gate_value       REAL,
  gate_ci_low      REAL,
  gate_ci_high     REAL,
  gate_measured_at TIMESTAMPTZ,

  created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  -- 반쪽짜리 게이트 기록을 금지한다. 재봤으면 전부 있어야 한다.
  CONSTRAINT gate_complete CHECK (
    gate_metric IS NULL OR (
      gate_n IS NOT NULL AND gate_runs IS NOT NULL AND gate_value IS NOT NULL
      AND gate_ci_low IS NOT NULL AND gate_ci_high IS NOT NULL
    )
  )
);

-- 배포 가능한 체크포인트는 이 뷰에서만 고른다 -------------------------------
CREATE VIEW deployable_checkpoint AS
SELECT * FROM checkpoint
WHERE gate_metric = 'rollout_success'
  AND gate_n     >= 100        -- n=20 은 신뢰구간 반폭이 게이트 폭과 같다
  AND gate_runs  >= 3          -- 학습 실행 간 성공률이 25%p 갈린다
  AND gate_value  > 0.20
  AND gate_ci_low > 0.20;      -- 하한도 게이트 위여야 "통과"다

-- 배포 ----------------------------------------------------------------------
CREATE TABLE deployment (
  deployment_id UUID PRIMARY KEY,
  robot_id      UUID NOT NULL,
  skill_id      TEXT NOT NULL REFERENCES skill(skill_id),
  ckpt_id       UUID NOT NULL REFERENCES checkpoint(ckpt_id),
  vlm_version   TEXT,
  status        TEXT NOT NULL CHECK (status IN ('active','retired')),
  deployed_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- 로봇 하나에 스킬 하나당 활성 배포는 하나뿐. 은퇴본은 이력으로 남긴다.
CREATE UNIQUE INDEX one_active_per_robot_skill
  ON deployment (robot_id, skill_id) WHERE status = 'active';
```

### 왜 게이트를 DB 에 넣어달라고 하는가

게이트가 애플리케이션 코드에만 있으면 언젠가 우회된다. "일단 시연 전이니까"가
가장 흔한 우회 사유다. `deployable_checkpoint` 뷰에서만 배포를 고르면 우회하려면
스키마를 고쳐야 하고, 스키마를 고치는 건 눈에 띈다.

숫자의 근거는 우리 실측이다 🟢 2026-09-01:
- `gate_n >= 100` — n=20 은 95% 신뢰구간 반폭이 약 ±18%p 로 게이트 폭(20%p)과 같다.
  그 표본으로는 통과·실패를 구분할 수 없다
- `gate_runs >= 3` — 같은 데이터·같은 설정으로 학습을 두 번 했더니 성공률이
  0% 와 25% 로 갈렸다. 단일 실행의 수치는 체크포인트를 대표하지 못한다
- `gate_ci_low > 0.20` — 점추정만 보면 "25%니까 통과"인데 신뢰구간이 9~49% 다

---

## 3. `skill_id` 는 enum 이다

정본은 `AI/contract/ids.py` 의 `SKILL_IDS` 다.

```
pick_place        집어 옮기기          tutorial
sort_two          두 곳으로 분류        tutorial   VLM 필요
align_fixture     지그에 정렬          tutorial
present_inspect   검사 자세 제시        tutorial
line_up           순서대로 늘어놓기      tutorial
```

FE 드롭다운, VLM 출력, 학습 잡, 배포가 전부 이 값을 쓴다. **자유 문자열로 두면
안 되는 이유**: 네 주체가 같은 것을 같은 이름으로 불러야 하는데 그중 하나가 VLM 이고,
제약이 없으면 그럴듯한 여섯 번째 스킬 이름을 만들어낸다.

값이 늘거나 바뀌면 AI 쪽에서 먼저 알린다. BE 는 `skill` 테이블 시드로 관리하면 된다.

---

## 4. VLM 출력 스키마 (FE·BE 경계)

```json
{
  "skill_id": "sort_two",
  "target": {
    "description_ko": "긁힘 있는 부품",
    "bbox_norm": [0.31, 0.44, 0.39, 0.52]
  },
  "destination": "right_tray",
  "confidence": 0.87,
  "abstain": false
}
```

- `skill_id`, `destination` 은 enum. 스킬마다 허용 `destination` 이 다르다
- `bbox_norm` 은 [0,1] 정규화 좌상단·우하단
- **`abstain=true` 이거나 `confidence < 0.70` 이면 로봇을 움직이지 않고 사용자에게 되묻는다**

`0.70` 은 🟡 잠정값이다. VLM baseline 측정(M0) 후 조정될 수 있고, 조정하면 알린다.
BE 는 이 값을 **설정으로 빼두면** 좋겠다 — 코드 상수로 박지 말아달라.

---

## 5. 아직 우리가 못 준 것

| 항목 | 언제 |
|---|---|
| 실제 체크포인트 | 파지 정책이 게이트를 넘은 뒤. **아직 0개** |
| 데모 층 스킬 행 | 페르소나 확정 후 |
| VLM 모델 버전 문자열 | 후보 2종(Qwen2.5-VL-3B / SmolVLM2-500M) fp16 안전성만 통과. 선택 미완 |
| `action_space` 확정값 | 지금 `joint_absolute` 인데 `joint_delta` 로 바뀔 가능성이 높다. **컬럼은 지금 넣어달라** |
| UMI 실시연 데이터 | 0편. 지금 있는 건 시뮬 스크립트 데이터뿐이다 |

`action_space` 컬럼을 지금 넣어달라는 이유: 어제 측정에서 절대 관절각 표현이
정밀도를 숨긴다는 게 드러났고(6개 관절 중 4개에서 예측 오차 > 요구 이동량),
델타 표현으로 바꿀 예정이다. 그러면 **같은 스킬의 체크포인트인데 행동 표현이 다른
두 개가 공존**하게 된다. 컬럼이 없으면 어느 것이 어느 표현인지 구분이 안 된다.

---

## 6. 질문 (BE → AI 로 답 주면 되는 것)

1. `storage_uri` 는 S3 인가 로컬 볼륨인가? 학습 서버(V100)가 직접 읽을 수 있어야 한다
2. 에피소드 업로드 단위 — 1건씩인가, 세션(수십 건) 묶음인가?
   6.8GB 규모라 묶음 업로드면 재시도 정책이 필요하다
3. 학습 큐가 잡을 던질 때 GPU 인덱스를 BE 가 고르나, 학습 쪽이 고르나?
   V100 은 NVLink 가 없어 **GPU 당 독립 잡**이 맞다
