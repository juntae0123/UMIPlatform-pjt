# PROPOSAL — 저장·배포 계약 (DB 스키마 · API) (2026-08-28)

- 작성자: 김준태 (트랙 B)
- 대상: [BE] S15P21A103-18 (저장·배포 계약) · -19 (DB 스키마) · -21~24
- 관련: S15P21A103-27 (데이터 계약)
- 확신도: 🟡 **제안. 합의되지 않았다.** 합의 없이 구현하면 두 번 짠다

## 0. 3계층 원칙

| 계층 | 무엇 | 어디 | 소유 |
|---|---|---|---|
| 파일 | 이미지·state·action 시계열 `.npz` (**평균 9.20MB/에피소드** 🟢) | 오브젝트 스토리지 또는 서버 디스크 | BE 제공, **포맷은 AI** |
| DB | 메타 + 파일 URI | PostgreSQL | BE |
| API | 경계 | HTTP/JSON + multipart | BE 구현, **필수 필드는 AI 지정** |

**원칙: 검증은 보내는 쪽이 한다.** 받는 쪽이 나중에 발견하면 원인 추적이 불가능하다.
계약 위반 업로드는 서버가 거부하고, 사유는 **배열**로 돌려준다(첫 위반에서 멈추지 않는다).

> BE 확인 사항: "파일을 통째로 올린다"는 것에는 이견이 없다 — 처음부터 multipart 통째였다.
> 확인이 필요한 것은 **받은 파일 본문을 DB 컬럼(`bytea`)에 넣는가, 스토리지에 두고 DB 엔 경로만 두는가**다.
> 전자면 1000건에 9.2GB 테이블이 되고 백업·덤프가 매번 그 용량을 통과한다.

## 1. 테이블

`skill · robot · episode · dataset · dataset_episode · train_run · checkpoint · vlm_adapter · deployment · rollout`

AI 가 "없으면 판정·추적이 불가능하다"고 지정하는 컬럼:

- **`episode`** — `source('sim'|'real')`, `cameras jsonb`, `contract_version`, `npz_uri`,
  `max_sync_offset_ms`, `instruction`(VLM용, 계약 확장 필요), `excluded_reason`(행은 지우지 않는다)
- **`dataset`** — `contract_version`, `frozen_at`(얼린 뒤 구성 변경 금지)
- **`dataset_episode`** — `split('train'|'val')` 고정
- **`train_run`** — `config_sha`, `code_sha`, 큐 인덱스 `(status, priority, queued_at)`
- **`checkpoint`** — `gate_metric / gate_value / gate_threshold / gate_passed / gate_eval_env /
  gate_n_trials` + `baselines jsonb`. **이 필드가 없는 체크포인트는 배포 후보가 아니다**
- **`vlm_adapter`** — `base_model`, `base_quant`, `base_sha`, `train_quant`(미스매치 감지), `skills jsonb`
- **`deployment`** — `CREATE UNIQUE INDEX ON deployment (robot_id) WHERE status='active'`
- **`rollout`** — `success`, `failure_reason`(값 목록은 ROS 와 합의), `client_rollout_id`(멱등성)

> `checkpoint` 의 게이트 컬럼과 `deployment` 의 부분 유니크 인덱스가 이 스키마의 핵심이다.
> 전자는 "안 재본 모델은 배포 못 한다", 후자는 "지금 이 로봇에 뭐가 올라가 있나"를
> **DB 제약으로** 강제한다. 문서로만 두면 발표 전날 깨진다.

## 2. 반드시 있어야 할 쿼리

| # | 쿼리 | 없으면 |
|---|---|---|
| 1 | 학습 큐 `FOR UPDATE SKIP LOCKED` | 워커 2대가 같은 잡을 집어 GPU 시간을 두 배로 태운다 |
| 2 | **데이터셋 계약버전 혼입 감지** | 섞인 데이터로 학습 → 손실은 정상, 성공률만 낮다. 추적 최악 |
| 3 | 데이터셋 현황 + `max(max_sync_offset_ms) > 10 → 학습 금지` | 어긋난 데이터를 사후에 걸러낼 방법이 없다 |
| 4 | 배포 후보 (`gate_passed` AND `contract_version` AND `cameras` AND `gate_n_trials >= 20`) | 5회 중 2회로 40% 라고 우기는 것을 못 막는다 |
| 5 | 감사 — 게이트 미통과가 `active` 인가 | 발표 수치의 출처가 사라진다 |
| 6 | 시뮬 vs 실물 성공률 `gap_pp` | Sim2Real 격차를 숫자로 말할 수 없다 |
| 7 | 실패 원인 분포 | "성공률 30%"만 알고 다음에 뭘 고칠지 모른다 |
| 8 | 관제 상단 현재 배포 (`LEFT JOIN`) | 시연 중 "이거 어느 버전이에요"에 답 못 한다 |
| 9 | 어댑터-베이스 정밀도 미스매치 | bf16 어댑터를 Q4 베이스에 붙인 것을 못 잡는다 |

전문(DDL·SQL)은 `바탕화면/pjt2자료/01_보고서/AI파트_전체지도_스키마_0828.html` §3.

## 3. API 경계 5개

| 경계 | 요청 | 거부 |
|---|---|---|
| B1 수집→서버 | `POST /episodes` multipart(meta json 그대로 + npz) | **422 `contract_violation` + `problems[]`** |
| B2 데이터셋→학습 | `POST /datasets` → `/freeze` → `POST /train-runs` | 409 `dataset_not_frozen` / `contract_version_mixed` |
| B2' 워커 | `GET /train-runs/next?gpu=H200` (204=빈 큐), `PATCH` 진행 | — |
| B3 학습→배포등록 | `POST /checkpoints` (**`gate` 필수**) | **409 `gate_not_passed` / 422 `gate_missing`** |
| B4 배포→Jetson | `POST /deployments`, `GET /robots/{name}/deployment/current` | 계약버전·카메라·sha 불일치 시 **로드 거부 후 관제 보고** |
| B5 Jetson→관제 | `POST /rollouts` (+`:batch`), `client_rollout_id` | 중복은 에러가 아니라 200 `{"duplicate":true}` |

`meta` 는 우리가 이미 쓰고 있는 `ep_00000.json` 을 **그대로** 올린다. 필드를 새로 만들지 않는다.

## 4. BE 에 요청하는 것 — 세 줄

1. **체크포인트 등록에서 `gate` 필수 + `passed=false` 는 409 로 거부.**
   규칙을 문서에만 두면 안 지켜지고 API 에 두면 지켜진다.
2. **계약 검증을 서버가 수행.** 검증기 코드는 우리가 제공한다 (`contract/episode.py`).
3. **`deployment(robot_id) WHERE status='active'` 부분 유니크 인덱스.**

## 5. VLM 데이터 — 통째 파일로는 학습이 안 된다

BE 의 "파일 통째로 올리면 VLM 학습에도 용이하다"는 절반만 맞다. 보관은 맞지만 학습에 바로는 안 들어간다.

| 이유 | 내용 |
|---|---|
| 라벨이 없다 | VLM 학습에 필요한 것은 (이미지, 지시문, 정답) 3종. 계약에 `instruction` 도 `skill` 도 없다. **이게 진짜 병목** |
| 프레임 중복 | 에피소드 1건 = 141스텝 × 카메라 2대 = **282장**인데 30Hz 연속이라 거의 같은 그림. VLM 은 지시 시점 1~2장이면 된다 |
| 해상도 | 우리 배열은 224×224 로 이미 축소돼 있다. 대상물이 **1.5~2.5cm** 라 bbox grounding 에는 빠듯할 수 있다 |

**제안 🟡** — 원본 `.npz` 통째 보관은 그대로(불변), VLM 용은 파생물로 뽑는다.

```sql
CREATE TABLE vlm_sample (
  id           BIGSERIAL PRIMARY KEY,
  episode_id   BIGINT NOT NULL REFERENCES episode(id),
  frame_index  INT    NOT NULL,   -- 원본 npz 의 몇 번째 프레임인가 (재현성)
  image_uri    TEXT   NOT NULL,   -- 지시 시점 프레임, 원본 해상도 JPEG
  instruction  TEXT   NOT NULL,
  skill        TEXT   NOT NULL,   -- skill.key 와 동일해야 한다
  target_bbox  JSONB,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

수집 시 **지시 시점 프레임 1장만 원본 해상도로 별도 저장**한다. 에피소드당 수백 KB 로
전체 용량 영향은 미미하고, 나중에 "해상도가 모자랐다"는 이유로 재수집하는 것보다 훨씬 싸다.
