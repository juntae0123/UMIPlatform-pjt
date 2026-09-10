"""Turn a DAgger collection result into a MEASURE draft.
DAgger 수집 결과를 MEASURE 초안으로 만든다.

Written whether the gates passed or not. A collection that failed its gate is a
measurement too, and the failure is the part that stops it from being repeated.
게이트 통과 여부와 무관하게 쓴다. 게이트에서 떨어진 수집도 계측이고, **그 실패
기록이 같은 실패를 막는 부분이다.**

    python tools/write_dagger_measure.py --result <json> --log <log> --out <md>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--result", type=Path, required=True)
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--prereg", type=str,
                   default="docs/PREREG_dagger_segments_0909.md")
    args = p.parse_args()

    r = json.loads(args.result.read_text(encoding="utf-8"))
    g = r.get("gates", {})
    q = r.get("queries", 0) or 1

    def mark(name: str) -> str:
        v = g.get(name)
        return "통과" if v else ("실패" if v is not None else "—")

    lines: list[str] = [
        "# MEASURE — DAgger 1b 유효 구간 수집 (2026-09-09)",
        "",
        "- 이슈 S15P21A103-34 · 작성 김준태(트랙B)",
        f"- 사전등록 `{args.prereg}` · 확신도 🟢 실행·로그 확인",
        f"- 원본 로그 `{args.log}` · 결과 `{args.result}`",
        "",
        "## 측정 조건 — 조건 없이 인용 금지",
        "",
        "```",
        f"행동 정책      v5 seed0/1/2 순환 (에피소드마다 교대)",
        f"물체 시드      4000~4099 · 지터 0.05m",
        f"라벨 구간      tick {r.get('label_start_tick', 30)}~199",
        f"전문가         ScriptedFeedbackPolicy — 질의만 하고 실행하지 않는다",
        f"장치           정책 추론 cpu · 렌더 egl",
        "```",
        "",
        "## 결과",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| 에피소드 | {r.get('episodes')} |",
        f"| 전문가 질의 | {r.get('queries')} |",
        f"| 유효 라벨 | {r.get('valid_ticks')} ({pct(r.get('valid_label_rate', 0))}) |",
        f"| 저장 라벨 | {r.get('stored_ticks')} ({pct(r.get('stored_label_rate', 0))}) |",
        f"| 구간 수 | {r.get('segments')} |",
        f"| 짧아서 버린 유효 틱 | {r.get('short_valid_ticks_dropped')} |",
        f"| 행동 정책 성공 | {r.get('behavior_successes')}/{r.get('episodes')} |",
        "",
        "### 무효 틱의 내역",
        "",
        "| 원인 | 틱 | 질의 대비 |",
        "|---|---|---|",
        f"| IK 실패 | {r.get('invalid_ik_ticks')} | {pct(r.get('invalid_ik_ticks', 0) / q)} |",
        f"| 범위 초과 | {r.get('invalid_range_ticks')} | {pct(r.get('invalid_range_ticks', 0) / q)} |",
        f"| 둘 다 | {r.get('invalid_both_ticks')} | {pct(r.get('invalid_both_ticks', 0) / q)} |",
        "",
        "### 범위 초과의 크기",
        "",
        "```",
        f"state  최대 초과 {r.get('max_state_excess', 0):.3e}",
        f"action 최대 초과 {r.get('max_action_excess', 0):.3e}",
        "계약 허용치 RANGE_TOLERANCE = 1e-4",
        "```",
        "",
        "⚠️ `state` 는 **측정된** 관절 위치다. clip 하지 않았고 "
        "`RANGE_TOLERANCE` 도 바꾸지 않았다. 정책이 운전하면 "
        "`BCPolicy.act` 가 출력을 [-1,1] 로 클립하므로 관절 한계를 정확히 "
        "명령할 수 있고, 위치 제어기가 한계로 밀면 넘어간다 — MuJoCo 관절 "
        "한계는 soft constraint 다. 측정값을 고치면 **정책이 한계를 때린다는 "
        "사실이 데이터에서 사라진다.**",
        "",
        "## 게이트 (결과 보기 전 확정)",
        "",
        "| 게이트 | 기준 | 판정 |",
        "|---|---|---|",
        f"| 유효 라벨 비율 | ≥95% | {mark('valid_label_rate_ge_95pct')} |",
        f"| 저장 라벨 비율 | ≥95% | {mark('stored_label_rate_ge_95pct')} |",
        f"| 계약 위반 | 0건 (실측 {r.get('contract_violations')}) | {mark('contract_violations_zero')} |",
        f"| **학습 진행** | 위 셋 전부 | **{mark('train_go')}** |",
        "",
    ]

    ph = r.get("phase_invalid") or {}
    if ph:
        lines += ["### IK 실패의 위상 분포", "",
                  "| 위상 | 틱 |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in sorted(ph.items(), key=lambda x: -x[1])]
        lines += [""]

    if not g.get("train_go"):
        lines += [
            "## 게이트 실패 — 학습으로 넘어가지 않았다",
            "",
            "**게이트를 옮기지 않는다.** 게이트 변경은 사후 합리화이고 n 증가는 "
            "정밀도 개선이다. 기준을 바꿔야 한다고 판단되면 **새 사전등록 문서**를 "
            "쓰고 이 문서에 취소선을 긋는다.",
            "",
            "다음에 잴 것:",
            "",
            "- 무효 틱이 IK 쪽인가 범위 쪽인가 — 위 표가 가른다",
            "- 범위 쪽이면 초과량이 1e-3 미만인가. 그렇다면 "
            "`RANGE_TOLERANCE` 상향을 **D-AI 안건**으로 올린다 "
            "(`contract/episode.py` 는 공용이라 혼자 못 바꾼다)",
            "- IK 쪽이면 위상 분포를 본다. 특정 위상에 몰리면 전문가 문제이지 "
            "DAgger 문제가 아니다",
            "",
        ]

    lines += [
        "## 이 수집이 말하지 않는 것",
        "",
        "- 라벨이 tick 30 부터다. 앞 30틱은 전문가 라벨이 없어 버렸다 — "
        "BC 자기 행동을 라벨로 쓰면 자기모방이 된다",
        "- 행동 정책 성공률은 **수집 조건(지터 0.05m, 시드 4000~4099)**의 것이고 "
        "평가 시드 블록(3000~3099)의 롤아웃 수치와 다른 값이다. 나란히 인용하지 마라",
        "- 학습 3회는 **고장검사**다. 조건 비교가 아니다 — D-AI-30 은 조건 비교에 "
        "**5회**를 요구한다. 이 결과로 \"DAgger 가 낫다\" 를 주장할 수 없다",
        "",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
