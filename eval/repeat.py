"""Train the same configuration several times and report the spread.
같은 설정으로 여러 번 학습하고 그 폭을 보고한다.

Why a single run cannot stand for a checkpoint.
왜 한 번의 실행이 체크포인트를 대표하지 못하는가.

On 2026-09-01 two trainings on identical data with identical settings scored 0%
and 25% on the same seed block. Reporting either number alone would have been a
claim the measurement does not support -- one would have said the policy fails,
the other that it passes the gate. The difference was training nondeterminism,
and nothing in the loss curve hinted at it.
2026-09-01, 동일한 데이터·동일한 설정의 학습 두 번이 같은 시드 블록에서 0% 와 25% 를
냈다. 둘 중 하나만 보고했다면 계측이 뒷받침하지 않는 주장이 됐을 것이다 — 하나는
정책이 실패한다고, 다른 하나는 게이트를 통과한다고 말했을 테니까. 차이는 학습의
비결정성이었고, 손실 곡선에는 아무 힌트도 없었다.

So the deployable-checkpoint gate requires `runs >= 3` and this tool produces it:
N trainings with different seeds, each scored under the same conditions, reported
as mean and range rather than a single number.
그래서 배포 가능 체크포인트 게이트는 `runs >= 3` 을 요구하고, 이 도구가 그것을
만든다. 서로 다른 시드로 N 회 학습하고 각각을 같은 조건에서 채점한 뒤, 단일 수치가
아니라 평균과 범위로 보고한다.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from contract.skills import GATE_MIN_RUNS, ROLLOUT_GATE
from eval.stats import wilson_ci
from paths import AI_ROOT, DEFAULT_EXP_LOG
from sim.mujoco.build_scene import DEFAULT_CONFIG
from tracking.exp_log import file_digest, log_run


@dataclass
class RunResult:
    """One training and its rollout score.
    학습 한 번과 그 롤아웃 점수."""

    seed: int
    ckpt: Path
    rate: float
    n_episodes: int
    action_space: str
    val_loss: float


def _run(cmd: list[str]) -> None:
    """Run a tool as a subprocess so it logs to EXP_LOG exactly as it normally does.
    도구를 하위 프로세스로 돌린다. 평소와 똑같이 EXP_LOG 에 기록되게 하려는 것이다."""
    print(f"\n$ {' '.join(str(c) for c in cmd)}\n", flush=True)
    proc = subprocess.run(cmd, cwd=AI_ROOT)
    if proc.returncode not in (0, 1):
        # 1 은 게이트 실패를 뜻한다. 그건 결과이지 오류가 아니다.
        raise SystemExit(f"명령이 종료코드 {proc.returncode} 로 실패했다: {cmd[0]}")


def _latest_record(experiment: str, match: dict[str, Any]) -> dict[str, Any] | None:
    """Newest EXP_LOG record whose conditions contain every key/value in `match`.
    conditions 가 `match` 의 모든 키·값을 담은 가장 최근 EXP_LOG 기록."""
    found: dict[str, Any] | None = None
    for line in DEFAULT_EXP_LOG.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("experiment") != experiment:
            continue
        cond = rec.get("conditions") or {}
        if all(cond.get(k) == v for k, v in match.items()):
            found = rec
    return found


def repeat(
    data: Path,
    *,
    runs: int,
    episodes: int,
    seed_base: int,
    eval_seed_base: int,
    device: str,
    jitter: float,
) -> list[RunResult]:
    """Train `runs` times, score each, and collect the results.
    `runs` 회 학습하고 각각 채점해 결과를 모은다."""
    out: list[RunResult] = []
    for i in range(runs):
        seed = seed_base + i
        ckpt = AI_ROOT / "checkpoints" / "bc" / f"{data.name}_seed{seed}.pt"

        _run([sys.executable, "tools/train_bc.py", "--data", str(data),
              "--seed", str(seed), "--out", str(ckpt), "--device", device, "--log"])
        _run([sys.executable, "tools/eval_rollout.py", "--episodes", str(episodes),
              "--seed-base", str(eval_seed_base), "--jitter", str(jitter), "--render",
              "--policy-ckpt", str(ckpt), "--log"])

        roll = _latest_record("rollout_baselines", {"policy_ckpt": str(ckpt)})
        train = _latest_record("train_bc", {"trained_on": str(data), "seed": seed})
        if roll is None:
            raise SystemExit(
                f"EXP_LOG 에서 {ckpt.name} 의 롤아웃 기록을 찾지 못했다. "
                "eval_rollout 이 --log 로 돌았는지 확인하라."
            )
        rate = float(roll["result"]["success_rates"]["bc"])
        out.append(RunResult(
            seed=seed,
            ckpt=ckpt,
            rate=rate,
            n_episodes=int(roll["conditions"]["episodes"]),
            action_space=str(roll["conditions"].get("policy_action_space", "?")),
            val_loss=float((train or {}).get("result", {}).get("best_val_loss", float("nan"))),
        ))
    return out


def summarise(results: list[RunResult]) -> dict[str, Any]:
    """Mean, range, and a pooled interval — with the pooling caveat attached.
    평균·범위와 합산 구간. 합산의 한계를 함께 붙인다."""
    rates = [r.rate for r in results]
    n_each = results[0].n_episodes
    total_n = n_each * len(results)
    total_ok = int(round(sum(rates) * n_each))
    lo, hi = wilson_ci(total_ok, total_n)
    return {
        "runs": len(results),
        "episodes_each": n_each,
        "mean": float(np.mean(rates)),
        "min": float(np.min(rates)),
        "max": float(np.max(rates)),
        "spread": float(np.max(rates) - np.min(rates)),
        "pooled_successes": total_ok,
        "pooled_n": total_n,
        "ci95": [lo, hi],
        "ci95_caveat": (
            "구간은 실행 전체를 합산해 계산했다. 서로 다른 학습은 엄밀히는 같은 "
            "정책의 표본이 아니므로, 이 구간은 '이 설정이 내놓는 정책'의 구간으로 "
            "읽어야 한다. 개별 체크포인트의 구간이 아니다."
        ),
    }


def format_report(results: list[RunResult], summary: dict[str, Any]) -> str:
    lines = [
        f"{'시드':>6s} {'행동공간':>16s} {'val_loss':>10s} {'롤아웃':>9s}  체크포인트",
        "-" * 78,
    ]
    for r in results:
        lines.append(
            f"{r.seed:6d} {r.action_space:>16s} {r.val_loss:10.5f} "
            f"{r.rate * 100:8.1f}%  {r.ckpt.name}"
        )
    lines.append("-" * 78)
    lines.append(
        f"{'평균':>6s} {'':>16s} {'':>10s} {summary['mean'] * 100:8.1f}%   "
        f"범위 {summary['min'] * 100:.1f}~{summary['max'] * 100:.1f}% "
        f"(폭 {summary['spread'] * 100:.1f}%p)"
    )
    lines.append(
        f"{'합산':>6s} {'':>16s} {'':>10s} "
        f"{summary['pooled_successes']}/{summary['pooled_n']}   "
        f"95% 구간 {summary['ci95'][0] * 100:.1f}~{summary['ci95'][1] * 100:.1f}%"
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=GATE_MIN_RUNS)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=0, help="학습 시드 시작값")
    parser.add_argument("--eval-seed-base", type=int, default=3000, help="평가 시드 블록")
    parser.add_argument("--jitter", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--author", type=str, default="김준태(트랙B)")
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()

    if args.runs < GATE_MIN_RUNS:
        print(
            f"⚠️ runs={args.runs} 는 배포 게이트의 최소치 {GATE_MIN_RUNS} 보다 작다. "
            "이 결과로는 배포 판정을 할 수 없다."
        )

    print(f"학습 {args.runs}회 × 롤아웃 {args.episodes}편 · 데이터 {args.data}")
    print(f"평가 시드 블록 {args.eval_seed_base}~{args.eval_seed_base + args.episodes - 1} "
          "(모든 실행이 동일)\n")

    results = repeat(
        args.data,
        runs=args.runs,
        episodes=args.episodes,
        seed_base=args.seed_base,
        eval_seed_base=args.eval_seed_base,
        device=args.device,
        jitter=args.jitter,
    )
    summary = summarise(results)

    print("\n" + "=" * 78)
    print(format_report(results, summary))
    print()

    spaces = {r.action_space for r in results}
    if len(spaces) > 1:
        print(f"⚠️ 행동 공간이 섞였다: {spaces}. 같은 조건의 반복이 아니다.")

    mean_ok = summary["mean"] > ROLLOUT_GATE
    ci_ok = summary["ci95"][0] > ROLLOUT_GATE
    runs_ok = summary["runs"] >= GATE_MIN_RUNS
    passed = mean_ok and ci_ok and runs_ok

    print("배포 게이트 판정 (contract/skills.py 의 상수):")
    print(f"  평균 {summary['mean']:.3f} > {ROLLOUT_GATE:.2f} → {'통과' if mean_ok else '실패'}")
    print(f"  95% 구간 하한 {summary['ci95'][0]:.3f} > {ROLLOUT_GATE:.2f} → "
          f"{'통과' if ci_ok else '실패'}")
    print(f"  실행 {summary['runs']} >= {GATE_MIN_RUNS} → {'통과' if runs_ok else '실패'}")
    print(f"\n→ {'배포 가능' if passed else '배포 불가'}")
    print(f"\n⚠️ {summary['ci95_caveat']}")

    if args.log:
        rec = log_run(
            experiment="repeat_runs",
            author=args.author,
            issue="S15P21A103-34",
            conditions={
                "data": str(args.data),
                "runs": args.runs,
                "episodes": args.episodes,
                "train_seeds": [r.seed for r in results],
                "eval_seed_base": args.eval_seed_base,
                "jitter_m": args.jitter,
                "config_sha": file_digest(DEFAULT_CONFIG),
                "gate": {"rollout": ROLLOUT_GATE, "min_runs": GATE_MIN_RUNS},
            },
            result={
                **summary,
                "passed": passed,
                "per_run": [
                    {"seed": r.seed, "rate": r.rate, "val_loss": r.val_loss,
                     "action_space": r.action_space, "ckpt": str(r.ckpt)}
                    for r in results
                ],
            },
        )
        print(f"\nEXP_LOG.jsonl 기록 (git {rec['git_rev']}, dirty={rec['git_dirty']})")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
