"""Merge contract datasets into a new directory, validating every episode.
계약 데이터셋들을 새 디렉터리로 병합하고 모든 에피소드를 검증한다.

Hard-links where the filesystem allows it, so merging a 6.8GB dataset costs no
disk. Falls back to copying across filesystems.
가능하면 하드링크를 쓴다. 6.8GB 데이터셋을 병합해도 디스크를 안 먹는다.
파일시스템이 다르면 복사로 내려간다.

⚠️ Refuses to write into an existing directory. Reusing a dataset name is how
   98 completed episodes lost their first 69 on 2026-09-07.
⚠️ 이미 있는 디렉터리에는 쓰지 않는다. 이름 재사용이 2026-09-07 에 완주한 98편의
   앞 69편을 날린 경로다.

    python tools/merge_datasets.py datasets/sim_pick_v5 out/labels out/merged
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contract.episode import read_episode, validate, write_dataset_index  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("sources", type=Path, nargs="+", help="병합할 데이터셋들")
    p.add_argument("out", type=Path, help="새 디렉터리. 기존 경로는 거부한다")
    args = p.parse_args()

    out: Path = args.out
    sources: list[Path] = args.sources
    if out.exists():
        raise SystemExit(
            f"{out} 가 이미 있다. 데이터셋 이름을 재사용하지 않는다 — "
            "2026-09-07 에 완주한 v4 98편의 앞 69편을 그렇게 날렸다"
        )
    for s in sources:
        if not s.is_dir():
            raise SystemExit(f"입력 데이터셋이 없다: {s}")

    out.mkdir(parents=True)
    linked = copied = 0
    per_source: dict[str, int] = {}

    for src_dir in sources:
        n = 0
        for npz in sorted(src_dir.glob("*.npz")):
            for src in (npz, npz.with_suffix(".json")):
                if not src.exists():
                    continue
                dst = out / src.name
                if dst.exists():
                    raise SystemExit(
                        f"이름 충돌: {src.name} 가 이미 {out} 에 있다. "
                        "두 데이터셋의 에피소드 이름이 겹친다 — 병합 전에 "
                        "한쪽 이름을 바꿔라"
                    )
                try:
                    os.link(src, dst)
                    linked += 1
                except OSError:
                    shutil.copy2(src, dst)
                    copied += 1
            n += 1
        per_source[str(src_dir)] = n
        print(f"  {src_dir}: {n}편")

    eps = sorted(out.glob("*.npz"))
    problems: dict[str, list[str]] = {}
    for path in eps:
        got = validate(read_episode(path))
        if got:
            problems[path.name] = got
    if problems:
        for name, got in list(problems.items())[:5]:
            print(f"  ❌ {name}: {got}")
        raise SystemExit(
            f"병합 결과에 계약 위반 {len(problems)}편. "
            "입력 데이터셋 중 하나가 이미 위반을 담고 있다"
        )

    write_dataset_index(out, {
        "experimental_only": True,
        "sources": {k: v for k, v in per_source.items()},
        "n_episodes": len(eps),
        "contract_violations": 0,
    })
    print(f"\n병합 {len(eps)}편 · 하드링크 {linked} · 복사 {copied} · 계약 위반 0")
    print(f"→ {out}/dataset.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
