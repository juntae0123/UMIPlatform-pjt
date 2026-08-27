"""Append-only experiment tracker.
추가 전용 실험 트래커.

Every measurement script writes one JSON line per run into EXP_LOG.jsonl.
The point is that a number can always be traced back to the code and config
that produced it — a result without its conditions is not a result.
모든 계측 스크립트는 실행마다 EXP_LOG.jsonl 에 JSON 한 줄을 남긴다.
수치를 만들어낸 코드와 설정으로 언제나 되짚어갈 수 있어야 한다 —
조건 없는 결과는 결과가 아니다.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_AI_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = REPO_AI_ROOT / "EXP_LOG.jsonl"


def _git_rev() -> str:
    """Current commit hash, or 'unknown' outside a repo.
    현재 커밋 해시. 저장소 밖이면 'unknown'."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_AI_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _git_dirty() -> bool | None:
    """True when the worktree has uncommitted changes.
    커밋 안 된 변경이 있으면 True."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_AI_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return bool(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None


def file_digest(path: Path) -> str:
    """Short sha256 of a file, so config drift is detectable after the fact.
    파일의 짧은 sha256. 나중에 설정이 바뀐 것을 알아챌 수 있게."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def log_run(
    experiment: str,
    author: str,
    result: dict[str, Any],
    conditions: dict[str, Any] | None = None,
    issue: str | None = None,
    log_path: Path = DEFAULT_LOG,
) -> dict[str, Any]:
    """Append one experiment record and return it.
    실험 기록 한 건을 append 하고 그 레코드를 반환한다."""
    import mujoco  # local import: the tracker must work without mujoco too

    record: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment": experiment,
        "author": author,
        "issue": issue,
        "git_rev": _git_rev(),
        "git_dirty": _git_dirty(),
        "env": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "mujoco": mujoco.__version__,
        },
        "conditions": conditions or {},
        "result": result,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record
