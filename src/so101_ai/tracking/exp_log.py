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

from so101_ai.paths import AI_ROOT, DEFAULT_EXP_LOG

REPO_AI_ROOT = AI_ROOT
DEFAULT_LOG = DEFAULT_EXP_LOG


def _git(*args: str) -> tuple[int, str]:
    """Run git in the AI root and return (returncode, stdout).
    AI 루트에서 git 을 돌리고 (종료코드, stdout) 을 반환한다."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=REPO_AI_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return out.returncode, out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return 127, ""


def _git_rev() -> str:
    """Current commit hash, or 'unknown' outside a repo.
    현재 커밋 해시. 저장소 밖이면 'unknown'."""
    code, out = _git("rev-parse", "--short", "HEAD")
    return out if code == 0 and out else "unknown"


def _git_dirty() -> bool | None:
    """True when the worktree has uncommitted changes, None when not a repo.
    커밋 안 된 변경이 있으면 True. **저장소가 아니면 None.**

    This used to return False outside a repo, because `git status` printed
    nothing and empty output was read as "clean". Every logged run therefore
    claimed a clean tree while sitting in a directory git had never heard of.
    예전에는 저장소 밖에서 False 를 반환했다. `git status` 가 아무것도 출력하지
    않고, 그 빈 출력을 "깨끗함"으로 읽었기 때문이다. 그래서 모든 기록이
    git 이 알지도 못하는 디렉터리에서 "clean" 을 주장했다.
    """
    code, out = _git("status", "--porcelain")
    if code != 0:
        return None
    return bool(out)


def file_digest(path: Path) -> str:
    """Short sha256 of a file, so config drift is detectable after the fact.
    파일의 짧은 sha256. 나중에 설정이 바뀐 것을 알아챌 수 있게."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


# What counts as "the code that produced this number".
# "이 수치를 만든 코드"에 해당하는 것.
CODE_GLOBS: tuple[str, ...] = ("src/so101_ai/**/*.py", "configs/*.yaml", "scenes/*.xml")


def code_digest(root: Path = REPO_AI_ROOT, globs: tuple[str, ...] = CODE_GLOBS) -> str:
    """Short sha256 over the source tree — ties a number to code without git.
    소스 트리 전체의 짧은 sha256. git 없이도 수치를 코드에 묶는다.

    Measurements run wherever MuJoCo is installed, which is not always a git
    checkout. When it is not, `git_rev` is 'unknown' and the record loses its
    only link back to the code — which defeats the entire purpose of the log.
    This hash does not care about git: same sources, same digest.
    계측은 MuJoCo 가 깔린 곳에서 돌고, 그곳이 항상 git 체크아웃은 아니다.
    아닐 때 `git_rev` 는 'unknown' 이 되고, 기록은 코드로 되짚어갈 유일한 연결을
    잃는다. 로그의 존재 이유가 무너지는 지점이다. 이 해시는 git 을 보지 않는다.
    같은 소스면 같은 값이다.
    """
    h = hashlib.sha256()
    files: list[Path] = []
    for pattern in globs:
        files.extend(p for p in root.glob(pattern) if p.is_file())
    for path in sorted(set(files), key=lambda p: p.relative_to(root).as_posix()):
        h.update(path.relative_to(root).as_posix().encode("utf-8"))
        h.update(hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()[:12]


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

    rev = _git_rev()
    dirty = _git_dirty()
    if rev == "unknown":
        print(
            f"⚠️ EXP_LOG: {REPO_AI_ROOT} 는 git 체크아웃이 아니다. "
            "git_rev='unknown' 으로 기록한다 — 커밋으로 되짚어갈 수 없다. "
            "code_sha 로만 코드를 특정할 수 있다."
        )

    record: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment": experiment,
        "author": author,
        "issue": issue,
        "git_rev": rev,
        "git_dirty": dirty,
        "code_sha": code_digest(),
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
