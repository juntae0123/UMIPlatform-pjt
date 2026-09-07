"""Shared-machine etiquette, enforced in code instead of pasted into commands.
공유 머신 규약을 명령에 붙여 쓰는 대신 코드로 강제한다.

SSAFY's rule is **one allocated GPU, one job**. On 2026-09-07 that was broken three
separate ways in one afternoon, all by hand-written command lines 🟢:

- `MUJOCO_GL=egl` was omitted once and the job died on an OpenGL error
- the same `probe_freeze` command was launched **four times** (queued input from a
  busy terminal), all four writing to the same `>` log, making the log unusable
- `CUDA_VISIBLE_DEVICES=1` took a second GPU, and the server operators warned that
  the machine was fully loaded and jobs would be killed
- every process span BLAS threads equal to the core count (80), so six processes
  produced load average 165 on 80 cores

SSAFY 규약은 **할당 GPU 한 장, 잡 하나**다. 2026-09-07 하루 오후에 그것이 세 가지로
깨졌고 전부 손으로 쓴 명령줄 때문이었다 🟢 — `MUJOCO_GL=egl` 누락으로 OpenGL 에러,
같은 `probe_freeze` 를 **4번** 실행(바쁜 터미널에 큐잉된 입력)해서 넷이 같은 로그에
써서 로그가 무효, `CUDA_VISIBLE_DEVICES=1` 로 두 번째 장을 잡아 서버 측 경고,
그리고 프로세스마다 BLAS 스레드를 코어 수(80)만큼 띄워 6개에서 load average 165.

None of those are knowledge failures -- the rules were known and written down. They
are failures of a workflow that re-types the rules every time. So this module is
imported **first**, before numpy or torch, and does it once.
넷 다 지식의 실패가 아니다 — 규칙은 알고 있었고 적혀 있었다. 규칙을 매번 다시
타이핑하는 작업방식의 실패다. 그래서 이 모듈을 numpy·torch **보다 먼저** 임포트해
한 번에 처리한다.

    import runtime_limits          # noqa: F401  — 반드시 numpy/torch 앞
    runtime_limits.claim("probe_freeze")   # 중복 실행이면 여기서 종료
"""

from __future__ import annotations

import atexit
import os
import sys
from pathlib import Path

# Threads. These must be set before numpy/torch load their BLAS backend, which is
# why this module has to be imported first -- setting them later is silently
# ignored and the process still takes every core.
# 스레드. numpy/torch 가 BLAS 백엔드를 로드하기 전에 정해져야 하므로 이 모듈이
# 먼저 임포트돼야 한다. 나중에 설정하면 조용히 무시되고 코어를 다 먹는다.
DEFAULT_THREADS = "2"
_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

# Headless EGL. Omitting it is a hard crash inside mujoco.MjrContext, and the
# message ("an OpenGL platform library has not been loaded") does not say which
# variable is missing.
# 헤드리스 EGL. 빼먹으면 mujoco.MjrContext 안에서 크래시하고, 메시지는 어느 변수가
# 없는지 말해주지 않는다.
DEFAULT_MUJOCO_GL = "egl"

LOCK_DIR = Path(__file__).resolve().parent / "out"


def _apply() -> None:
    for var in _THREAD_VARS:
        os.environ.setdefault(var, DEFAULT_THREADS)
    os.environ.setdefault("MUJOCO_GL", DEFAULT_MUJOCO_GL)
    # One GPU. `setdefault` so an explicit choice still wins -- but the default is
    # never "all of them".
    # GPU 한 장. 명시 지정은 존중하되 기본값이 "전부" 인 일은 없게 한다.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


_apply()


def torch_threads(n: int | None = None) -> None:
    """Cap torch's intra-op threads. Safe to call after torch is imported.
    torch 의 intra-op 스레드를 제한한다. torch 임포트 후에 불러도 된다."""
    try:
        import torch

        torch.set_num_threads(int(n or DEFAULT_THREADS))
    except Exception:
        pass


def claim(name: str, stale_ok: bool = False) -> None:
    """Refuse to start if another instance of this tool is already running.
    같은 도구가 이미 돌고 있으면 시작을 거부한다.

    A duplicate run is worse than a slow one: four copies of `probe_freeze` all
    redirected to one log produced a file that could not be read, and the wasted
    load is charged to everyone on the machine.
    중복 실행은 느린 것보다 나쁘다. `probe_freeze` 사본 4개가 한 로그로 리다이렉트돼
    읽을 수 없는 파일이 됐고, 낭비된 부하는 머신을 쓰는 모두에게 청구된다.
    """
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lock = LOCK_DIR / f".lock_{name}"
    if lock.exists():
        try:
            pid = int(lock.read_text(encoding="utf-8").split()[0])
        except (ValueError, IndexError, OSError):
            pid = -1
        alive = pid > 0 and Path(f"/proc/{pid}").exists()
        if alive and not stale_ok:
            print(f"✗ `{name}` 이 이미 PID {pid} 로 돌고 있다. "
                  f"중복 실행하지 않는다.\n"
                  f"  끝내려면: kill {pid}\n"
                  f"  락만 지우려면: rm {lock}", file=sys.stderr)
            raise SystemExit(3)
        if not alive:
            print(f"· 죽은 락 정리: {lock.name} (PID {pid})")

    lock.write_text(f"{os.getpid()} {name}\n", encoding="utf-8")
    atexit.register(lambda: lock.unlink(missing_ok=True))


def banner() -> str:
    """One line naming the limits actually in effect.
    실제로 적용된 제한을 한 줄로."""
    return (f"[런타임] GPU {os.environ.get('CUDA_VISIBLE_DEVICES')} · "
            f"스레드 {os.environ.get('OMP_NUM_THREADS')} · "
            f"MUJOCO_GL {os.environ.get('MUJOCO_GL')}")
