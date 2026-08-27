"""CLI entry point for `so101_ai.eval.rollout`.
`so101_ai.eval.rollout` 의 CLI 진입점.

Equivalent to `python -m so101_ai.eval.rollout`. This file exists so the runnable
commands are discoverable in one directory.
`python -m so101_ai.eval.rollout` 과 같다. 실행 가능한 명령이 한 디렉터리에서 보이도록
두는 파일이다.
"""

from so101_ai.eval.rollout import main

if __name__ == "__main__":
    raise SystemExit(main())
