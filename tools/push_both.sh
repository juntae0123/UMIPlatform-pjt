#!/usr/bin/env bash
# Push to the team GitLab and to the personal GitHub mirror in one step.
# 팀 GitLab 과 개인 GitHub 미러에 한 번에 올린다.
#
# The mirror is a subtree split of AI/, so its history is the AI-part commits and
# nothing else. `subtree split` recomputes hashes every time, so the mirror push
# is a force push -- that repository holds nothing the split does not regenerate.
# 미러는 AI/ 의 subtree split 이라 히스토리가 AI 파트 커밋만으로 이뤄진다.
# `subtree split` 은 매번 해시를 새로 계산하므로 미러 push 는 force 다. 그 저장소에는
# split 이 다시 만들어내지 못하는 것이 없다.
#
# 사용법:  bash AI/tools/push_both.sh          (저장소 어디서든)

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
MIRROR_BRANCH="ai-standalone"

# Only tracked changes block the push. Untracked files are not going anywhere --
# refusing because of one is refusing for a reason that does not exist.
# push 를 막는 것은 추적 중인 변경뿐이다. 추적되지 않는 파일은 어차피 올라가지
# 않으므로, 그것 때문에 거부하는 것은 존재하지 않는 이유로 거부하는 것이다.
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "✗ 커밋되지 않은 변경이 있다. 커밋하고 다시 실행하라:"
  git status --short --untracked-files=no
  exit 1
fi

UNTRACKED="$(git ls-files --others --exclude-standard | head -5)"
if [ -n "${UNTRACKED}" ]; then
  echo "· 추적되지 않는 파일이 있다 (올라가지 않는다):"
  echo "${UNTRACKED}" | sed 's/^/    /'
  echo
fi

echo "=============================================="
echo " 1/2  팀 GitLab (origin ${BRANCH})"
echo "=============================================="
git pull --rebase origin "${BRANCH}"
git push origin "${BRANCH}"

if ! git remote get-url gh >/dev/null 2>&1; then
  echo
  echo "⚠️ 리모트 'gh' 가 없다. GitHub 미러는 건너뛴다."
  echo "   추가하려면: git remote add gh https://github.com/juntae0123/UMIPlatform-pjt.git"
  exit 0
fi

echo
echo "=============================================="
echo " 2/2  GitHub 미러 (gh main) — AI/ 만"
echo "=============================================="
git branch -D "${MIRROR_BRANCH}" >/dev/null 2>&1 || true
git subtree split --prefix=AI -b "${MIRROR_BRANCH}" -q
echo "AI/ 커밋 $(git rev-list --count "${MIRROR_BRANCH}") 개"
git push gh "${MIRROR_BRANCH}:main" --force

echo
echo "완료. GitLab ${BRANCH} · GitHub main 둘 다 최신이다."
