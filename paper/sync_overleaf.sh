#!/bin/bash
# Sync paper/ with an Overleaf project over Overleaf's git bridge.
#
# The cluster's git has no `git subtree`, so this uses built-in plumbing:
#   pull: fetch Overleaf, merge its tree into paper/ (subtree merge strategy)
#   push: commit paper/'s tree on top of Overleaf's head, push it as master
#
# Usage (from anywhere in the repo):
#   paper/sync_overleaf.sh init    # first sync with a fresh Overleaf project (keeps our files)
#   paper/sync_overleaf.sh pull    # bring Overleaf edits into paper/ (creates a merge commit)
#   paper/sync_overleaf.sh push    # send committed paper/ to Overleaf
# Remote: `git remote add overleaf https://git.overleaf.com/<project-id>` (see README.md).

set -euo pipefail

REMOTE="${OVERLEAF_REMOTE:-overleaf}"
BRANCH="${OVERLEAF_BRANCH:-main}"   # GitHub-synced Overleaf uses `main`; git bridge uses `master`
PREFIX=paper

cd "$(git rev-parse --show-toplevel)"

fetch() { git fetch "${REMOTE}" "${BRANCH}"; }

case "${1:-}" in
    init)
        fetch
        # Unrelated histories; on conflicts (e.g. Overleaf's default main.tex) keep ours.
        git merge --allow-unrelated-histories -X subtree="${PREFIX}" -X ours \
            -m "chore(paper): merge Overleaf project into ${PREFIX}/" FETCH_HEAD
        ;;
    pull)
        fetch
        git merge -X subtree="${PREFIX}" -m "chore(paper): pull from Overleaf" FETCH_HEAD
        ;;
    push)
        fetch
        remote_head="$(git rev-parse FETCH_HEAD)"
        # Refuse to overwrite Overleaf edits we have not merged yet.
        if ! git merge-base --is-ancestor "${remote_head}" HEAD; then
            echo "Overleaf has changes not merged here: run '$0 pull' first." >&2
            exit 1
        fi
        if ! git diff --quiet HEAD -- "${PREFIX}" || ! git diff --cached --quiet -- "${PREFIX}"; then
            echo "Uncommitted changes in ${PREFIX}/ are not pushed; commit them first." >&2
        fi
        tree="$(git rev-parse "HEAD:${PREFIX}")"
        if [[ "${tree}" == "$(git rev-parse "${remote_head}^{tree}")" ]]; then
            echo "Overleaf is up to date."
            exit 0
        fi
        commit="$(git commit-tree "${tree}" -p "${remote_head}" \
            -m "Sync from $(git rev-parse --short HEAD): $(git log -1 --format=%s)")"
        git push "${REMOTE}" "${commit}:refs/heads/${BRANCH}"
        # Record the pushed commit in our history (no content change) so the next
        # push sees Overleaf's head as already merged.
        git merge -X subtree="${PREFIX}" -m "chore(paper): record push to Overleaf" "${commit}"
        ;;
    *)
        echo "usage: $0 {init|pull|push}" >&2
        exit 2
        ;;
esac
