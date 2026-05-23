---
name: release
description: Cut a BVID-FE release to PyPI. Use when the user asks to release, publish, cut a tag, ship a new version, or bump the version. Walks through the version bump, CHANGELOG roll, commit, and tag flow that triggers the OIDC publish workflow.
---

# Release a new BVID-FE version

BVID-FE publishes to PyPI through **OIDC Trusted Publishing** driven by
`.github/workflows/publish.yml`. There are no PyPI tokens in the repo —
all that matters is that a tag `vX.Y.Z` exists on `main` and the version
in `pyproject.toml` matches the tag.

The publish workflow runs in the GitHub Actions environment `release`,
which must exist on the repo (Settings → Environments). If it doesn't,
the workflow queues indefinitely as an intentional fail-safe.

## Pre-flight

Before doing anything else, confirm with the user:

1. The version number to release (`X.Y.Z`). Patch/minor/major decision
   is the user's call — look at `## [Unreleased]` in `CHANGELOG.md` and
   recommend, but don't decide unilaterally.
2. That `main` is in the state they want to ship (no surprise pending
   PRs they wanted in).

Then verify the workspace is clean:

```bash
git status
git branch --show-current   # must be main
git pull origin main
```

If we're on a non-`main` branch, stop and ask — release tags ship from
`main` only.

## Steps

### 1. Bump the version in `pyproject.toml`

Change exactly the `[project] version = "X.Y.Z"` line. Nothing else.
Do not bump dependency pins or touch other metadata in the same commit.

### 2. Roll the changelog

In `CHANGELOG.md`:

- Rename the existing `## [Unreleased]` heading to
  `## [X.Y.Z] - YYYY-MM-DD` (today's date, ISO format).
- Add a fresh empty `## [Unreleased]` block **above** the new release
  section, with empty `### Added`, `### Changed`, `### Fixed` stubs
  matching the existing house style — look at prior releases in the
  file for the exact subsection set to mirror.

If `## [Unreleased]` is empty, stop and ask the user — releases with no
notes are almost always a mistake.

### 3. Commit and push to main

```bash
git add pyproject.toml CHANGELOG.md
git commit -m "Release vX.Y.Z"
git push origin main
```

### 4. Tag and push the tag

The leading `v` is required — the publish workflow trigger is `v*`:

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

### 5. Confirm the publish workflow

The `Publish to PyPI` workflow now runs. It will:

- Verify the tag matches `[project].version` in `pyproject.toml`
- Build sdist + wheel
- Run `twine check`
- Upload via OIDC

Report the workflow URL to the user and stop there. Do **not**
auto-merge anything else or attempt to monitor the workflow with
polling — if the user wants live status, suggest
`subscribe_pr_activity` or watching it in the browser.

## Recovery if the tag check fails

The most common failure is a tag/version mismatch (forgot the
`pyproject.toml` bump, typo, etc.). Recover with:

```bash
git push --delete origin vX.Y.Z
git tag -d vX.Y.Z
# fix pyproject.toml, commit, push, then re-tag
```

Do NOT force-push to `main` to "fix" the release commit — make a new
fix commit and a new tag.

## Things this skill must never do

- Never `--force` push the tag or `main`.
- Never amend a release commit that has already been pushed.
- Never bypass `pre-commit` with `--no-verify`.
- Never invent a version number — always confirm with the user.
- Never tag from a branch other than `main`.
