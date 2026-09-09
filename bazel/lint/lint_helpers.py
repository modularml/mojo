# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

# Provides utilities for setting up linters

import os
import subprocess


def is_check() -> bool:
    # Default to format
    return os.getenv("CHECK", "0").lower() in {"1", "true"}


def is_fast() -> bool:
    # Default to fast
    return os.getenv("FAST", "1").lower() in {"1", "true"}


def _has_jj(cwd: str = ".") -> bool:
    # Prefer jj when available: in a non-colocated jj workspace `git ls-files` /
    # `git diff` can't see the working copy. `jj root` only succeeds inside a jj
    # repo, so this gracefully falls through to git in plain git checkouts.
    # `cwd` lets us probe submodules separately, which are their own colocated
    # jj clones in a jj workspace.
    return (
        subprocess.run(
            ["bash", "-c", "command -v jj"], capture_output=True
        ).returncode
        == 0
        and subprocess.run(
            ["jj", "root"], cwd=cwd, capture_output=True
        ).returncode
        == 0
    )


def _get_files(stdout: bytes) -> set[str]:
    return set(stdout.decode().splitlines())


def _submodule_paths() -> list[str]:
    # Submodule paths relative to the superproject root. `git ls-files` /
    # `git diff` treat a submodule as a single gitlink and don't descend into
    # it, so we run git inside each submodule separately (see below).
    result = subprocess.run(
        ["git", "submodule", "--quiet", "foreach", "echo $sm_path"],
        capture_output=True,
    )
    if result.returncode != 0:
        return []
    return result.stdout.decode().splitlines()


def _prefixed(prefix: str, files: set[str]) -> set[str]:
    return {f"{prefix}/{f}" for f in files}


def _git_all_files(cwd: str = ".") -> set[str]:
    # `core.quotepath=off` keeps non-ASCII paths raw (see `_git_changed_files`).
    tracked = subprocess.check_output(
        ["git", "-c", "core.quotepath=off", "ls-files"], cwd=cwd
    )
    deleted = subprocess.check_output(
        ["git", "-c", "core.quotepath=off", "ls-files", "--deleted"], cwd=cwd
    )
    return _get_files(tracked) - _get_files(deleted)


def _jj_all_files(cwd: str = ".") -> set[str]:
    return _get_files(subprocess.check_output(["jj", "file", "list"], cwd=cwd))


def _git_changed_files(diff_target: str, cwd: str = ".") -> set[str]:
    # `core.quotepath=off` keeps non-ASCII paths (e.g. an em-dash in a
    # filename) as raw UTF-8 instead of C-quoting them into `"...\342\200\224..."`,
    # so a linter given this list finds the file instead of erroring on a
    # quoted path that doesn't exist. Mirrors `lint_file_paths.py`.
    result = subprocess.check_output(
        [
            "git",
            "-c",
            "core.quotepath=off",
            "diff",
            "--diff-filter=d",
            "--name-only",
            diff_target,
        ],
        cwd=cwd,
    )
    return _get_files(result)


def _jj_changed_files(diff_target: str, cwd: str = ".") -> set[str]:
    result = subprocess.check_output(
        ["jj", "diff", "--name-only", "--from", diff_target, "--to", "@"],
        cwd=cwd,
    )
    return _get_files(result)


def _submodule_diff_target(sm_path: str, super_target: str) -> str | None:
    # The submodule commit recorded in the superproject at `super_target`; this
    # is the right base to diff a submodule's working tree against. Returns None
    # if the submodule isn't a gitlink there or hasn't fetched that commit.
    rev = subprocess.run(
        ["git", "rev-parse", f"{super_target}:{sm_path}"], capture_output=True
    )
    if rev.returncode != 0:
        return None
    target = rev.stdout.decode().strip()
    has_commit = subprocess.run(
        ["git", "cat-file", "-e", f"{target}^{{commit}}"],
        cwd=sm_path,
        capture_output=True,
    )
    return target if has_commit.returncode == 0 else None


def _oss_filter(files: set[str]) -> set[str]:
    # Only used for testing with the OSS overlay
    if not os.getenv("USE_OSS_FILTER"):
        return files

    return {
        f.removeprefix("oss/modular/")
        for f in files
        if f.startswith("oss/modular/")
    }


def _all_files(cwd: str = ".") -> set[str]:
    return _jj_all_files(cwd) if _has_jj(cwd) else _git_all_files(cwd)


def get_all_files() -> set[str]:
    # `jj file list` / `git ls-files` treat a submodule as one opaque gitlink
    # and don't descend into it, so recurse and union each submodule's files
    # (each is its own jj or git checkout).
    files = _all_files()
    for sm_path in _submodule_paths():
        files |= _prefixed(sm_path, _all_files(sm_path))
    return _oss_filter(files)


def _diff_target() -> str | None:
    # Commit the superproject is diffed against: the merge base of the working
    # copy and main, as a git commit id. Resolving it concretely (rather than as
    # a jj revset) lets `_submodule_diff_target` reuse it to read each
    # submodule's pinned gitlink, so submodules diff from the same base on both
    # the jj and git paths.
    #
    # If determining the merge base fails, returns None, which should indicate
    # falling back to a full lint.
    if _has_jj():
        return (
            subprocess.check_output(
                [
                    "jj",
                    "log",
                    "--no-graph",
                    "-r",
                    "heads(ancestors(@) & ancestors(main@origin))",
                    "-T",
                    'commit_id ++ "\n"',
                ]
            )
            .decode()
            .splitlines()[0]
        )
    if lint_diff_target := os.getenv("LINT_DIFF_TARGET"):
        return lint_diff_target
    result = subprocess.run(
        ["git", "merge-base", "origin/main", "HEAD"],
        check=False,
        capture_output=True,
    )
    if result.returncode == 0:
        return result.stdout.decode().rstrip("\n")
    else:
        return None


def _changed_files(diff_target: str, cwd: str = ".") -> set[str]:
    if _has_jj(cwd):
        return _jj_changed_files(diff_target, cwd=cwd)
    return _git_changed_files(diff_target, cwd=cwd)


def get_changed_files() -> set[str]:
    diff_target = _diff_target()
    if not diff_target:
        return get_all_files()
    files = _changed_files(diff_target)
    for sm_path in _submodule_paths():
        # Diff each submodule from the commit the superproject pins at the merge
        # base (its gitlink there) to the submodule working copy. This matches
        # how CI lints a submodule bump and keeps the jj and git paths in sync.
        sm_target = _submodule_diff_target(sm_path, diff_target)
        if sm_target is None:
            continue
        files |= _prefixed(sm_path, _changed_files(sm_target, cwd=sm_path))
    return _oss_filter(files)
