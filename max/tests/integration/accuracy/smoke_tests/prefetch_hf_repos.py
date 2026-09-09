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

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "click>=8,<9",
#   "huggingface_hub==1.8.0",
#   "pydantic>=2.0,<3",
#   "pyyaml",
#   "requests",
# ]
# ///

"""Make sure every HuggingFace repo a smoke test needs is already cached.

Two modes, switched by the HF_HUB_OFFLINE env var:

* Offline: probe the cache; exit non-zero with a one-line cache-miss message
  if anything is missing.
* Online: download anything missing, canonicalize the base repo's casing if
  needed.

Stdout carries one thing only: the resolved base repo name, emitted last and
only on full success. Progress lines go to stderr.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.constants import HF_HUB_OFFLINE
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
from smoke_test import hf_repos_for_model

for _noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


def _log(msg: str) -> None:
    click.echo(msg, err=True)


def _manifest_files(snapshot: Path) -> dict[str, int] | None:
    """The file map (path -> size) the cache populator recorded, or None.

    The populator writes manifests/<repo-dirname>/<revision>.json beside the
    hub/ tree, listing the repo's upstream files. None means no usable
    manifest: an unmanifested repo, a non-hub layout, or an unknown schema.
    """
    if len(snapshot.parents) < 4 or snapshot.parents[2].name != "hub":
        return None
    manifest = (
        snapshot.parents[3]
        / "manifests"
        / snapshot.parents[1].name
        / f"{snapshot.name}.json"
    )
    try:
        doc = json.loads(manifest.read_bytes())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(doc, dict) or doc.get("version") != 1:
        return None
    files = doc.get("files")
    return files if isinstance(files, dict) else None


def _snapshot_incomplete_reason(path: str) -> str | None:
    """Return why the snapshot can't serve the model, or None if it looks whole.

    A cached snapshot directory can resolve while missing content:
    snapshot_download(local_files_only=True) only checks that the directory
    exists. With a populator manifest, compare against the repo's exact file
    list. Otherwise fall back to heuristics: every cached entry readable, at
    least one weight file, and every shard named by a safetensors index.
    """
    root = Path(path)
    if (files := _manifest_files(root)) is not None:
        for name in sorted(files):
            try:
                actual = (root / name).stat().st_size
            except OSError:
                return f"{name} is in the populator manifest but not cached"
            if actual != files[name]:
                return (
                    f"{name} is {actual} bytes, the populator manifest "
                    f"says {files[name]}"
                )
        return None
    weight_files: list[Path] = []
    index_files: list[Path] = []
    for entry in root.rglob("*"):
        if entry.is_symlink() and not entry.exists():
            return f"broken symlink (evicted blob?): {entry.relative_to(root)}"
        if not entry.is_file():
            continue
        if entry.name.endswith((".safetensors", ".gguf")):
            weight_files.append(entry)
        elif entry.name.endswith(".safetensors.index.json"):
            index_files.append(entry)
    if not weight_files:
        return "no weight files (*.safetensors / *.gguf) in the snapshot"
    for index in index_files:
        try:
            weight_map = json.loads(index.read_text()).get("weight_map", {})
        except (OSError, json.JSONDecodeError):
            return f"unreadable safetensors index: {index.relative_to(root)}"
        shards = {str(shard) for shard in weight_map.values()}
        missing = sorted(
            shard for shard in shards if not (index.parent / shard).is_file()
        )
        if missing:
            return (
                f"{index.relative_to(root)} names {len(missing)} missing "
                f"shard(s), e.g. {missing[0]}"
            )
    return None


def _pinned_revision(repo: str) -> str | None:
    """The revision the local cache pins for the repo, if it has one."""
    try:
        return Path(snapshot_download(repo, local_files_only=True)).name
    except LocalEntryNotFoundError:
        return None


def _cache_path(repo: str) -> str | None:
    """Return the snapshot path if fully cached locally, else None.

    Resolves the default revision, which the cache populator points at the
    revision hf-repo-lock.tsv pins, so offline this is the locked snapshot.
    Uses local_files_only=True so this never opens a socket.
    """
    try:
        path = snapshot_download(repo, local_files_only=True)
    except LocalEntryNotFoundError:
        return None
    if reason := _snapshot_incomplete_reason(path):
        _log(f"  Cached snapshot at {path} is incomplete: {reason}.")
        return None
    return path


def _ensure(repo: str, *, allow_canonicalize: bool) -> str:
    """Cache the repo, returning the resolved name.

    In offline mode a probe miss exits non-zero. Online, the base repo gets
    a canonical-name fallback since users type any casing.
    """
    _log(f"Checking the cache for '{repo}'...")
    path = _cache_path(repo)
    if path is not None:
        _log(f"  Already cached at {path}")
        return repo

    _log("  Not fully cached locally.")

    if HF_HUB_OFFLINE:
        _log(
            "  Offline mode is set, so we can't fetch from Hugging Face. "
            "Re-run with network access to download."
        )
        sys.exit(1)

    resolved = repo
    if allow_canonicalize:
        _log(
            "  Looking up the canonical name on Hugging Face (the cache "
            "is case-sensitive, so the casing must match exactly)..."
        )
        try:
            resolved = HfApi().model_info(repo).id
        except HfHubHTTPError as e:
            _log(
                f"  Failed to resolve canonical Hugging Face repo ID for "
                f"'{repo}': {e}"
            )
            sys.exit(1)
        if resolved != repo:
            _log(f"  Resolved '{repo}' to '{resolved}'.")
            path = _cache_path(resolved)
            if path is not None:
                _log(f"  Already cached under the canonical name at {path}")
                return resolved
        else:
            _log("  Canonical name matches the input.")

    _log(f"  Downloading '{resolved}' from Hugging Face...")
    path = snapshot_download(resolved, revision=_pinned_revision(resolved))
    _log(f"  Cached '{resolved}' to {path}")
    return resolved


@click.command()
@click.argument("model", type=str)
def main(model: str) -> None:
    repos = hf_repos_for_model(model)
    if not repos:
        _log(f"Nothing to pre-fetch for '{model}'.")
        return
    base_repo, *extras = repos

    _log(f"Base model: '{base_repo}'")
    resolved_base = _ensure(base_repo, allow_canonicalize=True)

    for repo in extras:
        _log(f"Also needed: '{repo}'")
        _ensure(repo, allow_canonicalize=False)

    # Stdout is the resolved base name; emit last so a partial run leaves it
    # empty for the caller.
    click.echo(resolved_base)


if __name__ == "__main__":
    main()
