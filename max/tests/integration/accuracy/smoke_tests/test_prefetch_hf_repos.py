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

import json
from pathlib import Path

from pytest import MonkeyPatch
from smoke_tests import prefetch_hf_repos


def _snapshot(tmp_path: Path, *names: str) -> Path:
    snap = tmp_path / "snapshots" / "0123456789abcdef"
    snap.mkdir(parents=True)
    for name in names:
        (snap / name).write_bytes(b"x")
    return snap


def _hub_snapshot(tmp_path: Path, *names: str, manifest: object) -> Path:
    """A snapshot in the real cache layout, with a populator manifest beside it."""
    revision = "0123456789abcdef"
    dirname = "models--org--model"
    snap = tmp_path / "hub" / dirname / "snapshots" / revision
    snap.mkdir(parents=True)
    for name in names:
        (snap / name).write_bytes(b"x")
    manifest_file = tmp_path / "manifests" / dirname / f"{revision}.json"
    manifest_file.parent.mkdir(parents=True)
    manifest_file.write_text(json.dumps(manifest))
    return snap


def test_complete_single_file_snapshot_is_ok(tmp_path: Path) -> None:
    snap = _snapshot(tmp_path, "config.json", "model.safetensors")
    assert prefetch_hf_repos._snapshot_incomplete_reason(str(snap)) is None


def test_gguf_only_snapshot_is_ok(tmp_path: Path) -> None:
    snap = _snapshot(tmp_path, "config.json", "model-q4_k.gguf")
    assert prefetch_hf_repos._snapshot_incomplete_reason(str(snap)) is None


def test_config_only_snapshot_is_incomplete(tmp_path: Path) -> None:
    """An interrupted download leaves configs but no weight files."""
    snap = _snapshot(tmp_path, "config.json", "tokenizer_config.json")
    reason = prefetch_hf_repos._snapshot_incomplete_reason(str(snap))
    assert reason is not None
    assert "no weight files" in reason


def test_broken_symlink_is_incomplete(tmp_path: Path) -> None:
    """Cache cleaners can evict blobs, leaving dangling snapshot symlinks."""
    snap = _snapshot(tmp_path, "config.json")
    (snap / "model.safetensors").symlink_to(tmp_path / "blobs" / "gone")
    reason = prefetch_hf_repos._snapshot_incomplete_reason(str(snap))
    assert reason is not None
    assert "model.safetensors" in reason


def test_missing_shard_is_incomplete(tmp_path: Path) -> None:
    snap = _snapshot(
        tmp_path, "config.json", "model-00001-of-00002.safetensors"
    )
    index = {
        "weight_map": {
            "a.weight": "model-00001-of-00002.safetensors",
            "b.weight": "model-00002-of-00002.safetensors",
        }
    }
    (snap / "model.safetensors.index.json").write_text(json.dumps(index))
    reason = prefetch_hf_repos._snapshot_incomplete_reason(str(snap))
    assert reason is not None
    assert "model-00002-of-00002.safetensors" in reason


def test_complete_sharded_snapshot_is_ok(tmp_path: Path) -> None:
    snap = _snapshot(
        tmp_path,
        "config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    )
    index = {
        "weight_map": {
            "a.weight": "model-00001-of-00002.safetensors",
            "b.weight": "model-00002-of-00002.safetensors",
        }
    }
    (snap / "model.safetensors.index.json").write_text(json.dumps(index))
    assert prefetch_hf_repos._snapshot_incomplete_reason(str(snap)) is None


def test_manifest_match_is_ok(tmp_path: Path) -> None:
    """A snapshot matching its manifest is whole; the manifest here names no
    weight files, which the weight heuristic would reject, so it also proves
    the manifest wins over heuristics."""
    snap = _hub_snapshot(
        tmp_path,
        "config.json",
        manifest={"version": 1, "files": {"config.json": 1}},
    )
    assert prefetch_hf_repos._snapshot_incomplete_reason(str(snap)) is None


def test_manifest_missing_file_is_incomplete(tmp_path: Path) -> None:
    """Regression: weights cached but the tokenizer vocabulary missing passed
    every heuristic and died later inside AutoTokenizer.from_pretrained."""
    snap = _hub_snapshot(
        tmp_path,
        "config.json",
        "model.safetensors",
        "tokenizer_config.json",
        manifest={
            "version": 1,
            "files": {
                "config.json": 1,
                "model.safetensors": 1,
                "tokenizer_config.json": 1,
                "tokenizer.json": 9000,
            },
        },
    )
    reason = prefetch_hf_repos._snapshot_incomplete_reason(str(snap))
    assert reason is not None
    assert "tokenizer.json" in reason


def test_manifest_size_mismatch_is_incomplete(tmp_path: Path) -> None:
    snap = _hub_snapshot(
        tmp_path,
        "config.json",
        "model.safetensors",
        manifest={
            "version": 1,
            "files": {"config.json": 1, "model.safetensors": 5000},
        },
    )
    reason = prefetch_hf_repos._snapshot_incomplete_reason(str(snap))
    assert reason is not None
    assert "model.safetensors" in reason


def test_unknown_manifest_version_falls_back_to_heuristics(
    tmp_path: Path,
) -> None:
    snap = _hub_snapshot(
        tmp_path,
        "config.json",
        manifest={"version": 2, "files": {"config.json": 1}},
    )
    reason = prefetch_hf_repos._snapshot_incomplete_reason(str(snap))
    assert reason is not None
    assert "no weight files" in reason


def test_snapshot_without_manifest_uses_heuristics(tmp_path: Path) -> None:
    snap = _snapshot(tmp_path, "config.json", "model.safetensors")
    assert prefetch_hf_repos._manifest_files(snap) is None
    assert prefetch_hf_repos._snapshot_incomplete_reason(str(snap)) is None


def test_online_repair_downloads_the_pinned_revision(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """A repair fetches the revision the cache pins, not upstream main."""
    snap = _snapshot(tmp_path, "config.json")
    revisions = []

    def fake_snapshot_download(
        repo: str, revision: str | None = None, local_files_only: bool = False
    ) -> str:
        if not local_files_only:
            revisions.append(revision)
        return str(snap)

    monkeypatch.setattr(
        prefetch_hf_repos, "snapshot_download", fake_snapshot_download
    )
    monkeypatch.setattr(prefetch_hf_repos, "HF_HUB_OFFLINE", False)
    prefetch_hf_repos._ensure("org/model", allow_canonicalize=False)
    assert revisions == [snap.name]


def test_cache_path_treats_incomplete_snapshot_as_miss(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Regression: snapshot_download(local_files_only=True) returns a snapshot
    directory on mere existence, so a partial snapshot (configs cached, weights
    never downloaded) must not count as a cache hit or the online pass skips
    the weights download and the server later fails with "compatible weights
    cannot be found"."""
    snap = _snapshot(tmp_path, "config.json")
    monkeypatch.setattr(
        prefetch_hf_repos,
        "snapshot_download",
        lambda *args, **kwargs: str(snap),
    )
    assert prefetch_hf_repos._cache_path("org/model") is None


def test_cache_path_returns_complete_snapshot(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    snap = _snapshot(tmp_path, "config.json", "model.safetensors")
    monkeypatch.setattr(
        prefetch_hf_repos,
        "snapshot_download",
        lambda *args, **kwargs: str(snap),
    )
    assert prefetch_hf_repos._cache_path("org/model") == str(snap)
