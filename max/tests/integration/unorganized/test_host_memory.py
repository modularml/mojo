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
"""Tests for host-memory introspection."""

from unittest.mock import mock_open, patch

import pytest
from max.pipelines.lib.host_memory import (
    _cgroup_memory_limit_paths,
    _host_memory_limit,
)


def test_host_memory_limit__reports_a_plausible_size() -> None:
    """The limit is discoverable on the platforms MAX serves from."""
    limit = _host_memory_limit()
    assert limit is not None
    assert limit > 0


@pytest.mark.parametrize(
    ("proc_self_cgroup", "expected"),
    [
        pytest.param(
            "0::/system.slice/max-serve.service\n",
            "/sys/fs/cgroup/system.slice/max-serve.service/memory.max",
            id="v2-systemd-unit",
        ),
        pytest.param(
            "4:memory:/docker/abc123\n",
            "/sys/fs/cgroup/memory/docker/abc123/memory.limit_in_bytes",
            id="v1-memory-controller",
        ),
    ],
)
def test_cgroup_paths__include_this_process_own_cgroup(
    proc_self_cgroup: str, expected: str
) -> None:
    """A unit-level limit lives below the mount, not at its root.

    Outside a container the cgroup mount is not namespaced, so reading only
    ``/sys/fs/cgroup/memory.max`` reports the root's limit and misses a
    ``MemoryMax=`` on the service -- overcommitting by exactly the cap.
    """
    with patch("builtins.open", mock_open(read_data=proc_self_cgroup)):
        paths = _cgroup_memory_limit_paths()

    assert expected in paths
    # The namespaced paths still come first, so a container is unaffected.
    assert paths[0] == "/sys/fs/cgroup/memory.max"


def test_cgroup_paths__root_cgroup_adds_nothing() -> None:
    """At the root there is nothing below the mount to look at."""
    with patch("builtins.open", mock_open(read_data="0::/\n")):
        assert _cgroup_memory_limit_paths() == [
            "/sys/fs/cgroup/memory.max",
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        ]


def test_cgroup_paths__unreadable_proc_falls_back() -> None:
    """No /proc (macOS, restricted sandboxes) must not raise."""
    with patch("builtins.open", side_effect=OSError):
        assert _cgroup_memory_limit_paths() == [
            "/sys/fs/cgroup/memory.max",
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        ]
