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

"""Host-memory introspection for sizing host-side caches."""

from __future__ import annotations

import os

# The preprocessed-media caches hold host tensors in the API server process, so
# unlike everything device memory planning sizes they never touch the device.
# They still have to fit somewhere: the image and video budgets default to
# several GiB each, which is a fine ceiling on a serving host and enough to OOM
# a small container.
_PREPROCESS_CACHE_MAX_FRACTION_OF_HOST_MEMORY = 0.25


def _cgroup_memory_limit_paths() -> list[str]:
    """Returns candidate memory-limit files for *this process's* cgroup.

    In a container the cgroup mount is namespaced, so the well-known paths are
    already this process's own. On a host they are the *root* cgroup's, which
    says nothing about a unit-level limit -- a systemd service with
    ``MemoryMax=`` sits several levels down. Reading only the root there would
    report no limit and overcommit by exactly the amount the unit was capped to.

    ``/proc/self/cgroup`` names this process's cgroup relative to the mount, so
    it resolves both cases.
    """
    paths = [
        "/sys/fs/cgroup/memory.max",  # cgroup v2, namespaced
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # cgroup v1, namespaced
    ]
    try:
        with open("/proc/self/cgroup") as cgroup_file:
            entries = cgroup_file.readlines()
    except OSError:
        return paths

    for entry in entries:
        fields = entry.strip().split(":", 2)
        if len(fields) != 3:
            continue
        hierarchy, controllers, cgroup_path = fields
        relative = cgroup_path.lstrip("/")
        if not relative:
            # Already the root cgroup; the well-known paths cover it.
            continue
        if hierarchy == "0":  # the v2 unified hierarchy
            paths.append(f"/sys/fs/cgroup/{relative}/memory.max")
        elif "memory" in controllers.split(","):
            paths.append(
                f"/sys/fs/cgroup/memory/{relative}/memory.limit_in_bytes"
            )
    return paths


def _host_memory_limit() -> int | None:
    """Returns the host memory this process may use, or ``None`` if unknown.

    Prefers a cgroup limit over physical RAM. A container is typically granted a
    fraction of its host, and it is that grant the OOM killer enforces, so
    sizing a host cache off ``SC_PHYS_PAGES`` alone would overcommit by exactly
    the ratio between the two. Takes the smallest limit found, since a nested
    cgroup is bounded by every ancestor as well as by itself.
    """
    limits: list[int] = []

    for path in _cgroup_memory_limit_paths():
        try:
            with open(path) as limit_file:
                raw = limit_file.read().strip()
        except OSError:
            continue
        try:
            limit = int(raw)
        except ValueError:
            # cgroup v2 writes the literal "max" when unlimited.
            continue
        # cgroup v1 has no such keyword and reports a near-2**63 sentinel.
        if 0 < limit < (1 << 62):
            limits.append(limit)

    try:
        limits.append(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        # Not POSIX, or the platform does not publish these names.
        pass

    return min(limits) if limits else None
