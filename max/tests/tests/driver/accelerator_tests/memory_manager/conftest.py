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


from enum import Enum

import pytest
from max.driver import Accelerator, Buffer, Usage
from max.dtype import DType
from max.support import to_human_readable_bytes

KiB = 1024
MiB = 1024 * 1024
GiB = 1024 * 1024 * 1024


# The buffer cache is persistent in a process so it is recommended to run tests
# one at time in its own process. We split each test into its own file to isolate them.
# TODO: Figure out a way to consolidate all the tests into a single file.


@pytest.fixture
def memory_manager_config(monkeypatch: pytest.MonkeyPatch) -> None:
    # Max cache size is 100MiB
    max_cache_size = 100 * MiB
    # 2% of 100MiB is 2MiB per chunk
    chunk_percent = 2

    # These tests assert classic freelist-allocator semantics (exact chunk
    # sizing, the 100MiB cap, and the OOM message). The VMM allocator (default
    # on for NVIDIA) rounds the pool up to a 256MiB granule and satisfies
    # requests from a single arena, which changes those invariants, so pin it
    # off here to keep exercising the classic path.
    monkeypatch.setenv("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_VMM", "False")

    # Set config for device memory manager
    monkeypatch.setenv(
        "MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE", f"{max_cache_size}"
    )
    monkeypatch.setenv(
        "MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_CHUNK_PERCENT",
        f"{chunk_percent}",
    )

    # Set config for host memory manager
    monkeypatch.setenv(
        "MODULAR_DEVICE_CONTEXT_HOST_MEMORY_MANAGER_SIZE", f"{max_cache_size}"
    )
    monkeypatch.setenv(
        "MODULAR_DEVICE_CONTEXT_HOST_MEMORY_MANAGER_CHUNK_PERCENT",
        f"{chunk_percent}",
    )

    # This toggles asserts in the buffer cache
    monkeypatch.setenv(
        "MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SELF_CHECK", "True"
    )
    # This can be manually enabled for debugging
    monkeypatch.setenv("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_LOG", "False")


def align_down(size: int, alignment: int = 256 * KiB) -> int:
    multiples = size // alignment
    return multiples * alignment


class MemType(str, Enum):
    PINNED = "pinned"
    DEVICE = "device"

    def alloc(self, size: int) -> Buffer:
        print(
            f"Allocating {self.value} buffer of size {to_human_readable_bytes(size)}"
        )
        return Buffer(
            shape=(size,),
            dtype=DType.int8,
            device=Accelerator(),
            usage=Usage.STAGING if self == MemType.PINNED else Usage.DEFAULT,
        )
