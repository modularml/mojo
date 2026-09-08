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
"""Tests that abort() prints its message on GPU.

Launches multiple threads and blocks to verify the message is printed
exactly once (gated to thread 0, block 0) rather than once per thread.
"""

from max.gpu.host import DeviceContext
from std.os import abort


def abort_kernel():
    abort("gpu abort message")


# Verify message appears exactly once despite multiple threads/blocks,
# and includes block/thread IDs.
# CHECK-COUNT-1: ABORT:
# CHECK-SAME: block: [0,0,0] thread: [0,0,0]: gpu abort message
# CHECK-NOT: gpu abort message
def main() raises:
    with DeviceContext() as ctx:
        ctx.enqueue_function[abort_kernel](grid_dim=2, block_dim=32)
        ctx.synchronize()
