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
# DOC: max/gpu/intro-tutorial.mdx

from std.sys import has_accelerator

from max.gpu.host import DeviceContext
from max.gpu import block_idx, thread_idx


def print_threads():
    """Print thread IDs."""

    print(
        block_idx.x,
        block_idx.y,
        block_idx.z,
        thread_idx.x,
        thread_idx.y,
        thread_idx.z,
        sep="\t",
    )


def main() raises:
    comptime if not has_accelerator():
        print("No compatible GPU found")
    else:
        var ctx = DeviceContext()
        print("block_idx\t\tthread_idx")
        print("x\ty\tz", "x\ty\tz", sep="\t")
        print("-" * 20, "-" * 20, sep="\t")
        ctx.enqueue_function[print_threads](
            grid_dim=(2, 2, 1), block_dim=(16, 4, 2)
        )
        ctx.synchronize()
        print("Program finished")
