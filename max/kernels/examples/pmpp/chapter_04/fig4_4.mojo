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

# Figure 4.4: Incorrect barrier usage example (from PMPP Chapter 4)
# This demonstrates a common mistake with conditional synchronization

from max.gpu import thread_idx
from max.gpu.sync import barrier

# ========================== KERNEL CODE ==========================


def incorrect_barrier(n: Int):
    """Demonstrates incorrect barrier usage with conditional synchronization.

    WARNING: This is an EXAMPLE OF INCORRECT CODE! Do not use this pattern.
    All threads in a block must reach the same barrier. Conditional barriers
    can cause deadlock.

    Args:
        n: Parameter (unused in this snippet).
    """
    if thread_idx.x % 2 == 0:
        # Even threads do some work...
        barrier()
    else:
        # Odd threads do different work...
        barrier()

    # This is problematic because if threads take different paths,
    # they may not all reach the same barrier, causing deadlock.
