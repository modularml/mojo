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
"""Provides memory barrier utilities for preventing compiler optimizations.

This module includes the `clobber_memory()` function which acts as a memory
fence to prevent the compiler from reordering or eliminating memory operations.
This is essential for accurate benchmarking when memory access patterns need to
be preserved exactly as written.
"""

from std.atomic import Ordering, fence

# ===-----------------------------------------------------------------------===#
# clobber_memory
# ===-----------------------------------------------------------------------===#


@always_inline
def clobber_memory():
    """Forces all pending memory writes to be flushed to memory.

    This ensures that the compiler does not optimize away memory writes if it
    deems them to be not necessary. In effect, this operation acts as a barrier
    to memory reads and writes.
    """

    # This operation corresponds to  atomic_signal_fence(memory_order_acq_rel)
    # in C++.
    fence[Ordering.ACQUIRE_RELEASE, scope="singlethread"]()
