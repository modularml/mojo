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
"""Custom kernels that read uninitialized memory for testing.

Used by the uninit check end-to-end tests to verify that the debug
allocator poison + MOJO_STDLIB_SIMD_UNINIT_CHECK detection pipeline
works correctly.
"""

import extensibility

from max.gpu.host import DeviceContext
from extensibility import InputTensor, OutputTensor


@extensibility.register("read_uninit_output")
struct ReadUninitOutput:
    """Reads from the output tensor before writing to it.

    The execute method runs on CPU.  When the op is assigned to GPU, the
    output tensor is allocated in device memory, which the debug
    allocator fills with the largest-finite poison pattern.  Reading via
    unsafe_ptr().unsafe_load() triggers _check_not_poison on the CPU side.
    """

    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor,
        x: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        ctx: DeviceContext,
    ) raises:
        # Read from the output BEFORE writing — this is uninitialized.
        # On GPU, the debug allocator has poisoned this with the
        # largest-finite bit pattern.
        var uninit = output.unsafe_ptr().unsafe_load()

        # Write to prevent dead-code elimination.
        output[0] = x[0] + uninit
