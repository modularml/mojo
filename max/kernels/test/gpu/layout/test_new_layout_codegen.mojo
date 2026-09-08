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
"""Test for MixedLayout GPU memory codegen with runtime indices."""

import std.sys

from max.gpu import thread_idx
from max.gpu.host.compile import _compile_code, get_gpu_target
from layout.tile_layout import Layout
from layout import Coord, Idx
from std.testing import assert_true


def test_codegen_memory[
    func_type: TrivialRegisterPassable, //, func: func_type
]() raises:
    """Generic function to test codegen memory patterns for any kernel function.

    Tests that the given kernel function compiles for both AMD and NVIDIA GPUs
    without using local/shared memory for compile-time known values.

    Parameters:
        func_type: The type of the kernel function to test (inferred).
        func: The kernel function to test.
    """

    # Test AMD GPU codegen
    var amd_asm = _compile_code[
        func, target=get_gpu_target["amdgpu:gfx942"]()
    ]().asm

    # Should not load from buffer for compile-time known values
    assert_true("ds_read" not in amd_asm and "ds_write" not in amd_asm)

    # Test NVIDIA GPU codegen
    var nvidia_asm = _compile_code[func, target=get_gpu_target["sm_80"]()]().asm

    # Should not use local memory for compile-time known values
    assert_true("ld.local" not in nvidia_asm and "st.local" not in nvidia_asm)


def kernel_mixed_dimensions(x: Int, ptr: MutPointer[Int32, MutAnyOrigin]):
    # Create layout with mixed compile-time and runtime dimensions
    var layout = Layout(shape=(Idx[8], x), stride=(x, Idx[1]))
    ptr[0] = Int32(layout(Coord(Idx[0], x - 1)))


def kernel_thread_idx(ptr: MutPointer[Int32, MutAnyOrigin]):
    comptime layout = Layout(shape=(Idx[8], Idx[2]), stride=(Idx[1], Idx[1]))
    ptr[0] = Int32(layout(Coord(thread_idx.x, thread_idx.y)))


def main() raises:
    test_codegen_memory[kernel_mixed_dimensions]()
    test_codegen_memory[kernel_thread_idx]()
