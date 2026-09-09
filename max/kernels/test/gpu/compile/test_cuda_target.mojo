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

from std.io.io import _printf
from std.math import erf
from std.math.uutils import ufloordiv, udivmod
from std.sys.info import is_nvidia_gpu, simd_width_of

import max.gpu.primitives.warp as warp
from max.algorithm.functional import elementwise
from std.bit import log2_floor
from max.gpu import (
    WARP_SIZE,
    thread_idx,
    block_dim,
    block_idx,
    global_idx,
    lane_id,
    warp_id,
)
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext, get_gpu_target
from max.gpu.host.compile import _compile_code
from std.memory import unsafe_memset_zero, unsafe_stack_allocation
from std.testing import *

from std.utils.coord import Coord
from std.utils.index import IndexList

# ===-----------------------------------------------------------------------===#
# Check parameterization
# ===-----------------------------------------------------------------------===#


# COM: Checks if we can do parameterization on the is_nvidia_gpu check.
# COM: In this case the code that would run on CUDA would return 42 and the
# COM: one that does not would return -1.


def parameterized_on_cuda() -> Int:
    comptime if is_nvidia_gpu():
        return 42
    else:
        return -1


@always_inline
def _verify_parameterized_on_cuda(asm: StringSlice) raises -> None:
    assert_true("test_cuda_target_parameterized" in asm)

    # Now make sure that we have something like this:
    #     st.param.b64 	[func_retval0], 42;
    # PTX ISA 8.3+ spells the function-param state space explicitly, so newer
    # targets emit `st.param::func.b64` for the same store.
    var instruction_start_loc = asm.find("st.param.b64")
    if instruction_start_loc < 0:
        instruction_start_loc = asm.find("st.param::func.b64")
    assert_true(instruction_start_loc >= 0)  # Assert it's present
    var instruction_end_loc = asm.find(";", instruction_start_loc)
    assert_true(instruction_end_loc >= 0)
    var instruction_str = asm[byte=instruction_start_loc:instruction_end_loc]
    # Make sure 42 appears somewhere in the instruction
    assert_true("42" in instruction_str)


def test_parameterized_on_cuda_sm80() raises:
    var asm = _compile_code[
        parameterized_on_cuda, target=get_gpu_target["sm_80"]()
    ]().asm
    _verify_parameterized_on_cuda(asm)


def test_parameterized_on_cuda_sm90() raises:
    var asm = _compile_code[
        parameterized_on_cuda, target=get_gpu_target["sm_90"]()
    ]().asm
    _verify_parameterized_on_cuda(asm)


# ===-----------------------------------------------------------------------===#
# Check print
# ===-----------------------------------------------------------------------===#


def hello_mojo():
    _printf["Hello"]()


@always_inline
def _verify_hello(asm: StringSlice) raises -> None:
    assert_true("test_cuda_target_hello_mojo" in asm)
    assert_true("vprintf" in asm)


def test_hello_mojo_sm80() raises:
    var asm = _compile_code[hello_mojo, target=get_gpu_target["sm_80"]()]().asm
    _verify_hello(asm)


def test_hello_mojo_sm90() raises:
    var asm = _compile_code[hello_mojo, target=get_gpu_target["sm_90"]()]().asm
    _verify_hello(asm)


# ===-----------------------------------------------------------------------===#
# Check elementwise kernel
# ===-----------------------------------------------------------------------===#


def erf_elementwise(
    buf: MutPointer[Float32, MutAnyOrigin], len: Int, ctx: DeviceContext
) raises:
    # Each thread will process 4 * simd_width elements.
    comptime granularity = 4 * simd_width_of[DType.float32]()
    var tid = granularity * global_idx.x

    @always_inline
    def func[simd_width: Int, alignment: Int = 1](idx: Coord) {var}:
        var offset = tid + Int(idx[0].value())
        if offset >= len:
            return
        buf[offset] = erf(buf[offset])

    elementwise[simd_width=simd_width_of[DType.float32](), target="gpu"](
        func, Coord(granularity), ctx
    )


def _verify_erf_elementwise(asm: StringSlice) raises:
    assert_true("test_cuda_target_erf_elementwis" in asm)
    assert_true("tid.x" in asm)
    assert_true("ntid.x" in asm)
    assert_true("ctaid.x" in asm)


def test_erf_elementwise_sm80() raises:
    var asm = _compile_code[
        erf_elementwise, target=get_gpu_target["sm_80"]()
    ]().asm
    _verify_erf_elementwise(asm)


def test_erf_elementwise_sm90() raises:
    var asm = _compile_code[
        erf_elementwise, target=get_gpu_target["sm_90"]()
    ]().asm
    _verify_erf_elementwise(asm)


# ===-----------------------------------------------------------------------===#
# Check full kernel
# ===-----------------------------------------------------------------------===#


def erf_kernel(buf: MutPointer[Float32, MutAnyOrigin], len: Int):
    var tid = thread_idx.x + block_dim.y * block_idx.y

    if tid >= len:
        return

    buf[tid] = erf(buf[tid])


@always_inline
def _verify_erf_kernel(asm: StringSlice) raises -> None:
    assert_true("erf_kernel" in asm)
    assert_true("tid.x" in asm)
    assert_true("ntid.y" in asm)
    assert_true("ctaid.y" in asm)


def test_erf_kernel_sm80() raises:
    var asm = _compile_code[erf_kernel, target=get_gpu_target["sm_80"]()]().asm
    _verify_erf_kernel(asm)


def test_erf_kernel_sm90() raises:
    var asm = _compile_code[erf_kernel, target=get_gpu_target["sm_90"]()]().asm
    _verify_erf_kernel(asm)


# ===-----------------------------------------------------------------------===#
# Check stack allocation
# ===-----------------------------------------------------------------------===#


def test_shared_stack_allocation() -> (
    MutPointer[Int8, MutUntrackedOrigin, address_space=.SHARED]
):
    return unsafe_stack_allocation[999, DType.int8, 8, address_space=.SHARED]()


@always_inline
def _verify_shared_stack_allocation(asm: StringSlice) raises -> None:
    assert_true("test_cuda_target_test_shared_" in asm)
    assert_true(".shared .align 8 .b8" in asm)


def test_shared_stack_allocation_sm80() raises:
    var asm = _compile_code[
        test_shared_stack_allocation, target=get_gpu_target["sm_80"]()
    ]().asm
    _verify_shared_stack_allocation(asm)


def test_shared_stack_allocation_sm90() raises:
    var asm = _compile_code[
        test_shared_stack_allocation, target=get_gpu_target["sm_90"]()
    ]().asm
    _verify_shared_stack_allocation(asm)


# ===-----------------------------------------------------------------------===#
# Check barrier
# ===-----------------------------------------------------------------------===#


def test_barrier():
    barrier()


@always_inline
def _verify_barrier(asm: StringSlice) raises -> None:
    assert_true("barrier" in asm)
    assert_true("bar.sync 	0" in asm)


def test_barrier_sm80() raises:
    var asm = _compile_code[
        test_barrier, target=get_gpu_target["sm_80"]()
    ]().asm
    _verify_barrier(asm)


def test_barrier_sm90() raises:
    var asm = _compile_code[
        test_barrier, target=get_gpu_target["sm_90"]()
    ]().asm
    _verify_barrier(asm)


# ===-----------------------------------------------------------------------===#
# SGEMM with register coarsening
# ===-----------------------------------------------------------------------===#


def gemm(
    c: MutPointer[Float32, MutAnyOrigin],
    a: ImmPointer[Float32, ImmutAnyOrigin],
    b: ImmPointer[Float32, ImmutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
):
    # Compute C = A x B
    #   where A is a (m x k) matrix
    #   where B is a (k x n) matrix
    #   where C is a (m x n) matrix
    #
    # Use register and shared memory tiling and thread coarsening
    #
    # NOTE: A and C are column major, B is row major.

    comptime TILE_SZ_A = 128
    comptime TILE_SZ_B = 16
    comptime TILE_SZ_RATIO = TILE_SZ_A // TILE_SZ_B

    # Utilities for accessing flattened matrices.
    @always_inline
    @__parameter
    def get_a(row: Int, col: Int) -> Float32:
        return a.load(row + m * col)

    @always_inline
    @__parameter
    def get_b(row: Int, col: Int) -> Float32:
        return b.load(row * n + col)

    @always_inline
    @__parameter
    def set_c(row: Int, col: Int, val: Float32):
        c[row + col * m] = val

    # Allocate B array into shared memory for tiling.
    var b_shared = unsafe_stack_allocation[
        TILE_SZ_RATIO * TILE_SZ_B,
        DType.float32,
        address_space=.SHARED,
    ]()

    # Thread indexing offsets.
    var row = global_idx.x
    var col = block_idx.y * TILE_SZ_B

    # Privatization of the C matrix.
    var c_reg = unsafe_stack_allocation[TILE_SZ_B, DType.float32]()

    unsafe_memset_zero(c_reg, TILE_SZ_B)

    # Loop over each input tile.
    for tile_idx in range((k - 1) // TILE_SZ_RATIO + 1):
        var i, j = udivmod(thread_idx.x, TILE_SZ_B)

        # Load the B matrix into shared memory.
        var b_val: Float32
        var r = tile_idx * TILE_SZ_RATIO + i
        if r < k and col + j < n:
            b_val = get_b(r, col + j)
        else:
            b_val = 0
        b_shared[i * TILE_SZ_B + j] = b_val

        barrier()

        # Loop within the tile.
        comptime for idx in range(TILE_SZ_RATIO):
            # Load the A tile into the register.
            var a_reg: Float32
            if row < m and tile_idx * TILE_SZ_RATIO + idx < k:
                a_reg = get_a(row, tile_idx * TILE_SZ_RATIO + idx)
            else:
                a_reg = 0

            # Compute the output element for each thread.
            for out_idx in range(TILE_SZ_B):
                c_reg[out_idx] += (
                    a_reg * b_shared[idx * TILE_SZ_RATIO + out_idx]
                )
        barrier()

    # Store the values into the output matrix.
    for out_idx in range(TILE_SZ_B):
        if row < m and col + out_idx < n:
            set_c(row, col + out_idx, c_reg.load(out_idx))


def _verify_gemm(asm: StringSlice) raises:
    assert_true("gemm" in asm)
    assert_true(".shared .align 4 .b8" in asm)
    assert_true("st.shared.b32" in asm)
    assert_true("ld.shared.b32" in asm)


def test_gemm_sm80() raises:
    var asm = _compile_code[gemm, target=get_gpu_target["sm_80"]()]().asm
    _verify_gemm(asm)


def test_gemm_sm90() raises:
    var asm = _compile_code[gemm, target=get_gpu_target["sm_90"]()]().asm
    _verify_gemm(asm)


# ===-----------------------------------------------------------------------===#
# shuffle ops
# ===-----------------------------------------------------------------------===#


def test_warp_shuffle_up(val: Float32) -> Float32:
    var res = val

    comptime limit = log2_floor(WARP_SIZE)

    comptime for mask in reversed(range(limit)):
        res += warp.shuffle_up(res, UInt32(1 << mask))
    return res


@always_inline
def _verify_warp_shuffle_up(asm: StringSlice) raises -> None:
    assert_true("test_warp_shuf" in asm)
    assert_true("shfl.sync.up.b32" in asm)


def test_warp_shuffle_up_sm80() raises:
    var asm = _compile_code[
        test_warp_shuffle_up, target=get_gpu_target["sm_80"]()
    ]().asm
    _verify_warp_shuffle_up(asm)


def test_warp_shuffle_up_sm90() raises:
    var asm = _compile_code[
        test_warp_shuffle_up, target=get_gpu_target["sm_90"]()
    ]().asm
    _verify_warp_shuffle_up(asm)


def test_warp_shuffle_down(val: Int32) -> Int32:
    var res = val

    comptime limit = log2_floor(WARP_SIZE)

    comptime for mask in reversed(range(limit)):
        res += warp.shuffle_down(res, UInt32(1 << mask))
    return res


@always_inline
def _verify_warp_shuffle_down(asm: StringSlice) raises -> None:
    assert_true("test_warp_shuf" in asm)
    assert_true("shfl.sync.down.b32" in asm)


def test_warp_shuffle_down_sm80() raises:
    var asm = _compile_code[
        test_warp_shuffle_down, target=get_gpu_target["sm_80"]()
    ]().asm
    _verify_warp_shuffle_down(asm)


def test_warp_shuffle_down_sm90() raises:
    var asm = _compile_code[
        test_warp_shuffle_down, target=get_gpu_target["sm_90"]()
    ]().asm
    _verify_warp_shuffle_down(asm)


# ===-----------------------------------------------------------------------===#
# Reduction
# ===-----------------------------------------------------------------------===#


def warp_sum_reduce(val: Float32) -> Float32:
    var res = val

    comptime limit = log2_floor(WARP_SIZE)

    comptime for mask in reversed(range(limit)):
        res += warp.shuffle_xor(res, UInt32(1 << mask))
    return res


@always_inline
def _verify_warp_sum_reduce(asm: StringSlice) raises -> None:
    assert_true("warp_sum_" in asm)
    assert_true("shfl.sync.bfly.b32" in asm)


def test_warp_sum_reduce_sm80() raises:
    var asm = _compile_code[
        warp_sum_reduce, target=get_gpu_target["sm_80"]()
    ]().asm
    _verify_warp_sum_reduce(asm)


def test_warp_sum_reduce_sm90() raises:
    var asm = _compile_code[
        warp_sum_reduce, target=get_gpu_target["sm_90"]()
    ]().asm
    _verify_warp_sum_reduce(asm)


def block_reduce(val: Float32) -> Float32:
    var shared = unsafe_stack_allocation[
        WARP_SIZE, DType.float32, address_space=.SHARED
    ]()

    comptime warp_shift = log2_floor(WARP_SIZE)

    var lane = lane_id()
    var warp = warp_id()

    var warp_sum = warp_sum_reduce(val)

    if lane == 0:
        shared[warp] = warp_sum

    barrier()

    return warp_sum_reduce(
        shared.load(lane) if thread_idx.x
        < ufloordiv(block_dim.x, WARP_SIZE) else 0
    )


@always_inline
def _verify_block_reduce(asm: StringSlice) raises -> None:
    assert_true("block_reduce" in asm)
    assert_true("mov.u32" in asm)


def test_block_reduce_sm80() raises:
    var asm = _compile_code[
        block_reduce, target=get_gpu_target["sm_80"]()
    ]().asm
    _verify_block_reduce(asm)


def test_block_reduce_sm90() raises:
    var asm = _compile_code[
        block_reduce, target=get_gpu_target["sm_90"]()
    ]().asm
    _verify_block_reduce(asm)


def main() raises:
    test_parameterized_on_cuda_sm80()
    test_parameterized_on_cuda_sm90()
    test_hello_mojo_sm80()
    test_hello_mojo_sm90()
    #    FIXME: MSTDL-1794
    #    Compilation for GPU calls to another compilation for GPU
    #    test_erf_elementwise_sm80()
    #    test_erf_elementwise_sm90()
    test_erf_kernel_sm80()
    test_erf_kernel_sm90()
    test_shared_stack_allocation_sm80()
    test_shared_stack_allocation_sm90()
    test_barrier_sm80()
    test_barrier_sm90()
    test_gemm_sm80()
    test_gemm_sm90()
    test_warp_shuffle_up_sm80()
    test_warp_shuffle_up_sm90()
    test_warp_shuffle_down_sm80()
    test_warp_shuffle_down_sm90()
    test_warp_sum_reduce_sm80()
    test_warp_sum_reduce_sm90()
    test_block_reduce_sm80()
    test_block_reduce_sm90()
