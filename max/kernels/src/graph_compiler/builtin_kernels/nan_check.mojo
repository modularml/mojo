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
"""NaN/Inf detection kernels for the max-debug.nan-check feature.

These kernels are registered as custom ops in the `kernels` package and
inserted by the NanCheckPass compiler pass. The architecture is:

1. nan_check_count: read-only reduction that counts NaN/Inf values in a
   floating-point tensor. Outputs two single-element int32 tensors with
   the counts. CPU: vectorized scan with atomics. GPU: parallel reduction.

2. nan_check_raise: host-side kernel that reads the NaN/Inf counts
   (transferred to host via mo.transfer for GPU graphs) and raises a
   diagnostic error if any are non-zero.
"""

from max.algorithm import elementwise
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu.host.info import is_cpu
from std.memory import alloc, dealloc, unsafe_stack_allocation
from std.memory.alloc import Layout as AllocLayout
from std.atomic import Atomic
from std.sys import simd_width_of
from std.utils.numerics import isinf, isnan

from std.math import ceildiv

from std.utils.coord import Coord
from std.utils.index import IndexList
from extensibility import InputTensor, OutputTensor


@__name(t"nan_check_gpu_{dtype}")
def _nan_check_gpu_kernel[
    dtype: DType,
](
    src_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    total_elements: Int32,
    out_nan: UnsafePointer[Int32, MutAnyOrigin],
    out_inf: UnsafePointer[Int32, MutAnyOrigin],
):
    """GPU kernel: count NaN/Inf values via parallel reduction."""
    var nan_local = unsafe_stack_allocation[1, Int32, address_space=.SHARED]()
    var inf_local = unsafe_stack_allocation[1, Int32, address_space=.SHARED]()
    if thread_idx.x == 0:
        nan_local[0] = Int32(0)
        inf_local[0] = Int32(0)
    barrier()

    var _total_elements = Int(total_elements)
    var tid = block_idx.x * block_dim.x + thread_idx.x
    var my_nan = Int32(0)
    var my_inf = Int32(0)
    if tid < _total_elements:
        var val = src_ptr.load(tid)
        if isnan(val):
            my_nan = Int32(1)
        elif isinf(val):
            my_inf = Int32(1)
    if my_nan > 0:
        _ = Atomic.fetch_add(nan_local, my_nan)
    if my_inf > 0:
        _ = Atomic.fetch_add(inf_local, my_inf)

    barrier()

    if thread_idx.x == 0:
        if nan_local[0] > 0:
            _ = Atomic.fetch_add(out_nan, nan_local[0])
        if inf_local[0] > 0:
            _ = Atomic.fetch_add(out_inf, inf_local[0])


def nan_check_count[
    dtype: DType,
    rank: Int,
    target: StaticString,
](
    nan_count_out: OutputTensor[dtype=.int32, rank=1, ...],
    inf_count_out: OutputTensor[dtype=.int32, rank=1, ...],
    input: InputTensor[dtype=dtype, rank=rank, ...],
    ctx: DeviceContext,
) raises:
    """Counts NaN/Inf values in a floating-point tensor.

    Read-only: does not modify the input tensor. Outputs two single-element
    int32 tensors with the NaN and Inf counts. No D2H transfer or
    synchronization is performed: the counts stay on the same device as
    the input.
    """
    var total = input.size()

    comptime if is_cpu[target]():
        # CPU path: vectorized scan using elementwise with atomic accumulators.
        var nan_acc = alloc(AllocLayout[Int32](count=1)).into_managed()
        var inf_acc = alloc(AllocLayout[Int32](count=1)).into_managed()

        var nan_acc_ptr = nan_acc.unsafe_ptr()
        var inf_acc_ptr = inf_acc.unsafe_ptr()

        nan_acc_ptr[] = Int32(0)
        inf_acc_ptr[] = Int32(0)

        @always_inline
        def scan[width: Int, alignment: Int = 1](idx: Coord) {var}:
            var flat = idx[0].value()
            var ptr = input.unsafe_ptr()
            var val = ptr.load[width=width](flat)
            var nans = isnan(val).cast[.int32]().reduce_add()
            var infs = isinf(val).cast[.int32]().reduce_add()
            if nans > 0:
                _ = Atomic.fetch_add(nan_acc_ptr, nans)
            if infs > 0:
                _ = Atomic.fetch_add(inf_acc_ptr, infs)

        elementwise[simd_width_of[dtype]()](scan, Coord(total), ctx)

        nan_count_out.unsafe_ptr()[] = nan_acc_ptr[]
        inf_count_out.unsafe_ptr()[] = inf_acc_ptr[]
    else:
        # GPU path: parallel reduction writing directly to output tensors.
        # Zero the output counts first via a single-thread init kernel,
        # then run the reduction that atomically accumulates into them.
        var gpu_ctx = ctx
        var out_nan_ptr = nan_count_out.unsafe_ptr()
        var out_inf_ptr = inf_count_out.unsafe_ptr()

        @__parameter
        @__name(t"nan_check_zero_counts")
        def zero_counts(
            nan_ptr: UnsafePointer[Int32, MutAnyOrigin],
            inf_ptr: UnsafePointer[Int32, MutAnyOrigin],
        ):
            nan_ptr[] = Int32(0)
            inf_ptr[] = Int32(0)

        gpu_ctx.enqueue_function[zero_counts](
            out_nan_ptr, out_inf_ptr, grid_dim=1, block_dim=1
        )

        # `ceildiv(0, BLOCK) == 0`, and a zero-sized grid launch is rejected.
        if total == 0:
            return

        comptime BLOCK = 256
        var grid = ceildiv(total, BLOCK)

        comptime kernel = _nan_check_gpu_kernel[dtype]
        gpu_ctx.enqueue_function[kernel](
            rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
                input.unsafe_ptr()
            ),
            Int32(total),
            out_nan_ptr,
            out_inf_ptr,
            grid_dim=grid,
            block_dim=BLOCK,
        )


def nan_check_raise[
    label: StaticString,
    type_str: StaticString,
](
    nan_count: InputTensor[dtype=.int32, rank=1, ...],
    inf_count: InputTensor[dtype=.int32, rank=1, ...],
) raises:
    """Raises an error if NaN or Inf counts are non-zero.

    Reads two single-element int32 tensors on the host and raises a
    diagnostic error if either is > 0.
    """
    var nans = Int(nan_count.unsafe_ptr()[])
    var infs = Int(inf_count.unsafe_ptr()[])
    if nans > 0 or infs > 0:
        raise Error(
            "NaN/Inf detected in '"
            + String(label)
            + "' ("
            + String(type_str)
            + "): "
            + String(nans)
            + " NaN, "
            + String(infs)
            + " Inf"
        )
