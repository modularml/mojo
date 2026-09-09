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
"""Generates tensors filled with values drawn from a normal (Gaussian) distribution for CPU and GPU."""

from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext
from std.math import ceildiv
from std.random import NormalRandom
from extensibility import _dot_prod

from std.utils import IndexList
from std.utils.coord import Coord, coord_to_index_list


def random_normal[
    dtype: DType,
    rank: Int,
    //,
    target: StaticString,
    OutputFn: ImplicitlyCopyable
    & RegisterPassable
    & def[width: SIMDLength, _rank: Int](
        idx: IndexList[_rank], val: SIMD[dtype, width]
    ),
](
    shape: IndexList[rank],
    mean: Float32,
    stddev: Float32,
    seed_ptr: UnsafePointer[UInt64, ImmutAnyOrigin],
    ctx: DeviceContext,
    output_fn: OutputFn,
) raises:
    """Call `output_fn` with values from a normal distribution, matching
    PyTorch CUDA's `torch.randn` element-to-counter mapping.

    For element `i`, mirrors PyTorch's per-thread Philox state:

        thread_id     = i mod GRID_BLOCK
        within_thread = i div GRID_BLOCK   (0..3)

    where `GRID_BLOCK = 256 * min(num_SMs * blocks_per_sm, ceil(numel/256))`.

    A single Philox step at counter `(0, 0, thread_id, 0)` produces 4 normals
    via :func:`std.random.NormalRandom.step_normal_4`; the within_thread
    index selects which lane to write to `output[i]`.

    Bit-exact for `numel <= 4 * GRID_BLOCK_max` (≈ 1.2M elements on B200).

    Parameters:
        dtype: The data type to generate.
        rank: The rank of the underlying buffer.
        target: The target to run on.
        OutputFn: The type of the function which stores the generated values.

    Args:
        shape: The shape of the output being stored into by output_fn.
        mean: The mean of the normal distribution.
        stddev: The standard deviation of the normal distribution.
        seed_ptr: Pointer to a single uint64 in device memory containing
            the Philox seed.
        ctx: The device context.
        output_fn: The function which stores the generated values.
    """

    if stddev <= 0:
        raise Error("stddev must be positive")

    var numel = shape.flattened_length()
    if numel == 0:
        return

    var strides = shape.get_row_major_strides()

    comptime BLOCK_SIZE: Int = 256

    # GRID_BLOCK mirrors PyTorch CUDA's calc_execution_policy. On GPU it
    # depends on the device's SM count (comptime via default_device_info).
    # On CPU we treat the whole tensor as one "thread group", which collapses
    # to within_thread = 0 for every element.
    var grid_block: Int

    comptime if target == "gpu":
        comptime info = DeviceContext.default_device_info
        comptime MAX_GRID = (
            info.sm_count * (info.threads_per_multiprocessor // BLOCK_SIZE)
        )
        var nblocks = ceildiv(numel, BLOCK_SIZE)
        var grid_x = min(nblocks, MAX_GRID)
        grid_block = grid_x * BLOCK_SIZE
    else:
        grid_block = numel

    @always_inline
    def generate[width: Int, alignment: Int = 1](idx: Coord) {var}:
        comptime assert (
            width == 1
        ), "PyTorch-compat normal kernel uses scalar lanes"
        var i = _dot_prod(
            rebind[type_of(strides)](coord_to_index_list(idx)), strides
        )
        var thread_id = UInt64(i % grid_block)
        var within_thread = i // grid_block

        var rng = NormalRandom(seed=seed_ptr[0], subsequence=thread_id)
        var four = rng.step_normal_4(mean=mean, stddev=stddev)
        var value = four[within_thread].cast[dtype]()
        output_fn[width=1](coord_to_index_list(idx), SIMD[dtype, 1](value))

    elementwise[simd_width=1, target=target](generate, Coord(shape), ctx)
