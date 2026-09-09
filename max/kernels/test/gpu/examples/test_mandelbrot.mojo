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

from std.math import ceildiv, iota
from std.sys.info import simd_width_of

from std.algorithm import vectorize
from std.complex import ComplexSIMD
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.testing import assert_equal

from std.sys import has_apple_gpu_accelerator
from layout import TileTensor, Idx, row_major

comptime float_type = DType.float32 if has_apple_gpu_accelerator() else DType.float64
comptime int_type = DType.int


comptime width = 4096
comptime height = 4096
comptime MAX_ITERS = 1000
comptime BLOCK_SIZE = 32

comptime min_x = -2.0
comptime max_x = 0.47
comptime min_y = -1.12
comptime max_y = 1.12


@always_inline
def mandelbrot_kernel[
    simd_width: SIMDLength
](c: ComplexSIMD[float_type, simd_width]) -> SIMD[int_type, simd_width]:
    """A vectorized implementation of the inner mandelbrot computation."""
    var z = ComplexSIMD[float_type, simd_width](0, 0)
    var iters = SIMD[int_type, simd_width](0)

    var in_set_mask = SIMD[.bool, simd_width](fill=True)
    for _ in range(MAX_ITERS):
        if not in_set_mask.reduce_or():
            break
        in_set_mask = z.squared_norm().le(4)
        iters = in_set_mask.select(iters + 1, iters)
        z = z.squared_add(c)

    return iters


def mandelbrot(out_ptr: MutPointer[Scalar[int_type], MutAnyOrigin]):
    # Each task gets a row.
    var row = global_idx.x
    if row >= height:
        return

    var out = TileTensor(out_ptr, row_major(height, width))

    comptime scale_x = (max_x - min_x) / width
    comptime scale_y = (max_y - min_y) / height

    @always_inline
    def compute_vector[simd_width: Int](col: Int) {mut}:
        """Each time we operate on a `simd_width` vector of pixels."""
        if col >= width:
            return
        var cx = (
            min_x
            + (Scalar[float_type](col) + iota[float_type, simd_width]())
            * scale_x
        )
        var cy = min_y + SIMD[float_type, 1](row) * SIMD[
            float_type, simd_width
        ](scale_y)
        var c = ComplexSIMD[float_type, simd_width](cx, cy)
        out.store[width=simd_width](
            (row, col), mandelbrot_kernel[simd_width](c)
        )

    # We vectorize the call to compute_vector where call gets a chunk of
    # pixels.
    vectorize[simd_width_of[float_type]()](width, compute_vector)


def run_mandelbrot(ctx: DeviceContext) raises:
    var out_host = alloc[Scalar[int_type]](width * height)

    var out_device = ctx.enqueue_create_buffer[int_type](width * height)

    @always_inline
    def run_mandelbrot(ctx: DeviceContext) raises {imm}:
        ctx.enqueue_function[mandelbrot](
            out_device,
            grid_dim=(ceildiv(height, BLOCK_SIZE),),
            block_dim=(BLOCK_SIZE,),
        )

    run_mandelbrot(ctx)  # Warmup
    print(
        "Computation took:",
        Float64(ctx.execution_time(run_mandelbrot, 1)) / 1_000_000_000.0,
    )

    ctx.enqueue_copy(out_host, out_device)

    ctx.synchronize()

    var accum = Scalar[int_type](0)
    for i in range(width):
        for j in range(height):
            accum += out_host[i * width + j]

    comptime ref_result = 4687767697 if float_type == DType.float64 else 4687810683
    assert_equal(Scalar[int_type](ref_result), accum)


def main() raises:
    with DeviceContext() as ctx:
        run_mandelbrot(ctx)
