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
#
# Fuzz target: rms_norm (`rms_norm_gpu`) (see gpu-kernels-fuzzing-design.md).
#
# Fully runtime-shapeable: fuzzes (rows, cols), with cols swept across boundary
# classes around WARP_SIZE*simd_width (the warp-tiling vs block dispatch pivot).
# In-place via capturing input/output closures. Memory-safety oracle.

from std.math import sqrt
from std.random import rand, seed
from std.sys.defines import get_defined_int

from max.gpu.host import DeviceContext
from layout import Coord, TileTensor, row_major
from nn.normalization import *
from std.utils.index import Index

from _fuzz import boundary_int, collect_args, flag, flag_int, numeric_check

comptime rn_type = DType.float32
comptime rn_rank = 2
comptime TILE = 128  # warp-tiling vs block dispatch pivot on cols.
comptime fuzz_seed = get_defined_int["fuzz_seed", 12345]()
comptime budget = get_defined_int["budget", 16]()


@fieldwise_init
struct CaseSpec(Copyable, Movable, Writable):
    var rows: Int
    var cols: Int

    def write_to(self, mut writer: Some[Writer]):
        writer.write("rows=", self.rows, " cols=", self.cols)


def gen_specs(n: Int) -> List[CaseSpec]:
    var specs = List[CaseSpec]()
    for _ in range(n):
        specs.append(
            CaseSpec(boundary_int(1, 256, 8), boundary_int(1, 8192, TILE))
        )
    return specs^


def _rms_norm_ref(
    src: Span[Scalar[rn_type], _],
    gamma: Span[Scalar[rn_type], _],
    dst: Span[mut=True, Scalar[rn_type], _],
    rows: Int,
    cols: Int,
    eps: Float64,
    weight_offset: Float64,
):
    """FP64 CPU rms_norm matching the kernel: out = (x/rms)*(gamma+offset),
    rms = sqrt(mean(x^2) + eps)."""
    for r in range(rows):
        var base = r * cols
        var ss = Float64(0)
        for c in range(cols):
            var v = src[base + c].cast[.float64]()
            ss += v * v
        var rms = sqrt(ss / Float64(cols) + eps)
        for c in range(cols):
            var v = src[base + c].cast[.float64]()
            var g = gamma[c].cast[.float64]() + weight_offset
            dst[base + c] = ((v / rms) * g).cast[rn_type]()


def run_one_case(
    ctx: DeviceContext, spec: CaseSpec, check: Bool = False
) raises:
    var rows = spec.rows
    var cols = spec.cols
    var shape = Index(rows, cols)

    var data_h = ctx.enqueue_create_host_buffer[rn_type](rows * cols)
    var gamma_h = ctx.enqueue_create_host_buffer[rn_type](cols)
    rand(data_h.as_span())
    rand(gamma_h.as_span())

    var data_d = ctx.enqueue_create_buffer[rn_type](rows * cols)
    var gamma_d = ctx.enqueue_create_buffer[rn_type](cols)
    ctx.enqueue_copy(data_d, data_h)
    ctx.enqueue_copy(gamma_d, gamma_h)

    var data_buf = TileTensor(data_d, row_major(Coord(shape)))
    var gamma = TileTensor(gamma_d, row_major(Coord(Index(cols))))
    var epsilon = Float32(0.001)
    var weight_offset = Scalar[rn_type](0.0)

    @always_inline
    @__copy_capture(data_buf)
    @__parameter
    def input_fn[width: Int](coords: Coord) -> SIMD[rn_type, width]:
        var idx = data_buf.layout(coords)
        return data_buf.raw_load[width=width](idx)

    @always_inline
    @__copy_capture(data_buf)
    @__parameter
    def identity_output_fn[
        width: SIMDLength, alignment: Int
    ](coords: Coord, val: SIMD[rn_type, width]) -> None:
        var idx = data_buf.layout(coords)
        data_buf.raw_store[width=width, alignment=alignment](idx, val)

    rms_norm_gpu[
        rn_rank, input_fn, identity_output_fn, multiply_before_cast=True
    ](Coord(shape), gamma, epsilon, weight_offset, ctx)
    ctx.synchronize()

    if check:
        var out_h = ctx.enqueue_create_host_buffer[rn_type](rows * cols)
        ctx.enqueue_copy(out_h, data_d)
        ctx.synchronize()
        var ref_h = ctx.enqueue_create_host_buffer[rn_type](rows * cols)
        _rms_norm_ref(
            data_h.as_span(),
            gamma_h.as_span(),
            ref_h.as_span(),
            rows,
            cols,
            epsilon.cast[.float64](),
            weight_offset.cast[.float64](),
        )
        if not numeric_check(out_h.as_span(), ref_h.as_span()):
            raise Error("rms_norm numeric mismatch")

    _ = data_d
    _ = gamma_d
    _ = data_buf


def main() raises:
    var args = collect_args()
    var mode = flag(args, "--mode", "fuzz")
    var the_seed = flag_int(args, "--seed", fuzz_seed)
    var the_budget = flag_int(args, "--budget", budget)
    var check = flag_int(args, "--check", 0) == 1
    seed(the_seed)

    if mode == "list-specs":
        var specs = gen_specs(the_budget)
        for i in range(len(specs)):
            print(
                "FUZZ_SPEC idx=",
                i,
                "rows=",
                specs[i].rows,
                "cols=",
                specs[i].cols,
            )
        return

    if mode == "single":
        var rows = flag_int(args, "--rows", 8)
        var cols = flag_int(args, "--cols", 128)
        print("FUZZ_SINGLE rows=", rows, "cols=", cols)
        with DeviceContext() as ctx:
            run_one_case(ctx, CaseSpec(rows, cols), check)
        print("FUZZ_RESULT verdict=PASS")
        return

    print("=== fuzz_rms_norm seed=", the_seed, "budget=", the_budget, "===")
    var specs = gen_specs(the_budget)
    with DeviceContext() as ctx:
        for i in range(len(specs)):
            print("case", i, ":", specs[i])
            run_one_case(ctx, specs[i], check)
    print("=== done:", len(specs), "cases ===")
