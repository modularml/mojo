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
# Fuzz target: layer_norm (`layer_norm`) (see gpu-kernels-fuzzing-design.md).
#
# Fully runtime-shapeable: fuzzes (rows, cols). Memory-safety oracle by default;
# with --check, an FP64 CPU reference (out = (x-mean)*rsqrt(var+eps)*gamma+beta,
# mean/var over the inner axis) drives the numerical (`ref`) oracle.

from std.math import sqrt
from std.random import rand, seed
from std.sys.defines import get_defined_int

from max.gpu.host import DeviceContext
from layout import Coord, TileTensor, row_major
from nn.normalization import *
from std.utils.index import Index

from _fuzz import boundary_int, collect_args, flag, flag_int, numeric_check

comptime ln_type = DType.float32
comptime ln_rank = 2
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


def _layer_norm_ref[
    dtype: DType
](
    src: Span[Scalar[dtype], _],
    gamma: Span[Scalar[dtype], _],
    beta: Span[Scalar[dtype], _],
    dst: Span[mut=True, Scalar[dtype], _],
    rows: Int,
    cols: Int,
    eps: Float64,
):
    """FP64 CPU layer_norm: out = (x-mean)*rsqrt(var+eps)*gamma + beta."""
    for r in range(rows):
        var base = r * cols
        var mean = Float64(0)
        for c in range(cols):
            mean += src[base + c].cast[.float64]()
        mean /= Float64(cols)
        var var_ = Float64(0)
        for c in range(cols):
            var d = src[base + c].cast[.float64]() - mean
            var_ += d * d
        var_ /= Float64(cols)
        var norm = 1.0 / sqrt(var_ + eps)
        for c in range(cols):
            var x = src[base + c].cast[.float64]()
            var g = gamma[c].cast[.float64]()
            var b = beta[c].cast[.float64]()
            dst[base + c] = (((x - mean) * norm) * g + b).cast[dtype]()


def run_one_case[
    dtype: DType = ln_type
](ctx: DeviceContext, spec: CaseSpec, check: Bool = False) raises:
    var rows = spec.rows
    var cols = spec.cols
    var shape = Index(rows, cols)

    var data_h = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var gamma_h = ctx.enqueue_create_host_buffer[dtype](cols)
    var beta_h = ctx.enqueue_create_host_buffer[dtype](cols)
    rand(data_h.as_span())
    rand(gamma_h.as_span())
    rand(beta_h.as_span())

    var data_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    # Distinct output buffer: input_fn (reads) and output_fn (writes) are
    # separate value-closure args, so they must reference distinct buffer
    # origins (writing in place into `data_d` would alias the read; mirrors
    # test_layer_norm.mojo's `run_layer_norm_gpu`).
    var out_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var gamma_d = ctx.enqueue_create_buffer[dtype](cols)
    var beta_d = ctx.enqueue_create_buffer[dtype](cols)
    ctx.enqueue_copy(data_d, data_h)
    ctx.enqueue_copy(gamma_d, gamma_h)
    ctx.enqueue_copy(beta_d, beta_h)

    var param_shape = Index(cols)
    var data_buf = TileTensor(data_d, row_major(Coord(shape)))
    var out_buf = TileTensor(out_d, row_major(Coord(shape)))
    var gamma = TileTensor(gamma_d, row_major(Coord(param_shape)))
    var beta = TileTensor(beta_d, row_major(Coord(param_shape)))
    var epsilon = Float32(1e-5)

    @always_inline
    def input_fn[
        width: Int, alignment: Int
    ](coords: Coord) {var data_buf} -> SIMD[dtype, width]:
        var idx = data_buf.layout(coords)
        return data_buf.raw_load[width=width, alignment=alignment](idx)

    @always_inline
    def output_fn[
        width: SIMDLength, alignment: Int
    ](coords: Coord, val: SIMD[dtype, width]) {var out_buf}:
        var idx = out_buf.layout(coords)
        out_buf.raw_store[width=width, alignment=alignment](
            idx, rebind[SIMD[dtype, width]](val)
        )

    layer_norm[dtype, 2, target="gpu"](
        input_fn,
        output_fn,
        Coord(shape),
        Int(cols),
        gamma,
        beta,
        epsilon.cast[dtype](),
        ctx,
    )
    ctx.synchronize()

    if check:
        var out_h = ctx.enqueue_create_host_buffer[dtype](rows * cols)
        ctx.enqueue_copy(out_h, out_d)
        ctx.synchronize()
        var ref_h = ctx.enqueue_create_host_buffer[dtype](rows * cols)
        _layer_norm_ref(
            data_h.as_span(),
            gamma_h.as_span(),
            beta_h.as_span(),
            ref_h.as_span(),
            rows,
            cols,
            epsilon.cast[.float64](),
        )
        if not numeric_check(out_h.as_span(), ref_h.as_span()):
            raise Error("layer_norm numeric mismatch")

    _ = data_d
    _ = out_d
    _ = gamma_d
    _ = beta_d
    _ = data_buf
    _ = out_buf


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

    print("=== fuzz_layer_norm seed=", the_seed, "budget=", the_budget, "===")
    var specs = gen_specs(the_budget)
    with DeviceContext() as ctx:
        for i in range(len(specs)):
            print("case", i, ":", specs[i])
            run_one_case(ctx, specs[i], check)
    print("=== done:", len(specs), "cases ===")
