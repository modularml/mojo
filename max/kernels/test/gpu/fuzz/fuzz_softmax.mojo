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
# Fuzz target: softmax (`_softmax_gpu`) (see gpu-kernels-fuzzing-design.md).
#
# Fully runtime-shapeable: fuzzes a rank-3 shape, with the inner (softmax) axis
# swept across boundary classes around WARP_SIZE (the warp-vs-block dispatch
# pivot). Memory-safety oracle (memcheck / redzone); values are irrelevant.

from std.math import exp
from std.random import rand, random_ui64, seed
from std.sys.defines import get_defined_int

from max.gpu.host import DeviceContext
from layout import (
    Coord,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    coord_to_index_list,
    row_major,
)
from nn.softmax import _softmax_gpu
from std.utils import IndexList
from std.utils.numerics import isinf, isnan

from _fuzz import (
    boundary_int,
    collect_args,
    fill_by_dist,
    fill_with_specials,
    flag,
    flag_int,
    numeric_check,
    value_dist_name,
)

comptime sm_type = DType.float32
comptime sm_rank = 3
comptime WARP = 32  # warp-vs-block dispatch pivot on the inner axis.
comptime fuzz_seed = get_defined_int["fuzz_seed", 12345]()
comptime budget = get_defined_int["budget", 16]()


@fieldwise_init
struct CaseSpec(Copyable, Movable, Writable):
    var d0: Int
    var d1: Int
    var inner: Int  # the softmax axis (rank - 1)
    var dist: Int  # value-distribution id (see _fuzz: VD_*)

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "d0=",
            self.d0,
            " d1=",
            self.d1,
            " inner=",
            self.inner,
            " dist=",
            value_dist_name(self.dist),
        )


def gen_specs(n: Int) -> List[CaseSpec]:
    var specs = List[CaseSpec]()
    for _ in range(n):
        var d0 = boundary_int(1, 32, 8)
        var d1 = boundary_int(1, 64, 8)
        var inner = boundary_int(1, 4096, WARP)
        # Fuzz the value distribution too (softmax is overflow-safe, so its FP64
        # reference stays robust across these). Bias to uniform; the rest spread
        # over normal/sparse/large/all-equal. NaN/Inf "specials" (id 5) are left
        # out of the auto-mix (NaN-vs-reference is a separate contract) but stay
        # reachable via `--dist 5`.
        var dist = 0 if Int(random_ui64(0, 2)) != 0 else Int(random_ui64(0, 4))
        specs.append(CaseSpec(d0, d1, inner, dist))
    return specs^


def _softmax_ref(
    src: Span[Scalar[sm_type], _],
    dst: Span[mut=True, Scalar[sm_type], _],
    rows: Int,
    inner: Int,
):
    """FP64 CPU softmax over the inner axis (the higher-precision oracle)."""
    for r in range(rows):
        var base = r * inner
        var m = src[base].cast[.float64]()
        for c in range(1, inner):
            var v = src[base + c].cast[.float64]()
            if v > m:
                m = v
        var s = Float64(0)
        for c in range(inner):
            s += exp(src[base + c].cast[.float64]() - m)
        for c in range(inner):
            var e = exp(src[base + c].cast[.float64]() - m)
            dst[base + c] = (e / s).cast[sm_type]()


def _check_softmax_contract(vals: Span[Scalar[sm_type], _]) -> Bool:
    """softmax contract, robust to NaN/Inf inputs: every output is NaN (legit
    propagation) or in [0, 1]; it is never +-Inf and never out of range. A
    violation (an Inf, a negative, or a >1 probability) is a real bug regardless
    of input -- unlike a tolerance diff, this has no FP-divergence false positive.
    """
    for i in range(len(vals)):
        var v = vals[i]
        if Bool(isnan(v)):
            continue  # NaN propagation from a NaN/Inf input is allowed
        if (
            Bool(isinf(v))
            or Bool(v < Scalar[sm_type](-0.001))
            or Bool(v > Scalar[sm_type](1.001))
        ):
            print("FUZZ_CONTRACT_FAIL idx=", i, "val=", v)
            return False
    return True


def run_one_case(
    ctx: DeviceContext,
    spec: CaseSpec,
    check: Bool = False,
    contract: Bool = False,
) raises:
    var shape = IndexList[sm_rank](spec.d0, spec.d1, spec.inner)
    var length = shape.flattened_length()

    var in_host = ctx.enqueue_create_host_buffer[sm_type](length)
    # Contract mode injects NaN/Inf/+-0/large specials to drive the
    # finite/propagation contract; otherwise use the spec's value distribution.
    if contract:
        fill_with_specials(in_host.as_span(), density=0.3)
    else:
        fill_by_dist(in_host.as_span(), spec.dist)

    var in_dev = ctx.enqueue_create_buffer[sm_type](length)
    var out_dev = ctx.enqueue_create_buffer[sm_type](length)
    ctx.enqueue_copy(in_dev, in_host)

    comptime layout_dyn = Layout.row_major[sm_rank]()
    var in_tt = LayoutTensor[sm_type, layout_dyn](
        in_dev.unsafe_ptr(), RuntimeLayout[layout_dyn].row_major(shape)
    )

    @__parameter
    @__copy_capture(in_tt)
    def input_fn_device[
        _simd_width: Int
    ](coords: Coord) -> SIMD[sm_type, _simd_width]:
        return in_tt.load[width=_simd_width](coord_to_index_list(coords))

    _softmax_gpu[sm_type, 1, sm_rank, input_fn_device](
        Coord(shape),
        TileTensor(out_dev, row_major(Coord(shape))),
        sm_rank - 1,  # axis must be the inner dim
        ctx,
    )
    ctx.synchronize()

    if contract:
        var out_h = ctx.enqueue_create_host_buffer[sm_type](length)
        ctx.enqueue_copy(out_h, out_dev)
        ctx.synchronize()
        if not _check_softmax_contract(out_h.as_span()):
            raise Error("softmax finite/range contract violated")
    elif check:
        var out_h = ctx.enqueue_create_host_buffer[sm_type](length)
        var ref_h = ctx.enqueue_create_host_buffer[sm_type](length)
        ctx.enqueue_copy(out_h, out_dev)
        ctx.synchronize()
        _softmax_ref(
            in_host.as_span(), ref_h.as_span(), spec.d0 * spec.d1, spec.inner
        )
        if not numeric_check(out_h.as_span(), ref_h.as_span()):
            raise Error("softmax numeric mismatch")

    _ = in_dev
    _ = out_dev
    _ = in_tt


def main() raises:
    var args = collect_args()
    var mode = flag(args, "--mode", "fuzz")
    var the_seed = flag_int(args, "--seed", fuzz_seed)
    var the_budget = flag_int(args, "--budget", budget)
    var check = flag_int(args, "--check", 0) == 1
    var contract = flag_int(args, "--contract", 0) == 1
    seed(the_seed)

    if mode == "list-specs":
        var specs = gen_specs(the_budget)
        for i in range(len(specs)):
            print(
                "FUZZ_SPEC idx=",
                i,
                "d0=",
                specs[i].d0,
                "d1=",
                specs[i].d1,
                "inner=",
                specs[i].inner,
                "dist=",
                specs[i].dist,
            )
        return

    if mode == "single":
        var d0 = flag_int(args, "--d0", 2)
        var d1 = flag_int(args, "--d1", 8)
        var inner = flag_int(args, "--inner", 128)
        var dist = flag_int(args, "--dist", 0)
        print("FUZZ_SINGLE d0=", d0, "d1=", d1, "inner=", inner, "dist=", dist)
        with DeviceContext() as ctx:
            run_one_case(ctx, CaseSpec(d0, d1, inner, dist), check, contract)
        print("FUZZ_RESULT verdict=PASS")
        return

    print("=== fuzz_softmax seed=", the_seed, "budget=", the_budget, "===")
    var specs = gen_specs(the_budget)
    with DeviceContext() as ctx:
        for i in range(len(specs)):
            print("case", i, ":", specs[i])
            run_one_case(ctx, specs[i], check, contract)
    print("=== done:", len(specs), "cases ===")
