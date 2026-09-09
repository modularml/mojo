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
# Fuzz target: GEMV split-K (`_matmul_gpu` -> SM100 GEMV_SPLIT_K).
#
# This pins the production FP32 decode router/gate GEMM (KERN-3076): on SM100
# `matmul_dispatch_sm100` sends transpose_b FP32 with static_N<=256,
# static_K>=2048, M<=64 to GEMV_SPLIT_K with an M-adaptive tile_m (buckets:
# M<=6 -> 1, M<=12 -> 2, M<=64 -> 4; sm100_structured/default/dispatch.mojo).
# GEMV_SPLIT_K partitions K across warps and reduces per output row via a
# warp.sum + fixed-order shared-memory tree WITHIN one block (gemv.mojo:488-510)
# -- no cross-block atomics -- so the result must be bit-stable run-to-run.
#
# M is the runtime fuzz axis (crosses the tile_m bucket boundaries); N, K are
# comptime (the tuned path reads them from the STATIC shape). N=128, K=6144 is
# the decode router shape.
#
# We call `_matmul_gpu` with `use_tf32=False`: for M<=64 this selects the exact
# same GEMV_SPLIT_K kernel as the default, but it additionally arms
# `matmul_dispatch_sm100`'s `has_precise_f32_gemv` comptime assert. If a
# dispatch-heuristic edit (or a `-D N=` / `-D K=` override) ever moved this
# shape off the split-K path, the target fails to COMPILE rather than silently
# pinning a different kernel -- the "assert the selected algorithm" guard here.
#
# `determinism` oracle (--rerun N): re-launch the SAME input N times and require
# bit-exact output (atol=rtol=0). A divergence is a real race / order-dependent
# reduction, not a legitimate reduction-order FP wobble.

from std.random import rand, seed
from std.sys.defines import get_defined_int

from max.gpu.host import DeviceContext
from layout import Coord, Idx, TileTensor, row_major
from linalg.matmul.gpu import _matmul_gpu

from _fuzz import boundary_int, collect_args, flag, flag_int, numeric_check

comptime dtype = DType.float32  # FP32 router/gate GEMM (KERN-3076 split-K).
comptime N = get_defined_int[
    "N", 128
]()  # COMPTIME (tuned SM100 reads static N)
comptime K = get_defined_int[
    "K", 6144
]()  # COMPTIME; >= 2048 for the split-K gate.
comptime TILE = 12  # tile_m bucket boundary (M<=12 -> tile_m=2, M<=64 -> 4).
comptime fuzz_seed = get_defined_int["fuzz_seed", 12345]()
comptime budget = get_defined_int["budget", 16]()


@fieldwise_init
struct CaseSpec(Copyable, Movable, Writable):
    var m: Int

    def write_to(self, mut writer: Some[Writer]):
        writer.write("m=", self.m, " N=", N, " K=", K)


def gen_specs(n: Int) -> List[CaseSpec]:
    var specs = List[CaseSpec]()
    for _ in range(n):
        # M<=64 keeps the shape on the GEMV_SPLIT_K path (M=65.. exits to the
        # tile GEMM). Boundary-biased around the tile_m=2/4 bucket flip.
        specs.append(CaseSpec(boundary_int(1, 64, TILE)))
    return specs^


def run_one_case(ctx: DeviceContext, spec: CaseSpec, rerun: Int = 0) raises:
    var m = spec.m
    var a_size = m * K
    var b_size = N * K  # transpose_b => B is [N, K]
    var c_size = m * N

    var a_host = ctx.enqueue_create_host_buffer[dtype](a_size)
    var b_host = ctx.enqueue_create_host_buffer[dtype](b_size)
    rand(a_host.as_span())
    rand(b_host.as_span())

    var a_dev = ctx.enqueue_create_buffer[dtype](a_size)
    var b_dev = ctx.enqueue_create_buffer[dtype](b_size)
    var c_dev = ctx.enqueue_create_buffer[dtype](c_size)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    # M runtime (bare Int); N, K comptime (Idx[...]) so the tuned split-K path
    # engages (it keys on the static N/K).
    var a = TileTensor(a_dev, row_major(Coord(m, Idx[K])))
    var b = TileTensor(b_dev, row_major(Coord(Idx[N], Idx[K])))
    var c = TileTensor(c_dev, row_major(Coord(m, Idx[N])))

    # use_tf32=False pins the IEEE-fp32 split-K GEMV and makes the dispatcher's
    # `has_precise_f32_gemv` comptime assert enforce it (see module header).
    _matmul_gpu[use_tensor_core=True, transpose_b=True, use_tf32=False](
        c, a, b, ctx
    )
    ctx.synchronize()

    if rerun > 0:
        # Run-to-run determinism: re-launch the SAME input `rerun` times and
        # require bit-exact output. The K reduction is a fixed-order in-block
        # tree, so any difference is a real race / nondeterminism.
        var first_h = ctx.enqueue_create_host_buffer[dtype](c_size)
        ctx.enqueue_copy(first_h, c_dev)
        ctx.synchronize()
        for _ in range(rerun - 1):
            _matmul_gpu[use_tensor_core=True, transpose_b=True, use_tf32=False](
                c, a, b, ctx
            )
            ctx.synchronize()
            var rep_h = ctx.enqueue_create_host_buffer[dtype](c_size)
            ctx.enqueue_copy(rep_h, c_dev)
            ctx.synchronize()
            if not numeric_check(
                rep_h.as_span(), first_h.as_span(), atol=0.0, rtol=0.0
            ):
                raise Error("GEMV split-K run-to-run nondeterminism (rerun)")

    _ = a_dev
    _ = b_dev
    _ = c_dev


def main() raises:
    var args = collect_args()
    var mode = flag(args, "--mode", "fuzz")
    var the_seed = flag_int(args, "--seed", fuzz_seed)
    var the_budget = flag_int(args, "--budget", budget)
    var rerun = flag_int(args, "--rerun", 0)
    seed(the_seed)

    if mode == "list-specs":
        var specs = gen_specs(the_budget)
        for i in range(len(specs)):
            print("FUZZ_SPEC idx=", i, "m=", specs[i].m)
        return

    if mode == "single":
        var m = flag_int(args, "--m", 4)
        print("FUZZ_SINGLE m=", m, "N=", N, "K=", K)
        with DeviceContext() as ctx:
            run_one_case(ctx, CaseSpec(m), rerun=rerun)
        print("FUZZ_RESULT verdict=PASS")
        return

    print(
        "=== fuzz_gemv_split_k seed=",
        the_seed,
        "budget=",
        the_budget,
        "N=",
        N,
        "K=",
        K,
        "===",
    )
    var specs = gen_specs(the_budget)
    with DeviceContext() as ctx:
        for i in range(len(specs)):
            print("case", i, ":", specs[i])
            run_one_case(ctx, specs[i], rerun=rerun)
    print("=== done:", len(specs), "cases ===")
