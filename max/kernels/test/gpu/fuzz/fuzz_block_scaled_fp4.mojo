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
# Fuzz target: block-scaled FP4 SM100 matmul (small_bn).
#
# A thin wrapper over the testbed (which does the packed-FP4 + scale-factor setup
# and an internal reference check). We bake the small_bn config (mma_n=24,
# N=1024, K=1056) and fuzz M; run under memcheck+pool-off, a small_bn boundary
# over-read of the scale-factor tensor surfaces from generated M values. B200/
# SM100 only. N/K/mma_n are compile-time (`-D N=.. -D K=.. -D mma_n=..`).

from std.sys import size_of
from std.random import seed
from max.gpu.host import DeviceContext
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from std.sys.defines import get_defined_int
from std.utils.index import Index
from std.utils.static_tuple import StaticTuple

from layout import Idx
from linalg.fp4_utils import NVFP4_SF_DTYPE, NVFP4_SF_VECTOR_SIZE
from linalg.matmul.gpu.sm100.testbed_block_scaled_fp4 import (
    test_blackwell_block_scaled_matmul_tma_umma_warp_specialized,
)

from _fuzz import boundary_int, collect_args, flag, flag_int

comptime a_dtype = DType.uint8  # packed FP4 (placeholder, per the testbed)
comptime out_dtype = DType.bfloat16
comptime scales_dtype = NVFP4_SF_DTYPE
comptime SF_VECTOR_SIZE = NVFP4_SF_VECTOR_SIZE
comptime swizzle = TensorMapSwizzle.SWIZZLE_128B
comptime BK = swizzle.bytes() // size_of[a_dtype]()
comptime MMA_K = 32
comptime mma_n = get_defined_int["mma_n", 24]()  # small_bn boundary
comptime block_tile_shape = Index(128, mma_n, BK)
comptime umma_shape = Index(128, mma_n, MMA_K)
comptime N = get_defined_int["N", 1024]()
comptime K = get_defined_int["K", 1056]()  # 1024 + 32

comptime fuzz_seed = get_defined_int["fuzz_seed", 12345]()
comptime budget = get_defined_int["budget", 12]()


@fieldwise_init
struct CaseSpec(Copyable, Movable, Writable):
    var m: Int

    def write_to(self, mut writer: Some[Writer]):
        writer.write("m=", self.m, " N=", N, " K=", K, " mma_n=", mma_n)


def gen_specs(n: Int) -> List[CaseSpec]:
    var specs = List[CaseSpec]()
    for _ in range(n):
        specs.append(CaseSpec(boundary_int(1, 2048, 128)))
    return specs^


def run_one_case(ctx: DeviceContext, spec: CaseSpec) raises:
    # The testbed allocates the packed-FP4 operands + scale factors, runs the
    # small_bn kernel, and checks an internal reference; we only vary M.
    test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
        a_dtype,
        a_dtype,
        out_dtype,
        scales_dtype,
        block_tile_shape,
        umma_shape,
        cluster_shape=StaticTuple[Int32, 3](1, 1, 1),
        cta_group=1,
        a_swizzle=swizzle,
        b_swizzle=swizzle,
        block_swizzle_size=8,
        SF_VECTOR_SIZE=SF_VECTOR_SIZE,
        is_small_bn=True,
    ](ctx, spec.m, Idx[N], Idx[K])


def main() raises:
    var args = collect_args()
    var mode = flag(args, "--mode", "fuzz")
    var the_seed = flag_int(args, "--seed", fuzz_seed)
    var the_budget = flag_int(args, "--budget", budget)
    seed(the_seed)

    if mode == "list-specs":
        var specs = gen_specs(the_budget)
        for i in range(len(specs)):
            print("FUZZ_SPEC idx=", i, "m=", specs[i].m)
        return

    if mode == "single":
        var m = flag_int(args, "--m", 100)
        print("FUZZ_SINGLE m=", m, "N=", N, "K=", K, "mma_n=", mma_n)
        with DeviceContext() as ctx:
            run_one_case(ctx, CaseSpec(m))
        print("FUZZ_RESULT verdict=PASS")
        return

    print(
        "=== fuzz_block_scaled_fp4 seed=",
        the_seed,
        "budget=",
        the_budget,
        "N=",
        N,
        "K=",
        K,
        "mma_n=",
        mma_n,
        "===",
    )
    var specs = gen_specs(the_budget)
    with DeviceContext() as ctx:
        for i in range(len(specs)):
            print("case", i, ":", specs[i])
            run_one_case(ctx, specs[i])
    print("=== done:", len(specs), "cases ===")
