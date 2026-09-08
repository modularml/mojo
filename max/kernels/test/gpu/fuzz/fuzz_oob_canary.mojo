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
# OOB canary fuzz target -- the positive control for the whole fuzz pipeline
# (see gpu-kernels-fuzzing-design.md). The kernel writes one element `overflow`
# positions past the end of its buffer, i.e. a *deliberate* out-of-bounds write
# iff `overflow > 0`. It exists to prove the oracle dispatch works end to end:
#
#   * --oracle redzone / memcheck  -> DETECTS the OOB write   (verdict FAIL)
#   * --oracle diff                -> does NOT (pool masks it) (verdict PASS)
#
# i.e. it is the fuzz-pipeline analogue of positive_control_memcheck_oob, but
# driven by the standard list-specs/single/fuzz modes so the orchestrator
# exercises the same path it uses for real kernels. A generated batch mixes
# overflow==0 (clean) and overflow>0 (OOB) cases, so the fuzzer "finds" the
# planted bug from generated specs.

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.math import ceildiv
from std.random import random_ui64, seed
from std.sys.defines import get_defined_int

from _fuzz import boundary_int, collect_args, flag, flag_int

comptime fuzz_seed = get_defined_int["fuzz_seed", 12345]()
comptime budget = get_defined_int["budget", 16]()
comptime BLOCK = 256


def canary_kernel(
    dst: MutPointer[Float32, MutAnyOrigin], n_dev: Int32, overflow_dev: Int32
):
    # `Int` is not device-passable; widen the fixed-width args.
    var n = Int(n_dev)
    var overflow = Int(overflow_dev)
    var gid = global_idx.x
    if gid < n:
        dst[gid] = Float32(gid)
    # One thread writes `overflow` elements past the end of the n-element
    # buffer: an OOB write iff overflow > 0 (lands in the redzone / unmapped).
    if gid == 0 and overflow > 0:
        dst[n - 1 + overflow] = Float32(1.0)


@fieldwise_init
struct CanarySpec(Copyable, Movable, Writable):
    var num_elems: Int
    var overflow: Int

    def write_to(self, mut writer: Some[Writer]):
        writer.write("num_elems=", self.num_elems, " overflow=", self.overflow)


def gen_specs(n: Int) -> List[CanarySpec]:
    """Boundary-aware sizes with a mix of clean (overflow==0) and OOB cases."""
    var specs = List[CanarySpec]()
    for _ in range(n):
        var ne = boundary_int(1, 4096, 256)
        var ov = Int(random_ui64(0, 3))  # 0 = clean; 1..3 = OOB write
        specs.append(CanarySpec(ne, ov))
    return specs^


def run_one_case(ctx: DeviceContext, spec: CanarySpec) raises:
    var n = spec.num_elems
    var dst = ctx.enqueue_create_buffer[.float32](n)
    ctx.enqueue_function[canary_kernel](
        dst,
        Int32(n),
        Int32(spec.overflow),
        grid_dim=ceildiv(n, BLOCK),
        block_dim=BLOCK,
    )
    ctx.synchronize()
    _ = dst


def main() raises:
    var args = collect_args()
    var mode = flag(args, "--mode", "fuzz")
    var the_seed = flag_int(args, "--seed", fuzz_seed)
    var the_budget = flag_int(args, "--budget", budget)
    seed(the_seed)

    if mode == "list-specs":
        var specs = gen_specs(the_budget)
        for i in range(len(specs)):
            print(
                "FUZZ_SPEC idx=",
                i,
                "num_elems=",
                specs[i].num_elems,
                "overflow=",
                specs[i].overflow,
            )
        return

    if mode == "single":
        var ne = flag_int(args, "--num_elems", 64)
        var ov = flag_int(args, "--overflow", 0)
        print("FUZZ_SINGLE num_elems=", ne, "overflow=", ov)
        with DeviceContext() as ctx:
            run_one_case(ctx, CanarySpec(ne, ov))
        print("FUZZ_RESULT verdict=PASS")
        return

    print("=== fuzz_oob_canary seed=", the_seed, "budget=", the_budget, "===")
    var specs = gen_specs(the_budget)
    with DeviceContext() as ctx:
        for i in range(len(specs)):
            print("case", i, ":", specs[i])
            run_one_case(ctx, specs[i])
    print("=== done:", len(specs), "cases ===")
